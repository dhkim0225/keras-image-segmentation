# NOTICE !!
# OUTPUT STRIDE == 16
# BASE NN == MODIFIED ALIGNED XCEPTION

from keras.engine.topology import Layer
from keras.utils.conv_utils import conv_output_length
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import AveragePooling2D
from keras.layers.core import Activation, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add, concatenate
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf

from model.metrics import dice_coef, m_iou, precision, recall


class AtrousDepthwiseConv2D(Layer):
    def __init__(self, k_size, strides, padding, dilation_rate, kernel_regularizer=None, **kwargs):
        self.k_size = k_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.regularizer = kernel_regularizer

        super(AtrousDepthwiseConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.depthwise_kernel = self.add_weight(name='kernel',
                                                shape=(self.k_size[0], self.k_size[1], int(input_shape[3]), 1),
                                                initializer='glorot_normal',
                                                regularizer=self.regularizer,
                                                trainable=True)

        super(AtrousDepthwiseConv2D, self).build(input_shape)

    def call(self, x):
        return K.depthwise_conv2d(x, depthwise_kernel=self.depthwise_kernel,
                                  strides=self.strides,
                                  padding=self.padding,
                                  data_format="channels_last",
                                  dilation_rate=self.dilation_rate)

    def compute_output_shape(self, input_shape):
        rows = conv_output_length(input_shape[1], self.k_size[0], self.padding, self.strides[0])
        cols = conv_output_length(input_shape[2], self.k_size[1], self.padding, self.strides[1])
        output_shape = (input_shape[0], rows, cols, input_shape[3])
        return output_shape


def seperable_conv(input_tensor, filters, strides=(1, 1), dilation_rate=(1, 1)):
    x = AtrousDepthwiseConv2D((3, 3),
                              strides=strides,
                              padding='same',
                              dilation_rate=dilation_rate)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), use_bias=False)(x)
    x = BatchNormalization()(x)
    return Activation('relu')(x)


def entry_flow(input_tensor, filters):
    shortcut = Conv2D(filters, (1, 1), strides=(2, 2), padding='same')(input_tensor)
    shortcut = BatchNormalization()(shortcut)
    shortcut = Activation('relu')(shortcut)

    x = seperable_conv(input_tensor, filters=filters)
    x = seperable_conv(x, filters=filters)
    x = seperable_conv(x, filters=filters, strides=(2, 2))
    return add([shortcut, x])


def middle_flow(input_tensor):
    x = seperable_conv(input_tensor, filters=728)
    x = seperable_conv(x, filters=728)
    x = seperable_conv(x, filters=728)
    return add([input_tensor, x])


def exit_flow(input_tensor):
    shortcut = Conv2D(1024, (1, 1), padding='same')(input_tensor)
    shortcut = BatchNormalization()(shortcut)
    shortcut = Activation('relu')(shortcut)

    x = seperable_conv(input_tensor, filters=728)
    x = seperable_conv(x, filters=1024)
    x = seperable_conv(x, filters=1024, dilation_rate=(2, 2))

    x = add([shortcut, x])
    x = seperable_conv(x, filters=1536)
    x = seperable_conv(x, filters=1536)
    return seperable_conv(x, filters=2048)


def aspp_module(input_tensor, rates=(6, 12, 18)):
    concat_list = []

    # image_pooling
    h = input_tensor.shape[1].value
    w = input_tensor.shape[2].value
    x = AveragePooling2D(pool_size=(h, w))(input_tensor)
    x = Conv2D(256, (1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Lambda(lambda x: tf.image.resize_images(x, (h, w)))(x)

    concat_list.append(x)

    # 1x1
    x = Conv2D(256, (1, 1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    concat_list.append(x)

    # 6x6, 12x12, 18x18
    for rate in rates:
        x = seperable_conv(input_tensor, filters=256, dilation_rate=(rate, rate))
        concat_list.append(x)

    x = concatenate(concat_list)
    x = Conv2D(256, (1, 1))(x)
    x = BatchNormalization()(x)
    return Activation('relu')(x)


def deeplab_v3_plus(num_classes, input_shape, lr_init, lr_decay):
    img_input = Input(input_shape)

    # entry flow
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, kernel_size=3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = entry_flow(x, 128)
    shortcut = Conv2D(256, (1, 1))(x)
    shortcut = BatchNormalization()(shortcut)
    shortcut = Activation('relu')(shortcut)

    x = entry_flow(x, 256)
    x = entry_flow(x, 728)

    # middle flow
    for i in range(16):
        x = middle_flow(x)

    # exit flow
    x = exit_flow(x)
    x = aspp_module(x)
    x = Dropout(0.1)(x)

    # decoder
    x = Conv2DTranspose(256, kernel_size=(8, 8), strides=(4, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = add([x, shortcut])

    x = Conv2DTranspose(256, kernel_size=(8, 8), strides=(4, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_classes, (1, 1), activation='softmax')(x)

    model = Model(img_input, x)
    model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef, m_iou, precision, recall])

    return model
