# -*- coding: utf-8 -*-
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Reshape, Permute, Dense, Activation, Flatten, Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPool2D, BatchNormalization
from keras.layers import Convolution2D, UpSampling2D, AtrousConvolution2D, ZeroPadding2D, Lambda, Conv2DTranspose
from keras.layers import multiply, add, concatenate
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras.utils import np_utils, conv_utils

from os.path import splitext, join, isfile
from os import environ
from math import ceil
import argparse
import numpy as np
from scipy import misc, ndimage


class CroppingLike2D(Layer):
    def __init__(self, target_shape, offset=None, data_format=None, **kwargs):
        super(CroppingLike2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.target_shape = target_shape
        if offset is None or offset == 'centered':
            self.offset = 'centered'
        elif isinstance(offset, int):
            self.offset = (offset, offset)
        elif hasattr(offset, '__len__'):
            if len(offset) != 2:
                raise ValueError('`offset` should have two elements. '
                                 'Found: ' + str(offset))
            self.offset = offset
        self.input_spec = InputSpec(ndim=4)


    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0],
                    input_shape[1],
                    self.target_shape[2],
                    self.target_shape[3])
        else:
            return (input_shape[0],
                    self.target_shape[1],
                    self.target_shape[2],
                    input_shape[3])


    def call(self, inputs):
        input_shape = K.int_shape(inputs)
        if self.data_format == 'channels_first':
            input_height = input_shape[2]
            input_width = input_shape[3]
            target_height = self.target_shape[2]
            target_width = self.target_shape[3]
            if target_height > input_height or target_width > input_width:
                raise ValueError('The Tensor to be cropped need to be smaller'
                                 'or equal to the target Tensor.')

            if self.offset == 'centered':
                self.offset = [
                        int((input_height - target_height) / 2),
                        int((input_width - target_width) / 2)]

            if self.offset[0] + target_height > input_height:
                raise ValueError('Height index out of range: '
                                 + str(self.offset[0] + target_height))
            if self.offset[1] + target_width > input_width:
                raise ValueError('Width index out of range:'
                                 + str(self.offset[1] + target_width))

            return inputs[:,
                          :,
                          self.offset[0]:self.offset[0] + target_height,
                          self.offset[1]:self.offset[1] + target_width]
        elif self.data_format == 'channels_last':
            input_height = input_shape[1]
            input_width = input_shape[2]
            target_height = self.target_shape[1]
            target_width = self.target_shape[2]
            if target_height > input_height or target_width > input_width:
                raise ValueError('The Tensor to be cropped need to be smaller'
                                 'or equal to the target Tensor.')

            if self.offset == 'centered':
                self.offset = [int((input_height - target_height) / 2),
                               int((input_width - target_width) / 2)]

            if self.offset[0] + target_height > input_height:
                raise ValueError('Height index out of range: '
                                 + str(self.offset[0] + target_height))
            if self.offset[1] + target_width > input_width:
                raise ValueError('Width index out of range:'
                                 + str(self.offset[1] + target_width))
            output = inputs[
                    :,
                    self.offset[0]:self.offset[0] + target_height,
                    self.offset[1]:self.offset[1] + target_width,
                    :]
            return output


class BilinearUpSampling2D(Layer):
    def __init__(self, target_shape=None,factor=None, data_format=None, **kwargs):
        # conmpute dataformat
        if data_format is None:
            data_format = K.image_data_format()
        assert data_format in {
            'channels_last', 'channels_first'}

        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        self.target_shape = target_shape
        self.factor = factor
        if self.data_format == 'channels_first':
            self.target_size = (target_shape[2], target_shape[3])
        elif self.data_format == 'channels_last':
            self.target_size = (target_shape[1], target_shape[2])
        super(BilinearUpSampling2D, self).__init__(**kwargs)


    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return (input_shape[0], self.target_size[0],
                    self.target_size[1], input_shape[3])
        else:
            return (input_shape[0], input_shape[1],
                    self.target_size[0], self.target_size[1])


    def call(self, inputs):
        return K.resize_images(inputs, self.factor, self.factor, self.data_format)


    def get_config(self):
        config = {'target_shape': self.target_shape,
                'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# residual module
def identity_block(input_tensor, kernel_size, filters, stage, block, dilation_rate=1, multigrid=[1,2,1], use_se=True):
    # conv filters
    filters1, filters2, filters3 = filters

    # compute dataformat
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # layer names
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # dilated rate
    if dilation_rate < 2:
        multigrid = [1, 1, 1]

    # forward
    x = Conv2D(
            filters1,
            (1, 1),
            name=conv_name_base + '2a',
            dilation_rate=dilation_rate*multigrid[0])(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2,
            kernel_size,
            padding='same',
            name=conv_name_base + '2b',
            dilation_rate=dilation_rate*multigrid[1])(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3,
            (1, 1),
            name=conv_name_base + '2c',
            dilation_rate=dilation_rate*multigrid[2])(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # stage 5 after squeeze and excinttation layer
    if use_se and stage < 5:
        se = _squeeze_excite_block(x, filters3, k=1, name=conv_name_base+'_se')
        x = multiply([x, se])
    x = add([x, input_tensor])
    x = Activation('relu')(x)

    return x


def _conv(**conv_params):
    # conv params
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault('dilation_rate',(1,1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    block = conv_params.setdefault("block", "assp")


    def f(input):
        conv = Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                dilation_rate=dilation_rate,
                kernel_initializer=kernel_initializer,activation='linear')(input)
        return conv
    return f


# Atrous Spatial Pyramid Pooling block
def aspp_block(x, num_filters=256, rate_scale=1, output_stride=16, input_shape=(512, 512, 3)):
    # compute dataformat
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # forward
    conv3_3_1 = ZeroPadding2D(padding=(6*rate_scale, 6*rate_scale))(x)
    conv3_3_1 = _conv(
            filters=num_filters,
            kernel_size=(3, 3),
            dilation_rate=(6*rate_scale, 6*rate_scale),
            padding='valid',
            block='assp_3_3_1_%s'%output_stride)(conv3_3_1)
    conv3_3_1 = BatchNormalization(axis=bn_axis, name='bn_3_3_1_%s'%output_stride)(conv3_3_1)

    conv3_3_2 = ZeroPadding2D(padding=(12*rate_scale, 12*rate_scale))(x)
    conv3_3_2 = _conv(
            filters=num_filters,
            kernel_size=(3, 3),
            dilation_rate=(12*rate_scale, 12*rate_scale),
            padding='valid',
            block='assp_3_3_2_%s'%output_stride)(conv3_3_2)
    conv3_3_2 = BatchNormalization(axis=bn_axis, name='bn_3_3_2_%s'%output_stride)(conv3_3_2)

    conv3_3_3 = ZeroPadding2D(padding=(18*rate_scale, 18*rate_scale))(x)
    conv3_3_3 = _conv(
            filters=num_filters,
            kernel_size=(3, 3),
            dilation_rate=(18*rate_scale, 18*rate_scale),
            padding='valid',
            block='assp_3_3_3_%s'%output_stride)(conv3_3_3)
    conv3_3_3 = BatchNormalization(axis=bn_axis, name='bn_3_3_3_%s'%output_stride)(conv3_3_3)

    conv1_1 = _conv(
            filters=num_filters,
            kernel_size=(1, 1),
            padding='same',
            block='assp_1_1_%s'%output_stride)(x)
    conv1_1 = BatchNormalization(axis=bn_axis, name='bn_1_1_%s'%output_stride)(conv1_1)

    # global_feat = AveragePooling2D((input_shape[0]/output_stride,input_shape[1]/output_stride))(x)
    # global_feat = _conv(filters=num_filters, kernel_size=(1, 1),padding='same')(global_feat)
    # global_feat = BatchNormalization()(global_feat)
    # global_feat = BilinearUpSampling2D((256,input_shape[0]/output_stride,input_shape[1]/output_stride),factor=input_shape[1]/output_stride)(global_feat)

    # channel merge
    y = merge([
        conv3_3_1,
        conv3_3_2,
        conv3_3_3,
        conv1_1,
        ],
        # global_feat,
        mode='concat', concat_axis=3)

    # y = _conv_bn_relu(filters=1, kernel_size=(1, 1),padding='same')(y)
    y = _conv(
            filters=256,
            kernel_size=(1, 1),
            padding='same',
            block='assp_out_%s'%output_stride)(y)
    y = BatchNormalization(axis=bn_axis,name='bn_out_%s'%output_stride)(y)

    return y


# residual module
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), dilation_rate=1, multigrid=[1, 2, 1], use_se=True):
    # conv filters
    filters1, filters2, filters3 = filters

    # compute dataformat
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # dailated rate
    if dilation_rate > 1:
        strides=(1,1)
    else:
        multigrid = [1, 1, 1]

    # forward
    x = Conv2D(
            filters1,
            (1, 1),
            strides=strides,
            name=conv_name_base + '2a',
            dilation_rate=dilation_rate*multigrid[0])(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(
            filters2,
            kernel_size,
            padding='same',
            name=conv_name_base + '2b',
            dilation_rate=dilation_rate*multigrid[1])(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(
            filters3,
            (1, 1),
            name=conv_name_base + '2c',
            dilation_rate=dilation_rate*multigrid[2])(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(
            filters3,
            (1, 1),
            strides=strides,
            name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    # stage after 5 squeeze and excittation
    if use_se and stage < 5:
        se = _squeeze_excite_block(x, filters3, k=1,name=conv_name_base+'_se')
        x = multiply([x, se])
    x = add([x, shortcut])
    x = Activation('relu')(x)

    return x


def duc(x, factor=8, output_shape=(512, 512, 1)):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    H, W, c, r = output_shape[0], output_shape[1], output_shape[2], factor
    h = H / r
    w = W / r
    x = Conv2D(
            c*r*r,
            (3, 3),
            padding='same',
            name='conv_duc_%s'%factor)(x)
    x = BatchNormalization(axis=bn_axis,name='bn_duc_%s'%factor)(x)
    x = Activation('relu')(x)
    x = Permute((3, 1, 2))(x)
    x = Reshape((c, r, r, h, w))(x)
    x = Permute((1, 4, 2, 5, 3))(x)
    x = Reshape((c, H, W))(x)
    x = Permute((2, 3, 1))(x)

    return x


# interpolation
def Interp(x, shape):
    from keras.backend import tf as ktf
    new_height, new_width = shape
    resized = ktf.image.resize_images(x, [int(new_height), int(new_width)], align_corners=True)
    return resized


# interpolation block
def interp_block(x, num_filters=512, level=1, input_shape=(512, 512, 3), output_stride=16):
    feature_map_shape = (input_shape[0] / output_stride, input_shape[1] / output_stride)

    # compute dataformat
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    if output_stride == 16:
        scale = 5
    elif output_stride == 8:
        scale = 10

    kernel = (level*scale, level*scale)
    strides = (level*scale, level*scale)
    global_feat = AveragePooling2D(kernel, strides=strides, name='pool_level_%s_%s'%(level, output_stride))(x)
    global_feat = _conv(
            filters=num_filters,
            kernel_size=(1, 1),
            padding='same',
            name='conv_level_%s_%s'%(level,output_stride))(global_feat)
    global_feat = BatchNormalization(axis=bn_axis, name='bn_level_%s_%s'%(level, output_stride))(global_feat)
    global_feat = Lambda(Interp, arguments={'shape': feature_map_shape})(global_feat)

    return global_feat


# squeeze and excitation function
def _squeeze_excite_block(input, filters, k=1, name=None):
    init = input
    se_shape = (1, 1, filters * k) if K.image_data_format() == 'channels_last' else (filters * k, 1, 1)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense((filters * k) // 16, activation='relu', kernel_initializer='he_normal', use_bias=False,name=name+'_fc1')(se)
    se = Dense(filters * k, activation='sigmoid', kernel_initializer='he_normal', use_bias=False,name=name+'_fc2')(se)
    return se


# pyramid pooling function
def pyramid_pooling_module(x, num_filters=512, input_shape=(512, 512, 3), output_stride=16, levels=[6, 3, 2, 1]):
    # compute data format
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    pyramid_pooling_blocks = [x]
    for level in levels:
        pyramid_pooling_blocks.append(
            interp_block(
                x,
                num_filters=num_filters,
                level=level,
                input_shape=input_shape,
                output_stride=output_stride))

    y = concatenate(pyramid_pooling_blocks)
    #y = merge(pyramid_pooling_blocks, mode='concat', concat_axis=3)
    y = _conv(
            filters=num_filters,
            kernel_size=(3, 3),
            padding='same',
            block='pyramid_out_%s'%output_stride)(y)
    y = BatchNormalization(axis=bn_axis, name='bn_pyramid_out_%s'%output_stride)(y)
    y = Activation('relu')(y)

    return y


def crop_deconv(
        classes,
        scale=1,
        kernel_size=(4, 4),
        strides=(2, 2),
        crop_offset='centered',
        weight_decay=0.,
        block_name='featx'):
    def f(x, y):
        def scaling(xx, ss=1):
            return xx * ss

        scaled = Lambda(
                scaling,
                arguments={'ss': scale},
                name='scale_{}'.format(block_name))(x)
        score = Conv2D(
                filters=classes,
                kernel_size=(1, 1),
                activation='linear',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(weight_decay),
                name='score_{}'.format(block_name))(scaled)

        if y is None:
            upscore = Conv2DTranspose(
                    filters=classes,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='valid',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay),
                    use_bias=False,
                    name='upscore_{}'.format(block_name))(score)
        else:
            crop = CroppingLike2D(
                    target_shape=K.int_shape(y),
                    offset=crop_offset,
                    name='crop_{}'.format(block_name))(score)
            merge = add([y, crop])
            upscore = Conv2DTranspose(
                    filters=classes,
                    kernel_size=kernel_size,
                    strides=strides, padding='valid',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay),
                    use_bias=False,
                    name='upscore_{}'.format(block_name))(merge)
        return upscore
    return f


def PSPNet50(
        input_shape=(512, 512, 3),
        n_labels=3,
        output_stride=16,
        num_blocks=4,
        multigrid=[1, 1, 1],
        levels=[6, 3, 2, 1],
        use_se=True,
        output_mode="sigmoid",
        upsample_type='transpose'):
    

    # Input shape
    img_input = Input(shape=input_shape)

    # compute input shape
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), use_se=use_se)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', use_se=use_se)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', use_se=use_se)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', use_se=use_se)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', use_se=use_se)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', use_se=use_se)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', use_se=use_se)

    if output_stride==8:
        rate_scale=2
    elif output_stride==16:
        rate_scale=1

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', dilation_rate=1*rate_scale, multigrid=multigrid, use_se=use_se)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', dilation_rate=1*rate_scale, multigrid=multigrid, use_se=use_se)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', dilation_rate=1*rate_scale, multigrid=multigrid, use_se=use_se)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', dilation_rate=1*rate_scale, multigrid=multigrid, use_se=use_se)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', dilation_rate=1*rate_scale, multigrid=multigrid, use_se=use_se)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', dilation_rate=1*rate_scale, multigrid=multigrid, use_se=use_se)

    init_rate = 2
    for block in range(4, num_blocks+1):
        if block==4:
            block=''
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a%s'%block,dilation_rate=init_rate*rate_scale, multigrid=multigrid, use_se=use_se)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b%s'%block,dilation_rate=init_rate*rate_scale, multigrid=multigrid, use_se=use_se)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c%s'%block,dilation_rate=init_rate*rate_scale, multigrid=multigrid, use_se=use_se)
        init_rate*=2

    # x1 = aspp_block(x,256,rate_scale=rate_scale,output_stride=output_stride,input_shape=input_shape)

    x = pyramid_pooling_module(x, num_filters=512, input_shape=input_shape, output_stride=output_stride, levels=levels)

    # x = merge([
    #         x1,
    #         x2,
    #         ], mode='concat', concat_axis=3)

    # upsample_type
    if upsample_type=='duc':
        x = duc(x, factor=output_stride, output_shape=(input_shape[0], input_shape[1], n_labels))
        out = _conv(filters=n_labels, kernel_size=(1, 1), padding='same', block='out_duc_%s'%output_stride)(x)

    elif upsample_type=='bilinear':
        x = _conv(filters=n_labels, kernel_size=(1, 1), padding='same', block='out_bilinear_%s'%output_stride)(x)
        out = BilinearUpSampling2D((n_labels, input_shape[0], input_shape[1]), factor=output_stride)(x)

    elif upsample_type=='transpose':
        out =  Conv2DTranspose(
                filters=n_labels,
                kernel_size=(output_stride*2, output_stride*2),
                strides=(output_stride, output_stride),
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=None,
                use_bias=False,
                name='upscore_{}'.format('out'))(x)

    out = Reshape((input_shape[0] * input_shape[1], n_labels), input_shape=(input_shape[0], input_shape[1], n_labels))(out)
    # default "softmax"
    out = Activation(output_mode)(out)

    model = Model(inputs=img_input, outputs=out)

    return model

if __name__ == '__main__':
    model = PSPNet50()
    # model.compile(optimizer='adam', loss='bce')
    model.summary()