import cv2
import numpy as np

from keras import backend as K

from model.unet import unet
from model.fcn import fcn_8s

# Use only 3 classes.
labels = ['background', 'person', 'car', 'road']

model = fcn_8s(input_shape=(256, 512, 3), num_classes=len(labels), init_lr=5e-3)
model.load_weights('./model_weight.h5')


def result_map_to_img(res_map):
    img = np.zeros((256, 512, 3), dtype=np.uint8)
    res_map = np.squeeze(res_map)

    argmax_idx = np.argmax(res_map, axis=2)

    # For np.where calculation.
    person = (argmax_idx == 1)
    car = (argmax_idx == 2)
    road = (argmax_idx == 3)

    img[:, :, 0] = np.where(person, 255, 0)
    img[:, :, 1] = np.where(car, 255, 0)
    img[:, :, 2] = np.where(road, 255, 0)

    return img


x_img = cv2.imread('./img/test.png')
cv2.imshow('x_img', x_img)
x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
x_img = np.expand_dims(x_img, 0)

res_map = model.predict(x_img)

res = result_map_to_img(res_map)

cv2.imshow('res', res)
cv2.waitKey(0)
