from __future__ import print_function

import argparse
import cv2
import numpy as np

from model.unet import unet
from model.fcn import fcn_8s
from model.pspnet import pspnet50


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


# Parse Options
parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", required=True, choices=['fcn', 'unet', 'pspnet'],
                    help="Model to test. 'fcn', 'unet', 'pspnet' is available.")
parser.add_argument("-P", "--img_path", required=True, help="The image path you want to test")

args = parser.parse_args()
model_name = args.model
img_path = args.img_path

# Use only 3 classes.
labels = ['background', 'person', 'car', 'road']

# Choose model to train
if model_name == "fcn":
    model = fcn_8s(input_shape=(256, 512, 3), num_classes=len(labels), lr_init=1e-3, lr_decay=5e-4)
elif model_name == "unet":
    model = unet(input_shape=(256, 512, 3), num_classes=len(labels), lr_init=1e-3, lr_decay=5e-4)
elif model_name == "pspnet":
    model = pspnet50(input_shape=(256, 512, 3), num_classes=len(labels), lr_init=1e-3, lr_decay=5e-4)

try:
    model.load_weights(model_name + '_model_weight.h5')
except:
    print("You must train model and get weight before test.")

x_img = cv2.imread(img_path)
cv2.imshow('x_img', x_img)
x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
x_img = x_img / 127.5 - 1
x_img = np.expand_dims(x_img, 0)

pred = model.predict(x_img)
res = result_map_to_img(pred[0])

cv2.imshow('res', res)
cv2.waitKey(0)
