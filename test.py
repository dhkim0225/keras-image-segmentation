from __future__ import print_function

import argparse
import cv2
import numpy as np

from model.unet import unet
from model.fcn import fcn_8s
from model.pspnet import pspnet50
from model.deeplab import deeplab_v3_plus


def result_map_to_img(res_map, num_classes, height, width):
    img = np.zeros((height, width, 3), dtype=np.uint8)

    colors = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                       [255, 255, 0], [0, 255, 255], [255, 0, 255], [255, 255, 255]])

    res_map = np.squeeze(res_map)
    res_map = np.argmax(res_map, axis=-1)

    for i in range(num_classes):
        mask = (res_map == i)
        tmp = np.zeros((height, width, 3), dtype=np.uint8)
        tmp[:, :, 0] = np.where(mask, colors[i, 0], 0)
        tmp[:, :, 1] = np.where(mask, colors[i, 1], 0)
        tmp[:, :, 2] = np.where(mask, colors[i, 2], 0)
        img = np.add(img, tmp)

    return img


# Parse Options
parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", required=True, choices=['fcn', 'unet', 'pspnet', 'deeplab'],
                    help="Model to train. 'fcn', 'unet', 'pspnet', 'deeplab' is available.")
parser.add_argument("-P", "--img_path", required=True, help="The image path you want to test")
parser.add_argument("-S", "--scale", required=False, default=0.25, choices=[0.25, 0.5, 0.75, 1],
                    help="Scale of the image. You should use the same value as the rate you set in make_h5.py.")
parser.add_argument("-NC", "--num_classes", required=False, default=8, choices=[4, 8], help="Number of classes.")


args = parser.parse_args()
model_name = args.model
img_path = args.img_path
num_classes = int(args.num_classes)
scale_size = float(args.scale)

h = int(1024 * scale_size)
w = int(2048 * scale_size)

# Use only 3 classes.
labels = ['background', 'person', 'car', 'road']

# Choose model to train
if model_name == "fcn":
    model = fcn_8s(input_shape=(h, w, 3), num_classes=num_classes, lr_init=1e-3, lr_decay=5e-4)
elif model_name == "unet":
    model = unet(input_shape=(h, w, 3), num_classes=num_classes, lr_init=1e-3, lr_decay=5e-4)
elif model_name == "pspnet":
    model = pspnet50(input_shape=(h, w, 3), num_classes=num_classes, lr_init=1e-3, lr_decay=5e-4)
elif model_name == "deeplab":
    model = deeplab_v3_plus(input_shape=(h, w, 3), num_classes=num_classes, lr_init=1e-3, lr_decay=5e-4)

try:
    model.load_weights(model_name + '_model_weight.h5')
except:
    print("You must train model and get weight before test.")

img = cv2.imread(img_path)
x_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
x_img = x_img / 127.5 - 1
x_img = np.expand_dims(x_img, 0)

pred = model.predict(x_img)
res = result_map_to_img(pred[0], num_classes=num_classes, height=h, width=w)

res = cv2.addWeighted(img, 0.7, res, 0.2, 0.0)

cv2.imshow('res', res)
cv2.waitKey(0)

