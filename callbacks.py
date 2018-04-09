from __future__ import print_function
from keras.callbacks import Callback

import cv2
import numpy as np
import os


class TrainCheck(Callback):
    def __init__(self, output_path, model_name, num_classes, height, width):
        self.epoch = 0
        self.output_path = output_path
        self.model_name = model_name
        self.num_classes = num_classes
        self.h = height
        self.w = width

    def result_map_to_img(self, res_map):
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        colors = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                           [255, 255, 0], [0, 255, 255], [255, 0, 255], [255, 255, 255]])

        res_map = np.squeeze(res_map)
        res_map = np.argmax(res_map, axis=-1)

        for i in range(self.num_classes):
            mask = (res_map == i)
            tmp = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            tmp[:, :, 0] = np.where(mask, colors[i, 0], 0)
            tmp[:, :, 1] = np.where(mask, colors[i, 1], 0)
            tmp[:, :, 2] = np.where(mask, colors[i, 2], 0)
            img = np.add(img, tmp)

        return img

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch+1
        self.visualize('img/test.png')

    def visualize(self, path):
        img = cv2.imread(path)
        x_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x_img = np.expand_dims(x_img, 0)
        x_img = x_img / 127.5 - 1

        pred = self.model.predict(x_img)
        res_img = self.result_map_to_img(pred[0])
        res_img = cv2.addWeighted(img, 0.7, res_img, 0.2, 0.0)

        cv2.imwrite(os.path.join(self.output_path, self.model_name + '_epoch_' + str(self.epoch) + '.png'), res_img)
