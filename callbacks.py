from __future__ import print_function
from keras.callbacks import Callback

import cv2
import numpy as np
import os


class TrainCheck(Callback):
    def __init__(self, output_path, model_name):
        self.epoch = 0
        self.output_path = output_path
        self.model_name = model_name

    def result_map_to_img(self, res_map):
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

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch+1
        self.visualize('img/test.png')

    def visualize(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)
        img = img / 127.5 - 1

        pred = self.model.predict(img)
        res_img = self.result_map_to_img(pred[0])

        cv2.imwrite(os.path.join(self.output_path, self.model_name + '_epoch_' + str(self.epoch) + '.png'), res_img)
