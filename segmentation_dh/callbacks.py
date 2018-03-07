from __future__ import print_function
from keras.callbacks import Callback
import cv2
import numpy as np
import os


class TrainCheck(Callback):
    def __init__(self, output_path):
        self.epoch = 0
        self.output_path = output_path

    def result_map_to_img(self, res_map):
        img = np.zeros((256, 512, 3))
        res_map = np.squeeze(res_map)

        # For np.where calculation.
        person = (res_map[:, :, 1] == 1)
        car = (res_map[:, :, 2] == 1)
        road = (res_map[:, :, 3] == 1)

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

        pred = self.model.predict(img)
        res_img = self.result_map_to_img(pred[0])

        cv2.imwrite(os.path.join(self.output_path, 'epoch_' + str(self.epoch) + '_out.png'), res_img)
