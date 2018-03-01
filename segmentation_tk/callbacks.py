from __future__ import print_function
import keras
import cv2
import numpy as np
import os
from glob import glob

class trainCheck(keras.callbacks.Callback):
    def __init__(self, flag):
        self.flag = flag
        self.epoch = 0

    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch
        self.train_visualization_seg(self.model, epoch, './result/frankfurt_000000_014480_leftImg8bit.png')
        self.train_visualization_seg(self.model, epoch, './result/bremen_000097_000019_leftImg8bit.png')
        return

    def on_epoch_end(self, epoch, logs={}):
        # self.train_visualization_seg(self.model, epoch)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    
    def train_visualization_seg(self, model, epoch, path):
        # image_name_list = sorted(glob(os.path.join(self.flag.data_path,'val/IMAGE/*/frankfurt_000000_014480_leftImg8bit.png')))
        # print (image_name_list)

        image_name = path #'./result/frankfurt_000000_014480_leftImg8bit.png'
        image_height = self.flag.image_height
        image_width = self.flag.image_width
        
        imgInput = cv2.imread(image_name, self.flag.color_mode)
        imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2RGB)
        output_path = self.flag.output_dir
        input_data = imgInput.reshape((1,image_height,image_width,self.flag.color_mode*2+1))

        t_start = cv2.getTickCount()
        result = model.predict(input_data, 1)
        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
        print ("[*] Predict Time: %.3f ms"%t_total)
        
        imgMask = (result[0]*255).astype(np.uint8)
        imgShow = cv2.cvtColor(imgInput, cv2.COLOR_RGB2BGR).copy()
        #cv2.cvtColor(imgInput, cv2.COLOR_GRAY2BGR)
        # imgMaskColor = cv2.applyColorMap(imgMask, cv2.COLORMAP_JET)
        imgMaskColor = imgMask
        imgShow = cv2.addWeighted(imgShow, 0.5, imgMaskColor, 0.6, 0.0)
        output_path = os.path.join(self.flag.output_dir, '%04d_'%epoch+os.path.basename(image_name))
        mask_path = os.path.join(self.flag.output_dir, 'mask_%04d_'%epoch+os.path.basename(image_name))
        cv2.imwrite(output_path, imgShow)
        cv2.imwrite(mask_path, imgMaskColor)
        # print "SAVE:[%s]"%output_path
        # cv2.imwrite(os.path.join(output_path, 'img%04d.png'%epoch), imgShow)
        # cv2.namedWindow("show", 0)
        # cv2.resizeWindow("show", 800, 800)
        # cv2.imshow("show", imgShow)
        # cv2.waitKey(1)