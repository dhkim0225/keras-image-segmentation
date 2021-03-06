from __future__ import print_function
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras.models import model_from_json
import tensorflow as tf
import keras
import cv2
import numpy as np
import os
from glob import glob
import argparse

def predict_image(flag):
    t_start = cv2.getTickCount()
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    with open(os.path.join(flag.ckpt_dir, flag.ckpt_name, 'model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    weight_list = sorted(glob(os.path.join(flag.ckpt_dir, flag.ckpt_name, "weight*")))
    model.load_weights(weight_list[-1])
    print ("[*] model load : %s"%weight_list[-1])
    t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000 
    print ("[*] model loading Time: %.3f ms"%t_total)

    imgInput = cv2.imread(flag.test_image_path, 1)
    imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2RGB)
    input_data = imgInput.reshape((1,256,512,3)).astype(np.float32)

    t_start = cv2.getTickCount()
    result = model.predict(input_data, 1)
    t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
    print ("Predict Time: %.3f ms"%t_total)
    
    imgMask = (result[0]*255).astype(np.uint8)
    # print (imgMask.shape)
    # print (imgInput.dtype)
    imgShow = cv2.cvtColor(imgInput, cv2.COLOR_RGB2BGR)
    # _, imgMask = cv2.threshold(imgMask, int(255*flag.confidence_value), 255, cv2.THRESH_BINARY)
    
    # imgMaskColor = cv2.applyColorMap(imgMask, cv2.COLORMAP_JET)
    imgMaskColor = imgMask
    # imgZero = np.zeros((256,256), np.uint8)
    # imgMaskColor = cv2.merge((imgZero, imgMask, imgMask))
    imgShow = cv2.addWeighted(imgShow, 0.5, imgMaskColor, 0.5, 0.0)

    cv2.imshow('show', imgShow)
    key = cv2.waitKey()
    if key == 27:
        exit()
    # output_path = os.path.join(flag.output_dir, os.path.basename(flag.test_image_path))
    # cv2.imwrite(output_path, imgShow)
    # print "SAVE:[%s]"%output_path
        
def demo(flag):

    vc = cv2.VideoCapture('./demo/stuttgart_02.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    vw_name = './demo/result_stuttgart_02.mp4'
    vw = cv2.VideoWriter(vw_name, fourcc, 30.0, (512, 256))

    t_start = cv2.getTickCount()
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    with open(os.path.join(flag.ckpt_dir, flag.ckpt_name, 'model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    weight_list = sorted(glob(os.path.join(flag.ckpt_dir, flag.ckpt_name, "weight*")))
    model.load_weights(weight_list[-1])
    print ("[*] model load : %s"%weight_list[-1])
    t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000 
    print ("[*] model loading Time: %.3f ms"%t_total)

    while True:
        # imgInput = cv2.imread(flag.test_image_path, 1)
        ok, imgInput = vc.read()
        if ok is False:
            break
        imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2RGB)
        input_data = imgInput.reshape((1,256,512,3)).astype(np.float32)

        t_start = cv2.getTickCount()
        result = model.predict(input_data, 1)
        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
        print ("Predict Time: %.3f ms"%t_total)
        
        imgMask = (result[0]*255).astype(np.uint8)
        # human, car, road
        _, imgMask[:,:,0] = cv2.threshold(imgMask[:,:,0], int(255*0.5), 255, cv2.THRESH_BINARY)
        _, imgMask[:,:,1] = cv2.threshold(imgMask[:,:,1], int(255*0.5), 255, cv2.THRESH_BINARY)
        _, imgMask[:,:,2] = cv2.threshold(imgMask[:,:,2], int(255*0.5), 255, cv2.THRESH_BINARY)
        imgShow = cv2.cvtColor(imgInput, cv2.COLOR_RGB2BGR)
        
        imgMaskColor = imgMask
        imgShow = cv2.addWeighted(imgShow, 0.5, imgMaskColor, 0.6, 0.0)

        cv2.imshow('show', imgShow)
        vw.write(imgShow)
        key = cv2.waitKey(1)
        if key == 27:
            exit()