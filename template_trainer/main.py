import cv2
import numpy as np
import os
from glob import glob
import argparse
import sys
import train
from test import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", help="training data path", default="/run/media/tkwoo/myWorkspace/workspace/01.dataset/03.Mask_data/cityscape")
parser.add_argument("--output_dir", help="output directory", default="./result")
parser.add_argument("--image_width", help="image size", default=512, type=int)
parser.add_argument("--image_height", help="image size", default=256, type=int)
parser.add_argument("--color_mode", help="color", default=0, type=int)
parser.add_argument("--batch_size", help="batch size", default=8, type=int)
parser.add_argument("--total_epoch", help="number of epochs", default=500, type=int)
parser.add_argument("--initial_learning_rate", help="init lr", default=0.001, type=float)
parser.add_argument("--learning_rate_decay_factor", help="learning rate decay", default=0.5, type=float)
parser.add_argument("--epoch_per_decay", help="lr decay period", default=250, type=int)
parser.add_argument("--ckpt_dir", help="checkpoint root directory", default='./checkpoint')
parser.add_argument("--ckpt_name", help="[.../ckpt_dir/ckpt_name/weights.h5]", default='Unet')
parser.add_argument("--pretrained_weight_path", help="weight.h5 path", default=None)
parser.add_argument("--confidence_value", help="mask threshold value", default=0.5, type=float)
parser.add_argument("--debug", help="for debug [str: 'true' or 'false']", default='false')
parser.add_argument("--mode",
                    help="[train] or [predict_img] or [predict_imgDir]",
                    default='train')
parser.add_argument("--test_image_path", 
                    help="[mode:predict_img] ex) .../Image.png, [mode:predict_imgDir] ex).../dirname",
                    default=None)
parser.add_argument("--tf_log_level", help="0, 1, 2, 3", default='2', type=str)

flag = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = flag.tf_log_level # or any {'0', '1', '2', '3'}
# os.environ["CUDA_VISIBLE_DEVICES"]="0" 

def main():
    if not os.path.isdir(flag.output_dir):
        os.mkdir(flag.output_dir)
    if flag.mode == 'train':
        train_op = train.TrainModel(flag)
        train_op.train()
    elif flag.mode == 'predict_img':
        predict_image(flag)
    elif flag.mode == 'predict_imgDir':
        print 'not supported'
    else:
        print 'not supported'

if __name__ == '__main__':
    main()