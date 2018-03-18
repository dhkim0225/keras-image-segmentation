from __future__ import print_function

import os
import matplotlib.pyplot as plt
import argparse

from keras.callbacks import ModelCheckpoint, EarlyStopping
from callbacks import TrainCheck

from model.unet import unet
from model.fcn import fcn_8s
from model.pspnet import pspnet50
from dataset_parser.generator import data_generator

# Current python dir path
dir_path = os.path.dirname(os.path.realpath('__file__'))

# Parse Options
parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", required=True, choices=['fcn', 'unet', 'pspnet'],
                    help="Model to train. 'fcn', 'unet', 'pspnet' is available.")
parser.add_argument("-TB", "--train_batch", required=False, default=4, help="Batch size for train.")
parser.add_argument("-VB", "--val_batch", required=False, default=1, help="Batch size for validation.")
parser.add_argument("-LI", "--lr_init", required=False, default=1e-3, help="Initial learning rate.")
parser.add_argument("-LD", "--lr_decay", required=False, default=5e-4, help="How much to decay the learning rate.")
parser.add_argument("--vgg", required=False, default=None, help="Pretrained vgg16 weight path.")

args = parser.parse_args()
model_name = args.model
TRAIN_BATCH = args.train_batch
VAL_BATCH = args.val_batch
lr_init = args.lr_init
lr_decay = args.lr_decay
vgg_path = args.vgg

# Use only 3 classes.
labels = ['background', 'person', 'car', 'road']

# Choose model to train
if model_name == "fcn":
    model = fcn_8s(input_shape=(256, 512, 3), num_classes=len(labels),
                   lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "unet":
    model = unet(input_shape=(256, 512, 3), num_classes=len(labels),
                 lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "pspnet":
    model = pspnet50(input_shape=(256, 512, 3), num_classes=len(labels), lr_init=lr_init, lr_decay=lr_decay)

# Define callbacks
checkpoint = ModelCheckpoint(filepath=model_name + '_model_weight.h5',
                             monitor='val_dice_coef',
                             save_best_only=True,
                             save_weights_only=True)
train_check = TrainCheck(output_path='./img', model_name=model_name)
early_stopping = EarlyStopping(monitor='val_dice_coef', patience=10)

# training
history = model.fit_generator(data_generator('dataset_parser/data.h5', TRAIN_BATCH, 'train'),
                              steps_per_epoch=3475 // TRAIN_BATCH,
                              validation_data=data_generator('dataset_parser/data.h5', VAL_BATCH, 'val'),
                              validation_steps=500 // VAL_BATCH,
                              callbacks=[checkpoint, train_check],
                              epochs=300,
                              verbose=1)

plt.title("loss")
plt.plot(history.history["loss"], color="r", label="train")
plt.plot(history.history["val_loss"], color="b", label="val")
plt.legend(loc="best")
plt.savefig(model_name + '_loss.png')

plt.title("dice_coef")
plt.plot(history.history["dice_coef"], color="r", label="train")
plt.plot(history.history["val_dice_coef"], color="b", label="val")
plt.legend(loc="best")
plt.savefig(model_name + '_dice_coef.png')
