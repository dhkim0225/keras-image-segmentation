from __future__ import print_function

import os
import matplotlib.pyplot as plt
import argparse

from keras.callbacks import ModelCheckpoint, EarlyStopping
from callbacks import TrainCheck

from model.unet import unet
from model.fcn import fcn_8s
from model.pspnet import pspnet50
from model.deeplab import deeplab_v3_plus
from dataset_parser.generator import data_generator

# Current python dir path
dir_path = os.path.dirname(os.path.realpath('__file__'))

# Parse Options
parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", required=True, choices=['fcn', 'unet', 'pspnet', 'deeplab'],
                    help="Model to train. 'fcn', 'unet', 'pspnet', 'deeplab' is available.")
parser.add_argument("-S", "--scale", required=False, default=0.25, choices=[0.25, 0.5, 0.75, 1],
                    help="Scale of the image. You should use the same value as the rate you set in make_h5.py.")
parser.add_argument("-NC", "--num_classes", required=False, default=8, choices=[4, 8], help="Number of classes.")
parser.add_argument("-TB", "--train_batch", required=False, default=4, help="Batch size for train.")
parser.add_argument("-VB", "--val_batch", required=False, default=1, help="Batch size for validation.")
parser.add_argument("-LI", "--lr_init", required=False, default=1e-4, help="Initial learning rate.")
parser.add_argument("-LD", "--lr_decay", required=False, default=5e-4, help="How much to decay the learning rate.")
parser.add_argument("--vgg", required=False, default=None, help="Pretrained vgg16 weight path.")

args = parser.parse_args()
model_name = args.model
num_classes = int(args.num_classes)
scale_size = float(args.scale)
TRAIN_BATCH = int(args.train_batch)
VAL_BATCH = int(args.val_batch)
lr_init = float(args.lr_init)
lr_decay = float(args.lr_decay)
vgg_path = args.vgg

h = int(1024 * scale_size)
w = int(2048 * scale_size)

# Choose model to train
if model_name == "fcn":
    model = fcn_8s(input_shape=(h, w, 3), num_classes=num_classes,
                   lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "unet":
    model = unet(input_shape=(h, w, 3), num_classes=num_classes,
                 lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "pspnet":
    model = pspnet50(input_shape=(h, w, 3), num_classes=num_classes, lr_init=lr_init, lr_decay=lr_decay)
elif model_name == "deeplab":
    model = deeplab_v3_plus(input_shape=(h, w, 3), num_classes=num_classes, lr_init=lr_init, lr_decay=lr_decay)

# Define callbacks
checkpoint = ModelCheckpoint(filepath=model_name + '_model_weight.h5',
                             monitor='val_dice_coef',
                             save_best_only=True,
                             save_weights_only=True)
train_check = TrainCheck(output_path='./img',
                         model_name=model_name,
                         num_classes=num_classes,
                         height=h,
                         width=w)
#early_stopping = EarlyStopping(monitor='val_dice_coef', patience=10)

# generator
if num_classes == 8:
    t_gen = data_generator('data.h5', TRAIN_BATCH, 'train', is_hard=True, height=h, width=w)
    v_gen = data_generator('data.h5', VAL_BATCH, 'val', is_hard=True, height=h, width=w)
else:
    t_gen = data_generator('data.h5', TRAIN_BATCH, 'train', is_hard=False, height=h, width=w)
    v_gen = data_generator('data.h5', VAL_BATCH, 'val', is_hard=False, height=h, width=w)

# training
history = model.fit_generator(t_gen,
                              steps_per_epoch=3475 // TRAIN_BATCH,
                              validation_data=v_gen,
                              validation_steps=500 // VAL_BATCH,
                              callbacks=[checkpoint, train_check],
                              epochs=100,
                              verbose=1)

plt.title("loss")
plt.plot(history.history["loss"], color="r", label="train")
plt.plot(history.history["val_loss"], color="b", label="val")
plt.legend(loc="best")
plt.savefig('img/' + model_name + '_loss.png')

plt.gcf().clear()
plt.title("dice_coef")
plt.plot(history.history["dice_coef"], color="r", label="train")
plt.plot(history.history["val_dice_coef"], color="b", label="val")
plt.legend(loc="best")
plt.savefig('img/' + model_name + '_dice_coef.png')

plt.gcf().clear()
plt.title("mIoU")
plt.plot(history.history["m_iou"], color="r", label="train")
plt.plot(history.history["val_m_iou"], color="b", label="val")
plt.legend(loc="best")
plt.savefig('img/' + model_name + '_mIoU.png')

plt.gcf().clear()
plt.title("precision")
plt.plot(history.history["precision"], color="r", label="train")
plt.plot(history.history["val_precision"], color="b", label="val")
plt.legend(loc="best")
plt.savefig('img/' + model_name + '_precision.png')

plt.gcf().clear()
plt.title("recall")
plt.plot(history.history["recall"], color="r", label="train")
plt.plot(history.history["val_recall"], color="b", label="val")
plt.legend(loc="best")
plt.savefig('img/' + model_name + '_recall.png')
