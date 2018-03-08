import os
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint
from callbacks import TrainCheck

from model.unet import unet
from model.fcn import fcn_8s
from dataset_parser.generator import data_generator

# Use only 3 classes.
labels = ['background', 'person', 'car', 'road']

model = unet(input_shape=(256, 512, 3), num_classes=len(labels), init_lr=5e-3, vgg_weight_path="../vgg16_notop.h5")
#model = fcn_8s(input_shape=(256, 512, 3), num_classes=len(labels), init_lr=1e-3, vgg_weight_path="../vgg16_notop.h5")

model.summary()

# callbacks
checkpoint = ModelCheckpoint(filepath='model_weight.h5',
                             save_best_only=True,
                             save_weights_only=True)
train_check = TrainCheck(output_path='./img')

# training
history = model.fit_generator(data_generator('../dataset_parser/data.h5', 4, 'train'),
                              steps_per_epoch=(3475 // 4),
                              validation_data=data_generator('../dataset_parser/data.h5', 1, 'val'),
                              validation_steps=(500 // 1),
                              callbacks=[checkpoint, train_check],
                              epochs=300,
                              verbose=2)

plt.title("loss")
plt.plot(history.history["loss"], color="r", label="train")
plt.plot(history.history["val_loss"], color="b", label="val")
plt.legend(loc="best")
plt.show()
plt.imshow()

plt.title("dice_coef")
plt.plot(history.history["dice_coef"], color="r", label="train")
plt.plot(history.history["val_dice_coef"], color="b", label="val")
plt.legend(loc="best")
plt.show()
plt.imshow()
