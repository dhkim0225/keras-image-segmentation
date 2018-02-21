import h5py
import numpy as np
import random
import time
import cv2
from keras.preprocessing.image import ImageDataGenerator

data = h5py.File('data.h5', 'r')

# Use only 3 classes.
# labels = ['background', 'person', 'car', 'road']


# Centering method helps normalization image (-1 ~ 1)
def centering(np_image):
    return 2 * (np_image - 128)


# Get ImageDataGenerator arguments(options) depends on mode - (train, val, test)
def get_data_gen_args(mode):
    if mode == 'train' or mode == 'val':
        x_data_gen_args = dict(preprocessing_function=centering,
                               rescale=1./255,
                               shear_range=0.1,
                               zoom_range=0.1,
                               rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               horizontal_flip=True)

        y_data_gen_args = dict(shear_range=0.1,
                               zoom_range=0.1,
                               rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               horizontal_flip=True)

    elif mode == 'test':
        x_data_gen_args = dict(preprocessing_function=centering, rescale=1./255)
        y_data_gen_args = dict()

    else:
        print("Data_generator function should get mode arg 'train' or 'val' or 'test'.")
        return -1

    return x_data_gen_args, y_data_gen_args


# One hot encoding for y_img.
def get_result_map(y_img):
    result_map = np.zeros((256, 512, 4))

    # For np.where calculation.
    person = (y_img == 24)
    car = (y_img == 26)
    road = (y_img == 7)
    background = np.logical_not(person + car + road)

    result_map[:, :, 0] = np.where(background, 1, 0)
    result_map[:, :, 1] = np.where(person, 1, 0)
    result_map[:, :, 2] = np.where(car, 1, 0)
    result_map[:, :, 3] = np.where(road, 1, 0)

    return result_map


# Data generator for fit_generator.
def data_generator(b_size, mode):
    x_imgs = data.get('/' + mode + '/x')
    y_imgs = data.get('/' + mode + '/y')

    # Make ImageDataGenerator.
    x_data_gen_args, y_data_gen_args = get_data_gen_args(mode)
    x_data_gen = ImageDataGenerator(**x_data_gen_args)
    y_data_gen = ImageDataGenerator(**y_data_gen_args)

    # random index for random data access.
    d_size = x_imgs.shape[0]
    shuffled_idx = list(range(d_size))
    random.shuffle(shuffled_idx)

    x = []
    y = []
    while True:
        for i in range(d_size):
            idx = shuffled_idx[i]

            x.append(x_imgs[idx].reshape((256, 512, 3)))
            y_img = y_imgs[idx].reshape((256, 512))
            y.append(get_result_map(y_img))

            if len(x) == b_size:
                # Adapt ImageDataGenerator flow method for data augmentation.
                _ = np.zeros(b_size)
                seed = random.randrange(1, 1000)

                x_tmp_gen = x_data_gen.flow(np.array(x), _,
                                            batch_size=b_size,
                                            seed=seed)
                y_tmp_gen = y_data_gen.flow(np.array(y), _,
                                            batch_size=b_size,
                                            seed=seed)

                # Finally, yield x, y data.
                x_result, _ = next(x_tmp_gen)
                y_result, _ = next(y_tmp_gen)
                yield x_result, y_result

                x.clear()
                y.clear()
