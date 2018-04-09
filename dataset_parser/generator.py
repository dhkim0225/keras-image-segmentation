import h5py
import numpy as np
import random
import cv2

from keras.preprocessing.image import ImageDataGenerator

# You can see details of labels in cityscape_scripts github.
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py#L61


def pre_processing(img):
    # Random exposure and saturation (0.8 ~ 1.2 scale)
    rand_s = random.uniform(0.8, 1.2)
    rand_v = random.uniform(0.8, 1.2)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    tmp = np.ones_like(img[:, :, 1]) * 255
    img[:, :, 1] = np.where(img[:, :, 1] * rand_s > 255, tmp, img[:, :, 1] * rand_s)
    img[:, :, 2] = np.where(img[:, :, 2] * rand_v > 255, tmp, img[:, :, 2] * rand_v)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    # Normalization image (-1 ~ 1 value)
    return img / 127.5 - 1


# Get ImageDataGenerator arguments(options) depends on mode - (train, val, test)
def get_data_gen_args(mode):
    if mode == 'train' or mode == 'val':
        x_data_gen_args = dict(preprocessing_function=pre_processing,
                               shear_range=0.1,
                               zoom_range=0.2,
                               rotation_range=20,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               fill_mode='constant',
                               horizontal_flip=True)

        y_data_gen_args = dict(shear_range=0.1,
                               zoom_range=0.2,
                               rotation_range=20,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               fill_mode='constant',
                               horizontal_flip=True)

    elif mode == 'test':
        x_data_gen_args = dict(preprocessing_function=pre_processing)
        y_data_gen_args = dict()

    else:
        print("Data_generator function should get mode arg 'train' or 'val' or 'test'.")
        return -1

    return x_data_gen_args, y_data_gen_args


# One hot encoding for y_img (use only 3 classes).
def get_result_map_easy(b_size, y_img):
    y_img = np.squeeze(y_img, axis=3)
    result_map = np.zeros((b_size, y_img.shape[1], y_img.shape[2], 4))

    # For np.where calculation.
    person = (y_img == 24)
    car = (y_img == 26)
    road = (y_img == 7)
    background = np.logical_not(person + car + road)

    result_map[:, :, :, 0] = np.where(background, 1, 0)
    result_map[:, :, :, 1] = np.where(person, 1, 0)
    result_map[:, :, :, 2] = np.where(car, 1, 0)
    result_map[:, :, :, 3] = np.where(road, 1, 0)

    return result_map


# One hot encoding for y_img (use 8 classes).
def get_result_map_hard(b_size, y_img):
    y_img = np.squeeze(y_img, axis=3)
    result_map = np.zeros((b_size, y_img.shape[1], y_img.shape[2], 8))

    # For np.where calculation.
    flat = np.logical_and(7 <= y_img, y_img <= 10)
    construction = np.logical_and(11 <= y_img, y_img <= 16)
    obj = np.logical_and(17 <= y_img, y_img <= 20)
    nature = np.logical_and(21 <= y_img, y_img <= 22)
    sky = (y_img == 23)
    human = np.logical_and(24 <= y_img, y_img <= 25)
    vehicle = np.logical_and(26 <= y_img, y_img <= 33)
    background = np.logical_not(flat + construction + obj + nature + sky + human + vehicle)

    result_map[:, :, :, 0] = np.where(background, 1, 0)
    result_map[:, :, :, 1] = np.where(flat, 1, 0)
    result_map[:, :, :, 2] = np.where(construction, 1, 0)
    result_map[:, :, :, 3] = np.where(obj, 1, 0)
    result_map[:, :, :, 4] = np.where(nature, 1, 0)
    result_map[:, :, :, 5] = np.where(sky, 1, 0)
    result_map[:, :, :, 6] = np.where(human, 1, 0)
    result_map[:, :, :, 7] = np.where(vehicle, 1, 0)

    return result_map


# Data generator for fit_generator.
def data_generator(d_path, b_size, mode, is_hard, height, width):
    data = h5py.File(d_path, 'r')
    x_imgs = data.get('/' + mode + '/x')
    y_imgs = data.get('/' + mode + '/y')

    # Make ImageDataGenerator.
    x_data_gen_args, y_data_gen_args = get_data_gen_args(mode)
    x_data_gen = ImageDataGenerator(**x_data_gen_args)
    y_data_gen = ImageDataGenerator(**y_data_gen_args)

    # random index for random data access.
    d_size = x_imgs.shape[0]
    shuffled_idx = list(range(d_size))

    x = []
    y = []
    while True:
        random.shuffle(shuffled_idx)
        for i in range(d_size):
            idx = shuffled_idx[i]

            x.append(x_imgs[idx].reshape((height, width, 3)))
            y.append(y_imgs[idx].reshape((height, width, 1)))

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
                if is_hard:
                    yield x_result, get_result_map_hard(b_size, y_result)
                else:
                    yield x_result, get_result_map_easy(b_size, y_result)

                x.clear()
                y.clear()
