import cv2
import numpy as np

from keras import backend as K

from fcn import fcn_8s

# Use only 3 classes.
labels = ['background', 'person', 'car', 'road']

model = fcn_8s(len(labels), (256, 512, 3))
model.load_weights('./model_data/model_26.h5')

K.set_learning_phase(0)


def result_map_to_img(res_map):
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

x_img = cv2.imread('/home/anthony/Desktop/project/keras-image-segmentation/leftImg8bit/train/aachen/aachen_000032_000019_leftImg8bit.png')
x_img = cv2.resize(x_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
cv2.imshow('x_img', x_img)
x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
x_img = np.expand_dims(x_img, 0)

res_map = model.predict(x_img)

res = result_map_to_img(res_map)

cv2.imshow('res', res)
cv2.waitKey(0)
