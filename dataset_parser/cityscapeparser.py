from __future__ import print_function
import numpy as np
from glob import glob
import os
import cv2

area_img_path = '/Volumes/Samsung USB/dataset/cityscape/leftImg8bit/train/'
area_gt_path = '/Volumes/Samsung USB/dataset/cityscape/gtFine/train/'
area_img_list = sorted(glob(area_img_path+'*'))
area_gt_list = sorted(glob(area_gt_path+'*'))
# print (area_list)

def select_labels(gt):
    human = np.where(gt == 24, 1, 0)
    car = np.where(gt == 26, 2, 0)
    road = np.where(gt == 7, 3, 0)

    gt_new = road + car + human
    return gt_new

for area in area_img_list:
    # print (area)
    image_list = sorted(glob(os.path.join(area,'*')))
    # print (image_list)
    for image_name in image_list:
        gt_name = os.path.basename(image_name).replace('leftImg8bit', 'gtFine_labelIds')
        gt_name = os.path.join(area_gt_path, os.path.basename(area), gt_name)
        print (os.path.basename(image_name))

        img = cv2.imread(image_name, 1)
        gt = cv2.imread(gt_name, 1)

        img = cv2.resize(img, (img.shape[1]/2, img.shape[0]/2))
        gt = cv2.resize(gt, (gt.shape[1]/2, gt.shape[0]/2), interpolation=cv2.INTER_NEAREST)

        gt = select_labels(gt)

        show = img
        gt_show = gt.astype(np.float32)*255/33
        gt_show = gt_show.astype(np.uint8)
        gt_show = cv2.applyColorMap(gt_show, cv2.COLORMAP_JET)
        show = cv2.addWeighted(img, 0.5, gt_show, 0.6, 0)

        # cv2.imshow('gt', gt)
        cv2.imshow('show', show)
        key = cv2.waitKey()
        if key == 27:
            exit()
