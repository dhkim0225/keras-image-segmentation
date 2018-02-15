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

for area in area_img_list:
    print (area)
    image_list = sorted(glob(os.path.join(area,'*')))
    # print (image_list)
    for image_name in image_list:
        print (image_name)
        gt_name = os.path.basename(image_name).replace('leftImg8bit', 'gtFine_color')
        gt_name = os.path.join(area_gt_path, gt_name)
        print (gt_name)
        break
    break
# exit()
for area in area_gt_list:
    print (area)
    gt_list = sorted(glob(os.path.join(area, '*color.png')))
    for gt_name in gt_list:
        print (gt_name)
        break
    break

img = cv2.imread(image_name, 1)
gt = cv2.imread(gt_name, 1)

img = cv2.resize(img, (img.shape[1]/2, img.shape[0]/2))
gt = cv2.resize(gt, (gt.shape[1]/2, gt.shape[0]/2), interpolation=cv2.INTER_NEAREST)

# gt = gt[gt[:,:]==(128, 64, 128)]
# gt = gt[]
road_color = np.array([128, 64, 128], dtype=np.uint8)
human_color = np.array([60, 20, 220], dtype=np.uint8)
car_color = np.array([142, 0, 0], dtype=np.uint8)
road = np.where(gt[:,:] == road_color, gt, np.array([0,0,0], dtype=np.uint8))
car = np.where(gt[:,:] == car_color, gt, np.array([0,0,0], dtype=np.uint8))
human = np.where(gt[:,:] == human_color, gt, np.array([0,0,0], dtype=np.uint8))

gt = road + human + car

show = img
show = cv2.addWeighted(img, 0.5, gt, 0.6, 0)

cv2.imshow('gt', gt)
cv2.imshow('show', show)
cv2.waitKey()

exit()