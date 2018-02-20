from __future__ import print_function

import h5py
import os
import numpy as np
import cv2
import argparse

# Save current python dir path
dir_path = os.path.dirname(os.path.realpath('__file__'))

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="path of leftImg8bit folder.", default='/run/media/tkwoo/myWorkspace/workspace/01.dataset/cityscape/leftImg8bit')
parser.add_argument("--gtpath", help="path of gtFine folder.", default='/run/media/tkwoo/myWorkspace/workspace/01.dataset/cityscape/gtFine')

args = parser.parse_args()
img_folder_path = args.path
gt_folder_path = args.gtpath

# Use only 3 classes.
labels = ['background', 'person', 'car', 'road']

def get_data(mode):
    if mode == 'train' or mode == 'val' or mode == 'test':
        x_paths = []
        y_paths = []
        tmp_img_folder_path = os.path.join(img_folder_path, mode)
        tmp_gt_folder_path = os.path.join(gt_folder_path, mode)

        # os.walk helps to find all files in directory.
        for (path, dirname, files) in sorted(os.walk(tmp_img_folder_path)):
            for filename in sorted(files):
                x_paths.append(os.path.join(path, filename))

        # Find ground_truth file paths with x_paths.
        idx = len(tmp_img_folder_path)
        for x_path in x_paths:
            y_paths.append(tmp_gt_folder_path + x_path[idx:-15] + 'gtFine_labelIds.png')

        return x_paths, y_paths
    else:
        print("Please call get_data function with arg 'train', 'val', 'test'.")

def write_data(h5py_file, mode, x_paths, y_paths):
    num_data = len(x_paths)

    uint8_dt = h5py.special_dtype(vlen=np.uint8)
    string_dt = h5py.special_dtype(vlen=str)

    group = h5py_file.create_group(mode)
    h5_name = group.create_dataset('name', shape=(num_data,), dtype=string_dt)
    h5_image = group.create_dataset('image', shape=(num_data,), dtype=uint8_dt)
    h5_label = group.create_dataset('label', shape=(num_data,), dtype=uint8_dt)

    for i in range(num_data):
        x_img = cv2.imread(x_paths[i])
        y_img = cv2.imread(y_paths[i])
        x_img = cv2.resize(x_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
        y_img = cv2.resize(y_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)

        # x_img = np.reshape(x_img, (256*512*3,))
        # y_img = np.reshape(y_img, (256*512*3,))

        h5_image[i] = x_img.flatten()
        h5_label[i] = y_img.flatten()
        # h5_name[i] = np.array(os.path.basename(x_train_path[i]))
        h5_name[i] = os.path.basename(x_paths[i])

def make_h5py():
    x_train_paths, y_train_paths = get_data('train')
    x_val_paths, y_val_paths = get_data('val')
    x_test_paths, y_test_paths = get_data('test')

    h5py_file = h5py.File(os.path.join(dir_path, 'data.h5'), 'w')
    
    start = cv2.getTickCount()
    write_data(h5py_file, 'train', x_train_paths, y_train_paths)
    time = (cv2.getTickCount()-start)/cv2.getTickFrequency()
    print ('parsing train data, Time:%.3fs'%time)

    start = cv2.getTickCount()
    write_data(h5py_file, 'val', x_val_paths, y_val_paths)
    time = (cv2.getTickCount()-start)/cv2.getTickFrequency()
    print ('parsing val data, Time:%.3fs'%time)

    start = cv2.getTickCount()
    write_data(h5py_file, 'test', x_test_paths, y_test_paths)
    time = (cv2.getTickCount()-start)/cv2.getTickFrequency()
    print ('parsing test data, Time:%.3fs'%time)

def read_h5py_example():
    h5_in = h5py.File(os.path.join(dir_path, 'data.h5'), 'r')
    print (h5_in.keys())
    print (h5_in['train']['image'].dtype)
    print (h5_in['train']['image'][0].shape)

    x_img = np.reshape(h5_in['train']['image'][0], (256,512,3))
    y_img = np.reshape(h5_in['train']['label'][0], (256,512,3))
    name = h5_in['train']['name'][0]
    print (name)
    y_img = (y_img.astype(np.float32)*255/33).astype(np.uint8)
    y_show = cv2.applyColorMap(y_img, cv2.COLORMAP_JET)
    show = cv2.addWeighted(x_img, 0.5, y_show, 0.5, 0)
    cv2.imshow("show", show)
    cv2.waitKey()
    
if __name__=='__main__':
    # make_h5py()
    read_h5py_example()