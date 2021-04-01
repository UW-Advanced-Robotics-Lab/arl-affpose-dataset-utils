import numpy as np
import shutil
import glob
import os

import scipy.io
import scipy.misc
from PIL import Image

import matplotlib.pyplot as plt

#######################################
#######################################

import sys
sys.path.append('../..')
# print(sys.path)

#######################################
#######################################

import cfg as config

#######################################
#######################################

data_path     = config.ROOT_DATA_PATH + 'LabelFusion/dataset/'
new_data_path = config.ROOT_DATA_PATH + 'Real/'

image_exts = [
            config.RGB_EXT,
            config.DEPTH_EXT,
            config.LABEL_EXT,
            config.META_EXT
]

#######################################
#######################################

train_val_split = 0.8 # 80% train / 20% val
val_test_split = 0.75 # 15% val   / 5% test

########################
########################
offset_train, offset_val, offset_test = 0, 0, 0
train_files_len, val_files_len, test_files_len = 0, 0, 0

for image_ext in image_exts:
    file_path = data_path + 'elavator_down_*/*/' +'*' + image_ext
    files = np.array(sorted(glob.glob(file_path)))
    print("\nLoaded files: ", len(files))
    print("File path: ", file_path)

    ###############
    # split files
    ###############
    np.random.seed(0)
    total_idx = np.arange(0, len(files), 1)
    train_idx = np.random.choice(total_idx, size=int(train_val_split * len(total_idx)), replace=False)
    val_test_idx = np.delete(total_idx, train_idx)

    train_files = files[train_idx]
    val_test_files = files[val_test_idx]

    val_test_idx = np.arange(0, len(val_test_files), 1)
    val_idx = np.random.choice(val_test_idx, size=int(val_test_split * len(val_test_idx)), replace=False)
    test_idx = np.delete(val_test_idx, val_idx)
    val_files = val_test_files[val_idx]
    test_files = val_test_files[test_idx]

    print("Chosen Train Files {}/{}".format(len(train_files), len(files)))
    print("Chosen Val Files {}/{}".format(len(val_files), len(files)))
    print("Chosen Test Files {}/{}".format(len(test_files), len(files)))

    if image_ext == config.RGB_EXT:
        train_files_len = len(train_files)
        val_files_len = len(val_files)
        test_files_len = len(test_files)

    ###############
    # train
    ###############
    split_folder = 'train/'

    for idx, file in enumerate(train_files):
        old_file_name = file
        new_file_name = new_data_path + split_folder

        # object = old_file_name.split('/')[7]
        object = ''

        count = 1000000 + idx
        image_num = str(count)[1:]

        if image_ext == config.RGB_EXT:
            # move_file_name = new_file_name + 'rgb/' + np.str(object) + '_' + np.str(image_num) + '.png'
            move_file_name = new_file_name + 'rgb/' + np.str(image_num) + '.png'
            if idx == 0:
                print(f'Old file: {old_file_name}')
                print(f'New file: {move_file_name}')
            shutil.copyfile(old_file_name, move_file_name)

        elif image_ext == config.DEPTH_EXT:
            move_file_name = new_file_name + 'depth/' + np.str(image_num) + '_depth.png'
            if idx == 0:
                print(f'Old file: {old_file_name}')
                print(f'New file: {move_file_name}')
            shutil.copyfile(old_file_name, move_file_name)

        elif image_ext == config.LABEL_EXT:
            move_file_name = new_file_name + 'masks/' + np.str(image_num) + '_label.png'
            if idx == 0:
                print(f'Old file: {old_file_name}')
                print(f'New file: {move_file_name}')
            shutil.copyfile(old_file_name, move_file_name)

        elif image_ext == config.META_EXT:
            move_file_name = new_file_name + 'meta/' + np.str(image_num) + '_meta.mat'
            if idx == 0:
                print(f'Old file: {old_file_name}')
                print(f'New file: {move_file_name}')
            shutil.copyfile(old_file_name, move_file_name)

        else:
            assert "*** IMAGE EXT DOESN'T EXIST ***"

    ###############
    # val
    ###############
    split_folder = 'val/'

    for idx, file in enumerate(val_files):
        old_file_name = file
        new_file_name = new_data_path + split_folder

        # object = old_file_name.split('/')[7]
        object = ''

        count = 1000000 + idx
        image_num = str(count)[1:]

        if image_ext == config.RGB_EXT:
            move_file_name = new_file_name + 'rgb/' + np.str(image_num) + '.png'
            shutil.copyfile(old_file_name, move_file_name)

        elif image_ext == config.DEPTH_EXT:
            move_file_name = new_file_name + 'depth/' + np.str(image_num) + '_depth.png'
            shutil.copyfile(old_file_name, move_file_name)

        elif image_ext == config.LABEL_EXT:
            move_file_name = new_file_name + 'masks/' + np.str(image_num) + '_label.png'
            shutil.copyfile(old_file_name, move_file_name)

        elif image_ext == config.META_EXT:
            move_file_name = new_file_name + 'meta/' + np.str(image_num) + '_meta.mat'
            shutil.copyfile(old_file_name, move_file_name)

        else:
            assert "*** IMAGE EXT DOESN'T EXIST ***"

    ###############
    # test
    ###############
    split_folder = 'test/'

    for idx, file in enumerate(test_files):
        old_file_name = file
        new_file_name = new_data_path + split_folder

        # object = old_file_name.split('/')[7]
        object = ''

        count = 1000000 + idx
        image_num = str(count)[1:]

        if image_ext == config.RGB_EXT:
            move_file_name = new_file_name + 'rgb/' + np.str(image_num) + '.png'
            shutil.copyfile(old_file_name, move_file_name)

        elif image_ext == config.DEPTH_EXT:
            move_file_name = new_file_name + 'depth/' + np.str(image_num) + '_depth.png'
            shutil.copyfile(old_file_name, move_file_name)

        elif image_ext == config.LABEL_EXT:
            move_file_name = new_file_name + 'masks/' + np.str(image_num) + '_label.png'
            shutil.copyfile(old_file_name, move_file_name)

        elif image_ext == config.META_EXT:
            move_file_name = new_file_name + 'meta/' + np.str(image_num) + '_meta.mat'
            shutil.copyfile(old_file_name, move_file_name)

        else:
            assert "*** IMAGE EXT DOESN'T EXIST ***"
