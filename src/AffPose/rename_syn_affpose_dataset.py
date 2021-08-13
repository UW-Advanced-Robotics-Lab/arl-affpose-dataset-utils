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
sys.path.append('../../')

#######################################
#######################################

import cfg as config

#######################################
#######################################

data_path     = config.NDDS_PATH
new_data_path = config.ROOT_DATA_PATH + 'Syn_test/'

image_exts = [
            # config.SYN_RGB_EXT,
            # config.SYN_DEPTH_EXT,
            # config.SYN_OBJ_LABEL_EXT,
            # config.SYN_OBJ_PART_LABEL_EXT,
            # config.SYN_AFF_LABEL_EXT,
            config.SYN_META_EXT,
]

#######################################
#######################################

train_val_split = 0.8  # 80% train / 20% val
val_test_split = 0.95  # 15% val   / 1% test

########################
########################
offset_train, offset_val, offset_test = 0, 0, 0
train_files_len, val_files_len, test_files_len = 0, 0, 0

for image_ext in image_exts:
    file_path = data_path + "*/*/*/" + '??????' + image_ext
    files = np.sort(np.array(glob.glob(file_path)))
    print("\nLoaded files: ", len(files))
    print("File path: ", file_path)

    ###############
    # split files
    ###############
    np.random.seed(0)
    total_idx = np.arange(0, len(files), 1)
    train_idx = np.random.choice(total_idx, size=int(train_val_split * len(total_idx)), replace=False)
    val_test_idx = np.delete(total_idx, train_idx)

    train_files = np.sort(np.array(files[train_idx]))
    val_test_files = np.sort(np.array(files[val_test_idx]))

    val_test_idx = np.arange(0, len(val_test_files), 1)
    val_idx = np.random.choice(val_test_idx, size=int(val_test_split * len(val_test_idx)), replace=False)
    test_idx = np.delete(val_test_idx, val_idx)

    val_files = np.sort(np.array(val_test_files[val_idx]))
    test_files = np.sort(np.array(val_test_files[test_idx]))

    print("Chosen Train Files {}/{}".format(len(train_files), len(files)))
    print("Chosen Val Files {}/{}".format(len(val_files), len(files)))
    print("Chosen Test Files {}/{}".format(len(test_files), len(files)))

    if image_ext == config.SYN_RGB_EXT:
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
        if idx % 1000 == 0 and idx != 0:
            print(f'image:{idx+1}/{len(train_files)}, image file:{file} ..')

        # object = old_file_name.split('/')[7]
        object = ''

        count = 1000000 + idx
        image_num = str(count)[1:]

        if image_ext == config.SYN_RGB_EXT:
            move_file_name = new_file_name + 'rgb/' + str(image_num) + config.FORMATTED_RGB_EXT
            if idx == 0:
                print(f'Old file: {old_file_name}')
                print(f'New file: {move_file_name}')
            shutil.copyfile(old_file_name, move_file_name)

        elif image_ext == config.SYN_DEPTH_EXT:
            move_file_name = new_file_name + 'depth/' + str(image_num) + config.FORMATTED_DEPTH_EXT
            if idx == 0:
                print(f'Old file: {old_file_name}')
                print(f'New file: {move_file_name}')
            shutil.copyfile(old_file_name, move_file_name)

        elif image_ext == config.SYN_OBJ_LABEL_EXT:
            move_file_name = new_file_name + 'masks_obj/' + str(image_num) + config.FORMATTED_OBJ_LABEL_EXT
            if idx == 0:
                print(f'Old file: {old_file_name}')
                print(f'New file: {move_file_name}')
            shutil.copyfile(old_file_name, move_file_name)
            
        elif image_ext == config.SYN_OBJ_PART_LABEL_EXT:
            move_file_name = new_file_name + 'masks_obj_part/' + str(image_num) + config.FORMATTED_OBJ_PART_LABEL_EXT
            if idx == 0:
                print(f'Old file: {old_file_name}')
                print(f'New file: {move_file_name}')
            shutil.copyfile(old_file_name, move_file_name)

        elif image_ext == config.SYN_AFF_LABEL_EXT:
            move_file_name = new_file_name + 'masks_aff/' + str(image_num) + config.FORMATTED_AFF_LABEL_EXT
            if idx == 0:
                print(f'Old file: {old_file_name}')
                print(f'New file: {move_file_name}')
            shutil.copyfile(old_file_name, move_file_name)

        elif image_ext == config.SYN_META_EXT:
            move_file_name = new_file_name + 'meta/' + str(image_num) + config.FORMATTED_META_EXT
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

        if image_ext == config.SYN_RGB_EXT:
            move_file_name = new_file_name + 'rgb/' + str(image_num) + config.FORMATTED_RGB_EXT
            shutil.copyfile(old_file_name, move_file_name)

        elif image_ext == config.SYN_DEPTH_EXT:
            move_file_name = new_file_name + 'depth/' + str(image_num) + config.FORMATTED_DEPTH_EXT
            shutil.copyfile(old_file_name, move_file_name)
            
        elif image_ext == config.SYN_OBJ_LABEL_EXT:
            move_file_name = new_file_name + 'masks_obj/' + str(image_num) + config.FORMATTED_OBJ_LABEL_EXT
            shutil.copyfile(old_file_name, move_file_name)
        
        elif image_ext == config.SYN_OBJ_PART_LABEL_EXT:
            move_file_name = new_file_name + 'masks_obj_part/' + str(image_num) + config.FORMATTED_OBJ_PART_LABEL_EXT
            shutil.copyfile(old_file_name, move_file_name)

        elif image_ext == config.SYN_AFF_LABEL_EXT:
            move_file_name = new_file_name + 'masks_aff/' + str(image_num) + config.FORMATTED_AFF_LABEL_EXT
            shutil.copyfile(old_file_name, move_file_name)

        elif image_ext == config.SYN_META_EXT:
            move_file_name = new_file_name + 'meta/' + str(image_num) + config.FORMATTED_META_EXT
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

        if image_ext == config.SYN_RGB_EXT:
            move_file_name = new_file_name + 'rgb/' + str(image_num) + config.FORMATTED_RGB_EXT
            shutil.copyfile(old_file_name, move_file_name)

        elif image_ext == config.SYN_DEPTH_EXT:
            move_file_name = new_file_name + 'depth/' + str(image_num) + config.FORMATTED_DEPTH_EXT
            shutil.copyfile(old_file_name, move_file_name)
            
        elif image_ext == config.SYN_OBJ_LABEL_EXT:
            move_file_name = new_file_name + 'masks_obj/' + str(image_num) + config.FORMATTED_OBJ_LABEL_EXT
            shutil.copyfile(old_file_name, move_file_name)
        
        elif image_ext == config.SYN_OBJ_PART_LABEL_EXT:
            move_file_name = new_file_name + 'masks_obj_part/' + str(image_num) + config.FORMATTED_OBJ_PART_LABEL_EXT
            shutil.copyfile(old_file_name, move_file_name)

        elif image_ext == config.SYN_AFF_LABEL_EXT:
            move_file_name = new_file_name + 'masks_aff/' + str(image_num) + config.FORMATTED_AFF_LABEL_EXT
            shutil.copyfile(old_file_name, move_file_name)

        elif image_ext == config.SYN_META_EXT:
            move_file_name = new_file_name + 'meta/' + str(image_num) + config.FORMATTED_META_EXT
            shutil.copyfile(old_file_name, move_file_name)

        else:
            assert "*** IMAGE EXT DOESN'T EXIST ***"
