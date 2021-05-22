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
sys.path.append('../')
# print(sys.path)

#######################################
#######################################

import cfg as config

#######################################
#######################################

img_types = ['Real/', 'Syn/',]

data_path     = config.ROOT_DATA_PATH
new_data_path = config.ROOT_DATA_PATH + 'RealandSyn/'

image_exts = [
            config.FORMATTED_RGB_EXT,
            config.FORMATTED_DEPTH_EXT,
            config.FORMATTED_OBJ_LABEL_EXT,
            config.FORMATTED_OBJ_PART_LABEL_EXT,
            config.FORMATTED_AFF_LABEL_EXT,
            config.FORMATTED_META_EXT
]

########################
########################
offset_train, offset_val = 0, 0
train_files_len, val_files_len = 0, 0

for img_type in img_types:
    print(f"\n *** {img_type} ***")
    for image_ext in image_exts:

        train_path = data_path + img_type + 'train/*/' + '??????' + image_ext
        val_path = data_path + img_type + 'val/*/' + '??????' + image_ext

        train_files = np.array(sorted(glob.glob(train_path)))
        val_files = np.array(sorted(glob.glob(val_path)))

        print(f"\nLoaded Train files: {len(train_files)}, Offset: {offset_train}")
        print(f"Loaded Val files: {len(val_files)}, Offset: {offset_val}")
        print("File path: ", train_path)

        ###############
        # train
        ###############
        split_folder = 'train/'

        for idx, file in enumerate(train_files):
            old_file_name = file
            new_file_name = new_data_path + split_folder
            if idx % 1000 == 0 and idx != 0:
                print(f'image:{idx + 1}/{len(train_files)}, image file:{file} ..')

            count = 1000000 + idx + offset_train
            image_num = str(count)[1:]

            if image_ext == config.FORMATTED_RGB_EXT:
                move_file_name = new_file_name + 'rgb/' + np.str(image_num) + config.FORMATTED_RGB_EXT
                if idx == 0:
                    print(f'Old file: {old_file_name}')
                    print(f'New file: {move_file_name}')
                shutil.copyfile(old_file_name, move_file_name)

            elif image_ext == config.FORMATTED_DEPTH_EXT:
                move_file_name = new_file_name + 'depth/' + np.str(image_num) + config.FORMATTED_DEPTH_EXT
                if idx == 0:
                    print(f'Old file: {old_file_name}')
                    print(f'New file: {move_file_name}')
                shutil.copyfile(old_file_name, move_file_name)

            elif image_ext == config.FORMATTED_OBJ_LABEL_EXT:
                move_file_name = new_file_name + 'masks_obj/' + np.str(image_num) + config.FORMATTED_OBJ_LABEL_EXT
                if idx == 0:
                    print(f'Old file: {old_file_name}')
                    print(f'New file: {move_file_name}')
                shutil.copyfile(old_file_name, move_file_name)

            elif image_ext == config.FORMATTED_OBJ_PART_LABEL_EXT:
                move_file_name = new_file_name + 'masks_obj_part/' + np.str(image_num) + config.FORMATTED_OBJ_PART_LABEL_EXT
                if idx == 0:
                    print(f'Old file: {old_file_name}')
                    print(f'New file: {move_file_name}')
                shutil.copyfile(old_file_name, move_file_name)

            elif image_ext == config.FORMATTED_AFF_LABEL_EXT:
                move_file_name = new_file_name + 'masks_aff/' + np.str(image_num) + config.FORMATTED_AFF_LABEL_EXT
                if idx == 0:
                    print(f'Old file: {old_file_name}')
                    print(f'New file: {move_file_name}')
                shutil.copyfile(old_file_name, move_file_name)

            elif image_ext == config.FORMATTED_META_EXT:
                move_file_name = new_file_name + 'meta/' + np.str(image_num) + config.FORMATTED_META_EXT
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

            count = 1000000 + idx + offset_val
            image_num = str(count)[1:]

            if image_ext == config.FORMATTED_RGB_EXT:
                move_file_name = new_file_name + 'rgb/' + np.str(image_num) + config.FORMATTED_RGB_EXT
                shutil.copyfile(old_file_name, move_file_name)

            elif image_ext == config.FORMATTED_DEPTH_EXT:
                move_file_name = new_file_name + 'depth/' + np.str(image_num) + config.FORMATTED_DEPTH_EXT
                shutil.copyfile(old_file_name, move_file_name)

            elif image_ext == config.FORMATTED_OBJ_LABEL_EXT:
                move_file_name = new_file_name + 'masks_obj/' + np.str(image_num) + config.FORMATTED_OBJ_LABEL_EXT
                shutil.copyfile(old_file_name, move_file_name)

            elif image_ext == config.FORMATTED_OBJ_PART_LABEL_EXT:
                move_file_name = new_file_name + 'masks_obj_part/' + np.str(image_num) + config.FORMATTED_OBJ_PART_LABEL_EXT
                shutil.copyfile(old_file_name, move_file_name)

            elif image_ext == config.FORMATTED_AFF_LABEL_EXT:
                move_file_name = new_file_name + 'masks_aff/' + np.str(image_num) + config.FORMATTED_AFF_LABEL_EXT
                shutil.copyfile(old_file_name, move_file_name)

            elif image_ext == config.FORMATTED_META_EXT:
                move_file_name = new_file_name + 'meta/' + np.str(image_num) + config.FORMATTED_META_EXT
                shutil.copyfile(old_file_name, move_file_name)

            else:
                assert "*** IMAGE EXT DOESN'T EXIST ***"

        ###############
        ###############

        if image_ext == config.FORMATTED_META_EXT:
            offset_train += len(train_files)
            offset_val += len(val_files)