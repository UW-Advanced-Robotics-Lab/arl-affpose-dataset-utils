import glob
import numpy as np

import cv2
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

from utils import helper_utils
from utils.dataset import affpose_dataset_utils

#######################################
#######################################

def main():

    imgs_path = config.NDDS_PATH + "*/*/*/" + '??????' + config.SYN_RGB_EXT
    image_files = sorted(glob.glob(imgs_path))
    print('Loaded {} Images'.format(len(image_files)))

    # select random test images
    # np.random.seed(0)
    # num_files = 25
    # random_idx = np.random.choice(np.arange(0, int(len(image_files)), 1), size=int(num_files), replace=False)
    # image_files = np.array(image_files)[random_idx]
    # print("Selected Files: {}".format(len(image_files)))

    for image_idx, image_addr in enumerate(image_files):

        file_path = image_addr.split(config.SYN_RGB_EXT)[0]
        str_num = file_path.split('/')[-1]
        print(f'\n{image_idx + 1}/{len(image_files)}, image_addr:{file_path}')

        rgb_addr            = file_path + config.SYN_RGB_EXT
        depth_addr          = file_path + config.SYN_DEPTH_EXT
        obj_part_label_addr = file_path + config.SYN_OBJ_PART_LABEL_EXT

        rgb            = np.array(Image.open(rgb_addr))
        depth          = np.array(Image.open(depth_addr))
        obj_part_label = np.array(Image.open(obj_part_label_addr))

        #####################
        # obj and aff masks
        #####################
        label = affpose_dataset_utils.convert_obj_part_mask_to_obj_mask(obj_part_label)
        aff_label = affpose_dataset_utils.convert_obj_part_mask_to_aff_mask(obj_part_label)

        #####################
        # DEPTH INFO
        #####################

        helper_utils.print_depth_info(depth)
        depth = helper_utils.convert_16_bit_depth_to_8_bit(depth)

        #####################
        # LABEL INFO
        #####################

        helper_utils.print_class_labels(label)
        helper_utils.print_class_labels(aff_label)

        #####################
        # PLOTTING
        #####################

        rgb             = cv2.resize(rgb, config.RESIZE)
        depth           = cv2.resize(depth, config.RESIZE)
        label           = cv2.resize(label, config.RESIZE)
        color_label     = affpose_dataset_utils.colorize_obj_mask(label)
        aff_label       = cv2.resize(aff_label, config.RESIZE)
        color_aff_label = affpose_dataset_utils.colorize_aff_mask(aff_label)

        cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        cv2.imshow('depth', depth)
        cv2.imshow('heatmap', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
        cv2.imshow('label', cv2.cvtColor(color_label, cv2.COLOR_BGR2RGB))
        cv2.imshow('aff_label', cv2.cvtColor(color_aff_label, cv2.COLOR_BGR2RGB))

        cv2.waitKey(0)

if __name__ == '__main__':
    main()