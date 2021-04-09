import glob
import numpy as np

import cv2
from PIL import Image
import matplotlib.pyplot as plt

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
    np.random.seed(0)
    num_files = 25
    random_idx = np.random.choice(np.arange(0, int(len(image_files)), 1), size=int(num_files), replace=False)
    image_files = np.array(image_files)[random_idx]
    print("Selected Files: {}".format(len(image_files)))

    for image_idx, image_addr in enumerate(image_files):

        image_addr = image_addr.split(config.SYN_RGB_EXT)[0]
        str_num = image_addr.split('/')[-1]
        print(f'\n{image_idx + 1}/{len(image_files)}, image_addr:{image_addr}')

        rgb_addr            = image_addr + config.SYN_RGB_EXT
        depth_addr          = image_addr + config.SYN_DEPTH_EXT
        obj_label_addr      = image_addr + config.SYN_OBJ_LABEL_EXT
        aff_label_addr      = image_addr + config.SYN_AFF_LABEL_EXT

        rgb            = np.array(Image.open(rgb_addr))
        depth          = np.array(Image.open(depth_addr))
        obj_label      = np.array(Image.open(obj_label_addr))
        aff_label      = np.array(Image.open(aff_label_addr))

        #####################
        # DEPTH INFO
        #####################

        helper_utils.print_depth_info(depth)
        depth = helper_utils.convert_16_bit_depth_to_8_bit(depth)

        #####################
        # LABEL INFO
        #####################

        helper_utils.print_class_labels(obj_label)
        helper_utils.print_class_labels(aff_label)

        #####################
        # PLOTTING
        #####################

        rgb             = cv2.resize(rgb, config.RESIZE)
        depth           = cv2.resize(depth, config.RESIZE)
        obj_label       = cv2.resize(obj_label, config.RESIZE)
        color_obj_label = affpose_dataset_utils.colorize_obj_mask(obj_label)
        aff_label       = cv2.resize(aff_label, config.RESIZE)
        color_aff_label = affpose_dataset_utils.colorize_aff_mask(aff_label)

        cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        cv2.imshow('depth', depth)
        cv2.imshow('heatmap', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
        cv2.imshow('obj_label', cv2.cvtColor(color_obj_label, cv2.COLOR_BGR2RGB))
        cv2.imshow('aff_label', cv2.cvtColor(color_aff_label, cv2.COLOR_BGR2RGB))

        cv2.waitKey(0)

if __name__ == '__main__':
    main()