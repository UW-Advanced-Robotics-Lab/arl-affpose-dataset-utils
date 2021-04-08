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

    imgs_path = config.LABELFUSION_LOG_PATH + "*" + config.RGB_EXT
    # imgs_path = '/data/Akeaveny/Datasets/ARLAffPose/LabelFusion/train/' + "*/*/*" + config.RGB_EXT
    img_files = sorted(glob.glob(imgs_path))
    print('Loaded {} Images'.format(len(img_files)))

    # select random test images
    # np.random.seed(0)
    # num_files = 25
    # random_idx = np.random.choice(np.arange(0, int(len(img_files)), 1), size=int(num_files), replace=False)
    # img_files = np.array(img_files)[random_idx]
    # print("Chosen Files: {}".format(len(img_files)))

    for image_idx, image_addr in enumerate(img_files):

        str_num = image_addr.split('/')[-1].split(config.RGB_EXT)[0]
        print(f'\nimage:{image_idx+1}/{len(img_files)}, file:{image_addr}')

        rgb_addr   = config.LABELFUSION_LOG_PATH + str_num + config.RGB_EXT
        depth_addr = config.LABELFUSION_LOG_PATH + str_num + config.DEPTH_EXT
        label_addr = config.LABELFUSION_LOG_PATH + str_num + config.OBJ_LABEL_EXT

        rgb      = np.array(Image.open(rgb_addr))
        depth    = np.array(Image.open(depth_addr))
        label    = np.array(Image.open(label_addr))

        #####################
        # DEPTH INFO
        #####################

        helper_utils.print_depth_info(depth)
        depth = helper_utils.convert_16_bit_depth_to_8_bit(depth)

        #####################
        # LABEL INFO
        #####################

        helper_utils.print_class_labels(label)

        #####################
        # PLOTTING
        #####################

        rgb = cv2.resize(rgb, config.RESIZE)
        depth = cv2.resize(depth, config.RESIZE)
        label = cv2.resize(label, config.RESIZE)
        color_label = affpose_dataset_utils.colorize_obj_mask(label)

        cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        cv2.imshow('depth', depth)
        cv2.imshow('heatmap', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
        cv2.imshow('label', cv2.cvtColor(color_label, cv2.COLOR_BGR2RGB))

        cv2.waitKey(0)

if __name__ == '__main__':
    main()