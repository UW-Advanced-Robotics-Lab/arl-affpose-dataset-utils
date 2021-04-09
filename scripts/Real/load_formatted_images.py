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

    imgs_path = config.ROOT_DATA_PATH + "dataset/*/*/*" + config.RGB_EXT
    # imgs_path = config.LABELFUSION_AFF_DATASET_PATH + "*" + config.RGB_EXT
    img_files = sorted(glob.glob(imgs_path))
    print('Loaded {} Images'.format(len(img_files)))

    # select random test images
    np.random.seed(0)
    num_files = 25
    random_idx = np.random.choice(np.arange(0, int(len(img_files)), 1), size=int(num_files), replace=False)
    img_files = np.array(img_files)[random_idx]
    print("Chosen Files: {}".format(len(img_files)))

    for image_idx, image_addr in enumerate(img_files):

        file_path = image_addr.split(config.RGB_EXT)[0]
        print(f'\nimage:{image_idx+1}/{len(img_files)}, file:{file_path}')

        rgb_addr       = file_path + config.RGB_EXT
        depth_addr     = file_path + config.DEPTH_EXT
        label_addr     = file_path + config.OBJ_LABEL_EXT
        aff_label_addr = file_path + config.AFF_LABEL_EXT

        rgb       = np.array(Image.open(rgb_addr))
        depth     = np.array(Image.open(depth_addr))
        label     = np.array(Image.open(label_addr))
        aff_label = np.array(Image.open(aff_label_addr))

        ##################################
        ### RESIZE & CROP
        ##################################

        rgb = cv2.resize(rgb, config.RESIZE, interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth, config.RESIZE, interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, config.RESIZE, interpolation=cv2.INTER_NEAREST)
        aff_label = cv2.resize(aff_label, config.RESIZE, interpolation=cv2.INTER_NEAREST)

        rgb = helper_utils.crop(pil_img=rgb, crop_size=config.CROP_SIZE, is_img=True)
        depth = helper_utils.crop(pil_img=depth, crop_size=config.CROP_SIZE)
        label = helper_utils.crop(pil_img=label, crop_size=config.CROP_SIZE)
        aff_label = helper_utils.crop(pil_img=aff_label, crop_size=config.CROP_SIZE)

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

        color_label = affpose_dataset_utils.colorize_aff_mask(label)
        color_aff_label = affpose_dataset_utils.colorize_aff_mask(aff_label)

        cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        cv2.imshow('depth', depth)
        cv2.imshow('heatmap', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
        cv2.imshow('color_label', cv2.cvtColor(color_label, cv2.COLOR_BGR2RGB))
        cv2.imshow('aff_label', cv2.cvtColor(color_aff_label, cv2.COLOR_BGR2RGB))

        cv2.waitKey(0)

if __name__ == '__main__':
    main()