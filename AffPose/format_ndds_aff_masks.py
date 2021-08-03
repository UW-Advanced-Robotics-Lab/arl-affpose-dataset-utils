import numpy as np
import cv2

import scipy.io as scio

#######################################
#######################################

import sys
sys.path.append('../')

#######################################
#######################################

import cfg as config

from NDDS import dataloader as NDDSDataloader

from utils import helper_utils
from utils.dataset import affpose_dataset_utils

from utils.pose.load_obj_ply_files import load_obj_ply_files
from utils.pose.transform_obj_to_obj_part_pose import get_obj_part_pose_in_camera_frame

#######################################
#######################################

def main():

    # flag to show plotting.
    show_plot = False

    # Load ARL AffPose Images.
    dataloader = NDDSDataloader.ARLAffPose(scene = '6_*')

    for image_idx, image_addr in enumerate(dataloader.img_files):
        data = dataloader._get_ndds_item(image_idx)

        rgb = data["rgb"]
        obj_label = data["label"]
        aff_label = data["aff_label"]

        #####################
        # WRITING AFF DATASET
        #####################

        obj_label_addr = dataloader.file_path + config.SYN_OBJ_LABEL_EXT
        aff_label_addr = dataloader.file_path + config.SYN_AFF_LABEL_EXT

        cv2.imwrite(obj_label_addr, np.array(obj_label, dtype=np.uint8))
        cv2.imwrite(aff_label_addr, np.array(aff_label, dtype=np.uint8))

        #####################
        # PLOTTING
        #####################

        if show_plot:
            # obj mask.
            colour_obj_label = affpose_dataset_utils.colorize_obj_mask(obj_label)
            colour_obj_label = cv2.addWeighted(rgb, 0.35, colour_obj_label, 0.65, 0)
            cv2.imshow('obj_label', cv2.cvtColor(colour_obj_label, cv2.COLOR_BGR2RGB))
            # aff mask.
            colour_aff_label = affpose_dataset_utils.colorize_aff_mask(aff_label)
            colour_aff_label = cv2.addWeighted(rgb, 0.35, colour_aff_label, 0.65, 0)
            cv2.imshow('aff_label', cv2.cvtColor(colour_aff_label, cv2.COLOR_BGR2RGB))

            cv2.waitKey(0)

if __name__ == '__main__':
    main()