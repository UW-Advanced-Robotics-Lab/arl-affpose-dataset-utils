import os

import numpy as np
import cv2

import scipy.io as scio

import sys
sys.path.append('../../')

import cfg as config

from LabelFusion import dataloader as LabelFusionDataloader

from utils import helper_utils
from utils.dataset import affpose_dataset_utils


def main():

    # flag to show plotting.
    show_plot = False

    # Load ARL AffPose Images.
    dataloader = LabelFusionDataloader.ARLAffPose(subset='train',
                                                  subfolder='*',
                                                  _subdivide_images=False,
                                                  _subdivide_idx=3,
                                                  _num_subdivides=4,
                                                  )

    for image_idx, image_addr in enumerate(dataloader.img_files):
        data = dataloader.get_labelfusion_item(image_idx)

        rgb = data["rgb"]
        depth = data["depth"]
        label = data["label"]
        meta = data["meta"]
        colour_label = data["colour_label"]
        cv2_pose_img = data["cv2_pose_img"]

        #######################################
        # OBJECT
        #######################################

        aff_label = np.zeros(shape=(label.shape))
        obj_part_label = np.zeros(shape=(label.shape))

        #######################################
        # OBJECT
        #######################################

        obj_ids = np.array(meta['object_class_ids']).flatten()
        for idx, obj_id in enumerate(obj_ids):
            obj_color = affpose_dataset_utils.obj_color_map(obj_id)
            print(f"\tObject: {obj_id}, {dataloader.obj_classes[int(obj_id) - 1]}")

            obj_meta_idx = str(1000 + obj_id)[1:]
            obj_r = meta['obj_rotation_' + str(obj_meta_idx)]
            obj_t = meta['obj_translation_' + str(obj_meta_idx)]

            obj_r = np.array(obj_r, dtype=np.float64).reshape(3, 3)
            obj_t = np.array(obj_t, dtype=np.float64).reshape(-1, 3)

            #######################################
            # ITERATE OVER OBJ PARTS
            #######################################

            obj_part_ids = affpose_dataset_utils.map_obj_id_to_obj_part_ids(obj_id)
            print(f'\tobj_part_ids:{obj_part_ids}')#
            for obj_part_id in obj_part_ids:
                aff_id = affpose_dataset_utils.map_obj_part_id_to_aff_id(obj_part_id)
                aff_color = affpose_dataset_utils.aff_color_map(aff_id)
                print(f"\t\tAff: {aff_id}, {dataloader.obj_part_classes[int(obj_part_id) - 1]}")

                #######################################
                # DRAW OBJ POSE
                #######################################

                # projecting 3D model to 2D image
                obj_centered = dataloader.cld_obj_centered[obj_part_id]
                imgpts, jac = cv2.projectPoints(obj_centered * 1e3, obj_r, obj_t * 1e3, dataloader.cam_mat, dataloader.cam_dist)
                cv2_pose_img = cv2.polylines(cv2_pose_img, np.int32([np.squeeze(imgpts)]), True, obj_color)

                #######################################
                # AFF LABEL
                #######################################

                # Obj mask.
                mask_label = np.ma.getmaskarray(np.ma.masked_equal(label, obj_id))

                # Obj part mask.
                _obj_part_label = np.zeros(shape=(label.shape), dtype=np.uint8)
                _obj_part_label = cv2.polylines(_obj_part_label, np.int32([np.squeeze(imgpts)]), False, (obj_part_id))
                mask_obj_part_label = np.ma.getmaskarray(np.ma.masked_equal(_obj_part_label, obj_part_id))

                # Aff mask.
                mask_aff_label = mask_label * mask_obj_part_label
                res = cv2.findContours(mask_aff_label.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                contours = res[-2]  # for cv2 v3 and v4+ compatibility

                aff_label = cv2.drawContours(aff_label, contours, contourIdx=-1, color=aff_id, thickness=-1)
                obj_part_label = cv2.drawContours(obj_part_label, contours, contourIdx=-1, color=obj_part_id, thickness=-1)

        helper_utils.print_class_labels(aff_label)
        helper_utils.print_class_labels(obj_part_label)

        #####################
        # WRITING AFF DATASET
        #####################

        str_folder = dataloader.img_path.split('/')[-3]
        str_num = image_addr.split('/')[-1].split('_')[0]
        LABELFUSION_AFF_DATASET_FOLDER = config.ROOT_DATA_PATH + 'LabelFusion/dataset_wam_single/' + str_folder + '/images/'
        LABELFUSION_AFF_DATASET_PATH = LABELFUSION_AFF_DATASET_FOLDER + str_num

        if not os.path.exists(LABELFUSION_AFF_DATASET_FOLDER):
            os.makedirs(LABELFUSION_AFF_DATASET_FOLDER)

        aff_label_addr = LABELFUSION_AFF_DATASET_PATH + config.AFF_LABEL_EXT
        obj_part_label_addr = LABELFUSION_AFF_DATASET_PATH + config.OBJ_PART_LABEL_EXT

        cv2.imwrite(aff_label_addr, np.array(aff_label, dtype=np.uint8))
        cv2.imwrite(obj_part_label_addr, np.array(obj_part_label, dtype=np.uint8))

        #####################
        # PLOTTING
        #####################

        if show_plot:
            # aff mask.
            colour_aff_label = affpose_dataset_utils.colorize_aff_mask(aff_label)
            colour_aff_label = cv2.addWeighted(rgb, 0.35, colour_aff_label, 0.65, 0)
            cv2.imshow('aff_label', cv2.cvtColor(colour_aff_label, cv2.COLOR_BGR2RGB))
            # obj part mask.
            obj_label = affpose_dataset_utils.convert_obj_part_mask_to_obj_mask(obj_part_label)
            colour_obj_label = affpose_dataset_utils.colorize_obj_mask(obj_label)
            colour_obj_label = cv2.addWeighted(rgb, 0.35, colour_obj_label, 0.65, 0)
            cv2.imshow('obj_part_label', cv2.cvtColor(colour_obj_label, cv2.COLOR_BGR2RGB))
            # original obj mask.
            # cv2.imshow('obj_label', cv2.cvtColor(colour_label, cv2.COLOR_BGR2RGB))

            cv2.waitKey(1)


if __name__ == '__main__':
    main()