import os
import glob
import json
import numpy as np

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import scipy.io as scio

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

from utils.pose.load_obj_ply_files import load_obj_ply_files
from utils.pose.load_obj_6dof_pose import load_obj_6dof_pose

from utils.bbox.extract_bboxs_from_label import get_obj_bbox

#######################################
#######################################

def main():

    ###################################
    # Load Ply files
    ###################################

    cld, cld_obj_centered, cld_obj_part_centered, obj_classes, obj_part_classes = load_obj_ply_files()

    ##################################
    ##################################

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

        image_addr = image_addr.split(config.SYN_RGB_EXT)[0]
        str_num = image_addr.split('/')[-1]
        print(f'\n{image_idx+1}/{len(image_files)}, image_addr:{image_addr}')

        rgb_addr            = image_addr + config.SYN_RGB_EXT
        depth_addr          = image_addr + config.SYN_DEPTH_EXT
        obj_part_label_addr = image_addr + config.SYN_OBJ_PART_LABEL_EXT

        rgb            = np.array(Image.open(rgb_addr))
        depth          = np.array(Image.open(depth_addr))
        obj_part_label = np.array(Image.open(obj_part_label_addr))

        #####################
        # obj and aff masks
        #####################
        label = affpose_dataset_utils.convert_obj_part_mask_to_obj_mask(obj_part_label)
        aff_label = affpose_dataset_utils.convert_obj_part_mask_to_aff_mask(obj_part_label)

        obj_ids = np.unique(label)[1:]
        sorted_obj_idx = np.arange(start=0, stop=len(obj_ids))

        #####################
        # 6D POSE
        #####################
        cv2_obj_img = rgb.copy()

        json_addr = image_addr + config.SYN_JSON_EXT
        json_file = json.load(open(json_addr))
        object_parts = json_file['objects']

        #####################
        # affordances
        #####################

        obj_label_addr = image_addr + config.SYN_OBJ_LABEL_EXT
        aff_label_addr = image_addr + config.SYN_AFF_LABEL_EXT
        aff_meta_addr  = image_addr + config.META_EXT

        # for new affordances
        aff_meta = {}

        # ZED CAMERA
        aff_meta['width_' + np.str(str_num)] = config.WIDTH
        aff_meta['height_' + np.str(str_num)] = config.HEIGHT
        aff_meta['border_' + np.str(str_num)] = config.BORDER_LIST

        aff_meta['camera_scale_' + np.str(str_num)] = config.CAMERA_SCALE
        aff_meta['fx_' + np.str(str_num)] = config.CAM_FX
        aff_meta['fy_' + np.str(str_num)] = config.CAM_FY
        aff_meta['cx_' + np.str(str_num)] = config.CAM_CX
        aff_meta['cy_' + np.str(str_num)] = config.CAM_CY

        aff_meta['object_class_ids'] = obj_ids
        aff_meta['sorted_obj_idx'] = sorted_obj_idx

        ####################
        ####################

        for idx, object_part in enumerate(object_parts):

            ####################
            # aff id
            ####################
            actor_tag = object_part['class']

            object_part_id = np.int(actor_tag.split("_")[0])
            object_id = affpose_dataset_utils.map_obj_part_id_to_obj_id(object_part_id)
            print(f"Object: {object_part_id}, {obj_part_classes[int(object_part_id) - 1]}")
            object_mesh_id = actor_tag.split("_")[1]
            affordance_name = actor_tag.split("_")[-1]

            aff_id = affpose_dataset_utils.map_obj_part_id_to_aff_id(object_part_id)
            print(f"\tAff: {aff_id}, {obj_part_classes[int(object_part_id) - 1]}")

            ####################
            # SE3
            ####################
            rot = np.asarray(object_part['pose_transform'])[0:3, 0:3]

            # change LHS coordinates
            target_r = np.dot(np.array(rot), np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]))
            target_r = np.dot(target_r.T, np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]))

            # NDDS gives units in centimeters
            target_t = np.array(object_part['location']) * 10  # now in [mm]
            target_t /= 1e3  # now in [m]

            ####################
            # bbox
            ####################
            # x1,y1 ------
            # |          |
            # |          |
            # |          |
            # --------x2,y2
            object_part_y1, object_part_x1 = np.asarray(object_part['bounding_box']['top_left'])
            object_part_y2, object_part_x2 = np.asarray(object_part['bounding_box']['bottom_right'])
            object_part_x1, object_part_y1, object_part_x2, object_part_y2 = \
                int(object_part_x1), int(object_part_y1), int(object_part_x2), int(object_part_y2)

            x1, y1, x2, y2 = get_obj_bbox(label, object_id, config.HEIGHT, config.WIDTH, config.BORDER_LIST)

            # ###################
            # ###################
            obj_color = affpose_dataset_utils.obj_color_map(object_id)
            aff_color = affpose_dataset_utils.aff_color_map(aff_id)

            # projecting 3D model to 2D image
            imgpts, jac = cv2.projectPoints(cld_obj_part_centered[object_part_id] * 1e3, target_r, target_t * 1e3, config.CAM_MAT, config.CAM_DIST)
            cv2_obj_img = cv2.polylines(cv2_obj_img, np.int32([np.squeeze(imgpts)]), True, aff_color)

            # drawing bbox = (x1, y1), (x2, y2)
            cv2_obj_img = cv2.rectangle(cv2_obj_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2_obj_img = cv2.putText(cv2_obj_img,
                                      affpose_dataset_utils.map_obj_id_to_name(object_id),
                                      (x1, y1 - 5),
                                      cv2.FONT_ITALIC,
                                      0.4,
                                      (255, 255, 255))
            if aff_id in [1, 7]:
                # draw pose
                rotV, _ = cv2.Rodrigues(target_r)
                points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                axisPoints, _ = cv2.projectPoints(points, rotV, target_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (0, 0, 255), 3)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (255, 0, 0), 3)

            #######################################
            # meta
            #######################################
            obj_meta_idx = str(1000 + object_id)[1:]
            aff_meta['obj_bbox_' + np.str(obj_meta_idx)] = np.array([x1, y1, x2, y2])
            aff_meta['obj_rotation_' + np.str(obj_meta_idx)] = target_r
            aff_meta['obj_translation_' + np.str(obj_meta_idx)] = target_t

            obj_part_id_idx = str(1000 + object_part_id)[1:]
            aff_meta['obj_part_bbox_' + np.str(obj_part_id_idx)] = np.array([object_part_x1, object_part_y1, object_part_x2, object_part_y2])
            aff_meta['obj_part_rotation_' + np.str(obj_part_id_idx)] = target_r
            aff_meta['obj_part_translation_' + np.str(obj_part_id_idx)] = target_t
        aff_meta['aff_ids'] = np.unique(aff_label)[1:]

        #####################
        # WRITING AFF DATASET
        #####################

        ## CV2 does all operations in BGR
        cv2.imwrite(obj_label_addr, np.array(label, dtype=np.uint8))
        cv2.imwrite(aff_label_addr, np.array(aff_label, dtype=np.uint8))
        scio.savemat(aff_meta_addr, aff_meta)

        #####################
        # DEPTH INFO
        #####################

        helper_utils.print_depth_info(depth)
        depth = helper_utils.convert_16_bit_depth_to_8_bit(depth)

        #####################
        # LABEL INFO
        #####################

        label = affpose_dataset_utils.convert_obj_part_mask_to_obj_mask(obj_part_label)
        helper_utils.print_class_labels(label)

        #####################
        # PLOTTING
        #####################

        # rgb             = cv2.resize(rgb, config.RESIZE)
        # depth           = cv2.resize(depth, config.RESIZE)
        # label           = cv2.resize(label, config.RESIZE)
        # color_label     = affpose_dataset_utils.colorize_obj_mask(label)
        # aff_label       = cv2.resize(aff_label, config.RESIZE)
        # color_aff_label = affpose_dataset_utils.colorize_aff_mask(aff_label)
        # cv2_obj_img     = cv2.resize(cv2_obj_img, config.RESIZE)
        #
        # cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        # cv2.imshow('depth', depth)
        # cv2.imshow('heatmap', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
        # cv2.imshow('label', cv2.cvtColor(color_label, cv2.COLOR_BGR2RGB))
        # cv2.imshow('aff_label', cv2.cvtColor(color_aff_label, cv2.COLOR_BGR2RGB))
        # cv2.imshow('gt_pose', cv2.cvtColor(cv2_obj_img, cv2.COLOR_BGR2RGB))
        #
        # cv2.waitKey(0)

if __name__ == '__main__':
    main()