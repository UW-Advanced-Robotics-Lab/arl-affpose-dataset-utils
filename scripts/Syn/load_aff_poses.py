import glob
import json
import numpy as np

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import scipy.io as scio

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

        # gt pose
        meta_addr = image_addr + config.SYN_META_EXT
        aff_meta = scio.loadmat(meta_addr)

        cv2_obj_img = rgb.copy()
        cv2_obj_parts_img = rgb.copy()

        #####################
        # meta
        #####################

        obj_ids = np.array(aff_meta['object_class_ids']).flatten()

        #######################################
        #######################################

        for idx, obj_id in enumerate(obj_ids):
            ####################
            ####################
            print("Object:", obj_classes[int(obj_id) - 1])

            #######################################
            #######################################
            obj_color = affpose_dataset_utils.obj_color_map(obj_id)

            obj_meta_idx = str(1000 + obj_id)[1:]
            obj_bbox = np.array(aff_meta['obj_bbox_' + np.str(obj_meta_idx)]).flatten()
            x1, y1, x2, y2 = obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3]

            obj_r = aff_meta['obj_rotation_' + np.str(obj_meta_idx)]
            obj_t = aff_meta['obj_translation_' + np.str(obj_meta_idx)]

            #######################################
            # ITERATE OVER OBJ PARTS
            #######################################

            obj_part_ids = affpose_dataset_utils.map_obj_id_to_obj_part_ids(obj_id)
            print(f'obj_part_ids:{obj_part_ids}')
            for obj_part_id in obj_part_ids:
                aff_id = affpose_dataset_utils.map_obj_part_id_to_aff_id(obj_part_id)
                print(f"\tAff: {aff_id}, {obj_part_classes[int(obj_part_id) - 1]}")

                obj_part_id_idx = str(1000 + obj_part_id)[1:]
                obj_part_bbox = np.array(aff_meta['obj_part_bbox_' + np.str(obj_part_id_idx)]).flatten()
                obj_part_x1, obj_part_y1, obj_part_x2, obj_part_y2 = \
                    obj_part_bbox[0], obj_part_bbox[1], obj_part_bbox[2], obj_part_bbox[3]
                obj_part_r = aff_meta['obj_part_rotation_' + np.str(obj_part_id_idx)]
                obj_part_t = aff_meta['obj_part_translation_' + np.str(obj_part_id_idx)]

                #######################################
                #######################################
                aff_color = affpose_dataset_utils.aff_color_map(aff_id)

                # draw model
                obj_parts_imgpts, jac = cv2.projectPoints(cld_obj_part_centered[obj_part_id] * 1e3, obj_part_r,
                                                          obj_part_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                cv2_obj_parts_img = cv2.polylines(cv2_obj_parts_img, np.int32([np.squeeze(obj_parts_imgpts)]), False,
                                                  aff_color)

                # drawing bbox = (x1, y1), (x2, y2)
                cv2_obj_parts_img = cv2.rectangle(cv2_obj_parts_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # white
                # cv2_obj_parts_img = cv2.rectangle(cv2_obj_parts_img, (obj_part_x1, obj_part_y1), (obj_part_x2, obj_part_y2), aff_color, 2)

                cv2_obj_parts_img = cv2.putText(cv2_obj_parts_img,
                                                affpose_dataset_utils.map_obj_id_to_name(obj_id),
                                                (x1, y1 - 5),
                                                cv2.FONT_ITALIC,
                                                0.4,
                                                (255, 255, 255))  # red

                if aff_id == 1 or aff_id == 7:
                    # draw pose
                    rotV, _ = cv2.Rodrigues(obj_part_r)
                    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                    axisPoints, _ = cv2.projectPoints(points, rotV, obj_part_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                    cv2_obj_parts_img = cv2.line(cv2_obj_parts_img, tuple(axisPoints[3].ravel()),
                                                 tuple(axisPoints[0].ravel()), (0, 0, 255), 3)
                    cv2_obj_parts_img = cv2.line(cv2_obj_parts_img, tuple(axisPoints[3].ravel()),
                                                 tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
                    cv2_obj_parts_img = cv2.line(cv2_obj_parts_img, tuple(axisPoints[3].ravel()),
                                                 tuple(axisPoints[2].ravel()), (255, 0, 0), 3)

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

        rgb               = cv2.resize(rgb, config.RESIZE)
        depth             = cv2.resize(depth, config.RESIZE)
        obj_label         = cv2.resize(obj_label, config.RESIZE)
        color_obj_label   = affpose_dataset_utils.colorize_obj_mask(obj_label)
        aff_label         = cv2.resize(aff_label, config.RESIZE)
        color_aff_label   = affpose_dataset_utils.colorize_aff_mask(aff_label)
        cv2_obj_parts_img = cv2.resize(cv2_obj_parts_img, config.RESIZE)

        cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        cv2.imshow('depth', depth)
        cv2.imshow('heatmap', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
        cv2.imshow('obj_label', cv2.cvtColor(color_obj_label, cv2.COLOR_BGR2RGB))
        cv2.imshow('aff_label', cv2.cvtColor(color_aff_label, cv2.COLOR_BGR2RGB))
        cv2.imshow('gt_pose', cv2.cvtColor(cv2_obj_parts_img, cv2.COLOR_BGR2RGB))

        cv2.waitKey(0)

if __name__ == '__main__':
    main()