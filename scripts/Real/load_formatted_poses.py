import glob
import copy
import random

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

    imgs_path = config.ROOT_DATA_PATH + "dataset/*/*/*" + config.RGB_EXT
    # imgs_path = config.LABELFUSION_AFF_DATASET_PATH + "*" + config.RGB_EXT
    img_files = sorted(glob.glob(imgs_path))
    print('Loaded {} Images'.format(len(img_files)))

    # select random test images
    # np.random.seed(0)
    # num_files = 25
    # random_idx = np.random.choice(np.arange(0, int(len(img_files)), 1), size=int(num_files), replace=False)
    # img_files = np.array(img_files)[random_idx]
    # print("Chosen Files: {}".format(len(img_files)))

    for image_idx, image_addr in enumerate(img_files):

        file_path = image_addr.split(config.RGB_EXT)[0]
        print(f'\nimage:{image_idx+1}/{len(img_files)}, file:{file_path}')

        rgb_addr            = file_path + config.RGB_EXT
        depth_addr          = file_path + config.DEPTH_EXT
        label_addr          = file_path + config.OBJ_LABEL_EXT
        obj_part_label_addr = file_path + config.OBJ_PART_LABEL_EXT
        aff_label_addr      = file_path + config.AFF_LABEL_EXT

        rgb             = np.array(Image.open(rgb_addr))
        depth           = np.array(Image.open(depth_addr))
        label           = np.array(Image.open(label_addr))
        obj_part_label  = np.array(Image.open(obj_part_label_addr))
        aff_label       = np.array(Image.open(aff_label_addr))

        ##################################
        # RESIZE & CROP
        ##################################

        rgb = cv2.resize(rgb, config.RESIZE, interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth, config.RESIZE, interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, config.RESIZE, interpolation=cv2.INTER_NEAREST)
        obj_part_label = cv2.resize(obj_part_label, config.RESIZE, interpolation=cv2.INTER_NEAREST)
        aff_label = cv2.resize(aff_label, config.RESIZE, interpolation=cv2.INTER_NEAREST)

        rgb = helper_utils.crop(pil_img=rgb, crop_size=config.CROP_SIZE, is_img=True)
        depth = helper_utils.crop(pil_img=depth, crop_size=config.CROP_SIZE)
        label = helper_utils.crop(pil_img=label, crop_size=config.CROP_SIZE)
        obj_part_label = helper_utils.crop(pil_img=obj_part_label, crop_size=config.CROP_SIZE)
        aff_label = helper_utils.crop(pil_img=aff_label, crop_size=config.CROP_SIZE)

        ##################################
        ### META
        ##################################

        # gt pose
        meta_addr = file_path + config.META_EXT
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
            print("Object:", obj_classes[int(obj_id) - 1])

            #######################################
            # OBJECT
            #######################################
            obj_color = affpose_dataset_utils.obj_color_map(obj_id)

            obj_meta_idx = str(1000 + obj_id)[1:]
            obj_r = aff_meta['obj_rotation_' + np.str(obj_meta_idx)]
            obj_t = aff_meta['obj_translation_' + np.str(obj_meta_idx)]

            # obj_bbox = np.array(aff_meta['obj_bbox_' + np.str(obj_meta_idx)]).flatten()
            # x1, y1, x2, y2 = obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3]
            x1, y1, x2, y2 = get_obj_bbox(label, obj_id, config.HEIGHT, config.WIDTH, config.BORDER_LIST)

            # drawing bbox = (x1, y1), (x2, y2)
            cv2_obj_img = cv2.rectangle(cv2_obj_img, (x1, y1), (x2, y2), obj_color, 2)

            cv2_obj_img = cv2.putText(cv2_obj_img,
                                      affpose_dataset_utils.map_obj_id_to_name(obj_id),
                                      (x1, y1 - 5),
                                      cv2.FONT_ITALIC,
                                      0.4,
                                      obj_color)

            #######################################
            # ITERATE OVER OBJ PARTS
            #######################################

            obj_part_ids = affpose_dataset_utils.map_obj_id_to_obj_part_ids(obj_id)
            print(f'obj_part_ids:{obj_part_ids}')
            for obj_part_id in obj_part_ids:
                aff_id = affpose_dataset_utils.map_obj_part_id_to_aff_id(obj_part_id)
                print(f"\tAff: {aff_id}, {obj_part_classes[int(obj_part_id) - 1]}")

                #######################################
                # OBJECT
                #######################################
                obj_centered = cld_obj_centered[obj_part_id]

                # projecting 3D model to 2D image
                imgpts, jac = cv2.projectPoints(obj_centered * 1e3, obj_r, obj_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                cv2_obj_img = cv2.polylines(cv2_obj_img, np.int32([np.squeeze(imgpts)]), True, obj_color)

                # draw pose
                rotV, _ = cv2.Rodrigues(obj_r)
                points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                axisPoints, _ = cv2.projectPoints(points, rotV, obj_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()),(0, 0, 255), 3)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()),(0, 255, 0), 3)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()),(255, 0, 0), 3)

                #######################################
                # OBJECT PART AFF CENTERED
                #######################################
                aff_color = affpose_dataset_utils.aff_color_map(aff_id)

                obj_part_centered = cld_obj_part_centered[obj_part_id]
                obj_part_id_idx = str(1000 + obj_part_id)[1:]
                obj_part_r = aff_meta['obj_part_rotation_' + np.str(obj_part_id_idx)]
                obj_part_t = aff_meta['obj_part_translation_' + np.str(obj_part_id_idx)]

                # draw model
                obj_parts_imgpts, jac = cv2.projectPoints(obj_part_centered * 1e3, obj_part_r, obj_part_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                cv2_obj_parts_img = cv2.polylines(cv2_obj_parts_img, np.int32([np.squeeze(obj_parts_imgpts)]), False, aff_color)

                # obj_part_bbox = np.array(aff_meta['obj_part_bbox_' + np.str(obj_part_id_idx)]).flatten()
                # obj_part_x1, obj_part_y1, obj_part_x2, obj_part_y2 =\
                #     obj_part_bbox[0], obj_part_bbox[1], obj_part_bbox[2], obj_part_bbox[3]
                obj_part_x1, obj_part_y1, obj_part_x2, obj_part_y2 = get_obj_bbox(obj_part_label, obj_part_id,
                                                                                  config.HEIGHT, config.WIDTH,
                                                                                  config.BORDER_LIST)

                # drawing bbox = (x1, y1), (x2, y2)
                cv2_obj_parts_img = cv2.rectangle(cv2_obj_parts_img, (x1, y1), (x2, y2), (255, 0, 0), 2) # white
                # cv2_obj_parts_img = cv2.rectangle(cv2_obj_parts_img, (obj_part_x1, obj_part_y1), (obj_part_x2, obj_part_y2), aff_color, 2)

                cv2_obj_parts_img = cv2.putText(cv2_obj_parts_img,
                                                affpose_dataset_utils.map_obj_id_to_name(obj_id),
                                                (x1, y1 - 5),
                                                cv2.FONT_ITALIC,
                                                0.4,
                                                (255, 255, 255)) # red

                if aff_id == 1 or aff_id == 7:
                    # draw pose
                    rotV, _ = cv2.Rodrigues(obj_part_r)
                    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                    axisPoints, _ = cv2.projectPoints(points, rotV, obj_part_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                    cv2_obj_parts_img = cv2.line(cv2_obj_parts_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (0, 0, 255), 3)
                    cv2_obj_parts_img = cv2.line(cv2_obj_parts_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
                    cv2_obj_parts_img = cv2.line(cv2_obj_parts_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (255, 0, 0), 3)

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
        cv2.imshow('label', cv2.cvtColor(color_label, cv2.COLOR_BGR2RGB))
        cv2.imshow('obj_part_label', obj_part_label*50)
        cv2.imshow('aff_label', cv2.cvtColor(color_aff_label, cv2.COLOR_BGR2RGB))
        cv2.imshow('gt_obj_pose', cv2.cvtColor(cv2_obj_img, cv2.COLOR_BGR2RGB))
        cv2.imshow('gt_aff_pose', cv2.cvtColor(cv2_obj_parts_img, cv2.COLOR_BGR2RGB))

        cv2.waitKey(0)

if __name__ == '__main__':
    main()