import os
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

    # imgs_path = config.ROOT_DATA_PATH + "logs/*/*/*" + config.RGB_EXT
    imgs_path = config.LABELFUSION_LOG_PATH + "*" + config.RGB_EXT
    img_files = sorted(glob.glob(imgs_path))
    print('Loaded {} Images'.format(len(img_files)))

    for image_idx, image_addr in enumerate(img_files):

        # print(f'\nimage:{image_idx+1}/{len(img_files)}, file:{image_addr}')
        str_folder = image_addr.split(config.RGB_EXT)[0].split('/')[7]
        str_num    = image_addr.split(config.RGB_EXT)[0].split('/')[9]

        file_path = image_addr.split(config.RGB_EXT)[0]
        print(f'\nimage:{image_idx+1}/{len(img_files)}, file:{file_path}')

        # rgb_addr   = config.LABELFUSION_LOG_PATH + str_num + config.RGB_EXT
        rgb_addr   = file_path + config.RGB_EXT
        depth_addr = file_path + config.DEPTH_EXT
        label_addr = file_path + config.OBJ_LABEL_EXT

        rgb      = np.array(Image.open(rgb_addr))
        depth    = np.array(Image.open(depth_addr))
        label    = np.array(Image.open(label_addr))

        cv2_obj_img = rgb.copy()
        cv2_obj_parts_img = rgb.copy()

        #####################
        # 6D POSE
        #####################

        yaml_addr = file_path + config.POSE_EXT
        obj_ids, obj_poses = load_obj_6dof_pose(yaml_addr)

        sorted_obj_idx = config.SORTED_OBJ_IDX
        if sorted_obj_idx is None:
            sorted_obj_idx = np.arange(start=0, stop=len(obj_ids))
        obj_ids, obj_poses = obj_ids[sorted_obj_idx], obj_poses[:, :, sorted_obj_idx]

        #####################
        # affordances
        #####################

        LABELFUSION_AFF_DATASET_PATH = config.ROOT_DATA_PATH + 'dataset/' + str_folder + '/images/' + str_num

        aff_rgb_addr            = LABELFUSION_AFF_DATASET_PATH + config.RGB_EXT
        aff_depth_addr          = LABELFUSION_AFF_DATASET_PATH + config.DEPTH_EXT
        aff_obj_label_addr      = LABELFUSION_AFF_DATASET_PATH + config.OBJ_LABEL_EXT
        aff_obj_part_label_addr = LABELFUSION_AFF_DATASET_PATH + config.OBJ_PART_LABEL_EXT
        aff_aff_label_addr      = LABELFUSION_AFF_DATASET_PATH + config.AFF_LABEL_EXT
        aff_meta_addr           = LABELFUSION_AFF_DATASET_PATH + config.META_EXT

        strings = np.str(LABELFUSION_AFF_DATASET_PATH).split(' /')
        new_aff_dir = '/'.join(strings[:-1]) + '/'
        if not os.path.exists(new_aff_dir):
            os.makedirs(new_aff_dir)

        # for new affordances
        aff_label = np.zeros(shape=(label.shape))
        obj_part_label = np.zeros(shape=(label.shape), dtype=np.uint8)

        #######################################
        # TODO: meta
        #######################################
        aff_meta = {}
        aff_meta['object_class_ids'] = obj_ids
        aff_meta['sorted_obj_idx'] = sorted_obj_idx

        #######################################
        #######################################

        for idx, obj_id in enumerate(obj_ids):
            print("Object:", obj_classes[int(obj_id) - 1])

            #######################################
            # OBJECT
            #######################################
            obj_color = affpose_dataset_utils.obj_color_map(obj_id)

            target_r = obj_poses[0:3, 0:3, idx]
            target_t = obj_poses[0:3, -1, idx]

            target_r = np.array(target_r, dtype=np.float64).reshape(3, 3)
            target_t = np.array(target_t, dtype=np.float64).reshape(-1, 3)

            # drawing bbox = (x1, y1), (x2, y2)
            x1, y1, x2, y2 = get_obj_bbox(label, obj_id, config.HEIGHT, config.WIDTH, config.BORDER_LIST)
            cv2_obj_img = cv2.rectangle(cv2_obj_img, (x1, y1), (x2, y2), obj_color, 2)

            cv2_obj_img = cv2.putText(cv2_obj_img,
                                      affpose_dataset_utils.map_obj_id_to_name(obj_id),
                                      (x1, y1 - 5),
                                      cv2.FONT_ITALIC,
                                      0.4,
                                      obj_color)

            #######################################
            # TODO: meta
            #######################################
            obj_meta_idx = str(1000 + obj_id)[1:]
            aff_meta['obj_rotation_' + np.str(obj_meta_idx)] = target_r
            aff_meta['obj_translation_' + np.str(obj_meta_idx)] = target_t

            # print(f'Translation:{target_t}\nRotation:\n{target_r}\n')

            #######################################
            # ITERATE OVER OBJ PARTS
            #######################################

            obj_part_ids = affpose_dataset_utils.map_obj_id_to_obj_part_ids(obj_id)
            print(f'obj_part_ids:{obj_part_ids}')
            for obj_part_id in obj_part_ids:
                aff_id = affpose_dataset_utils.map_obj_part_id_to_aff_id(obj_part_id)
                print(f"\tAff: {aff_id}, {obj_part_classes[int(obj_part_id) - 1]}")

                #######################################
                # OBJECT CENTERED
                #######################################

                obj_centered = cld_obj_centered[obj_part_id]
                obj_r = copy.deepcopy(target_r)
                obj_t = copy.deepcopy(target_t)

                # projecting 3D model to 2D image
                imgpts, jac = cv2.projectPoints(obj_centered * 1e3, obj_r, obj_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                cv2_obj_img = cv2.polylines(cv2_obj_img, np.int32([np.squeeze(imgpts)]), True, obj_color)

                # draw pose
                rotV, _ = cv2.Rodrigues(obj_r)
                points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                axisPoints, _ = cv2.projectPoints(points, rotV, obj_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (0, 0, 255), 3)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (255, 0, 0), 3)

                #######################################
                # OBJECT PART BBOX
                #######################################

                if obj_part_id in affpose_dataset_utils.PROJECT_POINT_CLOUD:
                    obj_part_label = cv2.polylines(obj_part_label, np.int32([np.squeeze(imgpts)]), False, (obj_part_id))
                else:
                    max_row, max_col = label.shape
                    imgpts = np.squeeze(imgpts)
                    rows, cols = imgpts[:, 1], imgpts[:, 0]
                    rows = np.clip(rows, 0, max_row - 1)
                    cols = np.clip(cols, 0, max_col - 1)
                    for row, col in zip(rows, cols):
                        row, col = int(row), int(col)
                        obj_part_label[row][col] = obj_part_id

                obj_part_x1, obj_part_y1, obj_part_x2, obj_part_y2 = get_obj_bbox(obj_part_label, obj_part_id, config.HEIGHT, config.WIDTH, config.BORDER_LIST)

                #######################################
                # AFF LABEL
                #######################################
                if obj_part_id in affpose_dataset_utils.REQUIRE_INSIDE_CONTOURS:
                    res = cv2.findContours(obj_part_label.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                    contours = res[-2]  # for cv2 v3 and v4+ compatibility
                    aff_label = cv2.drawContours(aff_label, contours, contourIdx=-1, color=aff_id, thickness=-1)
                    if obj_part_id == affpose_dataset_utils.REQUIRE_INSIDE_CONTOURS[0]: # spatula
                        aff_label = cv2.drawContours(aff_label, contours, contourIdx=-1, color=aff_id, thickness=2)
                else:
                    res = cv2.findContours(obj_part_label.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    contours = res[-2]  # for cv2 v3 and v4+ compatibility
                    aff_label = cv2.drawContours(aff_label, contours, contourIdx=-1, color=aff_id, thickness=-1)
                    aff_label = cv2.fillPoly(aff_label, contours, color=aff_id)

                #######################################
                # Using PNP RANSAC to get new SE(3) matrix
                #######################################

                object_id_cld_2D = np.squeeze(np.array(imgpts, dtype=np.float32))
                object_id_cld_3D = np.squeeze(np.array(cld_obj_part_centered[obj_part_id], dtype=np.float32))

                # print(f'\tobject_id_cld_2D:{len(object_id_cld_2D)}, object_id_cld_3D:{len(object_id_cld_3D)}')
                if len(object_id_cld_2D) != len(object_id_cld_3D):
                    if len(object_id_cld_2D) > len(object_id_cld_3D):
                        min_points = len(object_id_cld_3D)
                        # idx = np.random.choice(np.arange(0, int(min_points), 1), size=int(min_points), replace=False)
                        idx = np.arange(0, int(min_points), 1)
                        object_id_cld_2D = object_id_cld_2D[idx]
                    else:
                        min_points = len(object_id_cld_2D)
                        # idx = np.random.choice(np.arange(0, int(min_points), 1), size=int(min_points), replace=False)
                        idx = np.arange(0, int(min_points), 1)
                        object_id_cld_3D = object_id_cld_3D[idx]
                    # print(f'\tmin_idx:{len(np.unique(idx))}')
                assert (len(object_id_cld_2D) == len(object_id_cld_3D))
                # print(f'\tobject_id_cld_2D:{len(object_id_cld_2D)}, object_id_cld_3D:{len(object_id_cld_3D)}')

                obj_rvec, _ = cv2.Rodrigues(obj_r)
                # _, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=object_id_cld_3D, imagePoints=object_id_cld_2D,
                _, rvec, tvec = cv2.solvePnP(objectPoints=object_id_cld_3D, imagePoints=object_id_cld_2D,
                                             cameraMatrix=config.CAM_MAT, distCoeffs=config.CAM_DIST,
                                             rvec=obj_rvec,
                                             tvec=obj_t,
                                             useExtrinsicGuess=True,
                                             flags=cv2.SOLVEPNP_ITERATIVE)

                obj_part_r, _ = cv2.Rodrigues(rvec)
                obj_part_r = np.array(obj_part_r, dtype=np.float64).reshape(3, 3)
                obj_part_t = np.array(tvec, dtype=np.float64).reshape(-1, 3)

                # print("R:{},{}\n{}".format(obj_part_r.shape, obj_part_r.dtype, obj_part_r))
                # print("t:{},{}\n{}".format(obj_part_t.shape, obj_part_t.dtype, obj_part_t))

                #######################################
                #######################################
                aff_color = affpose_dataset_utils.aff_color_map(aff_id)

                # draw model
                obj_parts_imgpts, jac = cv2.projectPoints(cld_obj_part_centered[obj_part_id] * 1e3, obj_part_r, obj_part_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                cv2_obj_parts_img = cv2.polylines(cv2_obj_parts_img, np.int32([np.squeeze(obj_parts_imgpts)]), False, aff_color)

                # drawing bbox = (x1, y1), (x2, y2)
                cv2_obj_parts_img = cv2.rectangle(cv2_obj_parts_img, (x1, y1), (x2, y2), (255, 0, 0), 2) # white
                # cv2_obj_parts_img = cv2.rectangle(cv2_obj_parts_img, (obj_part_x1, obj_part_y1), (obj_part_x2, obj_part_y2), aff_color, 2)

                cv2_obj_parts_img = cv2.putText(cv2_obj_parts_img,
                                                affpose_dataset_utils.map_obj_id_to_name(obj_id),
                                                (x1, y1 - 5),
                                                cv2.FONT_ITALIC,
                                                0.4,
                                                (255, 255, 255)) # red

                if obj_part_id in affpose_dataset_utils.DRAW_OBJ_PART_POSE:
                    # draw pose
                    rotV, _ = cv2.Rodrigues(obj_part_r)
                    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                    axisPoints, _ = cv2.projectPoints(points, rotV, obj_part_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                    cv2_obj_parts_img = cv2.line(cv2_obj_parts_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (0, 0, 255), 3)
                    cv2_obj_parts_img = cv2.line(cv2_obj_parts_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
                    cv2_obj_parts_img = cv2.line(cv2_obj_parts_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (255, 0, 0), 3)

                #######################################
                # TODO: meta
                #######################################
                obj_part_id_idx = str(1000 + obj_part_id)[1:]
                aff_meta['obj_part_rotation_' + np.str(obj_part_id_idx)] = obj_part_r
                aff_meta['obj_part_translation_' + np.str(obj_part_id_idx)] = obj_part_t
        aff_meta['aff_ids'] = np.unique(aff_label)[1:]

        #####################
        # WRITING AFF DATASET
        #####################

        ## CV2 does all operations in BGR
        cv2.imwrite(aff_rgb_addr, cv2.cvtColor(np.array(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(aff_depth_addr, np.array(depth).astype(np.uint16))
        cv2.imwrite(aff_obj_label_addr, np.array(label, dtype=np.uint8))
        cv2.imwrite(aff_obj_part_label_addr, np.array(obj_part_label, dtype=np.uint8))
        cv2.imwrite(aff_aff_label_addr, np.array(aff_label, dtype=np.uint8))
        scio.savemat(aff_meta_addr, aff_meta)

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

        # rgb               = cv2.resize(rgb, config.RESIZE)
        # depth             = cv2.resize(depth, config.RESIZE)
        # label             = cv2.resize(label, config.RESIZE)
        # color_label       = affpose_dataset_utils.colorize_obj_mask(label)
        # aff_label         = cv2.resize(aff_label, config.RESIZE)
        # color_aff_label   = affpose_dataset_utils.colorize_aff_mask(aff_label)
        # cv2_obj_img       = cv2.resize(cv2_obj_img, config.RESIZE)
        # cv2_obj_parts_img = cv2.resize(cv2_obj_parts_img, config.RESIZE)
        #
        # cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        # cv2.imshow('depth', depth)
        # cv2.imshow('heatmap', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
        # cv2.imshow('label', cv2.cvtColor(color_label, cv2.COLOR_BGR2RGB))
        # cv2.imshow('aff_label', cv2.cvtColor(color_aff_label, cv2.COLOR_BGR2RGB))
        # cv2.imshow('obj_part_label', obj_part_label*25)
        # cv2.imshow('gt_obj_pose', cv2.cvtColor(cv2_obj_img, cv2.COLOR_BGR2RGB))
        # cv2.imshow('gt_aff_pose', cv2.cvtColor(cv2_obj_parts_img, cv2.COLOR_BGR2RGB))
        #
        # cv2.waitKey(0)

if __name__ == '__main__':
    main()