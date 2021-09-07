import os

import numpy as np
import cv2

import scipy.io as scio


import sys
sys.path.append('../../')

from src import cfg as config

from src.LabelFusion import dataloader as LabelFusionDataloader

from src.utils import helper_utils
from src.utils.dataset import affpose_dataset_utils

from src.utils.pose.transform_obj_to_obj_part_pose import get_obj_part_pose_in_camera_frame


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
        cv2_pose_img = data["cv2_pose_img"]

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
                # TODO: META
                #######################################

                obj_part_r, obj_part_t = get_obj_part_pose_in_camera_frame(obj_r, obj_t,
                                                               dataloader.obj_to_obj_part_transforms[f'{obj_part_id}'],
                                                                           )

                obj_part_meta_idx = str(1000 + obj_part_id)[1:]
                meta['obj_part_rotation_' + str(obj_part_meta_idx)] = obj_part_r
                meta['obj_part_translation_' + str(obj_part_meta_idx)] = obj_part_t
                meta['cam_cx'] = dataloader.cam_cx
                meta['cam_cy'] = dataloader.cam_cy
                meta['cam_fx'] = dataloader.cam_fx
                meta['cam_fy'] = dataloader.cam_fy

                if show_plot:

                    #######################################
                    # DRAW OBJ POSE
                    #######################################

                    # projecting 3D model to 2D image
                    obj_centered = dataloader.cld_obj_centered[obj_part_id]
                    imgpts, jac = cv2.projectPoints(obj_centered * 1e3, obj_r, obj_t * 1e3, dataloader.cam_mat, dataloader.cam_dist)
                    cv2_pose_img = cv2.polylines(cv2_pose_img, np.int32([np.squeeze(imgpts)]), True, obj_color)

                    # modify YCB objects rotation matrix
                    _obj_r = affpose_dataset_utils.modify_obj_rotation_matrix_for_grasping(obj_id, obj_r.copy())

                    # draw pose
                    rotV, _ = cv2.Rodrigues(obj_r)
                    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                    axisPoints, _ = cv2.projectPoints(points, rotV, obj_t * 1e3, dataloader.cam_mat, dataloader.cam_dist)
                    cv2_pose_img = cv2.line(cv2_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
                    cv2_pose_img = cv2.line(cv2_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
                    cv2_pose_img = cv2.line(cv2_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 3)

                    #######################################
                    # DRAW OBJ PART POSE
                    #######################################

                    # projecting 3D model to 2D image
                    obj_part_centered = dataloader.cld_obj_part_centered[obj_part_id]
                    imgpts, jac = cv2.projectPoints(obj_part_centered * 1e3, obj_part_r, obj_part_t * 1e3, dataloader.cam_mat, dataloader.cam_dist)
                    cv2_pose_img = cv2.polylines(cv2_pose_img, np.int32([np.squeeze(imgpts)]), True, aff_color)

                    # modify YCB objects rotation matrix
                    _obj_part_r = affpose_dataset_utils.modify_obj_rotation_matrix_for_grasping(obj_id, obj_part_r.copy())

                    # draw pose
                    rotV, _ = cv2.Rodrigues(obj_part_r)
                    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                    axisPoints, _ = cv2.projectPoints(points, rotV, obj_part_t * 1e3, dataloader.cam_mat, dataloader.cam_dist)
                    cv2_pose_img = cv2.line(cv2_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
                    cv2_pose_img = cv2.line(cv2_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
                    cv2_pose_img = cv2.line(cv2_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 3)

        #####################
        # WRITING DATASET
        #####################

        str_folder = dataloader.img_path.split('/')[-3]
        str_num = image_addr.split('/')[-1].split('_')[0]
        LABELFUSION_AFF_DATASET_FOLDER = config.ROOT_DATA_PATH + 'LabelFusion/dataset_wam_single/' + str_folder + '/images/'
        LABELFUSION_AFF_DATASET_PATH = LABELFUSION_AFF_DATASET_FOLDER + str_num

        if not os.path.exists(LABELFUSION_AFF_DATASET_FOLDER):
            os.makedirs(LABELFUSION_AFF_DATASET_FOLDER)

        # init addr.
        rgb_addr = LABELFUSION_AFF_DATASET_PATH + config.RGB_EXT
        depth_addr = LABELFUSION_AFF_DATASET_PATH + config.DEPTH_EXT
        obj_label_addr = LABELFUSION_AFF_DATASET_PATH + config.OBJ_LABEL_EXT
        meta_addr = LABELFUSION_AFF_DATASET_PATH + config.META_EXT

        # write.
        cv2.imwrite(rgb_addr, cv2.cvtColor(np.array(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(depth_addr, np.array(depth).astype(np.uint16))
        cv2.imwrite(obj_label_addr, np.array(label, dtype=np.uint8))
        scio.savemat(meta_addr, meta)

        #####################
        # PLOTTING
        #####################

        if show_plot:
            cv2.imshow('cv2_pose_img', cv2.cvtColor(cv2_pose_img, cv2.COLOR_BGR2RGB))
            cv2.waitKey(0)


if __name__ == '__main__':
    main()