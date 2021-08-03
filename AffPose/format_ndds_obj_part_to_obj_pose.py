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

from utils.dataset import affpose_dataset_utils
from utils.pose.transform_obj_to_obj_part_pose import get_obj_pose_in_camera_frame

#######################################
#######################################

def main():

    # flag to show plotting.
    show_plot = False

    # Load ARL AffPose Images.
    dataloader = NDDSDataloader.ARLAffPose(scene = '6_*')

    for image_idx, image_addr in enumerate(dataloader.img_files):
        data = dataloader._get_ndds_item(image_idx)

        meta = data["meta"]
        cv2_pose_img = data["cv2_pose_img"]

        obj_ids = np.array(meta['object_class_ids']).flatten()
        for idx, obj_id in enumerate(obj_ids):
            print(f"\tObject: {obj_id}, {dataloader.obj_classes[int(obj_id) - 1]}")
            obj_color = affpose_dataset_utils.obj_color_map(obj_id)

            #######################################
            # ITERATE OVER OBJ PARTS
            #######################################

            obj_part_ids = affpose_dataset_utils.map_obj_id_to_obj_part_ids(obj_id)

            obj_t = np.zeros(shape=(len(obj_part_ids), 3))
            obj_r = np.zeros(shape=(len(obj_part_ids), 3, 3))

            print(f'\tobj_part_ids:{obj_part_ids}')
            for obj_part_idx, obj_part_id in enumerate(obj_part_ids):
                aff_id = affpose_dataset_utils.map_obj_part_id_to_aff_id(obj_part_id)
                aff_color = affpose_dataset_utils.aff_color_map(aff_id)
                print(f"\t\tAff: {aff_id}, {dataloader.obj_part_classes[int(obj_part_id) - 1]}")

                #######################################
                # OBJECT PART POSE
                #######################################

                obj_part_meta_idx = str(1000 + obj_part_id)[1:]
                obj_part_r = meta['obj_part_rotation_' + np.str(obj_part_meta_idx)]
                obj_part_t = meta['obj_part_translation_' + np.str(obj_part_meta_idx)]

                obj_part_r = np.array(obj_part_r, dtype=np.float64).reshape(3, 3)
                obj_part_t = np.array(obj_part_t, dtype=np.float64).reshape(-1, 3)

                #######################################
                # TODO: META
                #######################################

                obj_r[obj_part_idx, :, :], obj_t[obj_part_idx, :] = get_obj_pose_in_camera_frame(obj_part_r, obj_part_t,
                                                                dataloader.obj_to_obj_part_transforms[f'{obj_part_id}'],
                                                                                                 )

            # averaging obj pose.
            obj_t = np.mean(obj_t, axis=0).reshape(-1)
            obj_r = np.mean(obj_r, axis=0).reshape(3, 3)

            obj_meta_idx = str(1000 + obj_id)[1:]
            meta['obj_rotation_' + np.str(obj_meta_idx)] = obj_r
            meta['obj_translation_' + np.str(obj_meta_idx)] = obj_t
            meta['cam_cx'] = dataloader.cam_cx
            meta['cam_cy'] = dataloader.cam_cy
            meta['cam_fx'] = dataloader.cam_fx
            meta['cam_fy'] = dataloader.cam_fy

            #######################################
            #######################################

            if show_plot:

                # OBJ.
                cld_obj_centered = dataloader.cld[obj_id]
                imgpts, jac = cv2.projectPoints(cld_obj_centered * 1e3, obj_r, obj_t * 1e3, dataloader.cam_mat, dataloader.cam_dist)
                cv2_pose_img = cv2.polylines(cv2_pose_img, np.int32([np.squeeze(imgpts)]), True, obj_color)

                # modify YCB objects rotation matrix
                _obj_r = affpose_dataset_utils.modify_obj_rotation_matrix_for_grasping(obj_id, obj_r.copy())

                # draw pose
                rotV, _ = cv2.Rodrigues(_obj_r)
                points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                axisPoints, _ = cv2.projectPoints(points, rotV, obj_t * 1e3, dataloader.cam_mat, dataloader.cam_dist)
                cv2_pose_img = cv2.line(cv2_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
                cv2_pose_img = cv2.line(cv2_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
                cv2_pose_img = cv2.line(cv2_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 3)

        meta_addr = dataloader.file_path + config.SYN_META_EXT
        scio.savemat(meta_addr, meta)

        #####################
        # PLOTTING
        #####################

        if show_plot:
            cv2.imshow('cv2_pose_img', cv2.cvtColor(cv2_pose_img, cv2.COLOR_BGR2RGB))
            cv2.waitKey(0)


if __name__ == '__main__':
    main()
