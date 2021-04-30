import glob
import copy
import random

import numpy as np

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import scipy.io as scio

import open3d as o3d

#######################################
#######################################

import cfg as config

from utils import helper_utils
from utils.dataset import affpose_dataset_utils

from utils.pose.load_obj_ply_files import load_obj_ply_files
from utils.pose.load_obj_6dof_pose import load_obj_6dof_pose
from utils.pose.create_pointcloud_from_depth import create_pointcloud_from_depth_image, create_masked_pointcloud_from_depth_image

from utils.bbox.extract_bboxs_from_label import get_obj_bbox

#######################################
#######################################

def main():
    ###################################
    # Load Ply files
    ###################################

    # cld, obj_classes = load_ply_files()
    cld, cld_obj_centered, cld_obj_part_centered, obj_classes, obj_part_classes = load_obj_ply_files()

    ##################################
    ##################################

    # imgs_path = config.ROOT_DATA_PATH + "logs/*/*/*" + config.RGB_EXT
    imgs_path = config.LABELFUSION_LOG_PATH + "*" + config.RGB_EXT
    img_files = sorted(glob.glob(imgs_path))
    print('Loaded {} Images'.format(len(img_files)))

    # select random test images
    # np.random.seed(0)
    # num_files = 125 # 125 or len(img_files)
    # random_idx = np.random.choice(np.arange(0, int(len(img_files)), 1), size=int(num_files), replace=False)
    # img_files = np.array(img_files)[random_idx]
    # print("Chosen Files: {}".format(len(img_files)))

    for image_idx, image_addr in enumerate(img_files):

        file_path = image_addr.split(config.RGB_EXT)[0]
        print(f'\nimage:{image_idx + 1}/{len(img_files)}, file:{file_path}')

        rgb_addr = file_path + config.RGB_EXT
        depth_addr = file_path + config.DEPTH_EXT
        label_addr = file_path + config.OBJ_LABEL_EXT

        rgb = np.array(Image.open(rgb_addr))
        depth = np.array(Image.open(depth_addr))
        label = np.array(Image.open(label_addr))

        ##################################
        # RESIZE & CROP
        ##################################

        # rgb = cv2.resize(rgb, config.RESIZE, interpolation=cv2.INTER_CUBIC)
        # depth = cv2.resize(depth, config.RESIZE, interpolation=cv2.INTER_CUBIC)
        # label = cv2.resize(label, config.RESIZE, interpolation=cv2.INTER_NEAREST)
        #
        # rgb = helper_utils.crop(pil_img=rgb, crop_size=config.CROP_SIZE, is_img=True)
        # depth = helper_utils.crop(pil_img=depth, crop_size=config.CROP_SIZE)
        # label = helper_utils.crop(pil_img=label, crop_size=config.CROP_SIZE)

        #####################
        #####################

        cv2_mask_img = rgb.copy()
        cv2_obj_img = rgb.copy()

        #####################
        # MASKED RGB IMG
        #####################

        mask_label = np.ma.getmaskarray(np.ma.masked_not_equal(label, 0)).astype(np.uint8)
        cv2_mask_img = np.repeat(mask_label, 3).reshape(mask_label.shape[0], mask_label.shape[1], -1) * cv2_mask_img

        #####################
        # 6D POSE
        #####################

        yaml_addr = file_path + config.POSE_EXT
        obj_ids, obj_poses = load_obj_6dof_pose(yaml_addr)

        #####################
        #####################

        for idx, obj_id in enumerate(obj_ids):
            ####################
            ####################
            print("Object:", obj_classes[int(obj_id) - 1])

            target_r = obj_poses[0:3, 0:3, idx]
            target_t = obj_poses[0:3, -1, idx]

            target_r = np.array(target_r, dtype=np.float64).reshape(3, 3)
            target_t = np.array(target_t, dtype=np.float64).reshape(-1, 3)

            # print(f'Translation:{target_t}\nRotation:\n{target_r}\n')

            ####################
            # BBOX
            ####################
            obj_color = affpose_dataset_utils.obj_color_map(obj_id)

            # drawing bbox = (x1, y1), (x2, y2)
            # x1, y1, x2, y2 = get_obj_bbox(label, obj_id, config.HEIGHT, config.WIDTH, config.BORDER_LIST)
            # cv2_obj_img = cv2.rectangle(cv2_obj_img, (x1, y1), (x2, y2), obj_color, 2)
            #
            # cv2_obj_img = cv2.putText(cv2_obj_img,
            #                           affpose_dataset_utils.map_obj_id_to_name(obj_id),
            #                           # umd_utils.aff_id_to_name(label),
            #                           (x1, y1 - 5),
            #                           cv2.FONT_ITALIC,
            #                           0.4,
            #                           obj_color)

            #######################################
            # ICP REFINEMENT
            #######################################

            depth_pointcloud = create_pointcloud_from_depth_image(depth)
            mask_depth_pointcloud, bbox = create_masked_pointcloud_from_depth_image(obj_id, label, depth)

            depth_pcd = o3d.geometry.PointCloud()
            depth_pcd.points = o3d.utility.Vector3dVector(depth_pointcloud)
            depth_pcd.paint_uniform_color([0, 0, 1])

            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(mask_depth_pointcloud)
            # target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            target.paint_uniform_color([1, 0, 0])

            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(cld[obj_id])
            source.paint_uniform_color([0, 1, 0])

            threshold = 0.05
            init_SE3_transformation = np.eye(4)
            init_SE3_transformation[0:3, 0:3] = target_r
            init_SE3_transformation[0:3, -1] = target_t

            print("Initial Transformation is:")
            evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold,
                                                                          init_SE3_transformation)
            print(evaluation)
            print(init_SE3_transformation)

            # o3d.visualization.draw_geometries([target, copy.deepcopy(source).transform(init_SE3_transformation)])

            print("Apply point-to-point ICP")
            reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, init_SE3_transformation,
                                                                  o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                                  # o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
                                                                  )
            print(reg_p2p)
            print("ICP Transformation is:")
            print(reg_p2p.transformation)

            icp_r = reg_p2p.transformation[0:3, 0:3]
            icp_t = reg_p2p.transformation[0:3, -1]

            # o3d.visualization.draw_geometries([target, copy.deepcopy(source).transform(init_SE3_transformation)])

            ####################
            # OBJECT Pose: imgpts look messy after projection with full object models
            ####################

            # # projecting 3D model to 2D image
            # imgpts, jac = cv2.projectPoints(cld[obj_id] * 1e3, target_r, target_t * 1e3, config.CAM_MAT, config.CAM_DIST)
            # cv2_obj_img = cv2.polylines(cv2_obj_img, helper_utils.sort_imgpts(imgpts), True, obj_color)
            #
            # # modify YCB objects rotation matrix
            # target_r = affpose_dataset_utils.modify_obj_rotation_matrix_for_grasping(obj_id, target_r)
            #
            # # draw pose
            # rotV, _ = cv2.Rodrigues(target_r)
            # points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
            # axisPoints, _ = cv2.projectPoints(points, rotV, target_t * 1e3, config.CAM_MAT, config.CAM_DIST)
            # cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
            # cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
            # cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 3)

            #######################################
            # ITERATE OVER OBJ PARTS
            #######################################

            obj_part_ids = affpose_dataset_utils.map_obj_id_to_obj_part_ids(obj_id)
            print(f'obj_part_ids:{obj_part_ids}')
            for obj_part_id in obj_part_ids:
                aff_id = affpose_dataset_utils.map_obj_part_id_to_aff_id(obj_part_id)
                print(f"\tAff: {aff_id}, {obj_part_classes[int(obj_part_id) - 1]}")

                #######################################
                # 6-DOF POSE
                #######################################
                obj_centered = cld_obj_centered[obj_part_id]

                # projecting 3D model to 2D image
                imgpts, jac = cv2.projectPoints(obj_centered * 1e3, target_r, target_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                cv2_obj_img = cv2.polylines(cv2_obj_img, np.int32([np.squeeze(imgpts)]), True, obj_color)

                # modify YCB objects rotation matrix
                _target_r = affpose_dataset_utils.modify_obj_rotation_matrix_for_grasping(obj_id, target_r.copy())

                # draw pose
                rotV, _ = cv2.Rodrigues(_target_r)
                points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                axisPoints, _ = cv2.projectPoints(points, rotV, target_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 3)

                #######################################
                # ICP
                #######################################

                imgpts, jac = cv2.projectPoints(obj_centered * 1e3, icp_r, icp_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                cv2_obj_img = cv2.polylines(cv2_obj_img, np.int32([np.squeeze(imgpts)]), True, (0, 255, 255))

                # modify YCB objects rotation matrix
                _icp_r = affpose_dataset_utils.modify_obj_rotation_matrix_for_grasping(obj_id, icp_r.copy())

                # draw pose
                rotV, _ = cv2.Rodrigues(_icp_r)
                points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                axisPoints, _ = cv2.projectPoints(points, rotV, icp_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 255), 3)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (255, 0, 255), 3)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (255, 0, 255), 3)

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

        color_label = affpose_dataset_utils.colorize_obj_mask(label)

        # cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        # cv2.imshow('depth', depth)
        # cv2.imshow('heatmap', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
        # cv2.imshow('label', cv2.cvtColor(color_label, cv2.COLOR_BGR2RGB))
        cv2.imshow('mask', cv2.cvtColor(cv2_mask_img, cv2.COLOR_BGR2RGB))
        cv2.imshow('gt_pose', cv2.cvtColor(cv2_obj_img, cv2.COLOR_BGR2RGB))

        cv2.waitKey(0)

if __name__ == '__main__':
    main()