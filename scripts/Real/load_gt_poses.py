import glob
import copy

import numpy as np

import cv2
from PIL import Image
import matplotlib.pyplot as plt

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

from utils.pose.load_obj_ply_files import load_obj_ply_files, load_ply_files
from utils.pose.load_obj_6dof_pose import load_obj_6dof_pose

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

    imgs_path = config.ROOT_DATA_PATH + 'LabelFusion/train/' + '*/*/' + '*' + config.RGB_EXT
    # imgs_path = config.ROOT_DATA_PATH + "logs_test/*/*/*" + config.RGB_EXT
    # imgs_path = config.LABELFUSION_LOG_PATH + "*" + config.RGB_EXT
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
        print(f'\nimage:{image_idx+1}/{len(img_files)}, file:{file_path}')

        rgb_addr   = file_path + config.RGB_EXT
        depth_addr = file_path + config.DEPTH_EXT
        label_addr = file_path + config.OBJ_LABEL_EXT

        rgb      = np.array(Image.open(rgb_addr))
        depth    = np.array(Image.open(depth_addr))
        label    = np.array(Image.open(label_addr))

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
                # OBJECT CENTERED
                #######################################

                obj_centered = cld_obj_centered[obj_part_id]
                obj_r = copy.deepcopy(target_r)
                obj_t = copy.deepcopy(target_t)

                # projecting 3D model to 2D image
                imgpts, jac = cv2.projectPoints(obj_centered * 1e3, obj_r, obj_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                cv2_obj_img = cv2.polylines(cv2_obj_img, helper_utils.sort_imgpts(imgpts), True, obj_color)

                # modify YCB objects rotation matrix
                _obj_r = affpose_dataset_utils.modify_obj_rotation_matrix_for_grasping(obj_id, obj_r.copy())

                # draw pose
                rotV, _ = cv2.Rodrigues(_obj_r)
                points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                axisPoints, _ = cv2.projectPoints(points, rotV, obj_t * 1e3, config.CAM_MAT, config.CAM_DIST)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
                cv2_obj_img = cv2.line(cv2_obj_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 3)

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