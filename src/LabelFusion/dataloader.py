import os
import glob
import json

import numpy as np

import cv2
from PIL import Image

import scipy.io as scio

#######################################
#######################################

from src import cfg as config

from src.utils import helper_utils
from src.utils.dataset import affpose_dataset_utils

from src.utils.pose.load_obj_ply_files import load_obj_ply_files
from src.utils.pose.load_obj_6dof_pose import load_obj_6dof_pose

#######################################
#######################################

class ARLAffPose():

    def __init__(self,
                 subset = 'train',
                 subfolder = '*',
                 _subdivide_images = False,
                 _subdivide_idx = 3,
                 _num_subdivides = 4,
                 _select_random_images = False,
                 _select_every_ith_images = False,
                 ):

        ###################################
        # Load Ply files
        ###################################

        self.cld, self.cld_obj_centered, self.cld_obj_part_centered, \
        self.obj_classes, self.obj_part_classes = load_obj_ply_files()

        ###################################
        # Load Obj to Obj Part Offsets
        ###################################

        _file = open(f'{config.OBJ_TO_OBJ_PART_TRANSFORMS_FILE}', )
        self.obj_to_obj_part_transforms = json.load(_file)
        _file.close()

        ##################################
        # Load Images
        ##################################

        self.img_path = config.LABELFUSION_LOG_PATH + "*" + config.RGB_EXT
        # self.img_path = config.ROOT_DATA_PATH + f"LabelFusion/dataset_{subset}/" + f"{subfolder}/images/*" + config.RGB_EXT
        # self.img_path = "/data/Akeaveny/Datasets/ARLAffPose/LabelFusion/dataset_train/*_arl_lab_wam/images/*" + config.RGB_EXT
        self.img_files = np.sort(np.array(glob.glob(self.img_path)))
        print(f'Loaded {len(self.img_files)} Images')

        if _subdivide_images:
            # sub-divide images for formatting.
            img_subdivides = len(self.img_files)/_num_subdivides
            start = int(_subdivide_idx * img_subdivides)
            end = int((_subdivide_idx + 1 ) * img_subdivides)
            self.img_files = self.img_files[start:end+1]
            print("Subdivide Idx: {}, Num Subdivides: {}, Start: {}, End: {}, Chosen Files: {}"
                  .format(_subdivide_idx, _num_subdivides, start, end, len(self.img_files)))

        if _select_random_images:
            np.random.seed(0)
            num_files = int(len(self.img_files)/10)
            idx = np.arange(0, len(self.img_files), 1)
            random_idx = np.random.choice(idx, size=int(num_files), replace=False)
            self.img_files = np.array(self.img_files)[random_idx]
            print("Chosen Files: {}".format(len(self.img_files)))

        if _select_every_ith_images:
            every_ith_image = 5
            idx = np.arange(0, len(self.img_files), every_ith_image)
            self.img_files = np.sort(np.array(self.img_files)[idx])
            print("Chosen Files: {}".format(len(self.img_files)))

    def get_labelfusion_item(self, image_idx):

        self.image_addr = self.img_files[image_idx]
        self.file_path = self.image_addr.split(config.RGB_EXT)[0]
        print(f'\nimage:{image_idx+1}/{len(self.img_files)}, file:{self.file_path }')

        #######################################
        # Load LabelFusion data.
        #######################################

        rgb_addr = self.file_path + config.RGB_EXT
        depth_addr = self.file_path + config.DEPTH_EXT
        label_addr = self.file_path + config.OBJ_LABEL_EXT

        rgb = np.array(Image.open(rgb_addr))
        depth = np.array(Image.open(depth_addr))
        label = np.array(Image.open(label_addr))

        #######################################
        # Resize and Crop.
        #######################################

        rgb = cv2.resize(rgb, config.RESIZE, interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth, config.RESIZE, interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, config.RESIZE, interpolation=cv2.INTER_NEAREST)

        rgb = helper_utils.crop(pil_img=rgb, crop_size=config.CROP_SIZE, is_img=True)
        depth = helper_utils.crop(pil_img=depth, crop_size=config.CROP_SIZE)
        label = helper_utils.crop(pil_img=label, crop_size=config.CROP_SIZE)

        #####################
        # Image utils
        #####################

        # Convert Depth to 8-bit for plotting.
        helper_utils.print_depth_info(depth)
        depth = helper_utils.convert_16_bit_depth_to_8_bit(depth)

        # Label
        helper_utils.print_class_labels(label)
        colour_label = affpose_dataset_utils.colorize_obj_mask(label)
        colour_label = cv2.addWeighted(rgb, 0.35, colour_label, 0.65, 0)

        # Img to draw 6-DoF Pose.
        cv2_pose_img = colour_label.copy()

        #######################################
        # META
        #######################################

        yaml_addr = self.file_path + config.POSE_EXT
        meta = load_obj_6dof_pose(yaml_addr)

        # Load Camera Intrinsics.
        if 'arl_lab_wam' in self.file_path or 'logs_wam_single' in self.file_path:
            self.cam_cx = 615.583 * config.X_SCALE
            self.cam_cy = 359.161 * config.Y_SCALE
            self.cam_fx = 739.436
            self.cam_fy = 739.436
        else:
            self.cam_cx = 652.26074 * config.X_SCALE
            self.cam_cy = 335.50336 * config.Y_SCALE
            self.cam_fx = 680.72644
            self.cam_fy = 680.72644

        self.cam_mat = np.array([[self.cam_fx, 0, self.cam_cx], [0, self.cam_fy, self.cam_cy], [0, 0, 1]])
        self.cam_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        #####################
        #####################

        return {"rgb": rgb,
                "depth": depth,
                "label": label,
                "colour_label": colour_label,
                "cv2_pose_img": cv2_pose_img,
                "meta": meta,
                }

    def get_affpose_item(self, image_idx):

        data = self.get_labelfusion_item(image_idx)

        #######################################
        # OBJECT
        #######################################

        obj_part_label_addr = self.file_path + config.OBJ_PART_LABEL_EXT
        aff_label_addr = self.file_path + config.AFF_LABEL_EXT

        obj_part_label = np.array(Image.open(obj_part_label_addr))
        aff_label = np.array(Image.open(aff_label_addr))

        #######################################
        # Resize and Crop.
        #######################################

        obj_part_label = cv2.resize(obj_part_label, config.RESIZE, interpolation=cv2.INTER_NEAREST)
        aff_label = cv2.resize(aff_label, config.RESIZE, interpolation=cv2.INTER_NEAREST)

        obj_part_label = helper_utils.crop(pil_img=obj_part_label, crop_size=config.CROP_SIZE)
        aff_label = helper_utils.crop(pil_img=aff_label, crop_size=config.CROP_SIZE)

        #######################################
        #######################################

        colour_aff_label = affpose_dataset_utils.colorize_obj_mask(aff_label)
        colour_aff_label = cv2.addWeighted(data['rgb'], 0.35, colour_aff_label, 0.65, 0)

        #######################################
        #######################################

        data['obj_part_label'] = obj_part_label
        data['aff_label'] = aff_label
        data['colour_aff_label'] = colour_aff_label

        return data


