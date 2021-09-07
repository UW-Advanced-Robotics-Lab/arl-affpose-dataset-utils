import glob
import copy
import unittest

import numpy as np

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append('../../')

from src.LabelFusion import dataloader
from src.utils.dataset import affpose_dataset_utils


class TestLabelFusionDataloader(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestLabelFusionDataloader, self).__init__(*args, **kwargs)
        # load real images.
        self.dataloader = dataloader.ARLAffPose(_select_random_images=True)

    def load_images(self):

        for image_idx, image_addr in enumerate(self.dataloader.img_files):

            data = self.dataloader.get_labelfusion_item(image_idx)

            rgb = data["rgb"]
            depth = data["depth"]
            colour_label = data["colour_label"]

            #####################
            # PLOTTING
            #####################

            cv2.imshow('rgb', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            cv2.imshow('depth', depth)
            cv2.imshow('heatmap', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
            cv2.imshow('colour_label', cv2.cvtColor(colour_label, cv2.COLOR_BGR2RGB))

            cv2.waitKey(0)

    def load_gt_pose(self):

        for image_idx, image_addr in enumerate(self.dataloader.img_files):
            data = self.dataloader.get_labelfusion_item(image_idx)

            meta = data["meta"]
            cv2_pose_img = data["cv2_pose_img"]

            #######################################
            # OBJECT
            #######################################

            obj_ids = np.array(meta['object_class_ids']).flatten()
            for idx, obj_id in enumerate(obj_ids):
                print(f"\tObject: {obj_id}, {self.dataloader.obj_classes[int(obj_id) - 1]}")
                obj_color = affpose_dataset_utils.obj_color_map(obj_id)

                obj_meta_idx = str(1000 + obj_id)[1:]
                obj_r = meta['obj_rotation_' + str(obj_meta_idx)]
                obj_t = meta['obj_translation_' + str(obj_meta_idx)]

                obj_r = np.array(obj_r, dtype=np.float64).reshape(3, 3)
                obj_t = np.array(obj_t, dtype=np.float64).reshape(-1, 3)

                #######################################
                # ITERATE OVER OBJ PARTS
                #######################################

                obj_part_ids = affpose_dataset_utils.map_obj_id_to_obj_part_ids(obj_id)
                print(f'\tobj_part_ids:{obj_part_ids}')
                for obj_part_id in obj_part_ids:

                        aff_id = affpose_dataset_utils.map_obj_part_id_to_aff_id(obj_part_id)
                        print(f"\t\tAff: {aff_id}, {self.dataloader.obj_part_classes[int(obj_part_id) - 1]}")

                        #######################################
                        # OBJECT POSE
                        #######################################

                        # projecting 3D model to 2D image
                        obj_centered = self.dataloader.cld_obj_centered[obj_part_id]
                        imgpts, jac = cv2.projectPoints(obj_centered * 1e3, obj_r, obj_t * 1e3, self.dataloader.cam_mat, self.dataloader.cam_dist)
                        cv2_pose_img = cv2.polylines(cv2_pose_img, np.int32([np.squeeze(imgpts)]), True, obj_color)

                        # modify YCB objects rotation matrix
                        _obj_r = affpose_dataset_utils.modify_obj_rotation_matrix_for_grasping(obj_id, obj_r.copy())

                        # draw pose
                        rotV, _ = cv2.Rodrigues(_obj_r)
                        points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
                        axisPoints, _ = cv2.projectPoints(points, rotV, obj_t * 1e3, self.dataloader.cam_mat, self.dataloader.cam_dist)
                        cv2_pose_img = cv2.line(cv2_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
                        cv2_pose_img = cv2.line(cv2_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
                        cv2_pose_img = cv2.line(cv2_pose_img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 3)

            #####################
            # PLOTTING
            #####################

            cv2.imshow('cv2_pose_img', cv2.cvtColor(cv2_pose_img, cv2.COLOR_BGR2RGB))

            cv2.waitKey(0)

if __name__ == '__main__':
    # run all test.
    # unittest.main()

    # run desired test.
    suite = unittest.TestSuite()
    suite.addTest(TestLabelFusionDataloader("load_gt_pose"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
