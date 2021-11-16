import os
import glob
import copy
import random

import numpy as np

#######################################
#######################################

import sys
sys.path.append('..')
# print(sys.path)

import cfg as config

#######################################
#######################################

def main():
    np.random.seed(0)

    MIN_NUM_OBJECTS = 3
    MAX_NUM_OBJECTS = 6

    FRAMES_PER_VIDEO = int(625)

    TRAINING_DATASET_SIZE = int(30e3)
    TRAINING_NUM_VIDEOS = int(TRAINING_DATASET_SIZE / FRAMES_PER_VIDEO)

    TEST_DATASET_SIZE = int(TRAINING_DATASET_SIZE / 0.8 * 0.2)
    TEST_NUM_VIDEOS = int(TEST_DATASET_SIZE / FRAMES_PER_VIDEO)
    WAM_NUM_VIDEOS = int(10)

    SYN_DATASET_SIZE = int(80e3)
    SYN_TRAINING_VIDEOS = int(80)

    # #######################################
    # #######################################
    # print(f'\ngenerating training configs ..')
    #
    # for i in range(TRAINING_NUM_VIDEOS):
    #     video_idx = str(1000 + i)[1:]
    #     # SCENE
    #     random_scene_idx = np.random.randint(low=0, high=len(config.SCENES))
    #     _scene = np.array(config.SCENES)[random_scene_idx]
    #
    #     # select config.OBJECTS
    #     random_num_objs = np.random.randint(low=MIN_NUM_OBJECTS, high=MAX_NUM_OBJECTS)
    #     random_obj_idxs = np.random.choice(np.arange(0, int(len(config.OBJECTS)), 1), size=int(random_num_objs), replace=False)
    #     _objects = np.array(config.OBJECTS)[random_obj_idxs]
    #
    #     # print(f'Video:{i}/{NUM_VIDEOS},\tScene:{_scene},\tObjects:{_objects},\tObject Idxs:{random_obj_idxs}')
    #     print(f'{video_idx},\t{_scene},\t{_objects}')
    #
    # #######################################
    # #######################################
    # print(f'\ngenerating testing configs ..')
    #
    # for i in range(TEST_NUM_VIDEOS):
    #     video_idx = str(1000 + i)[1:]
    #     # TEST SCENE
    #     random_scene_idx = np.random.randint(low=0, high=len(config.TEST_SCENES))
    #     _scene = np.array(config.TEST_SCENES)[random_scene_idx]
    #
    #     # select config.OBJECTS
    #     random_num_objs = np.random.randint(low=MIN_NUM_OBJECTS, high=MAX_NUM_OBJECTS)
    #     random_obj_idxs = np.random.choice(np.arange(0, int(len(config.OBJECTS)), 1), size=int(random_num_objs), replace=False)
    #     _objects = np.array(config.OBJECTS)[random_obj_idxs]
    #
    #     # print(f'Video:{i}/{NUM_VIDEOS},\tScene:{_scene},\tObjects:{_objects},\tObject Idxs:{random_obj_idxs}')
    #     print(f'{video_idx},\t{_scene},\t{_objects}')
    #
    # #######################################
    # #######################################
    # print(f'\ngenerating ARL testing configs ..')
    #
    # for i in range(TEST_NUM_VIDEOS):
    #     video_idx = str(1000 + i)[1:]
    #     # TEST SCENE
    #     random_scene_idx = np.random.randint(low=0, high=len(config.ARL_TEST_SCENES))
    #     _scene = np.array(config.ARL_TEST_SCENES)[random_scene_idx]
    #
    #     # select config.OBJECTS
    #     random_num_objs = np.random.randint(low=MIN_NUM_OBJECTS, high=MAX_NUM_OBJECTS)
    #     random_obj_idxs = np.random.choice(np.arange(0, int(len(config.OBJECTS)), 1), size=int(random_num_objs), replace=False)
    #     _objects = np.array(config.OBJECTS)[random_obj_idxs]
    #
    #     # print(f'Video:{i}/{NUM_VIDEOS},\tScene:{_scene},\tObjects:{_objects},\tObject Idxs:{random_obj_idxs}')
    #     print(f'{video_idx},\t{_scene},\t{_objects}')
    #
    # #######################################
    # #######################################
    # print(f'\ngenerating synthetic ..')
    #
    # for i in range(SYN_TRAINING_VIDEOS):
    #     video_idx = str(1000 + i)[1:]
    #     # SCENE
    #     random_scene_idx = np.random.randint(low=0, high=len(config.SYN_SCENES))
    #     _scene = np.array(config.SYN_SCENES)[random_scene_idx]
    #
    #     # select config.OBJECTS
    #     random_num_objs = np.random.randint(low=MIN_NUM_OBJECTS, high=MAX_NUM_OBJECTS)
    #     random_obj_idxs = np.random.choice(np.arange(0, int(len(config.OBJECTS)), 1), size=int(random_num_objs), replace=False)
    #     _objects = np.array(config.OBJECTS)[random_obj_idxs]
    #
    #     # print(f'Video:{i}/{NUM_VIDEOS},\tScene:{_scene},\tObjects:{_objects},\tObject Idxs:{random_obj_idxs}')
    #     print(f'{video_idx},\t{_scene},\t{_objects}')
    #
    #######################################
    #######################################
    print(f'\ngenerating WAM configs ..')

    for i in range(WAM_NUM_VIDEOS):
        video_idx = str(1000 + i)[1:]
        # TEST SCENE
        _scene = 'arl_wam'

        # select config.OBJECTS
        random_num_objs = np.random.randint(low=MIN_NUM_OBJECTS, high=MAX_NUM_OBJECTS)
        random_obj_idxs = np.random.choice(np.arange(0, int(len(config.OBJECTS)), 1), size=int(random_num_objs), replace=False)
        _objects = np.array(config.OBJECTS)[random_obj_idxs]

        # print(f'Video:{i}/{NUM_VIDEOS},\tScene:{_scene},\tObjects:{_objects},\tObject Idxs:{random_obj_idxs}')
        print(f'{video_idx},\t{_scene},\t{_objects}')

    training_vid_idxs = np.sort(np.random.choice(np.arange(0, int(WAM_NUM_VIDEOS), 1), size=int(8), replace=False) - 1)
    print(f'training vid idxs: {training_vid_idxs}')

if __name__ == '__main__':
    main()