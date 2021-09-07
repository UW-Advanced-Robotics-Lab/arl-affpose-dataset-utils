import numpy as np

#######################################
# AffPose Prelims
#######################################

ROOT_PATH = '/home/akeaveny/git/ARLAffPoseDatasetUtils/src/'

OBJ_TO_OBJ_PART_TRANSFORMS_FILE = ROOT_PATH + 'AffPose/obj_to_obj_part_transforms.json'

#######################################
# Dataset Prelims
#######################################

ROOT_DATA_PATH = '/data/Akeaveny/Datasets/ARLAffPose/'

CLASSES_FILE   = ROOT_DATA_PATH + 'object_meshes/classes.txt'
CLASS_IDS_FILE = ROOT_DATA_PATH + 'object_meshes/classes_ids.txt'

OBJ_PART_CLASSES_FILE   = ROOT_DATA_PATH + 'object_meshes/obj_part_classes.txt'
OBJ_PART_CLASS_IDS_FILE = ROOT_DATA_PATH + 'object_meshes/obj_part_classes_ids.txt'

RGB_EXT            = "_rgb.png"
DEPTH_EXT          = "_depth.png"
OBJ_LABEL_EXT      = "_labels.png"
OBJ_PART_LABEL_EXT = "_obj_part_labels.png"
AFF_LABEL_EXT      = "_aff_labels.png"
POSE_EXT           = "_poses.yaml"
META_EXT           = "_meta.mat"

NDDS_PATH = ROOT_DATA_PATH + 'NDDS/'

SYN_RGB_EXT            = ".png"
SYN_DEPTH_EXT          = ".depth.mm.16.png"
SYN_OBJ_PART_LABEL_EXT = ".cs.png"
SYN_OBJ_LABEL_EXT      = "_obj_labels.png"
SYN_AFF_LABEL_EXT      = "_aff_labels.png"
SYN_JSON_EXT           = ".json"
SYN_META_EXT           = "_meta.mat"

#######################################
# Formatted Dataset Prelims
#######################################

FORMATTED_ROOT_DATA_PATH = '/data/Akeaveny/Datasets/ARLAffPose/'
FORMATTED_DATASET_PATH = FORMATTED_ROOT_DATA_PATH + 'Real'

FORMATTED_RGB_EXT            = '.png'
FORMATTED_DEPTH_EXT          = '_depth.png'
FORMATTED_OBJ_LABEL_EXT      = '_obj_label.png'
FORMATTED_OBJ_PART_LABEL_EXT = '_obj_part_labels.png'
FORMATTED_AFF_LABEL_EXT      = '_aff_label.png'
FORMATTED_META_EXT           = '_meta.mat'

#######################################
# ZED CAMERA
#######################################

WIDTH, HEIGHT = 1280, 720
ORIGINAL_SIZE = (WIDTH, HEIGHT)
RESIZE        = (int(WIDTH/1), int(HEIGHT/1))
CROP_SIZE     = (int(1280), int(720)) # (int(1280), int(720)) or (int(640), int(640))
WIDTH, HEIGHT = CROP_SIZE[0], CROP_SIZE[1]
MIN_SIZE = MAX_SIZE = 640

_step = 40
BORDER_LIST = np.arange(start=0, stop=np.max([WIDTH, HEIGHT])+_step, step=_step)

X_SCALE = CROP_SIZE[0] / ORIGINAL_SIZE[0]
Y_SCALE = CROP_SIZE[1] / ORIGINAL_SIZE[1]

CAMERA_SCALE = 1000

# # Real
# # # CAM_CX = 652.26074
# # # CAM_CY = 335.50336
# # CAM_CX = 652.26074 * X_SCALE
# # CAM_CY = 335.50336 * Y_SCALE
# # CAM_FX = 680.72644
# # CAM_FY = 680.72644
#
# # Syn
# CAM_CX = 653.5618286132812
# CAM_CY = 338.541748046875
# # # CAM_CX = 653.5618286132812 * X_SCALE
# # # CAM_CY = 338.541748046875  * Y_SCALE
# CAM_FX = 682.7849731445312
# CAM_FY = 682.7849731445312
#
# # # ARL
# # CAM_CX = 615.583
# # CAM_CY = 359.161
# # CAM_CX = 615.583 * X_SCALE
# # CAM_CY = 359.161 * Y_SCALE
# # CAM_FX = 739.436
# # CAM_FY = 739.436
#
# XMAP = np.array([[j for i in range(HEIGHT)] for j in range(WIDTH)])
# YMAP = np.array([[i for i in range(HEIGHT)] for j in range(WIDTH)])
#
# CAM_MAT = np.array([[CAM_FX, 0, CAM_CX], [0, CAM_FY, CAM_CY], [0, 0, 1]])
# CAM_DIST = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

#######################################
# Random Configs
#######################################

OBJECTS = [
    'mallet',  # 1
    'spatula',  # 2
    'wooden_spoon',  # 3
    'screwdriver',  # 4
    'garden_shovel',  # 5
    'pitcher_base',  # 6
    'bowl',  # 7
    'mug',  # 8
    'power_drill',  # 9
    'scissors',  # 10
    'large_clamp',  # 11
]

SCENES = [
    # Household
    'household_bedroom_floor',
    'household_bedroom_desk',
    'household_bedroom_bedside_table',
    'household_hallway_small_table',
    'household_kitchen_table',
    'household_kitchen_counter_top',
    'household_kitchen_small_table',
]

TEST_SCENES = [
    # Household
    'household_kitchen_tv',
    'household_kitchen_stairs',
    'household_commmon_area',
]

ARL_TEST_SCENES = [
    # Household
    'arl_lab_bench_top',
    'arl_lab_floor',
    'arl_lab_desk',
]

SYN_SCENES = [
    '1_bench',
    '2_work_bench',
    '3_coffee_table',
    '4_old_table',
    '5_bedside_table',
    'dr', # 1
    'dr', # 2
    'dr', # 3
    'dr', # 4
    'dr', # 5
]

#######################################
# LabelFusion: TRAIN
#######################################

# LOG_FILE = '000_household_kitchen_table/images/'
# LOG_FILE = '001_household_bedroom_floor/images/'
# LOG_FILE = '003_household_bedroom_small_table/images/'
# LOG_FILE = '005_household_kitchen_counter_top/images/'
# LOG_FILE = '006_household_bedroom_desk/images/'
# LOG_FILE = '007_household_kitchen_counter_top/images/'
# LOG_FILE = '008_household_bedroom_floor/images/'
# LOG_FILE = '011_household_bedroom_bedside_table/images/'
# LOG_FILE = '012_household_bedroom_bedside_table/images/'
# LOG_FILE = '013_household_kitchen_small_table/images/'
# LOG_FILE = '014_household_kitchen_small_table/images/'
# LOG_FILE = '015_household_kitchen_table/images/'
# LOG_FILE = '016_household_bedroom_floor/images/'
# LOG_FILE = '017_household_kitchen_table/images/'
# LOG_FILE = '020_household_bedroom_bedside_table/images/'
# LOG_FILE = '021_household_kitchen_counter_top/images/'
# LOG_FILE = '022_household_bedroom_small_table/images/'
# LOG_FILE = '023_household_bedroom_floor/images/'
# LOG_FILE = '024_household_bedroom_floor/images/'
# LOG_FILE = '026_household_kitchen_table/images/'
# LOG_FILE = '029_household_bedroom_small_table/images/'
# LOG_FILE = '030_household_bedroom_floor/images/'
# LOG_FILE = '031_household_bedroom_small_table/images/'
# LOG_FILE = '033_household_bedroom_small_table/images/'
# LOG_FILE = '034_household_kitchen_table/images/'
# LOG_FILE = '035_household_kitchen_small_table/images/'
# LOG_FILE = '036_household_bedroom_floor/images/'
# LOG_FILE = '037_household_bedroom_desk/images/'
# LOG_FILE = '039_household_kitchen_table/images/'
# LOG_FILE = '041_household_bedroom_floor/images/'
# LOG_FILE = '042_household_kitchen_table/images/'
# LOG_FILE = '043_household_bedroom_floor/images/'
# LOG_FILE = '045_household_bedroom_floor/images/'
# LOG_FILE = '046_household_kitchen_small_table/images/'
# LOG_FILE = '047_household_kitchen_counter_top/images/'
# LOG_FILE = '049_household_kitchen_tv/images/'
# LOG_FILE = '050_household_kitchen_stairs/images/'
# LOG_FILE = '051_household_kitchen_stairs/images/'
# LOG_FILE = '052_household_kitchen_tv/images/'
# LOG_FILE = '053_household_kitchen_tv/images/'
# LOG_FILE = '054_household_kitchen_stairs/images/'
# LOG_FILE = '055_household_commmon_area/images/'
# LOG_FILE = '056_household_commmon_area/images/'
# LOG_FILE = '057_household_kitchen_stairs/images/'
# LOG_FILE = '059_household_commmon_area/images/'
# LOG_FILE = '060_arl_lab_desk/images/'
# LOG_FILE = '061_arl_lab_bench_top/images/'
# LOG_FILE = '062_arl_lab_bench_top/images/'
# LOG_FILE = '063_arl_lab_bench_top/images/'
# LOG_FILE = '064_arl_lab_floor/images/'
# LOG_FILE = '065_arl_lab_floor/images/'
# LOG_FILE = '067_arl_lab_bench_top/images/'
# LOG_FILE = '068_arl_lab_desk/images/'
# LOG_FILE = '069_arl_lab_bench_top/images/'
# LOG_FILE = '070_arl_lab_floor/images/'

#######################################
# LabelFusion: TEST
#######################################

# LOG_FILE = '002_household_bedroom_floor/images/'
# LOG_FILE = '028_household_bedroom_bedside_table/images/'
# LOG_FILE = '032_household_kitchen_table/images/'
# LOG_FILE = '040_household_bedroom_desk/images/'
# LOG_FILE = '058_household_kitchen_tv/images/'
# LOG_FILE = '069_arl_lab_bench_top/images/'
LOG_FILE = '081_arl_lab_wam/images/'

#######################################
# LabelFusion: BAD
#######################################

# LOG_FILE = '004_household_bedroom_bedside_table/images/'
# LOG_FILE = '009_household_bedroom_small_table/images/'
# LOG_FILE = '010_household_kitchen_counter_top/images/'
# LOG_FILE = '018_household_bedroom_small_table/images/'
# LOG_FILE = '019_household_bedroom_small_table/images/'
# LOG_FILE = '025_household_kitchen_counter_top/images/'
# LOG_FILE = '027_household_kitchen_counter_top/images/'
# LOG_FILE = '038_household_bedroom_small_table/images/'
# LOG_FILE = '044_household_bedroom_small_table/images/'
# LOG_FILE = '048_household_commmon_area/images/'
# LOG_FILE = '066_arl_lab_bench_top/images/'
# LOG_FILE = '011_arl_lab_bench_top/images/'

#######################################
# LabelFusion: SINGLE OBJECTS
#######################################

# LOG_FILE = 'logs_wam_single/001_mallet/images/'
LOG_FILE = 'logs_wam_single/002_spatula/images/'
# LOG_FILE = 'logs_wam_single/003_wooden_spoon/images/'
# LOG_FILE = 'logs_wam_single/004_screwdriver/images/'
# LOG_FILE = 'logs_wam_single/005_garden_shovel/images/'
# LOG_FILE = 'logs_wam_single/019_pitcher_base/images/'
# LOG_FILE = 'logs_wam_single/024_bowl/images/'
# LOG_FILE = 'logs_wam_single/025_mug/images/'
# LOG_FILE = 'logs_wam_single/035_power_drill/images/'
# LOG_FILE = 'logs_wam_single/037_scissors/images/'
# LOG_FILE = 'logs_wam_single/051_large_clamp/images/'

#######################################
#######################################

LABELFUSION_LOG_PATH = ROOT_DATA_PATH + 'LabelFusion/arl_affposenet_dataset/' + LOG_FILE
LABELFUSION_AFF_DATASET_PATH = ROOT_DATA_PATH + 'dataset_arl_lab/' + LOG_FILE

# LABELFUSION_AFF_DATASET_PATH = ROOT_DATA_PATH + 'LabelFusion/train/' + LOG_FILE

