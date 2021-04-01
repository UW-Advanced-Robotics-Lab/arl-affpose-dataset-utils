import numpy as np

#######################################
# AffPose Dataset Prelims
#######################################

# from pathlib import Path
# ROOT_DIR_PATH = Path(__file__).parent.absolute().resolve(strict=True)
# ROOT_DIR_PATH = str(ROOT_DIR_PATH) + '/'
ROOT_DIR_PATH = '/home/akeaveny/git/ARLAffPoseDatasetUtils/'

ROOT_DATA_PATH = '/home/akeaveny/datasets/LabelFusion/affposenet_dataset/'
# ROOT_DATA_PATH = '/data/Akeaveny/Datasets/ARLAffPose/'

RGB_EXT       = "_rgb.png"
DEPTH_EXT     = "_depth.png"
LABEL_EXT     = "_labels.png"
AFF_LABEL_EXT = "_aff_labels.png"
POSE_EXT      = "_poses.yaml"
META_EXT      = "_meta.mat"

#######################################
# ZED CAMERA
#######################################

WIDTH, HEIGHT = 1280, 720
ORIGINAL_SIZE = (WIDTH, HEIGHT)
RESIZE        = (int(WIDTH/1), int(HEIGHT/1))

_step = 40
BORDER_LIST = np.arange(start=0, stop=np.max([WIDTH, HEIGHT])+_step, step=_step)

CAMERA_SCALE = 1000
CAM_CX = 653.5618286132812
CAM_CY = 338.541748046875
CAM_FX = 682.7849731445312
CAM_FY = 682.7849731445312

XMAP = np.array([[j for i in range(HEIGHT)] for j in range(WIDTH)])
YMAP = np.array([[i for i in range(HEIGHT)] for j in range(WIDTH)])

CAM_MAT = np.array([[CAM_FX, 0, CAM_CX], [0, CAM_FY, CAM_CY], [0, 0, 1]])
CAM_DIST = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

#######################################
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
# LabelFusion Log DIR
#######################################

### bedroom_floor
# LOG_FIlE = '001_household_bedroom_floor/images/'
# SORTED_OBJ_IDX = np.array([2, 0, 1]) # pitcher, wooden spoon, screwdriver

# LOG_FIlE = '002_household_bedroom_floor/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '008_household_bedroom_floor/images/'
# SORTED_OBJ_IDX = None

LOG_FIlE = '016_household_bedroom_floor/images/'
SORTED_OBJ_IDX = np.array([1, 2, 3, 0])

# LOG_FIlE = '023_household_bedroom_floor/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '024_household_bedroom_floor/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '030_household_bedroom_floor/images/'
# SORTED_OBJ_IDX = np.array([1, 0, 2])

# LOG_FIlE = '036_household_bedroom_floor/images/'
# SORTED_OBJ_IDX = np.array([3, 0, 1, 2, 4])

# LOG_FIlE = '041_household_bedroom_floor/images/'
# SORTED_OBJ_IDX = np.array([0, 1, 2, 3, 4]) # [1  2  8  9 11]

# LOG_FIlE = '043_household_bedroom_floor/images/'
# SORTED_OBJ_IDX = np.array([4, 0, 1, 2, 3]) # [ 1  2  3  7 10]

# LOG_FIlE = '045_household_bedroom_floor/images/'
# SORTED_OBJ_IDX = None

### bedroom_desk
# LOG_FIlE = '006_household_bedroom_desk/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '037_household_bedroom_desk/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '040_household_bedroom_desk/images/'
# SORTED_OBJ_IDX = np.array([4, 3, 0, 1, 2]) # [ 1  6  7  9 10]

### bedroom_bedside_table
# LOG_FIlE = '004_household_bedroom_bedside_table/images/'
# SORTED_OBJ_IDX = np.array([1, 2, 0])

# LOG_FIlE = '011_household_bedroom_bedside_table/images/' # 1 2 6 8 9
# SORTED_OBJ_IDX = np.array([2, 3, 4, 1, 0])

# LOG_FIlE = '012_household_bedroom_bedside_table/images/' # 2 6 10 11
# SORTED_OBJ_IDX = np.array([1, 0, 2, 3])

# LOG_FIlE = '020_household_bedroom_bedside_table/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '028_household_bedroom_bedside_table/images/'
# SORTED_OBJ_IDX = None

### bedroom_small_table
# LOG_FIlE = '003_household_bedroom_small_table/images/'
# SORTED_OBJ_IDX = np.array([2, 0, 1, 3])

# LOG_FIlE = '009_household_bedroom_small_table/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '018_household_bedroom_small_table/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '019_household_bedroom_small_table/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '022_household_bedroom_small_table/images/'
# SORTED_OBJ_IDX = np.array([3, 0, 1, 2])

# LOG_FIlE = '029_household_bedroom_small_table/images/'
# SORTED_OBJ_IDX = np.array([2, 0, 1, 3])

# LOG_FIlE = '031_household_bedroom_small_table/images/' # 2  3  7  9 11
# SORTED_OBJ_IDX = np.array([3, 0, 1, 2, 4])

# LOG_FIlE = '033_household_bedroom_small_table/images/'
# SORTED_OBJ_IDX = np.array([4, 1, 0, 2, 3])

# LOG_FIlE = '038_household_bedroom_small_table/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '044_household_bedroom_small_table/images/'
# SORTED_OBJ_IDX = None

### household_kitchen_counter_top
# LOG_FIlE = '005_household_kitchen_counter_top/images/' # [1 3 4 6 7]
# SORTED_OBJ_IDX = np.array([3, 0, 1, 2, 4])

# LOG_FIlE = '007_household_kitchen_counter_top/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '010_household_kitchen_counter_top/images/'
# SORTED_OBJ_IDX = np.array([2, 0, 1])

# LOG_FIlE = '021_household_kitchen_counter_top/images/'
# SORTED_OBJ_IDX = np.array([0, 1, 2, 3])

# LOG_FIlE = '025_household_kitchen_counter_top/images/' # [1 4 7 9]
# SORTED_OBJ_IDX = np.array([0, 1, 2, 3])

# LOG_FIlE = '027_household_kitchen_counter_top/images/' # [ 5  7  8  9 11]
# SORTED_OBJ_IDX = np.array([4, 0, 1, 2, 3])

# LOG_FIlE = '047_household_kitchen_counter_top/images/'
# SORTED_OBJ_IDX = None

### household_kitchen_small_table
# LOG_FIlE = '013_household_kitchen_small_table/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '014_household_kitchen_small_table/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '035_household_kitchen_small_table/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '046_household_kitchen_small_table/images/'
# SORTED_OBJ_IDX = None

### household_kitchen_table
# LOG_FIlE = '000_household_kitchen_table/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '015_household_kitchen_table/images/'
# SORTED_OBJ_IDX = np.array([3, 0, 1, 2])

# LOG_FIlE = '017_household_kitchen_table/images/'
# SORTED_OBJ_IDX = np.array([0, 2, 3, 1])

# LOG_FIlE = '026_household_kitchen_table/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '032_household_kitchen_table/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '034_household_kitchen_table/images/'
# SORTED_OBJ_IDX = np.array([0, 3, 4, 1, 2])

# LOG_FIlE = '039_household_kitchen_table/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '042_household_kitchen_table/images/'
# SORTED_OBJ_IDX = None

#######################################
# test logs
#######################################

# LOG_FIlE = '000_household_commmon_area/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '001_household_kitchen_tv/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '002_household_kitchen_stairs/images/' # 1 2 4 8 9
# SORTED_OBJ_IDX = np.array([4, 3, 0, 1, 2])

# LOG_FIlE = '003_household_kitchen_stairs/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '004_household_kitchen_tv/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '005_household_kitchen_tv/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '006_household_kitchen_stairs/images/' # 1  3  9 11
# SORTED_OBJ_IDX = np.array([2, 0, 1, 3])

# LOG_FIlE = '007_household_commmon_area/images/'
# SORTED_OBJ_IDX = np.array([2, 1, 0])

# LOG_FIlE = '008_household_commmon_area/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '009_household_kitchen_stairs/images/'
# SORTED_OBJ_IDX = np.array([2, 1, 0, 3])

# LOG_FIlE = '010_household_kitchen_tv/images/' # [1 4 5 6 7]
# SORTED_OBJ_IDX = np.array([0, 2, 1, 3, 4])

# LOG_FIlE = '011_household_commmon_area/images/'
# SORTED_OBJ_IDX = np.array([2, 0, 1])

#######################################
# ARL logs
#######################################

# LOG_FIlE = '000_arl_lab_desk/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '001_arl_lab_bench_top/images/'
# SORTED_OBJ_IDX = np.array([1, 0, 2])

# LOG_FIlE = '002_arl_lab_bench_top/images/'
# SORTED_OBJ_IDX = np.array([2, 0, 1])

# LOG_FIlE = '003_arl_lab_bench_top/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '004_arl_lab_floor/images/'
# SORTED_OBJ_IDX = np.array([3, 4, 1, 0, 2])

# LOG_FIlE = '005_arl_lab_floor/images/' # 1  2  7  8 10
# SORTED_OBJ_IDX = np.array([2, 3, 0, 1, 4])

# LOG_FIlE = '006_arl_lab_bench_top/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '007_arl_lab_bench_top/images/' #  1  4  9 11
# SORTED_OBJ_IDX = np.array([3, 1, 0, 2])

# LOG_FIlE = '008_arl_lab_desk/images/' # 1 2 4 5 9
# SORTED_OBJ_IDX = np.array([4, 0, 1, 2, 3])

# LOG_FIlE = '009_arl_lab_bench_top/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '010_arl_lab_floor/images/'
# SORTED_OBJ_IDX = None

# LOG_FIlE = '011_arl_lab_bench_top/images/'
# SORTED_OBJ_IDX = None

#######################################
#######################################

LABELFUSION_LOG_PATH = ROOT_DATA_PATH + 'logs/' + LOG_FIlE
LABELFUSION_AFF_DATASET_PATH = ROOT_DATA_PATH + 'dataset/' + LOG_FIlE
# LABELFUSION_AFF_DATASET_PATH = ROOT_DATA_PATH + 'LabelFusion/train/' + LOG_FIlE

