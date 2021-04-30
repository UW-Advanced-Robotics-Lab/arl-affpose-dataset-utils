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

    #######################################
    #######################################
    print(f'\ngenerating training configs ..')

    _FRAMES = np.array([0, 1, 2, 3, 5, 6, 7, 8,
                        10, 11, 12, 13, 14, 15, 16, 17,
                        20, 21, 22, 23, 24, 26, 28, 29,
                        30, 31, 32, 33, 34, 35, 36, 37, 39,
                        40, 41, 42, 43, 45, 46, 47, 49,
                        50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                        60, 61, 62, 63, 64, 65, 67, 68, 69,
                        70, 71
                        ])

    np.random.seed(14)
    _num_test_frames = 6
    TEST_FRAMES = np.random.choice(_FRAMES, size=int(_num_test_frames), replace=False)
    print(f"TEST_FRAMES: {np.sort(TEST_FRAMES)}")

if __name__ == '__main__':
    main()