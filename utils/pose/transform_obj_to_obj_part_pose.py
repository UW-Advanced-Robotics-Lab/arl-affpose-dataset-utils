import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation


def get_obj_part_pose_in_camera_frame(obj_r, obj_t, obj_to_obj_part):

    #######################################
    # OBJECT PART IN WORLD FRAME
    #######################################

    # cam_T_obj
    cam_T_obj = np.eye(4)
    cam_T_obj[0:3, 0:3] = obj_r
    cam_T_obj[0:3, -1] = obj_t

    # obj_to_obj_part
    obj_to_obj_part_t = np.array(obj_to_obj_part[0:3]).reshape(-1)
    obj_to_obj_part_q = np.array(obj_to_obj_part[3:7]).reshape(-1)
    obj_to_obj_part_r = SciPyRotation.from_quat(obj_to_obj_part_q).as_dcm().reshape(3, 3)

    obj_T_obj_part = np.eye(4)
    obj_T_obj_part[0:3, 0:3] = obj_to_obj_part_r
    obj_T_obj_part[0:3, -1] = obj_to_obj_part_t

    ''' cam_T_obj_part = cam_T_obj * obj_T_obj_part '''
    cam_T_obj_part = np.dot(cam_T_obj, obj_T_obj_part)
    R = cam_T_obj_part[0:3, 0:3].reshape(3, 3)
    t = cam_T_obj_part[0:3, -1].reshape(-1)

    return R, t

def get_obj_pose_in_camera_frame(obj_part_r, obj_part_t, obj_to_obj_part):

    #######################################
    # OBJECT IN WORLD FRAME
    #######################################

    # cam_T_obj_part
    cam_T_obj_part = np.eye(4)
    cam_T_obj_part[0:3, 0:3] = obj_part_r
    cam_T_obj_part[0:3, -1] = obj_part_t

    # obj_to_obj_part
    obj_to_obj_part_t = np.array(obj_to_obj_part[0:3]).reshape(-1)
    obj_to_obj_part_q = np.array(obj_to_obj_part[3:7]).reshape(-1)
    obj_to_obj_part_r = SciPyRotation.from_quat(obj_to_obj_part_q).as_dcm().reshape(3, 3)

    obj_part_T_obj = np.eye(4)
    obj_part_T_obj[0:3, 0:3] = obj_to_obj_part_r
    obj_part_T_obj[0:3, -1] = obj_to_obj_part_t

    obj_part_T_obj_t = np.eye(4)
    obj_part_T_obj_t[0:3, -1] = -1 * obj_to_obj_part_t

    obj_part_T_obj_R = np.eye(4)
    obj_part_T_obj_R[0:3, 0:3] = obj_to_obj_part_r.T

    obj_part_T_obj = np.dot(obj_part_T_obj_R, obj_part_T_obj_t)

    ''' cam_T_obj = cam_T_obj * obj_T_obj_part '''
    cam_T_obj_part = np.dot(cam_T_obj_part, obj_part_T_obj)
    R = cam_T_obj_part[0:3, 0:3].reshape(3, 3)
    t = cam_T_obj_part[0:3, -1].reshape(-1)

    return R, t

# def get_pose_in_camera_frame(obj_r, obj_t, cam_r, cam_t):
#     # Get object pose in world frame.
#     _obj_r, _obj_t = _get_pose_in_world_frame(obj_r, obj_t, cam_r, cam_t)
#     # Get object part pose in camera frame.
#     return _get_pose_in_cam_frame(_obj_r, _obj_t, cam_r, cam_t)

def _get_pose_in_world_frame(obj_r, obj_t, cam_r, cam_t):
    '''Function to get the object pose in the world frame.'''

    #######################################
    #######################################

    world_T_cam = np.eye(4)
    world_T_cam[0:3, 0:3] = cam_r
    world_T_cam[0:3, -1] = cam_t

    #######################################
    # OBJECT in WORLD FRAME
    #######################################

    # obj_t_zed
    cam_T_obj = np.eye(4)
    cam_T_obj[0:3, 0:3] = obj_r
    cam_T_obj[0:3, -1] = obj_t

    ''' obj_t_world = obj_t_zed * zed_T_world '''
    world_T_obj = np.dot(world_T_cam, cam_T_obj)
    R = world_T_obj[0:3, 0:3].reshape(3, 3)
    t = world_T_obj[0:3, -1].reshape(-1)

    # print(f'\tObject Pose in Cam Frame .. {obj_t}')
    # print(f'\tObject Pose in World Frame .. {t}')

    return R, t

def _get_pose_in_cam_frame(obj_part_r, obj_part_t, cam_r, cam_t):
    '''Function to get the object part pose in the camera frame.'''

    #######################################
    #######################################

    cam_T_world_t = np.eye(4)
    cam_T_world_t[0:3, -1] = -1 * cam_t

    cam_T_world_R = np.eye(4)
    cam_T_world_R[0:3, 0:3] = cam_r.T

    cam_T_world = np.dot(cam_T_world_R, cam_T_world_t)

    #######################################
    # OBJECT in CAMERA FRAME
    #######################################

    # obj_t_zed
    world_T_obj = np.eye(4)
    world_T_obj[0:3, 0:3] = obj_part_r
    world_T_obj[0:3, -1] = obj_part_t

    ''' obj_t_world = obj_t_zed * zed_T_world '''
    world_T_obj = np.dot(cam_T_world, world_T_obj)
    R = world_T_obj[0:3, 0:3].reshape(3, 3)
    t = world_T_obj[0:3, -1].reshape(-1)

    # print(f'\tObject Part Pose in World Frame .. {obj_part_t}')
    # print(f'\tObject Part Pose in Cam Frame .. {t}')

    return R, t