import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

def rotation_2d_around_center(pt, center, theta):
    """
    2d rotation on 3d vectors by ignoring y factor
    :param pt:
    :param center:
    :return:
    """
    pt = pt.copy()
    pt = pt - center
    x, y, z = pt
    new_pt = np.array([np.cos(theta) * x - np.sin(theta) * z, y, np.sin(theta) * x + np.cos(theta) * z]) + center
    return new_pt


def extend_along_center(pt, center, add_dist, min_dist, max_dist):
    pt = pt.copy()
    curr_dist = np.linalg.norm(pt - center)
    pt = pt - center
    new_dist = min(max(min_dist, curr_dist + add_dist), max_dist)
    pt = pt * (new_dist / curr_dist)
    pt = pt + center
    return pt


def vectorized_range(start, end):
    """  Return an array of NxD, iterating from the start to the end"""
    N = int(np.max(end - start)) + 1
    idxes = np.floor(np.arange(N) * (end - start)[:, None] / N + start[:, None]).astype('int')
    return idxes


def vectorized_meshgrid(vec_x, vec_y):
    """vec_x in NxK, vec_y in NxD. Return xx in Nx(KxD) and yy in Nx(DxK)"""
    N, K, D = vec_x.shape[0], vec_x.shape[1], vec_y.shape[1]
    vec_x = np.tile(vec_x[:, None, :], [1, D, 1]).reshape(N, -1)
    vec_y = np.tile(vec_y[:, :, None], [1, 1, K]).reshape(N, -1)
    return vec_x, vec_y

def get_camera_matrix(cam_pos, cam_angle, cam_size, cam_fov):
    focal_length = cam_size[0] / 2 / np.tan(cam_fov / 2)
    cam_intrinsics = np.array([[focal_length, 0, float(cam_size[1])/2],
                               [0, focal_length, float(cam_size[0])/2],
                               [0, 0, 1]])
    cam_pose = np.eye(4)
    rotation_matrix = Rotation.from_euler('xyz', [cam_angle[1], np.pi - cam_angle[0], np.pi], degrees=False).as_matrix()
    cam_pose[:3, :3] = rotation_matrix
    cam_pose[:3, 3] = cam_pos

    return cam_intrinsics, cam_pose

def rotate_rigid_object(center, axis, angle, pos=None, relative=None):
    '''
    rotate a rigid object (e.g. shape in flex).

    pos: np.ndarray 3x1, [x, y, z] coordinate of the object.
    relative: relative coordinate of the object to center.
    center: rotation center.
    axis: rotation axis.
    angle: rotation angle in radius.
    TODO: add rotaion of coordinates
    '''

    if relative is None:
        relative = pos - center

    quat = Quaternion(axis=axis, angle=angle)
    after_rotate = quat.rotate(relative)
    return after_rotate + center


def quatFromAxisAngle(axis, angle):
    '''
    given a rotation axis and angle, return a quatirian that represents such roatation.
    '''
    axis /= np.linalg.norm(axis)

    half = angle * 0.5
    w = np.cos(half)

    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two

    quat = np.array([axis[0], axis[1], axis[2], w])

    return quat