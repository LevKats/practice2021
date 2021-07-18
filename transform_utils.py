import numpy as np
from astropy.coordinates import Angle
from astropy.time import Time
import astropy.units as u
"""
ВНИМАНИЕ! ВСЕ УГЛЫ В РАДИАНАХ ЕСЛИ НЕ УКАЗАНО ИНОЕ
"""


def get_aug_vec(xyz):
    return np.vstack((xyz, np.ones(xyz.shape[1])))


# def get_vec(xyz1):
#     if xyz1.ndim == 1:
#         return xyz1[:3:]
#     elif xyz1.ndim == 2:
#         return xyz1[:3:, ::]
#     else:
#         raise RuntimeError("incodderct ndim {}".format(xyz1.ndim))


def create_aug(A, b=None):
    result = np.zeros((4, 4))
    result[:3:, :3:] = A
    if b is not None:
        b = b.flatten()
        result[:3:, 3] = b
    result[3, 3] = 1.0
    return result


def rotate_x(angle):
    return create_aug(np.array([
        [1, 0, 0],
        [0, np.cos(angle), np.sin(angle)],
        [0, -np.sin(angle), np.cos(angle)]
    ]))


def rotate_y(angle):
    return create_aug(np.array([
        [np.cos(angle), 0, -np.sin(angle)],
        [0, 1, 0],
        [np.sin(angle), 0, np.cos(angle)],
    ]))


def rotate_z(angle):
    return create_aug(np.array([
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ]))


def shift(vec):
    return create_aug(np.eye(3, dtype=float), -np.asarray(vec))


def swap(i, j):
    result = np.eye(4)
    result[i, i] = result[j, j] = 0
    result[i, j] = result[j, i] = 1
    return result


def flip(i):
    result = np.eye(4)
    result[i, i] = -1
    return result


def get_xyz(alpha, delta, is_left=False):
    """
    :param alpha: прямое восхождение, азимут, и т.д.
    :param delta: склонение, высота, и т.д.
    :param is_left: система левая?
    :return:
    """
    if is_left:
        alpha = 2*np.pi - alpha
    x = np.cos(delta) * np.cos(alpha)
    y = np.cos(delta) * np.sin(alpha)
    z = np.sin(delta)
    return np.vstack((x, y, z))


def get_angles(xyz, is_left=False):
    module = np.sqrt((xyz**2).sum(axis=0))
    xyz = xyz / module
    delta = np.arcsin(xyz[2, ::])
    cos = xyz[0, ::] / np.cos(delta)
    sin = xyz[1, ::] / np.cos(delta)
    alpha = (sin >= 0)*np.arccos(cos) + (sin < 0)*(2*np.pi - np.arccos(cos))
    if is_left:
        alpha = 2*np.pi - alpha
    return np.vstack((alpha, delta))


# def sidereal_time_rad(time: Time, longitude: float):
#     s = time.sidereal_time(
#         'apparent',
#         longitude=Angle(longitude, unit=u.rad)
#     )
#     return s.to(u.rad).value
#
#
# def get_xyz_ha(time, xyz_ad, latitude, longitude):
#     s = sidereal_time_rad(Time(time, scale="utc"), longitude)
#     return get_vec((rotate_y(np.pi / 2 - latitude) @ rotate_z(s)) @ get_aug_vec(xyz_ad))
