import numpy as np

from astropy.coordinates import Angle
from astropy import units as u
from astropy.time import Time
"""
ВНИМАНИЕ! ВСЕ УГЛЫ В РАДИАНАХ ЕСЛИ НЕ УКАЗАНО ИНОЕ
"""


def rotate_x(angle):
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), np.sin(angle)],
        [0, -np.sin(angle), np.cos(angle)]
    ])


def rotate_y(angle):
    return np.array([
        [np.cos(angle), 0, -np.sin(angle)],
        [0, 1, 0],
        [np.sin(angle), 0, np.cos(angle)]
    ])


def rotate_z(angle):
    return np.array([
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])


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


def sidereal_time_rad(time: Time, longitude: float):
    s = time.sidereal_time(
        'apparent',
        longitude=Angle(longitude, unit=u.rad)
    )
    # print(s.hms)
    return s.radians


def get_xyz_ha(time, xyz_ad, latitude, longitude):
    xyz = xyz_ad
    # print(xyz)
    s = sidereal_time_rad(time, longitude)
    # print(s)
    # t = s - alpha
    # print(get_deg(t) / 60)
    return (rotate_y(np.pi/2 - latitude) @ rotate_z(s)) @ xyz
