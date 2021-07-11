import numpy as np
from astropy.coordinates import Angle
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


def get_xyz(alpha, delta):
    x = np.cos(delta) * np.cos(alpha)
    y = np.cos(delta) * np.sin(alpha)
    z = np.sin(delta)
    return np.vstack((x, y, z))


def get_alpha_delta(xyz):
    module = np.sqrt((xyz**2).sum(axis=0))
    xyz = xyz / module
    delta = np.arcsin(xyz[2, ::])
    cos = xyz[0, ::] / np.cos(delta)
    sin = xyz[1, ::] / np.cos(delta)
    alpha = (sin >= 0)*np.arccos(cos) + (sin < 0)*(2*np.pi - np.arccos(cos))
    return np.vstack((alpha, delta))


def get_height(time, alpha, delta, latitude, longitude):
    def get_rad(d, m, s):
        return (d * 3600 + m * 60 + s) / 206265

    def get_deg(rad):
        return rad * 206265 / 3600

    xyz = get_xyz(alpha, delta)
    # print(xyz)
    s = time.sidereal_time(
        'apparent',
        longitude=Angle("{}d".format(get_deg(longitude)))
    )
    # print(s.hms)
    s = get_rad(*s.dms)
    # print(s)
    # t = s - alpha
    # print(get_deg(t) / 60)
    xyz_ha = (rotate_y(np.pi/2 - latitude) @ rotate_z(s)) @ xyz


def get_psi():
    # todo
    pass


def transform_pixels_to_equatorial(s, xy, eta_fits):
    # todo
    pass
