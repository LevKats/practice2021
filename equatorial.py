import numpy as np

# from transform_utils import get_xyz_ha
from transform_utils import get_xyz
# from transform_utils import get_angles
from transform_utils import rotate_x
from transform_utils import rotate_y
from transform_utils import rotate_z
from transform_utils import shift
# from transform_utils import swap
from transform_utils import flip
from transform_utils import get_aug_vec
# from transform_utils import get_xyz_ha
# from transform_utils import get_angles

from astropy.coordinates import SkyCoord
from astropy.coordinates import AltAz
from astropy.coordinates import EarthLocation
from astropy import units as u
from astropy.time import Time
"""
ВНИМАНИЕ! ВСЕ УГЛЫ В РАДИАНАХ ЕСЛИ НЕ УКАЗАНО ИНОЕ
"""


# def get_azimuth_height(
#         time, alpha, delta,
#         latitude, longitude, height,
#         temperature, press, humid,
#         wavelength):
#     # ВНИМАНИЕ! НУЖНО ТЕСТИРОВАНИЕ
#     return get_angles(get_xyz_ha(time, get_xyz(alpha, delta), latitude, longitude), True)


def get_azimuth_height(
        time, alpha, delta,
        latitude, longitude, height,
        temperature, press, humid,
        wavelength):
    # ВНИМАНИЕ! НУЖНО ТЕСТИРОВАНИЕ

    location = EarthLocation(lat=latitude*u.rad, lon=longitude*u.rad, height=height * u.m)
    observing_time = Time(time, scale="utc", location=location)
    altaz = AltAz(
        obstime=observing_time,
        location=location,
        pressure=press * u.Torr,
        temperature=temperature * u.deg_C,
        relative_humidity=humid/100,
        obswl=wavelength * u.m
    )
    coord = SkyCoord(ra=alpha * u.rad, dec=delta * u.rad)
    # print(coord)
    result = coord.transform_to(altaz)
    print(result)
    return result.az.radian - np.pi, result.alt.radian


def get_psi(azimuth, height, latitude):
    """
    :param azimuth: азимут
    :param height: высота
    :param latitude: широта
    :return: угол psi
    """
    rpn = get_xyz(np.pi, latitude, True)
    rz = get_xyz(0, np.pi/2, True)
    tranform_matrix = shift([0, 0, 1]) @ rotate_y(np.pi/2 - height) @ rotate_z(-azimuth)
    rpn2 = (tranform_matrix @ get_aug_vec(rpn)).flatten()[:3:]
    rz2 = (tranform_matrix @ get_aug_vec(rz)).flatten()[:3:]
    rpn3 = rpn2 * [1, 1, 0]
    rz3 = rz2 * [1, 1, 0]
    rpn3 /= np.linalg.norm(rpn3)
    rz3 /= np.linalg.norm(rz3)
    sinq = np.cross(rpn3, rz3)[2]
    acosq = np.arccos((rpn3 * rz3).sum())
    return (sinq < 0) * acosq + (sinq >= 0) * (2*np.pi - acosq)


def get_transform_matrix(theta, alpha, delta):
    return rotate_z(-alpha) @ rotate_y(np.pi/2 + delta) @ shift([0, 0, 1]) @ rotate_z(-theta) @ flip(1)


def pixels_to_equatorial_jac(s, theta, alpha, delta):
    (x,), (y,), (z,) = get_xyz(alpha, delta)
    r = np.sqrt(x**2 + y**2 + z**2)
    cos, sin = np.cos, np.sin
    rad_xyz_jac = np.array([
        [-r * cos(delta) * sin(alpha), -r * sin(delta) * cos(alpha), cos(delta) * cos(alpha)],
        [r * cos(delta) * cos(alpha), -r * sin(delta) * sin(alpha), cos(delta) * sin(alpha)],
        [0, r * cos(delta), sin(delta)]
    ])
    end_jac = np.zeros((2, 4))
    end_jac[::, :3:] = np.linalg.inv(rad_xyz_jac)[:2:, ::]
    # print(get_transform_matrix(theta, alpha, delta))
    # print(end_jac)
    return (end_jac @ get_transform_matrix(theta, alpha, delta)) * s


def pixels_to_equatorial_errors(edx, edy, jac):
    return np.sqrt((jac[::, :2:]**2 * np.array([edx, edy])[np.newaxis, ::]**2).sum(axis=1))
