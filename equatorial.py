from transform_utils import get_xyz_ha
from transform_utils import get_xyz
from transform_utils import get_angles
# from transform_utils import rotate_x
# from transform_utils import rotate_y
# from transform_utils import rotate_z
# матрицы поворота. Пример использования см. в transform_utils.py
"""
ВНИМАНИЕ! ВСЕ УГЛЫ В РАДИАНАХ ЕСЛИ НЕ УКАЗАНО ИНОЕ
"""


def get_azimuth_height(time, alpha, delta, latitude, longitude):
    # ВНИМАНИЕ! НУЖНО ТЕСТИРОВАНИЕ
    xyz_ha = get_xyz_ha(time, get_xyz(alpha, delta), latitude, longitude)
    return get_angles(xyz_ha, is_left=True)


def get_psi(alpha, delta, azimuth, height):
    """
    :param alpha: прямое восхождение
    :param delta: склонение
    :param azimuth: азимут
    :param height: высота
    :return: угол psi
    """
    # todo
    pass


def transform_pixels_to_equatorial(s, xy, h, psi, epsilon):
    """
    :param s: угловой масштаб
    :param xy: вектор разделения в пикселях. ВНИМАНИЕ! y направлено вниз!
    :param h: высота
    :param psi: параллактический угол
    :param epsilon: отношение потоков от компонент
    :return: экваториальные координаты компонент
    """
    # todo
    pass
