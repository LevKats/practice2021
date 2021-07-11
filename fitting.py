from scipy.optimize import curve_fit
from collections.abc import Callable
import numpy as np


def model(x_data: np.array, *args) -> np.array:
    """
    :param x_data -- двумерный массив (2, M), где x_data[i, j] --
    i-я координата в радиус-векторе точки j

    :param *args -- параметры модели. потом поймем, какие они будут, но
    к ним можно обращаться как к args[0], args[1], ...
    :return значение модельного спектра
    """
    # todo
    return x_data


def fit(x_data: np.array, y_data: np.array, func: Callable[[np.array, ...], np.array]):
    return curve_fit(func, x_data, y_data)
