from scipy.optimize import curve_fit
# from scipy.optimize import leastsq
from collections.abc import Callable
import numpy as np

from constants import MIN_FREQ_MASK
from constants import MAX_FREQ_MASK


def model(x_data: np.array, dx, dy, epsilon, A, fc) -> np.array:
    """
    :param x_data -- двумерный массив (2, M), где x_data[i, j] --
    i-я координата в радиус-векторе точки j

    :param dx: x координата вектора разделения
    :param dy: y координата вектора разделения
    :param epsilon: отношение потоков
    :param A: нормировочная константа
    :param fc: частота среза
    :return значение модельного спектра
    """
    d = np.array([dx, dy])
    if len(x_data.shape) == 2:
        d = d[::, np.newaxis]
    elif len(x_data.shape) == 3:
        d = d[::, np.newaxis, np.newaxis]
    else:
        raise RuntimeError(
            "invalid shape of x_data {}".format(x_data.shape)
        )
    return (A * (1 + epsilon ** 2 + 2 * epsilon * np.cos(2 * np.pi * (x_data * d).sum(axis=0))) *
            (np.linalg.norm(x_data, axis=0) <= MAX_FREQ_MASK * fc) *
            (np.linalg.norm(x_data, axis=0) >= MIN_FREQ_MASK * fc))


def fit(x_data: np.array, y_data: np.array, func: Callable[[np.array, ...], np.array], **kwargs):
    return curve_fit(func, x_data, y_data, **kwargs)
