from scipy.optimize import curve_fit
from collections.abc import Callable
import numpy as np


def model(x_data: np.array, dx, dy, epsilon, A) -> np.array:
    """
    :param x_data -- двумерный массив (2, M), где x_data[i, j] --
    i-я координата в радиус-векторе точки j

    :param dx: x координата вектора разделения
    :param dy: y координата вектора разделения
    :param epsilon: отношение потоков
    :param A: нормировочная константа
    :return значение модельного спектра
    """
    d = np.array([dx,dy])
    return A*(1+epsilon**2+2*epsilon*np.cos(2*np.pi*(x_data*d).sum(axis=0)))


def fit(x_data: np.array, y_data: np.array, func: Callable[[np.array, ...], np.array]):
    return curve_fit(func, x_data, y_data)