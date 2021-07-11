import numpy as np
from collections.abc import Callable
from functools import partial
from scipy.fft import fft2
from scipy.fft import fftshift


def get_master_bias(bias: np.array) -> np.array:
    """
    :param bias: подается трехмерный массив, нужно усреднить
    :return: усредненный bias, возможно надо найти
    """
    # todo
    pass


def process_bias(frame: np.array, master_bias: np.array) -> np.array:
    """
    :param frame: кадр
    :param master_bias: баес
    :return: вычитаем подложку
    """
    # todo
    pass


def mean_frame(frames: np.array) -> np.array:
    """
    :param frames: кадры
    :return: средний кадр
    """
    # todo
    pass


def get_fft_square_magnitude(frame: np.array) -> np.array:
    return np.abs(fftshift(fft2(frame)))**2


def bias_pipe(master_bias: np.array) -> Callable[[np.array], np.array]:
    return partial(process_bias, master_bias=master_bias)
