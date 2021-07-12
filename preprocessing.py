import numpy as np
from collections.abc import Callable
from functools import partial
from scipy.fft import fft2
from scipy.fft import fftshift


def crop_image(image: np.array, shape: tuple) -> np.array:
    """
    :param image: изображение
    :param shape: итоговый размер изображения
    :return: обрезанное изображение указанного размера (нужно для выбора
     луча поляриметра)
    """
    # todo
    pass


def crop_image_pipe(shape: tuple) -> Callable[[np.array], np.array]:
    return partial(crop_image, shape=shape)


def disable_weak_pixels(image: np.array, sigma_ron: float) -> np.array:
    """
    :param image: усредненное изображение
    :param sigma_ron: дисперсия шума считывания
    :return:
    """
    # todo
    pass


def disable_weak_pixels_pipe(sigma_ron: float) -> Callable[[np.array], np.array]:
    return partial(disable_weak_pixels, sigma_ron=sigma_ron)


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
