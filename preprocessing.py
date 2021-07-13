import numpy as np
from collections.abc import Callable
from functools import partial
from scipy.fft import fft2
from scipy.fft import fftshift


def crop_image(image: np.array, left_angle: tuple, shape: tuple) -> np.array:
    """
    :param image: изображение
    :param left_angle: левый верхний угол выделяемой области
    :param shape: итоговый размер изображения
    :return: обрезанное изображение указанного размера (нужно для выбора
     луча поляриметра)
    """
    return image[
        left_angle[0]:left_angle[0] + shape[0],
        left_angle[1]:left_angle[1] + shape[1]
    ]


def crop_image_pipe(left_angle: tuple, shape: tuple) -> Callable[[np.array], np.array]:
    return partial(crop_image, left_angle=left_angle, shape=shape)


def disable_weak_pixels(image: np.array, sigma_ron: float) -> np.array:
    """
    :param image: усредненное изображение
    :param sigma_ron: дисперсия шума считывания
    :return:
    """
    return (image < 5*sigma_ron)*image


def disable_weak_pixels_pipe(sigma_ron: float) -> Callable[[np.array], np.array]:
    return partial(disable_weak_pixels, sigma_ron=sigma_ron)


def process_bias(frame: np.array, master_bias: np.array) -> np.array:
    """
    :param frame: кадр
    :param master_bias: баес
    :return: вычитаем подложку
    """
    return frame - master_bias


def mean_frame(frames: np.array) -> np.array:
    """
    :param frames: кадры
    :return: средний кадр
    """
    return np.mean(frames, axis=0)


def get_fft_square_magnitude(frame: np.array) -> np.array:
    return np.abs(fftshift(fft2(frame)))**2


def bias_pipe(master_bias: np.array) -> Callable[[np.array], np.array]:
    return partial(process_bias, master_bias=master_bias)
