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
    if len(image.shape) == 2:
        return image[
               left_angle[0]:left_angle[0] + shape[0],
               left_angle[1]:left_angle[1] + shape[1]
        ]
    elif len(image.shape) == 3:
        return image[
               ::,
               left_angle[0]:left_angle[0] + shape[0],
               left_angle[1]:left_angle[1] + shape[1]
        ]
    else:
        raise RuntimeError(
            "invalid shape of image {}".format(image.shape)
        )


def crop_image_pipe(left_angle: tuple, shape: tuple) -> Callable[[np.array], np.array]:
    return partial(crop_image, left_angle=left_angle, shape=shape)


def disable_weak_pixels(image: np.array, sigma_ron: float) -> np.array:
    """
    :param image: усредненное изображение
    :param sigma_ron: дисперсия шума считывания
    :return:
    """
    return image * (image > 5*sigma_ron)


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


def get_fft_square_magnitude(frame: np.array, chunk_size: int, workers: int) -> np.array:
    # frame_count = frame.shape[0]
    # result = np.zeros_like(frame, dtype=complex)
    # last = 0
    # while last < frame_count:
    #     temp = np.asarray(frame[last:last + chunk_size:, ::, ::], dtype=complex)
    #     result[last:last + chunk_size:, ::, ::] = fft2(
    #         temp,
    #         workers=workers,
    #         overwrite_x=True
    #     )
    #     del temp
    #     last += chunk_size
    # return np.abs(fftshift(result, axes=(-2, -1)))**2
    return np.abs(fftshift(fft2(frame), axes=(-2, -1)))**2


def fft_pipe(chunk_size: int, workers: int) -> Callable[[np.array], np.array]:
    return partial(get_fft_square_magnitude, chunk_size=chunk_size, workers=workers)


def bias_pipe(master_bias: np.array) -> Callable[[np.array], np.array]:
    return partial(process_bias, master_bias=master_bias)
