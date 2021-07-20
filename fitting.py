from scipy.optimize import curve_fit
# from scipy.optimize import leastsq
from collections.abc import Callable
import numpy as np
from functools import partial

from scipy.fft import fftfreq
from scipy.fft import fftshift
from scipy.fft import ifft2

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


def fitting_mask(freqs, minfreq, maxfreq):
    return ((np.linalg.norm(freqs, axis=0) <= maxfreq) *
            (np.linalg.norm(freqs, axis=0) >= minfreq))


def obtain_fit_parameters(sci_spectrum, known_spectrum, fc, p0_mask_radius):
    y_data = sci_spectrum / known_spectrum
    y_size, x_size = y_data.shape
    x_freq, y_freq = np.meshgrid(
        fftshift(fftfreq(x_size)),
        fftshift(fftfreq(y_size))
    )
    x_data = np.vstack((x_freq.flatten(), y_freq.flatten()))
    freqs = np.stack((x_freq, y_freq))
    mask = fitting_mask(freqs, MIN_FREQ_MASK *fc, MAX_FREQ_MASK * fc)
    # mask = ((np.linalg.norm(freqs, axis=0) <= MAX_FREQ_MASK * fc) *
    #         (np.linalg.norm(freqs, axis=0) >= MIN_FREQ_MASK * fc))
    # mask *= np.abs(freqs[1, ::, ::]) >= 2 * MIN_FREQ_MASK * fc
    # for line in mask:
    #     print("".join(map(str, np.asarray(line, dtype=int))))

    inv = fftshift(ifft2(mask * y_data))
    # fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    # ax.set_xticks(np.arange(0, 256)[::10])
    # ax.set_xticklabels(np.arange(-128, 128)[::10])
    # ax.set_yticks(np.arange(0, 256)[::10])
    # ax.set_yticklabels(np.arange(-128, 128)[::10])
    # ax.imshow(np.abs(inv))
    # plt.show()
    p0_mask = np.ones_like(inv)
    p0_mask[
        (p0_mask.shape[0] // 2 - p0_mask_radius):(p0_mask.shape[0] // 2 + p0_mask_radius):,
        (p0_mask.shape[1] // 2 - p0_mask_radius):(p0_mask.shape[1] // 2 + p0_mask_radius):
    ] = 0
    # p0_mask[
    #     ::,
    #     (p0_mask.shape[1] // 2 - p0_mask_radius):(p0_mask.shape[1] // 2 + p0_mask_radius):
    # ] = 0
    dy, dx = np.unravel_index(np.argmax(inv * p0_mask, axis=None), inv.shape)
    dy -= inv.shape[0] // 2
    dx -= inv.shape[1] // 2
    A = (mask * y_data).max()

    # print(y_data.min(), y_data.max())
    values, errors = fit(
        x_data[::, ::],
        (y_data * mask).flatten()[::],
        partial(model, fc=fc),
        p0=(-dx, -dy, 0.5, A),
        # maxfev=500000,
        # nfev=100000,
        # bounds=([-1.0, -1.0, 0, 0.1 * y_data.max()], [1.0, 1.0, 1.0, y_data.max()]),
    )
    # return (25, -2, 0.5, 2.), errors
    return values, errors
