import matplotlib.pyplot as plt
import numpy as np

from scipy.fft import fftfreq
from scipy.fft import fftshift
from scipy.fft import ifft2

from os.path import join

from constants import MIN_FREQ_MASK
from constants import MAX_FREQ_MASK
from constants import X_TICK_SIZE
from constants import Y_TICK_SIZE
from constants import GRID_LINEWIDTH
from constants import MAX_SPECTRUM


def plot_spectrum(sci_spectrum, known_spectrum, func, obj_sci, obj_kno, fc,
                  minpowerspecturm, maxpowerspectrum, maxspectrum):
    y_data = (sci_spectrum / known_spectrum)

    # fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    # ax.plot(y_data[y_data.shape[0]//2, ::])
    # ax.set_ylim((-1, 2))
    # plt.show()

    y_size, x_size = y_data.shape
    x_values = fftshift(fftfreq(x_size))
    y_values = fftshift(fftfreq(y_size))
    x_freq, y_freq = np.meshgrid(
        x_values,
        y_values
    )
    x_data = np.stack((x_freq, y_freq))
    mask = ((np.linalg.norm(x_data, axis=0) <= MAX_FREQ_MASK * fc) *
            (np.linalg.norm(x_data, axis=0) >= MIN_FREQ_MASK * fc))
    y_fit = func(x_data)

    inv = fftshift(ifft2(mask * y_data))
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.set_title(obj_sci)
    ax.set_xticks(np.arange(0, 256)[::10])
    ax.set_xticklabels(np.arange(-128, 128)[::10])
    ax.set_yticks(np.arange(0, 256)[::10])
    ax.set_yticklabels(np.arange(-128, 128)[::10])
    ax.imshow(np.abs(inv))
    fname = join("images", obj_sci + "_invfft" + ".jpg")
    plt.savefig(fname)
    print("IMAGE SAVED {}".format(join(fname)))

    # print(sci_spectrum.min(), sci_spectrum.max())
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12), sharex="col", sharey="row")

    def generate_label(number):
        return "{:0.1f}".format(number)

    ax1.set_title(r"$|V(f)|^2$")
    im = ax1.imshow(
        y_data,
        vmin=max(min((y_data * mask).min(), y_fit.min()), minpowerspecturm),
        vmax=min(max((y_data * mask).max(), y_fit.max()), maxpowerspectrum)
    )
    ax1.set_xticks(np.arange(0, x_size, x_size // X_TICK_SIZE))
    ax1.set_xticklabels(map(generate_label, x_values[0:x_size:x_size // X_TICK_SIZE]))
    ax1.set_yticks(np.arange(0, y_size, y_size // Y_TICK_SIZE))
    ax1.set_yticklabels(map(generate_label, y_values[0:y_size:y_size // Y_TICK_SIZE]))
    fig.colorbar(im, ax=ax1)
    ax1.grid(linewidth=GRID_LINEWIDTH, linestyle='--')

    ax2.set_title("$|V(f)|^2$ fitting")
    im2 = ax2.imshow(
        y_fit,
        vmin=max(min((y_data * mask).min(), y_fit.min()), minpowerspecturm),
        vmax=min(max((y_data * mask).max(), y_fit.max()), maxpowerspectrum)
    )
    fig.colorbar(im2, ax=ax2)
    ax2.set_xticks(np.arange(0, x_size, x_size // X_TICK_SIZE))
    ax2.set_xticklabels(map(generate_label, x_values[0:x_size:x_size // X_TICK_SIZE]))
    ax2.grid(linewidth=GRID_LINEWIDTH, linestyle='--')
    # ax2.set_yticks(np.arange(0, y_size, y_size // Y_TICK_SIZE))
    # ax2.set_yticklabels(map(generate_label, y_values[0:y_size:y_size // Y_TICK_SIZE]))

    ax3.set_title(obj_sci)
    # im3 = ax3.imshow(np.log(sci_spectrum), vmax=8)
    im3 = ax3.imshow(sci_spectrum, vmax=maxspectrum)
    # fig.colorbar(im3, ax=ax3, label=r"$\log(\langle|\tilde{I}(f)|^2\rangle)$")
    fig.colorbar(im3, ax=ax3)
    # ax3.set_xticks(np.arange(0, x_size, x_size // X_TICK_SIZE))
    # ax3.set_xticklabels(map(generate_label, x_values[0:x_size:x_size // X_TICK_SIZE]))
    ax3.set_yticks(np.arange(0, y_size, y_size // Y_TICK_SIZE))
    ax3.set_yticklabels(map(generate_label, y_values[0:y_size:y_size // Y_TICK_SIZE]))
    ax3.grid(linewidth=GRID_LINEWIDTH, linestyle='--')

    ax4.set_title(obj_kno)
    # im4 = ax4.imshow(np.log(known_spectrum), vmax=8)
    im4 = ax4.imshow(known_spectrum, vmax=MAX_SPECTRUM)
    # fig.colorbar(im4, ax=ax4, label=r"$\log(\langle|\tilde{I}(f)|^2\rangle)$")
    fig.colorbar(im4, ax=ax4)
    ax4.grid(linewidth=GRID_LINEWIDTH, linestyle='--')
    # ax4.set_xticks(np.arange(0, x_size, x_size // X_TICK_SIZE))
    # ax4.set_xticklabels(map(generate_label, x_values[0:x_size:x_size // X_TICK_SIZE]))
    # ax4.set_yticks(np.arange(0, y_size, y_size // Y_TICK_SIZE))
    # ax4.set_yticklabels(map(generate_label, y_values[0:y_size:y_size // Y_TICK_SIZE]))
    plt.savefig(join("images", obj_sci + ".jpg"))
    print("IMAGE SAVED {}".format(join("images", obj_sci + ".jpg")))
