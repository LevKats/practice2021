import numpy as np
from scipy.fft import fftfreq
from scipy.fft import fftshift

import matplotlib.pyplot as plt

from pipeline import Pipeline

from preprocessing import bias_pipe
from preprocessing import mean_frame
from preprocessing import fft_pipe
from preprocessing import disable_weak_pixels_pipe
from preprocessing import crop_image_pipe

# from equatorial import get_psi
# from equatorial import get_azimuth_height
# from equatorial import transform_pixels_to_equatorial

from fitting import fit
from fitting import model

# from sys import argv
import argparse
from os.path import join

from astropy.io import fits
# fits.Conf.use_memmap = True
from astropy.time import Time
# from astropy.coordinates import Angle
from astropy import units as u

from functools import partial

from tqdm import tqdm


def process_bias(pipeline, args):
    with fits.open(args.bias) as full_fits:
        bias_fits = full_fits[0]
        print("\n" + "="*50)
        print("BIAS {}".format(args.bias))
        # print(bias_fits.info())
        print("HEADER")
        print(bias_fits.header.tostring(sep='\n'))
        bias = bias_fits.data
        master_bias = pipeline(bias)
        sigma_ron = full_fits[1].header["RONSIGMA"] / bias_fits.header["SNTVTY"]
        D = full_fits[1].header["APERTURE"]
        wavelength = full_fits[0].header["DTNWLGTH"] * 10**-9  # todo
        del bias
    return master_bias, (sigma_ron, D, wavelength)


def process_sci_star(pipeline, args, batch_size=10):
    with fits.open(args.sci) as full_fits:
        sci_fits = full_fits[0]
        print("\n" + "=" * 50)
        print("SCI_STAR {}".format(args.sci))
        # print(sci_fits.info())
        print("HEADER")
        print(sci_fits.header.tostring(sep='\n'))
        time = Time(sci_fits.header["FRAME"], scale="utc")
        latitude = u.Quantity(full_fits[1].header["LATITUDE"], unit=u.deg).to(u.rad).value
        longitude = u.Quantity(full_fits[1].header["LONGITUD"], unit=u.deg).to(u.rad).value
        alpha = u.Quantity(full_fits[0].header["RA"], unit=u.deg).to(u.rad).value
        delta = u.Quantity(full_fits[0].header["DEC"], unit=u.deg).to(u.rad).value
        sci = sci_fits.data

        sci_spectrum = None
        frame_count = sci.shape[0]
        print("process specturm...")
        print(flush=True, end="")
        for last in tqdm(np.arange(0, frame_count, batch_size), position=0, leave=True):
            count = min(batch_size, frame_count - last)
            temp = mean_frame(pipeline(sci[last:last + count:, ::, ::]))
            if sci_spectrum is None:
                sci_spectrum = np.zeros((frame_count, temp.shape[-2], temp.shape[-1]), dtype=float)
            sci_spectrum[last:last + count:, ::, ::] = temp
        del sci
        print('Done.')
    return mean_frame(sci_spectrum), ((time, latitude, longitude), (alpha, delta))


def process_known_star(pipeline, args, batch_size=10):
    with fits.open(args.cal) as known_fits:
        known_fits = known_fits[0]
        print("\n" + "=" * 50)
        print("KNOWN_STAR {}".format(args.cal))
        # print(known_fits.info())
        print("HEADER")
        print(known_fits.header.tostring(sep='\n'))
        known = known_fits.data

        known_spectrum = None
        frame_count = known.shape[0]
        print("process spectrum...")
        print(flush=True, end="")
        for last in tqdm(np.arange(0, frame_count, batch_size), position=0, leave=True):
            count = min(batch_size, frame_count - last)
            temp = mean_frame(pipeline(known[last:last + count:, ::, ::]))
            if known_spectrum is None:
                known_spectrum = np.zeros((frame_count, temp.shape[-2], temp.shape[-1]), dtype=float)
            known_spectrum[last:last + count:, ::, ::] = temp
        del known
        print('Done.')
    return mean_frame(known_spectrum)


def obtain_fit_parameters(sci_spectrum, known_spectrum, fc):
    y_data = sci_spectrum / known_spectrum
    y_size, x_size = y_data.shape
    x_freq, y_freq = np.meshgrid(
        fftshift(fftfreq(x_size)),
        fftshift(fftfreq(y_size))
    )
    x_data = np.vstack((x_freq.flatten(), y_freq.flatten()))
    print(y_data.min(), y_data.max())
    return fit(
        x_data[::, ::500],
        ((y_data.flatten() * (np.linalg.norm(x_data, axis=0) <= 0.6 * fc)
          * (np.linalg.norm(x_data, axis=0) >= 0.1 * fc)))[::500],
        partial(model, fc=fc),
        p0=(10, -10, 0.5*(y_data.max() - y_data.min()) / y_data.max(), 0.5*y_data.max()),
        maxfev=500000,
        # nfev=100000,
        # bounds=([-1.0, -1.0, 0, 0.1 * y_data.max()], [1.0, 1.0, 1.0, y_data.max()]),
    )


def plot_spectrum(sci_spectrum, known_spectrum, func, out_path):
    y_data = sci_spectrum / known_spectrum
    y_size, x_size = y_data.shape
    x_values = fftshift(fftfreq(x_size))
    y_values = fftshift(fftfreq(y_size))
    x_freq, y_freq = np.meshgrid(
        x_values,
        y_values
    )
    x_data = np.stack((x_freq, y_freq))
    y_fit = func(x_data)
    print(sci_spectrum.min(), sci_spectrum.max())
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12), sharex="col", sharey="row")
    X_TICK_SIZE = 10
    Y_TICK_SIZE = 10
    GRID_LINEWIDTH = 0.5

    def generate_label(number):
        return "{:0.1f}".format(number)

    ax1.set_title("Power spectrum")
    ax1.imshow(y_data, vmin=min(y_data.min(), y_fit.min()), vmax=max(y_data.max(), y_fit.max()))
    ax1.set_xticks(np.arange(0, x_size, x_size // X_TICK_SIZE))
    ax1.set_xticklabels(map(generate_label, x_values[0:x_size:x_size // X_TICK_SIZE]))
    ax1.set_yticks(np.arange(0, y_size, y_size // Y_TICK_SIZE))
    ax1.set_yticklabels(map(generate_label, y_values[0:y_size:y_size // Y_TICK_SIZE]))
    ax1.grid(linewidth=GRID_LINEWIDTH, linestyle='--')

    ax2.set_title("Estimation")
    ax2.imshow(y_fit, vmin=min(y_data.min(), y_fit.min()), vmax=max(y_data.max(), y_fit.max()))
    ax2.set_xticks(np.arange(0, x_size, x_size // X_TICK_SIZE))
    ax2.set_xticklabels(map(generate_label, x_values[0:x_size:x_size // X_TICK_SIZE]))
    ax2.grid(linewidth=GRID_LINEWIDTH, linestyle='--')
    # ax2.set_yticks(np.arange(0, y_size, y_size // Y_TICK_SIZE))
    # ax2.set_yticklabels(map(generate_label, y_values[0:y_size:y_size // Y_TICK_SIZE]))

    ax3.set_title("Scientific star")
    ax3.imshow(sci_spectrum, vmax=5*10**9)
    # ax3.set_xticks(np.arange(0, x_size, x_size // X_TICK_SIZE))
    # ax3.set_xticklabels(map(generate_label, x_values[0:x_size:x_size // X_TICK_SIZE]))
    ax3.set_yticks(np.arange(0, y_size, y_size // Y_TICK_SIZE))
    ax3.set_yticklabels(map(generate_label, y_values[0:y_size:y_size // Y_TICK_SIZE]))
    ax3.grid(linewidth=GRID_LINEWIDTH, linestyle='--')

    ax4.set_title("Known single star")
    ax4.imshow(known_spectrum, vmax=5*10**9)
    ax4.grid(linewidth=GRID_LINEWIDTH, linestyle='--')
    # ax4.set_xticks(np.arange(0, x_size, x_size // X_TICK_SIZE))
    # ax4.set_xticklabels(map(generate_label, x_values[0:x_size:x_size // X_TICK_SIZE]))
    # ax4.set_yticks(np.arange(0, y_size, y_size // Y_TICK_SIZE))
    # ax4.set_yticklabels(map(generate_label, y_values[0:y_size:y_size // Y_TICK_SIZE]))
    plt.savefig(out_path)
    print("IMAGE SAVED {}".format(out_path))


def main():
    # if len(argv) < 4:
    #     print(("usage: main.py bias_fits scientific_star_fits"
    #            " ordinary_star_fits"))
    #     return
    def left_angle_parser(s):
        try:
            y, x = map(int, s.split(','))
            return y, x
        except Exception:
            raise argparse.ArgumentTypeError("Coordinates must be y, x")

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bias", help="Bias path")
    parser.add_argument("-s", "--sci", help="Scientific star path")
    parser.add_argument("-c", "--cal", help="Known single star path")
    parser.add_argument("-ps", "--pixsize", help="Pixel size, m", type=float)
    parser.add_argument("-fs", "--fieldsize", help="Field size, px", type=int)
    parser.add_argument('-l', '--leftangle', help="Coordinate", dest="leftangle", type=left_angle_parser, nargs=1)
    parser.add_argument("-f", "--focal", help="F, m", type=float)
    parser.add_argument("-o", "--outimage", help="spectrum image path")

    args = parser.parse_args()
    print(args)

    pixel_size = args.pixsize
    field_size = args.fieldsize
    left_angle = args.leftangle[0]
    F = args.focal
    shape = (field_size, field_size)
    pipeline = Pipeline(
        mean_frame,
        crop_image_pipe(left_angle, shape)
    )
    master_bias, (sigma_ron, D, wavelength) = process_bias(pipeline, args)
    fc = 1.0 / (wavelength/D * F/pixel_size)
    # print(fc)
    # fc = 0.45

    pipeline = Pipeline(
        crop_image_pipe(left_angle, shape),
        bias_pipe(master_bias),
        disable_weak_pixels_pipe(sigma_ron),
        fft_pipe(30, 1),
        # mean_frame
    )

    sci_spectrum, ((time, latitude, longitude), (alpha, delta)) = process_sci_star(pipeline, args)
    # print(sci_spectrum)
    known_spectrum = process_known_star(pipeline, args)
    # print(sci_spectrum / known_spectrum)

    # <DEBUG>
    values, errors = obtain_fit_parameters(sci_spectrum, known_spectrum, fc)

    print("FIT PARAMETERS")
    print("\n".join(
        map(
            lambda tup: "{} +- {}".format(*tup),
            zip(values, np.sqrt(np.diag(errors)))
        )
    ))
    dx, dy, epsilon, A = values
    xy = np.array([dx, dy])[::, np.newaxis]
    # dx = 1
    # dy = 1
    # epsilon = 1
    # A = 10
    # </DEBUG>

    plot_spectrum(
        sci_spectrum, known_spectrum,
        partial(model, dx=dx, dy=dy, epsilon=epsilon, A=A, fc=fc),
        args.outimage
    )
    # azimuth, height = get_azimuth_height(time, alpha, delta, latitude, longitude)
    # psi = get_psi(alpha, delta, azimuth. azimuth, height)
    # s = 1.0
    # todo
    # coordinates = transform_pixels_to_equatorial(s, xy, height, psi, epsilon)
    # print("EQUATORIAL COORDINATES")
    # print(coordinates)


if __name__ == "__main__":
    main()
