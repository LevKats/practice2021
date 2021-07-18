import numpy as np
from scipy.fft import fftfreq
from scipy.fft import fftshift
from scipy.fft import ifft2

import matplotlib.pyplot as plt

from pipeline import Pipeline

from preprocessing import bias_pipe
from preprocessing import mean_frame
from preprocessing import fft_pipe
from preprocessing import disable_weak_pixels_pipe
from preprocessing import crop_image_pipe
from preprocessing import photon_pipe

from constants import MIN_FREQ_MASK
from constants import MAX_FREQ_MASK
from constants import PHOTON_NOISE_FREQ_MASK
from constants import X_TICK_SIZE
from constants import Y_TICK_SIZE
from constants import GRID_LINEWIDTH
from constants import MIN_POWERSPECTRUM
from constants import MAX_POWERSPECTRUM
from constants import MAX_SPECTRUM

from equatorial import get_psi
from equatorial import get_azimuth_height
from equatorial import pixels_to_equatorial_jac
from equatorial import pixels_to_equatorial_errors

from fitting import fit
from fitting import model

import argparse
from os.path import exists
from os.path import join

from astropy.io import fits
from astropy.time import Time
from astropy import units as u

from functools import partial

from tqdm import tqdm


def process_bias(pipeline, args):
    with fits.open(args.bias) as full_fits:
        bias_fits = full_fits[0]
        print("\n" + "="*50)
        print("METADATA")
        print(full_fits[1].header.tostring(sep='\n'))
        print("\n" + "="*50)
        print("BIAS {}".format(args.bias))
        # print(bias_fits.info())
        print("HEADER")
        print(bias_fits.header.tostring(sep='\n'))
        bias = bias_fits.data
        master_bias = pipeline(bias)
        sigma_ron = full_fits[1].header["RONSIGMA"] / bias_fits.header["SNTVTY"]
        D = full_fits[1].header["APERTURE"]
        latitude = u.Quantity(full_fits[1].header["LATITUDE"], unit=u.deg).to(u.rad).value
        longitude = u.Quantity(full_fits[1].header["LONGITUD"], unit=u.deg).to(u.rad).value
        height = full_fits[1].header["ALTITUDE"]
        # wavelength = full_fits[1].header["FILTLAM"] * 10**-9  # todo
        del bias
    return master_bias, ((sigma_ron, D), (latitude, longitude, height))


def process_spectrum(image_fits, batch_size, pipeline, filename, args):
    if exists(filename):
        if args.y:
            st = "yes"
        elif args.n:
            st = "no"
        else:
            st = input("{} found. Load? yes/no (delault yes) ".format(filename))
        if st == "yes" or st == "":
            with np.load(filename) as data:
                return data["spectrum"]
    frame = image_fits.data
    spectrum = None
    frame_count = frame.shape[0]
    print("process spectrum...")
    print(flush=True, end="")
    for last in tqdm(np.arange(0, frame_count, batch_size), position=0, leave=True):
        count = min(batch_size, frame_count - last)
        temp = mean_frame(pipeline(frame[last:last + count:, ::, ::]))
        if spectrum is None:
            spectrum = np.zeros((frame_count, temp.shape[-2], temp.shape[-1]), dtype=float)
        spectrum[last:last + count:, ::, ::] = temp
    del frame
    print('Done.')
    if args.y:
        st = "yes"
    elif args.n:
        st = "no"
    else:
        st = input("save spectrum to {}? yes/no (default yes) ".format(filename))
    mean_spectrum = mean_frame(spectrum)
    if st == "yes" or st == "":
        np.savez(filename, spectrum=mean_spectrum)
    return mean_spectrum


def process_sci_star(pipeline, args, batch_size=10):
    with fits.open(args.sci) as full_fits:
        sci_fits = full_fits[0]
        print("\n" + "=" * 50)
        print("SCI_STAR {}".format(args.sci))
        # print(sci_fits.info())
        print("HEADER")
        print(sci_fits.header.tostring(sep='\n'))
        time = Time(sci_fits.header["FRAME"], scale="utc")
        alpha = u.Quantity(full_fits[0].header["RA"], unit=u.deg).to(u.rad).value
        delta = u.Quantity(full_fits[0].header["DEC"], unit=u.deg).to(u.rad).value
        obj = sci_fits.header["OBJECT"]
        wavelength = full_fits[1].header["FILTLAM"] * 10 ** -9
        temp = full_fits[1].header["AMBTEMP"]
        press = full_fits[1].header["AMBPRESS"]
        humid = full_fits[1].header["AMBHUMID"]
        mean_spectrum = process_spectrum(sci_fits, batch_size, pipeline, join("spectra", obj + ".npz"), args)
    return mean_spectrum, ((time, wavelength, temp, press, humid), (alpha, delta), obj)


def process_known_star(pipeline, mean_spectrum_pipeline, args, batch_size=10):
    with fits.open(args.cal) as known_fits:
        known_fits = known_fits[0]
        print("\n" + "=" * 50)
        print("KNOWN_STAR {}".format(args.cal))
        # print(known_fits.info())
        print("HEADER")
        print(known_fits.header.tostring(sep='\n'))

        obj = known_fits.header["OBJECT"]
        mean_spectrum = process_spectrum(known_fits, batch_size, pipeline, join("spectra", obj + ".npz"), args)
    return mean_spectrum_pipeline(mean_spectrum), obj


def obtain_fit_parameters(sci_spectrum, known_spectrum, fc, p0_mask_radius):
    y_data = sci_spectrum / known_spectrum
    y_size, x_size = y_data.shape
    x_freq, y_freq = np.meshgrid(
        fftshift(fftfreq(x_size)),
        fftshift(fftfreq(y_size))
    )
    x_data = np.vstack((x_freq.flatten(), y_freq.flatten()))
    freqs = np.stack((x_freq, y_freq))
    mask = ((np.linalg.norm(freqs, axis=0) <= MAX_FREQ_MASK * fc) *
            (np.linalg.norm(freqs, axis=0) >= MIN_FREQ_MASK * fc))
    # mask *= (np.abs(np.arange(-128, 128, 256)) > 10)[::, np.newaxis]

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
    # parser.add_argument("-o", "--outimage", help="spectrum image path")
    parser.add_argument("-p0r", "--p0radius", help="p0 mask radius", type=int)
    parser.add_argument("-y", help="auto 'yes' in questions", action='store_true')
    parser.add_argument("-n", help="auto 'no' in questions", action='store_true')
    parser.add_argument("-mxps", help="max powerspecturm value", default=MAX_POWERSPECTRUM, type=float)
    parser.add_argument("-mnps", help="min powerspecturm value", default=MIN_POWERSPECTRUM, type=float)
    parser.add_argument("-ms", help="max specturm value", default=MAX_SPECTRUM, type=float)

    args = parser.parse_args()
    print(args)

    p0_mask_radius = args.p0radius
    pixel_size = args.pixsize
    field_size = args.fieldsize
    left_angle = args.leftangle[0]
    F = args.focal
    shape = (field_size, field_size)
    pipeline = Pipeline(
        mean_frame,
        crop_image_pipe(left_angle, shape)
    )
    master_bias, ((sigma_ron, D), (latitude, longitude, height)) = process_bias(pipeline, args)
    # fc = 0.45

    pipeline = Pipeline(
        crop_image_pipe(left_angle, shape),
        bias_pipe(master_bias),
        disable_weak_pixels_pipe(sigma_ron),
        fft_pipe(30, 1),
        # mean_frame
    )
    mean_spectrum, ((time, wavelength, temp, press, humid), (alpha, delta), obj_sci) = process_sci_star(
        pipeline, args
    )
    # fc = 1.0 / (wavelength / D * F / pixel_size)
    s = pixel_size / F
    fc = D/wavelength * s
    mean_spectrum_pipeline = Pipeline(
        photon_pipe(PHOTON_NOISE_FREQ_MASK * fc)
    )
    sci_spectrum = np.asarray(mean_spectrum_pipeline(mean_spectrum))
    # print(sci_spectrum)
    known_spectrum, obj_kno = process_known_star(pipeline, mean_spectrum_pipeline, args)
    # print(sci_spectrum / known_spectrum)
    print("\n" + "=" * 50)
    print("FC = {}".format(fc))
    # <DEBUG>
    values, errors = obtain_fit_parameters(sci_spectrum, known_spectrum, fc, p0_mask_radius)
    names = "dx", "dy", "epsilon", "A"

    print("FIT PARAMETERS")
    print("\n".join(
        map(
            lambda tup: "{} = {} +- {}".format(*tup),
            zip(names, values, np.sqrt(np.diag(errors)))
        )
    ))
    dx, dy, epsilon, A = values
    xyz = np.zeros(4)
    xyz[:2:] = dx, dy
    xyz = xyz[::, np.newaxis]

    plot_spectrum(
        sci_spectrum, known_spectrum,
        partial(model, dx=dx, dy=dy, epsilon=epsilon, A=A, fc=fc),
        obj_sci, obj_kno,
        fc,
        args.mnps, args.mxps, args.ms
    )
    azimuth, h = get_azimuth_height(time, alpha, delta, latitude, longitude, height, temp, press, humid, wavelength)
    psi = get_psi(azimuth, h, latitude)
    # todo
    jac = pixels_to_equatorial_jac(s, psi - h + np.pi, alpha, delta)
    (da,), (dd,) = jac @ xyz
    print("s", s)
    print("delta = ", delta)
    edx, edy, *_ = np.sqrt(np.diag(errors))
    eda, edd = pixels_to_equatorial_errors(edx, edy, jac)
    print("da = {} +- {}\ndd = {} +- {}".format(da * 206265, eda * 206265, dd * 206265, edd * 206265))
    print("sep", np.sqrt((da * np.cos(delta))**2 + dd**2) * 206265)
    print("sep2", s * np.sqrt(dx**2 + dy**2) * 206265)
    print("h, psi", h, psi)
    print("jac", pixels_to_equatorial_jac(s, psi - h + np.pi, alpha, delta))
    # print("EQUATORIAL COORDINATES")
    # print(coordinates)


if __name__ == "__main__":
    main()
