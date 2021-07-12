import numpy as np
from scipy.fft import fftfreq
from scipy.fft import fftshift

from pipeline import Pipeline

from preprocessing import bias_pipe
from preprocessing import mean_frame
from preprocessing import get_fft_square_magnitude
from preprocessing import disable_weak_pixels_pipe
from preprocessing import crop_image_pipe

from equatorial import get_psi
from equatorial import get_azimuth_height
from equatorial import transform_pixels_to_equatorial

from fitting import fit
from fitting import model

from sys import argv

from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import Angle
from astropy import units as u


def process_bias(pipeline):
    with fits.open(argv[1]) as bias_fits:
        bias_fits = bias_fits[0]
        print("BIAS {}".format(argv[1]))
        print(bias_fits.info())
        print("HEADER")
        print(bias_fits.header)
        bias = bias_fits.data
        master_bias = pipeline(bias)
    return master_bias


def process_sci_star(pipeline):
    with fits.open(argv[2]) as full_fits:
        sci_fits = full_fits[0]
        print("SCI_STAR {}".format(argv[2]))
        print(sci_fits.info())
        print("HEADER")
        print(sci_fits.header)
        time = Time(sci_fits.header["FRAME"], scale="utc")
        latitude = Angle(full_fits[1].header["LATITUDE"], unit=u.deg).radians
        longitude = Angle(full_fits[1].header["LONGITUD"], unit=u.deg).radians
        alpha = Angle(full_fits[0].header["RAAPP"], unit=u.deg).radians
        delta = Angle(full_fits[0].header["DECAPP"], unit=u.deg).radians
        sci = sci_fits.data
        sci_spectrum = np.asarray(pipeline(sci))
    return sci_spectrum, ((time, latitude, longitude), (alpha, delta))


def process_known_star(pipeline):
    with fits.open(argv[3]) as known_fits:
        known_fits = known_fits[0]
        print("KNOWN_STAR {}".format(argv[3]))
        print(known_fits.info())
        print("HEADER")
        print(known_fits.header)
        known = known_fits.data
        known_spectrum = np.asarray(pipeline(known))
    return known_spectrum


def obtain_fit_parameters(sci_spectrum, known_spectrum):
    y_data = sci_spectrum / known_spectrum
    y_size, x_size = y_data.shape()
    x_freq, y_freq = np.meshgrid(
        fftshift(fftfreq(x_size)),
        fftshift(fftfreq(y_size))
    )
    x_data = np.vstack((x_freq.flatten(), y_freq.flatten()))
    return fit(x_data, y_data.flatten(), model)


def main():
    if len(argv) < 4:
        print(("usage: main.py bias_fits scientific_star_fits"
               " ordinary_star_fits"))
        return

    shape = (512, 512)
    sigma_ron = 49
    # todo shape, sigma_ron
    pipeline = Pipeline(
        mean_frame,
        crop_image_pipe(shape)
    )
    master_bias = process_bias(pipeline)

    pipeline = Pipeline(
        mean_frame,
        crop_image_pipe(shape),
        bias_pipe(master_bias),
        disable_weak_pixels_pipe(sigma_ron),
        get_fft_square_magnitude
    )

    sci_spectrum, ((time, latitude, longitude), (alpha, delta)) = process_sci_star(pipeline)
    known_spectrum = process_known_star(pipeline)
    values, errors = obtain_fit_parameters(sci_spectrum, known_spectrum)

    print("FIT PARAMETERS")
    print("\n".join(
        map(
            lambda tup: "{} +- {}".format(*tup),
            zip(values, np.sqrt(np.diag(errors)))
        )
    ))
    dx, dy, epsilon, A = values
    xy = np.array([dx, dy])[::, np.newaxis]
    # todo plot spectrum
    azimuth, height = get_azimuth_height(time, alpha, delta, latitude, longitude)
    psi = get_psi(alpha, delta, azimuth. azimuth, height)
    s = 1.0
    # todo
    coordinates = transform_pixels_to_equatorial(s, xy, height, psi, epsilon)
    print("EQUATORIAL COORDINATES")
    print(coordinates)


if __name__ == "__main__":
    main()
