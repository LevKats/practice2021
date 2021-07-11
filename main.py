import numpy as np
from scipy.fft import fftfreq
from scipy.fft import fftshift

from pipeline import Pipeline

from preprocessing import get_master_bias
from preprocessing import bias_pipe
from preprocessing import mean_frame
from preprocessing import get_fft_square_magnitude

from fitting import fit
from fitting import model

from sys import argv
from astropy.io import fits


def process_bias():
    with fits.open(argv[1]) as bias_fits:
        bias_fits = bias_fits[0]
        print("BIAS {}".format(argv[1]))
        print(bias_fits.info())
        print("HEADER")
        print(bias_fits.header)
        bias = bias_fits.data
        master_bias = get_master_bias(bias)
    return master_bias


def process_sci_star(pipeline):
    with fits.open(argv[2]) as sci_fits:
        sci_fits = sci_fits[0]
        print("SCI_STAR {}".format(argv[2]))
        print(sci_fits.info())
        print("HEADER")
        print(sci_fits.header)
        sci = sci_fits.data
        sci_spectrum = np.asarray(pipeline(sci))
    return sci_spectrum


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

    master_bias = process_bias()

    pipeline = Pipeline(
        mean_frame,
        bias_pipe(master_bias),
        get_fft_square_magnitude
    )

    sci_spectrum = process_sci_star(pipeline)
    known_spectrum = process_known_star(pipeline)
    values, errors = obtain_fit_parameters(sci_spectrum, known_spectrum)

    print("FIT PARAMETERS")
    print("\n".join(
        map(
            lambda tup: "{} +- {}",
            zip(values, np.sqrt(np.diag(errors)))
        )
    ))
    # todo plot spectrum


if __name__ == "__main__":
    main()
