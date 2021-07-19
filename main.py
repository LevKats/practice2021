import numpy as np

from pipeline import Pipeline

from preprocessing import bias_pipe
from preprocessing import fft_pipe
from preprocessing import disable_weak_pixels_pipe
from preprocessing import crop_image_pipe
from preprocessing import photon_pipe
from preprocessing import mean_frame

from constants import PHOTON_NOISE_FREQ_MASK

from equatorial import get_psi
from equatorial import get_azimuth_height
from equatorial import pixels_to_equatorial_jac
from equatorial import pixels_to_equatorial_errors

from fitting import obtain_fit_parameters
from fitting import model

from plot import plot_spectrum

from fits_processing import process_bias
from fits_processing import process_sci_star
from fits_processing import process_known_star

from input_parameters import get_args

from functools import partial

from os import mkdir
from os.path import exists


def main():
    if not exists("spectra/"):
        mkdir("spectra")
        print("spectra/ created/")
    if not exists("images/"):
        mkdir("images")
        print("images/ created")
    args = get_args()
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
