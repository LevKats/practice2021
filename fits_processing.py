import numpy as np

from os.path import exists
from os.path import join

from astropy.io import fits
from astropy.time import Time
from astropy import units as u

from tqdm import tqdm

from preprocessing import mean_frame

from constants import SKIP


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

        filename = join("spectra", "bias" + full_fits[0].header["FRAME"] + ".npz")
        if exists(filename):
            if args.y:
                st = "yes"
            elif args.n:
                st = "no"
            else:
                st = input("{} found. Load? yes/no (delault yes) ".format(filename))
            if st == "yes" or st == "":
                with np.load(filename) as data:
                    master_bias = data["master_bias"]
        else:
            bias = bias_fits.data
            master_bias = pipeline(bias)
            del bias
            if args.y:
                st = "yes"
            elif args.n:
                st = "no"
            else:
                st = input("save bias to {}? yes/no (default yes) ".format(filename))
            if st == "yes" or st == "":
                np.savez(filename, master_bias=master_bias)

        sigma_ron = full_fits[1].header["RONSIGMA"] / bias_fits.header["SNTVTY"]
        D = full_fits[1].header["APERTURE"]
        latitude = u.Quantity(full_fits[1].header["LATITUDE"], unit=u.deg).to(u.rad).value
        longitude = u.Quantity(full_fits[1].header["LONGITUD"], unit=u.deg).to(u.rad).value
        height = full_fits[1].header["ALTITUDE"]
        # wavelength = full_fits[1].header["FILTLAM"] * 10**-9  # todo
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
    frame = image_fits.data[SKIP::, ::, ::]
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
