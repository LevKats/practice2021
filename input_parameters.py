import argparse

from constants import MAX_POWERSPECTRUM
from constants import MIN_POWERSPECTRUM
from constants import MAX_SPECTRUM


def get_args():
    # if len(argv) < 4:
    #     print(("usage: main.py bias_fits scientific_star_fits"
    #            " ordinary_star_fits"))
    #     return
    def left_angle_parser(st):
        try:
            y, x = map(int, st.split(','))
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

    return parser.parse_args()
