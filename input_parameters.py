import argparse

from constants import MAX_POWERSPECTRUM
from constants import MIN_POWERSPECTRUM
from constants import MAX_SPECTRUM
from constants import MASK_HORIZONTAL_WINDOW_SIZE
from constants import MIN_FREQ_MASK
from constants import MAX_FREQ_MASK


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
    parser.add_argument("-b", "--bias", help="Bias path", required=True)
    parser.add_argument("-s", "--sci", help="Scientific star path", required=True)
    parser.add_argument("-c", "--cal", help="Known single star path", required=True)
    parser.add_argument("-ps", "--pixsize", help="Pixel size, m", type=float, required=True)
    parser.add_argument("-fs", "--fieldsize", help="Field size, px", type=int, required=True)
    parser.add_argument(
        '-l', '--leftangle', help="Field left angle coordinate y,x px",
        dest="leftangle", type=left_angle_parser, nargs=1, required=True
    )
    parser.add_argument("-f", "--focal", help="F, m", type=float, required=True)
    # parser.add_argument("-o", "--outimage", help="spectrum image path", required=True)
    parser.add_argument("-p0r", "--p0radius", help="p0 mask radius, px", type=int, required=True)
    parser.add_argument("-y", help="auto 'yes' in questions", action='store_true')
    parser.add_argument("-n", help="auto 'no' in questions", action='store_true')
    parser.add_argument("-mxps", help="max powerspecturm value, optional", default=MAX_POWERSPECTRUM, type=float)
    parser.add_argument("-mnps", help="min powerspecturm value, optional", default=MIN_POWERSPECTRUM, type=float)
    parser.add_argument("-ms", help="max specturm value, optional", default=MAX_SPECTRUM, type=float)
    parser.add_argument("-ws", help="kaiser window size", default=MASK_HORIZONTAL_WINDOW_SIZE, type=int)
    parser.add_argument(
        "-mnfmsk", help="min freq mask (defalt {})".format(MIN_FREQ_MASK), default=MIN_FREQ_MASK, type=float
    )
    parser.add_argument(
        "-mxfmsk", help="max freq mask (defalt {})".format(MAX_FREQ_MASK), default=MAX_FREQ_MASK, type=float
    )
    parser.add_argument(
        '-p0', help="p0 dx,dy", dest="p0", type=left_angle_parser, nargs=1, default=(None,)
    )

    return parser.parse_args()
