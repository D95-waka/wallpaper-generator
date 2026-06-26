#!/usr/bin/env python

from matplotlib.axes import Axes
import logging
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

logger = logging.getLogger(__name__)

COLORS_CATPPUCCIN_MACHIATO = [
    '#24273a', # Base
    '#363a4f', # Base 0
    '#8aadf4', # Blue
    '#8bd5ca', # Teal
    '#a6da95', # Green
    '#eed49f', # Yellow
    '#f5a97f', # Peach
    '#ee99a0', # Maroon
    '#ed8796', # Red
]

def cache(f):
    cached_value = None
    def new_f(*args):
        nonlocal cached_value
        if cached_value == None:
            cached_value = f(*args)

        return cached_value

    return new_f

class GeneratorBase(object):
    def __init__(self) -> None:
        self.prev: GeneratorBase | None = None

    def generate(self) -> np.ndarray:
        logger.debug(f'generate called with {type(self)}')
        if self.prev != None:
            return self.prev.generate()
        
        raise Exception('undefined')

    def __truediv__(self, other):
        other.prev = self
        return other

class ConstantCanvasGenerator(GeneratorBase):
    def __init__(self, array: np.ndarray) -> None:
        super().__init__()
        self.__array = array

    def generate(self) -> np.ndarray:
        return self.__array.copy()

class ComplexCanvasGenerator(GeneratorBase):
    def __init__(self,
                 dims: tuple[int, int] = (1920, 1080),
                 center: np.complexfloating | complex = np.float128(0) + 0j,
                 horizontal_diameter: np.float128 | float = np.float128(1)) -> None:
        super().__init__()
        self.__dims = dims
        self.__center = center
        self.__horizontal_diameter = horizontal_diameter

    def generate(self):
        xmin = np.real(self.__center) - self.__horizontal_diameter / 2,
        xmax = np.real(self.__center) + self.__horizontal_diameter / 2,
        ymin = np.imag(self.__center) + self.__horizontal_diameter * self.__dims[1] / self.__dims[0] / 2,
        ymax = np.imag(self.__center) - self.__horizontal_diameter * self.__dims[1] / self.__dims[0] / 2,
        x = np.linspace(xmin, xmax, self.__dims[0])
        y = np.linspace(ymin, ymax, height)
        xx, yy = np.meshgrid(x, y)
        zz = xx + 1j * yy
        return zz

class MandelbrotScoreGenerator(GeneratorBase):
    def __init__(self, iteration_count = 100) -> None:
        super().__init__()
        self.__iteration_count = iteration_count

    @cache
    def generate(self):
        c: np.ndarray = super().generate()
        z = np.zeros_like(c, dtype=np.complex256)
        results = np.zeros_like(c, dtype=np.int16)
        to_compute = results == 0
        for i in range(self.__iteration_count):
            logger.debug(f'{round(100 * i / self.__iteration_count, 2)}% completed')
            z[to_compute] = np.power(z[to_compute], 2) + c[to_compute]
            to_compute &= np.abs(z) < 2
            results[to_compute] += 1

        results[to_compute] = -1
        return results / self.__iteration_count

class ScalarColorizerBase(object):
    def get_color(self, x):
        xmin = np.min(x)
        xmax = np.max(x)
        m = 255 * (x - xmin) / (xmax - xmin)
        return np.stack([m, m, m], axis=2).astype(int)

class DiscreteColorizer(ScalarColorizerBase):
    def __init__(self, map: list) -> None:
        super().__init__()
        self.__map = map

    def get_color(self, x):
        return self.__map[x]

class ColorizeGenerator(GeneratorBase):
    def __init__(self, colorizer: ScalarColorizerBase) -> None:
        super().__init__()
        self.__colorizer = colorizer

    def generate(self) -> np.ndarray:
        m = super().generate()
        return self.__colorizer.get_color(m)

class SaveAsImageGenerator(GeneratorBase):
    def __init__(self, filename: str) -> None:
        super().__init__()
        self.__filename = filename

    def generate(self) -> np.ndarray:
        m = super().generate()
        fig = plt.figure(figsize=(m.shape[1], m.shape[0]), dpi=1, frameon=False)
        ax = Axes(fig, (0, 0, 1, 1))
        ax.set_axis_off()
        fig.add_axes(ax)
        palette = colors.LinearSegmentedColormap.from_list('catppuccin-machiato',
                                                               colors=COLORS_CATPPUCCIN_MACHIATO,
                                                               under=COLORS_CATPPUCCIN_MACHIATO[0])
        plt.imshow(m,
                   cmap = palette,
                   vmin=0,
                   vmax=1,
                   aspect='equal',
                   interpolation='nearest')
        plt.savefig(self.__filename, bbox_inches="tight", pad_inches=0, dpi=1)
        return m

class CustomFormatter(logging.Formatter):
    __grey = "\x1b[38;21m"
    __yellow = "\x1b[33;21m"
    __red = "\x1b[31;21m"
    __bold_red = "\x1b[31;1m"
    __reset = "\x1b[0m"
    __format = "%(asctime)s, %(name)s, %(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: __grey + __format + __reset,
        logging.INFO: __grey + __format + __reset,
        logging.WARNING: __yellow + __format + __reset,
        logging.ERROR: __red + __format + __reset,
        logging.CRITICAL: __bold_red + __format + __reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def initialize_logger(level):
    logger.setLevel(level)
    logger.debug("BEGIN")
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

if __name__ == '__main__':
    # Args
    parser = argparse.ArgumentParser(
        prog='Image Generator',
        description='Create and colorize analytic images, such as fractals',
        epilog='for more help read the code')
    parser.add_argument('mode')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    initialize_logger(logging.DEBUG if args.verbose else logging.INFO)

    # Init mode
    mode = args.mode
    width, height = 1920, 1080
    modes = {
        'default': ComplexCanvasGenerator(
            center=-0.5 + 0j,
            dims=(width, height),
            horizontal_diameter=5) \
            / MandelbrotScoreGenerator(iteration_count=2**8) \
            / SaveAsImageGenerator(filename='bin/default.png'),
        'tails': ComplexCanvasGenerator(
            center=np.float128(-.74364085) + np.float128(.13182733) * 1j,
            horizontal_diameter=np.float128(.00012068),
            dims=(width, height)) \
            / MandelbrotScoreGenerator(iteration_count=2**10) \
            / SaveAsImageGenerator(filename='bin/tails.png'),
        'anthena': ComplexCanvasGenerator(
            center=np.float128(-.7436447860) + np.float128(.1318252536) * 1j,
            horizontal_diameter=np.float128(.0000029336),
            dims=(width, height)) \
            / MandelbrotScoreGenerator(iteration_count=2**11) \
            / SaveAsImageGenerator(filename='bin/anthena.png'),
    }
    if not mode in modes:
        print(f'available modes: {", ".join(modes.keys())}')
        exit(1)

    modes[mode].generate()
