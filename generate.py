#!/usr/bin/env python

from matplotlib.axes import Axes
import logging
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import sys

logger = logging.getLogger(__name__)

LOG_LEVEL = {
    "error": logging.ERROR,
    "info": logging.INFO,
    "debug": logging.DEBUG
}

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
        if self.prev != None:
            return self.prev.generate()
        
        raise Exception('undefined')

    def __truediv__(self, other):
        other.prev = self
        return other

class ComplexCanvasGenerator(GeneratorBase):
    def __init__(self,
                 dims: tuple[int, int] = (1920, 1080),
                 center = np.float128(0) + 0j,
                 horizontal_diameter: np.float128 = np.float128(1)) -> None:
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

class SaveAsImageGenerator(GeneratorBase):
    def __init__(self, filename: str) -> None:
        super().__init__()
        self.__filename = filename

    def generate(self) -> np.ndarray:
        m = super().generate()
        fig = plt.figure(figsize=(width, height), dpi=1, frameon=False)
        ax = Axes(fig, (0, 0, 1, 1))
        ax.set_axis_off()
        fig.add_axes(ax)
        palette = colors.LinearSegmentedColormap.from_list('catppuccin-machiato',
                                                               colors=COLORS_CATPPUCCIN_MACHIATO,
                                                               under=COLORS_CATPPUCCIN_MACHIATO[0])
        plt.imshow(m,
                   cmap = palette,
                   vmin=0, vmax=1,
                   aspect='equal', interpolation='nearest')
        plt.savefig(self.__filename, bbox_inches="tight", pad_inches=0, dpi=1)
        return m

def mandelbrot_image(x_center: np.float128,
                     y_center: np.float128,
                     relative_width: np.float128,
                     width: int,
                     height: int,
                     iteration_count = 100,
                     filename = 'a.png'):
    generator = ComplexCanvasGenerator(
        center=x_center + 1j * y_center,
        dims=(width, height),
        horizontal_diameter=relative_width) \
        / MandelbrotScoreGenerator(iteration_count=iteration_count) \
        / SaveAsImageGenerator(filename=filename)
    generator.generate()

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

def initialize_logger(config):
    logger.setLevel(LOG_LEVEL[config["log_level"]])
    logger.info("BEGIN")
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    logger.debug(f"Loaded config: {config}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('argument not provided')
        exit(1)

    width, height = 1920, 1080
    match sys.argv[1]:
        case 'default':
            # The default one
            x_center = np.float128(-0.5)
            y_center = np.float128(0)
            relative_width = np.float128(5)
            mandelbrot_image(x_center, y_center, relative_width, width, height,
                             iteration_count=2**8,
                             filename='bin/default.png')
        case 'tails':
            # seahorse tails
            x_center = np.float128(-.74364085)
            y_center = np.float128(.13182733)
            relative_width = np.float128(.00012068)
            mandelbrot_image(x_center, y_center, relative_width, width, height,
                             iteration_count=2**10,
                             filename='bin/tail.png')
        case 'anthena':
            x_center = np.float128(-.7436447860)
            y_center = np.float128(.1318252536)
            relative_width = np.float128(.0000029336)
            mandelbrot_image(x_center, y_center, relative_width, width, height,
                             iteration_count=2**11,
                             filename='bin/anthena.png')
        case _:
            print('invalid selection')
            exit(1)
