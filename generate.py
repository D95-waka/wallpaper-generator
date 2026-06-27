#!/usr/bin/env python

from matplotlib.axes import Axes
import logging
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.signal import convolve

logger = logging.getLogger(__name__)

COLOR_DEFS = {
    'catppuccin-latte': [
        '#eff1f5', # Base
        '#ccd0da', # Base 0
        '#1e66f5', # Blue
        '#179299', # Teal
        '#40a02b', # Green
        '#df8e1d', # Yellow
        '#fe640b', # Peach
        '#e64553', # Maroon
        '#d20f39', # Red
    ],
    'catppuccin-machiato': [
        '#24273a', # Base
        '#363a4f', # Base 0
        '#8aadf4', # Blue
        '#8bd5ca', # Teal
        '#a6da95', # Green
        '#eed49f', # Yellow
        '#f5a97f', # Peach
        '#ee99a0', # Maroon
        '#ed8796', # Red
    ],
}
DEFAULT_THEME_NAME = 'catppuccin-machiato'

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
        y = np.linspace(ymin, ymax, self.__dims[1])
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

class SetGenerator(GeneratorBase):
    def __init__(self, radius = 1) -> None:
        super().__init__()
        self.__radius = radius

    def generate(self) -> np.ndarray:
        m = super().generate()
        epsilon = np.abs(m[0, 0] - m[0, 1])
        result = np.zeros_like(m, dtype=float)
        result[self.inset(m, epsilon)] = 1
        result = convolve(result, self.convolve_with(), mode='same')
        result /= np.max(result)
        result[result < 0.001] = 0
        result[result > 0] = 1
        return result

    def inset(self, x, epsilon = 0):
        return np.abs(x) < epsilon

    @cache
    def convolve_with(self):
        l = np.arange(-self.__radius, self.__radius + 1)
        xx, yy = np.meshgrid(l, l)
        zz = xx + 1j * yy
        circle = np.abs(zz) < self.__radius
        circle = circle.astype(int)
        return circle

class FunctionGraphGenerator(SetGenerator):
    def __init__(self, fn, radius=1) -> None:
        super().__init__(radius)
        self.__fn = fn

    def inset(self, x, epsilon=0):
        return np.abs(np.imag(x) - self.__fn(np.real(x))) < epsilon

class ComplexToRealMapGenerator(GeneratorBase):
    def __init__(self, fn) -> None:
        super().__init__()
        self.__fn = fn

    def generate(self) -> np.ndarray:
        m = super().generate()
        return self.__fn(m)

def color_to_array(color: str | list) -> np.ndarray:
    if isinstance(color, str):
        hex_str = color.lstrip('#').strip()
        if len(hex_str) in (3, 4):
            hex_str = "".join(char * 2 for char in hex_str)

        if len(hex_str) not in (6, 8):
            raise ValueError(f"Invalid HEX color format: '{color}'")

        try:
            channels = [int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)]
        except ValueError as e:
            raise ValueError(f"Invalid hex characters in color: '{color}'") from e

        if len(channels) == 3:
            channels.append(255)

        return np.array(channels, dtype=np.float64) / 255.0

    elif isinstance(color, (list, tuple, np.ndarray)):
        arr = np.array(color, dtype=np.float64)
        if arr.size not in (3, 4):
            raise ValueError("Color array must have 3 (RGB) or 4 (RGBA) elements.")

        if np.any(arr > 1.0):
            arr /= 255.0

        if arr.size == 3:
            arr = np.append(arr, 1.0)

        return np.clip(arr, 0.0, 1.0)

class ColorProviderBase(object):
    def convert(self, score: np.ndarray) -> np.ndarray:
        return np.repeat([np.zeros_like(score)], 4, axis=0)

class ThresholdColorProvider(ColorProviderBase):
    def __init__(self,
                 mmin = None,
                 mmax = None,
                 under_color = '#000',
                 over_color = '#fff') -> None:
        super().__init__()
        self.__mmin = mmin
        self.__mmax = mmax
        self.__under_color = color_to_array(under_color)
        self.__over_color = color_to_array(over_color)

    def convert(self, score: np.ndarray) -> np.ndarray:
        mmin = self.__mmin if self.__mmin != None else np.min(score)
        mmax = self.__mmax if self.__mmax != None else np.max(score)
        m = (score - mmin) / (mmax - mmin)
        colored = np.zeros(list(m.shape) + [4])
        colored[m < 0, :] = self.__under_color
        colored[m >= 1, :] = self.__over_color
        in_range = (0 <= m) & (m < 1)
        colored[in_range, :] = self.convert_in_range(m[in_range])
        return colored

    def convert_in_range(self, score: np.ndarray) -> np.ndarray:
        return np.repeat([np.full_like(0.5, score)], 4, axis=0)


class LinearGradientColorProvider(ThresholdColorProvider):
    def __init__(self, colors: list, mmin=None, mmax=None, under_color='#000', over_color='#fff') -> None:
        super().__init__(mmin, mmax, under_color, over_color)
        self.__colors = np.array([color_to_array(x) for x in colors])

    def convert_in_range(self, score: np.ndarray) -> np.ndarray:
        lower_colors = self.__colors[np.floor((len(self.__colors) - 1) * score).astype(np.int64), :]
        upper_colors = self.__colors[np.ceil((len(self.__colors) - 1) * score).astype(np.int64), :]
        convex_combination = (((len(self.__colors) - 1) * score) % 1)[:, None]
        return lower_colors * (1 - convex_combination) + upper_colors * convex_combination

class ColorProviderHolder(ColorProviderBase):
    def __init__(self, default_provider: ColorProviderBase = ColorProviderBase()) -> None:
        super().__init__()
        self.__inner_provider = default_provider

    def convert(self, score: np.ndarray) -> np.ndarray:
        return self.__inner_provider.convert(score)

    def update_provider(self, provider: ColorProviderBase):
        self.__inner_provider = provider

class ColorizeGenerator(GeneratorBase):
    def __init__(self, color_provider: ColorProviderBase) -> None:
        super().__init__()
        self.__color_provider = color_provider

    def generate(self) -> np.ndarray:
        m = super().generate()
        return self.__color_provider.convert(m)

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
        plt.imshow(m,
                   cmap='gray',
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
    parser.add_argument('mode', help='predefined modes')
    parser.add_argument('-o', '--output', help='output file path, default to bin/<mode>-<theme>.png')
    parser.add_argument('-v', '--verbose', action='store_true', help='show debug logs')
    parser.add_argument('-t', '--theme', choices=COLOR_DEFS.keys(),
                        default=DEFAULT_THEME_NAME, help='color theme')
    args = parser.parse_args()
    initialize_logger(logging.DEBUG if args.verbose else logging.INFO)
    provider_updater = ColorProviderHolder()

    # Init mode
    mode = args.mode
    output_path = args.output if args.output != None else f'./bin/{mode}-{args.theme}.png'
    colors = COLOR_DEFS[args.theme]
    width, height = 1920, 1080
    color_provider = LinearGradientColorProvider(colors=colors,
                                                 under_color=colors[0],
                                                 mmin=0,
                                                 mmax=1)
    provider_updater.update_provider(color_provider)
    modes = {
        'default': ComplexCanvasGenerator(
            center=-0.5 + 0j,
            dims=(width, height),
            horizontal_diameter=5) \
            / MandelbrotScoreGenerator(iteration_count=2**7) \
            / ColorizeGenerator(provider_updater) \
            / SaveAsImageGenerator(filename=output_path),
        'tails': ComplexCanvasGenerator(
            center=np.float128(-.74364085) + np.float128(.13182733) * 1j,
            horizontal_diameter=np.float128(.00012068),
            dims=(width, height)) \
            / MandelbrotScoreGenerator(iteration_count=2**10) \
            / ColorizeGenerator(provider_updater) \
            / SaveAsImageGenerator(filename=output_path),
        'anthena': ComplexCanvasGenerator(
            center=np.float128(-.7436447860) + np.float128(.1318252536) * 1j,
            horizontal_diameter=np.float128(.0000029336),
            dims=(width, height)) \
            / MandelbrotScoreGenerator(iteration_count=2**11) \
            / ColorizeGenerator(provider_updater) \
            / SaveAsImageGenerator(filename=output_path),
        'halo': ComplexCanvasGenerator(
            center=np.float128(-.5503493176297569) + np.float128(.6259309572825709 ) * 1j,
            horizontal_diameter=np.float128(.00000000000054),
            dims=(width, height)) \
            / MandelbrotScoreGenerator(iteration_count=2**14) \
            / ColorizeGenerator(provider_updater) \
            / SaveAsImageGenerator(filename=output_path),
        'graph': ComplexCanvasGenerator(
            center=0,
            horizontal_diameter=5,
            dims=(width, height)) \
            / FunctionGraphGenerator(lambda x: np.sin(x),
                                     radius=30)
            / ColorizeGenerator(provider_updater) \
            / SaveAsImageGenerator(filename=output_path),
        'map': ComplexCanvasGenerator(
            center=0,
            horizontal_diameter=5,
            dims=(width, height)) \
            / ComplexToRealMapGenerator(lambda x: 1 - 3 * np.arctan(np.abs(x)) / np.pi)
            / ColorizeGenerator(provider_updater) \
            / SaveAsImageGenerator(filename=output_path),
    }
    if not mode in modes:
        print(f'available modes: {", ".join(modes.keys())}')
        exit(1)

    modes[mode].generate()
