#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.axes import Axes
import sys

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

def mandelbrot_score(c: np.ndarray, z: np.ndarray = np.array([0 + 0j]), count = 100):
    if z.shape != c.shape:
        z = np.zeros_like(c)

    results = np.zeros_like(c, dtype=np.int16)
    to_compute = results == 0
    for i in range(count):
        print(f'{round(100 * i / count, 2)}% completed')
        z[to_compute] = np.power(z[to_compute], 2) + c[to_compute]
        to_compute &= np.abs(z) < 2
        results[to_compute] += 1

    results[to_compute] = -1
    return results / count

def mandelbrot_compile(height=100, width=100,
                       xmin: np.float128 = np.float128(-1.5),
                       xmax: np.float128 = np.float128(0.5),
                       ymin: np.float128 = np.float128(-1),
                       ymax: np.float128 = np.float128(1),
                       iteration_count = 100):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    xx, yy = np.meshgrid(x, y)
    zz = xx + 1j * yy
    m = mandelbrot_score(zz, count=iteration_count)
    return m

def mandelbrot_image(x_center: np.float128,
                     y_center: np.float128,
                     relative_width: np.float128,
                     width: int,
                     height: int,
                     iteration_count = 100,
                     filename = 'a.png'):
    m = mandelbrot_compile(width = width, height = height,
                           xmin=x_center - relative_width / 2,
                           xmax=x_center + relative_width / 2,
                           ymin=y_center + relative_width * height / width / 2,
                           ymax=y_center - relative_width * height / width / 2,
                           iteration_count=iteration_count)
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
    plt.savefig(filename, bbox_inches="tight", pad_inches=0, dpi=1)

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
                             filename='default.png')
        case 'tails':
            # seahorse tails
            x_center = np.float128(-.74364085)
            y_center = np.float128(.13182733)
            relative_width = np.float128(.00012068)
            mandelbrot_image(x_center, y_center, relative_width, width, height,
                             iteration_count=2**10,
                             filename='tail.png')
        case 'anthena':
            x_center = np.float128(-.7436447860)
            y_center = np.float128(.1318252536)
            relative_width = np.float128(.0000029336)
            mandelbrot_image(x_center, y_center, relative_width, width, height,
                             iteration_count=2**11,
                             filename='anthena.png')
        case _:
            print('invalid selection')
            exit(1)
