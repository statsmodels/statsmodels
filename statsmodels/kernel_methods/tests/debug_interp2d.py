from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.interpolate import interp2d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


def interpolate2d(ax1, ax2, values, test_values):
    interp = interp2d(ax1, ax2, values)
    res = np.empty_like(test_values[:, 0])
    for i in range(test_values.shape[0]):
        res[i] = interp(test_values[i, 0], test_values[i, 1])
    return res


def draw_triangulation(xs, ys, values):
    trs = Triangulation(20 * xs, ys)
    trs.x = xs
    plt.tripcolor(trs, values, shading='gouraud')


def run():
    ax1 = np.r_[0:2 * np.pi:124j]
    ax2 = np.r_[-90:90:257j]
    grid = np.meshgrid(ax1, ax2, indexing='ij')

    N = 4096
    test_values = np.c_[np.random.rand(N) * 2 * np.pi,
                        np.random.rand(N) * 180 - 90]

    grid_vals = np.cos(grid[0]) + np.sin(grid[1] * np.pi / 180)
    vals = interpolate2d(ax1, ax2, grid_vals.T, test_values)

    plt.figure()
    draw_triangulation(test_values[:, 0], test_values[:, 1], vals)
    plt.title('scipy interp2d')

    interp = interp2d(ax1, ax2, grid_vals.T)
    xs = np.r_[0:2 * np.pi:100j]
    ys = np.r_[-90:90:200j]
    gr_vals = interp(xs, ys)
    gr = np.meshgrid(xs, ys)

    plt.figure()
    draw_triangulation(gr[0].flatten(), gr[1].flatten(), gr_vals.flatten())
    plt.title('Interpolated grid')

    plt.figure()
    draw_triangulation(grid[0].flatten(), grid[1].flatten(),
                       grid_vals.flatten())
    plt.title('Original Grid')


if __name__ == '__main__':
    run()
    plt.show()
