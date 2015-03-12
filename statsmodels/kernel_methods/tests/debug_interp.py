from statsmodels.kernel_methods._grid_interpolation import GridInterpolator
import statsmodels.api as sm
from scipy.interpolate import interp2d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

def interpolate2d(grid, values, test_values):
    interp = interp2d(grid.grid[0], grid.grid[1], values.T)
    res = np.empty_like(test_values[:, 0])
    for i in range(test_values.shape[0]):
        res[i] = interp(test_values[i, 0], test_values[i, 1])
    return res

def draw_triangulation(xs, ys, values):
    trs = Triangulation(20 * xs, ys)
    trs.x = xs
    plt.tripcolor(trs, values, shading='gouraud')
    plt.colorbar()

def run():
    N = 4096
    test_values = np.c_[np.random.rand(N) * 3 * np.pi - np.pi / 2,
                        np.random.rand(N) * 200 - 100]
    ax1 = np.r_[0:2 * np.pi:124j]
    ax1 = (ax1 + (ax1[1] - ax1[0]) / 2)[:-1]
    ax2 = np.r_[-90:90:257j]
    ax2 = (ax2 + (ax2[1] - ax2[0]) / 2)[:-1]
    grid2 = sm.kernel_methods.kde_utils.Grid([ax1, ax2])
    min_val0 = grid2.grid[0][0]
    max_val0 = grid2.grid[0][-1]
    min_val1 = grid2.grid[1][0]
    max_val1 = grid2.grid[1][-1]
    real_values = test_values.copy()
    real_values[real_values[:, 0] < min_val0, 0] = min_val0
    real_values[real_values[:, 0] > max_val0, 0] = max_val0
    real_values[real_values[:, 1] < min_val1, 1] = min_val1
    real_values[real_values[:, 1] > max_val1, 1] = max_val1
    sg = grid2.sparse()
    grid_val = np.cos(sg[0]) + np.sin(sg[1] * np.pi / 180)
    interp = GridInterpolator(grid2, grid_val)
    vals = interpolate2d(grid2, grid_val, real_values)
    gvals = interp(test_values)

    plt.figure()
    draw_triangulation(test_values[:, 0], test_values[:, 1], vals)
    plt.title('scipy interp2d')

    plt.figure()
    draw_triangulation(test_values[:, 0], test_values[:, 1], gvals)
    plt.title('statsmodels GridInterpolator')

    plt.figure()
    draw_triangulation(test_values[:, 0], test_values[:, 1], abs(gvals - vals))
    plt.title('Difference')

    fg = grid2.full('C')
    trs2 = Triangulation(fg[0].flatten() * 20, fg[1].flatten())
    trs2.x = fg[0].flatten()
    plt.figure()
    draw_triangulation(fg[0].flatten(), fg[1].flatten(), grid_val.flatten())
    plt.tripcolor(trs2, grid_val.flatten(), shading='gouraud')
    plt.title('Original Grid')

if __name__ == '__main__':
    run()
    plt.show()
