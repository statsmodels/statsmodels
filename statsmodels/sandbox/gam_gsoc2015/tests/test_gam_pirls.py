
from __future__ import division

__author__ = 'Luca Puggini: <lucapuggio@gmail.com>'
__date__ = '18/07/15'


import os
import pandas as pd
from numpy.testing import assert_allclose
import numpy as np
from statsmodels.sandbox.gam_gsoc2015.smooth_basis import CubicSplines, UnivariateCubicSplines
from statsmodels.sandbox.gam_gsoc2015.gam import GLMGam, get_sqrt


def test_splines_x():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results", "gam_PIRLS_results.csv")
    data = pd.read_csv(file_path)

    #x = data['x'].as_matrix()
    #y = data['y'].as_matrix()
    x = np.asarray(data['x'])
    y = np.asarray(data['y'])
    xk = np.array([0.2, .4, .6, .8])

    spl_x_R = data[['spl_x.1', 'spl_x.2', 'spl_x.3', 'spl_x.4', 'spl_x.5', 'spl_x.6']].as_matrix()

    cs = UnivariateCubicSplines(x, 4)
    cs.knots = xk
    cs._splines_x()
    spl_x = cs.basis_

    print(spl_x_R.shape, spl_x.shape)
    print(np.max(np.abs(spl_x - spl_x_R)))
    assert_allclose(spl_x_R, spl_x, atol=0.0001)


def test_spl_s():

    # matrix from R
    spl_s_R = [[0,    0,  0.000000000,  0.000000000,  0.000000000,  0.000000000],
               [0,    0,  0.000000000,  0.000000000,  0.000000000,  0.000000000],
               [0,    0,  0.001400000,  0.000200000, -0.001133333, -0.001000000],
               [0,    0,  0.000200000,  0.002733333,  0.001666667, -0.001133333],
               [0,    0, -0.001133333,  0.001666667,  0.002733333,  0.000200000],
               [0,    0, -0.001000000, -0.001133333,  0.000200000,  0.001400000]]

    x = np.random.normal(0, 1, 10)
    xk = np.array([0.2, .4, .6, .8])
    cs = UnivariateCubicSplines(x, df=4)
    cs.knots = xk

    spl_s = cs._splines_s()
    assert_allclose(spl_s_R, spl_s, atol=4.e-10)


test_spl_s()
test_splines_x()
