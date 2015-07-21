__author__ = 'Luca Puggini: <lucapuggio@gmail.com>'
__date__ = '18/07/15'


from statsmodels.sandbox.gam_gsoc2015.pirls import GamPirls, splines_x, splines_s, get_sqrt
import os
import pandas as pd
from numpy.testing import assert_allclose
import numpy as np
from statsmodels.sandbox.gam_gsoc2015.smooth_basis import CubicSplines

def test_splines_x():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results", "gam_PIRLS_results.csv")
    data = pd.read_csv(file_path)

    x = data['x'].as_matrix()
    y = data['y'].as_matrix()
    xk = np.array([0.2, .4, .6, .8])

    spl_x_R = data[['spl_x.1', 'spl_x.2', 'spl_x.3', 'spl_x.4', 'spl_x.5', 'spl_x.6']].as_matrix()

    cs = CubicSplines(x, 4)
    cs.knots = xk
    cs._splines_x()
    spl_x = cs.xs

    assert_allclose(spl_x_R, spl_x)


def test_spl_s():

    # matrix from R
    spl_s_R = [[0,    0,  0.000000000,  0.000000000,  0.000000000,  0.000000000],
               [0,    0,  0.000000000,  0.000000000,  0.000000000,  0.000000000],
               [0,    0,  0.001400000,  0.000200000, -0.001133333, -0.001000000],
               [0,    0,  0.000200000,  0.002733333,  0.001666667, -0.001133333],
               [0,    0, -0.001133333,  0.001666667,  0.002733333,  0.000200000],
               [0,    0, -0.001000000, -0.001133333,  0.000200000,  0.001400000]]

    xk = np.array([0.2, .4, .6, .8])
    cs = CubicSplines(None, 4)
    cs.knots = xk
    cs._splines_s()

    spl_s = cs.s
    assert_allclose(spl_s_R, spl_s, atol=4.e-10)


def test_gest_sqrt():

    xk = np.array([0.2, .4, .6, .8])
    cs = CubicSplines(None, 4)
    cs.knots = xk
    cs._splines_s()
    spl_s = cs.s

    b = get_sqrt(spl_s)
    assert_allclose(np.dot(b.T, b), spl_s)



#test_gest_sqrt()
#test_spl_s()
#test_splines_x()
