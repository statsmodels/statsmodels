"""
Created on May 12, 2025 5:05:16 p.m.

Author: Josef Perktold
License: BSD-3
"""

import os.path
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from statsmodels.multivariate._normality_mv import (
    normal_dh
    )


cur_dir = os.path.dirname(os.path.abspath(__file__))
iris_dir = os.path.join(cur_dir, '..', '..', 'genmod', 'tests', 'results')
iris_dir = os.path.abspath(iris_dir)
iris = pd.read_csv(os.path.join(iris_dir, 'iris.csv'), delimiter=",",
                     skip_header=1)

# recover original iris data from changed data set in iris.csv
idx = [34, 37]
diff = np.array([
    [ 0. ,  0. ,  0. ,  0.1, 0],
    [ 0. ,  0.5, -0.1,  0., 0]
    ])
iris[idx] -= diff


def test_normality_doornik_hansen():
    res_dh = normal_dh(iris)
    result_dh = 24.4145
    assert_allclose(res_dh[0], result_dh, atol= 5.e-4)


def test_normality_mardia():

    pass
