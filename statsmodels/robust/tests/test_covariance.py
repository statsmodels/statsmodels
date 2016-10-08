
import numpy as np
from scipy import linalg
from numpy.testing import assert_allclose, assert_equal

import statsmodels.robust.covariance as robcov


def test_robcov_SMOKE():
    # currently only smoke test
    nobs, k_vars = 100, 3

    mean = np.zeros(k_vars)
    cov = linalg.toeplitz(1. / np.arange(1, k_vars+1))

    np.random.seed(187649)
    x = np.random.multivariate_normal(mean, cov, size=nobs)
    x[0,:2] = 50

    xtx = x.T.dot(x)
    cov_emp = np.cov(x.T)


    robcov.cov_gk1(x[:, 0], x[:,2])
    robcov.cov_gk(x)
    robcov.cov_ogk(x)
