
import numpy as np
from scipy import linalg
from numpy.testing import assert_allclose, assert_equal

from statsmodels import robust
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

    # GK, OGK
    robcov.cov_gk1(x[:, 0], x[:,2])
    robcov.cov_gk(x)
    robcov.cov_ogk(x)

    # Tyler
    robcov.cov_tyler(x)
    robcov.cov_tyler_regularized(x, shrinkage_factor=0.1)

    x2 = np.array([x, x])
    x2_ = np.rollaxis(x2, 1)
    robcov.cov_tyler_pairs_regularized(x2_,
                                start_cov=np.diag(robust.mad(x)**2),
                                shrinkage_factor=0.1,
                                nobs=x.shape[0], k_vars=x.shape[1])

    # others, M-, ...

    # estimation for multivariate t
    r = robcov._cov_iter(x, robcov.weights_mvt, weights_args=(3, k_vars))
    # trimmed sample covariance
    r = robcov._cov_iter(x, robcov.weights_quantile, weights_args=(0.50, ),
                         rescale=True)
