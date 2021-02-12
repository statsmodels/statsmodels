# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:42:00 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from numpy.testing import assert_allclose
from scipy import stats
import statsmodels.distributions.tools as dt


def test_grid():
    # test bivariate independent beta
    k1, k2 = 3, 5
    xg1 = np.arange(k1) / (k1 - 1)
    xg2 = np.arange(k2) / (k2 - 1)

    # histogram values for distribution
    distr1 = stats.beta(2, 5)
    distr2 = stats.beta(4, 3)
    cdf1 = distr1.cdf(xg1)
    cdf2 = distr2.cdf(xg2)
    prob1 = np.diff(cdf1, prepend=0)
    prob2 = np.diff(cdf2, prepend=0)
    cd2d = cdf1[:, None] * cdf2
    pd2d = prob1[:, None] * prob2

    probs = dt.cdf2prob_grid(cd2d)
    cdfs = dt.prob2cdf_grid(pd2d)

    assert_allclose(cdfs, cd2d, atol=1e-12)
    assert_allclose(probs, pd2d, atol=1e-12)

    # check random sample
    nobs = 1000
    np.random.seed(789123)
    rvs = np.column_stack([distr1.rvs(size=nobs), distr2.rvs(size=nobs)])
    hist = np.histogramdd(rvs, [xg1, xg2])
    assert_allclose(probs[1:, 1:], hist[0] / len(rvs), atol=0.02)


def test_bernstein_1d():
    k = 5
    xg1 = np.arange(k) / (k - 1)
    xg2 = np.arange(2 * k) / (2 * k - 1)
    # verify linear coefficients are mapped to themselves
    res_bp = dt._eval_bernstein_1d(xg2, xg1)
    assert_allclose(res_bp, xg2, atol=1e-12)

    res_bp = dt._eval_bernstein_1d(xg2, xg1, method="beta")
    assert_allclose(res_bp, xg2, atol=1e-12)

    res_bp = dt._eval_bernstein_1d(xg2, xg1, method="bpoly")
    assert_allclose(res_bp, xg2, atol=1e-12)


def test_bernstein_2d():
    k = 5
    xg1 = np.arange(k) / (k - 1)
    cd2d = xg1[:, None] * xg1
    # verify linear coefficients are mapped to themselves
    for evalbp in (dt._eval_bernstein_2d, dt._eval_bernstein_dd):
        k_x = 2 * k
        # create flattened grid of bivariate values
        x2d = np.column_stack(
                np.unravel_index(np.arange(k_x * k_x), (k_x, k_x))
                ).astype(float)
        x2d /= x2d.max(0)

        res_bp = evalbp(x2d, cd2d)
        assert_allclose(res_bp, np.product(x2d, axis=1), atol=1e-12)

        # check univariate margins
        x2d = np.column_stack((np.arange(k_x) / (k_x - 1), np.ones(k_x)))
        res_bp = evalbp(x2d, cd2d)
        assert_allclose(res_bp, x2d[:, 0], atol=1e-12)

        # check univariate margins
        x2d = np.column_stack((np.ones(k_x), np.arange(k_x) / (k_x - 1)))
        res_bp = evalbp(x2d, cd2d)
        assert_allclose(res_bp, x2d[:, 1], atol=1e-12)
