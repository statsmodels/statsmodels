# -*- coding: utf-8 -*-
"""
Created on Mon May  9 17:58:43 2016

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from numpy.testing import assert_equal, assert_allclose

from statsmodels.stats._multinomial import (factmom_truncpoisson,
                                            chisquare_multinomial)


def _simulate_truncpoisson(rate, low, upp, n_rep=50000):

    rvs = np.random.poisson(rate, size=(n_rep, len(rate))).astype(float)
    mask = ((low <= rvs) & (rvs <= upp)) #.all(1)

    count = mask.sum(0)
    m1, m2, m3, m4 = [(mask * rvs**k).sum(0) / count for k in range(1, 5)]
    return m1, m2, m3, m4


def test_factmom_truncpoisson():
    # tests only the special case of poisson without truncation

    n_moms, rate, low, upp = 4, np.array([5., 10, 20]), 0, 1000
    k = np.arange(1, n_moms + 1)[:, None]

    fm2 = rate**k
    fm1 = factmom_truncpoisson(n_moms, rate, low, upp)[0]
    assert_equal(fm1, fm2)

    rate *= 1.3
    fm2 = np.array(rate)**k
    fm1 = factmom_truncpoisson(n_moms, rate, low, upp)[0]
    assert_equal(fm1, fm2)


def test_factmom_truncpoisson_mc():
    # this compares moments with simulation results
    # we keep n_rep=10000 in the test, precision is higher with n_rep=1000000
    np.random.seed(654321)
    low = [1, 2, 3, 30]
    upp = [7, 15, 25, 60.]
    rate = np.array([5., 10, 20, 50.])
    m = _simulate_truncpoisson(rate, low, upp, n_rep=10000) #1000000
    #fm1 = factmom_truncpoisson(4, rate, low, upp)[0]
    fm1 = factmom_truncpoisson(4, rate, np.array(low), np.array(upp))[0]

    from statsmodels.stats._multinomial import mfc2mnc
    mm = mfc2mnc(fm1)

    assert_allclose(np.array(m), np.array(mm), rtol=1e-2)
    # rtol=5e-3 for n_rep=1000000


def test_multinomial_pvalue():
    np.random.seed(12345)
    res1 = chisquare_multinomial([6,2,1,5,1,0]) #, kwds=dict(n_rep=500000, batch=50000))
    res2 = (11.8, 0.041514, 0.03763342488733476) # pvalue of MC
    # Cai and Krishnamoorthy have exact pvalue=0.042 at print precision
    assert_allclose(res1[0], res2[0], rtol=1e-12)  # this just checks scipy result
    assert_allclose(res1[2], res2[2], rtol=1e-12)  # this just checks scipy result
    assert_allclose(res1[1], 0.042, atol=0.005)
