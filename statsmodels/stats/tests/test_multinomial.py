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

def test_factmom_truncpoisson():
    # tests only the special case of poisson without truncation

    n_moms, rate, low, upp = 4, np.array([5., 10, 20]), -1, 1000
    k = np.arange(1, n_moms + 1)[:, None]

    fm2 = rate**k
    fm1 = factmom_truncpoisson(n_moms, rate, low, upp)[0]
    assert_equal(fm1, fm2)

    rate *= 1.3
    fm2 = np.array(rate)**k
    fm1 = factmom_truncpoisson(n_moms, rate, low, upp)[0]
    assert_equal(fm1, fm2)


def test_multinomial_pvalue():
    np.random.seed(12345)
    res1 = chisquare_multinomial([6,2,1,5,1,0]) #, kwds=dict(n_rep=500000, batch=50000))
    res2 = (11.8, 0.041514, 0.03763342488733476) # pvalue of MC
    # Cai and Krishnamoorthy have exact pvalue=0.042 at print precision
    assert_allclose(res1[0], res2[0], rtol=1e-12)  # this just checks scipy result
    assert_allclose(res1[2], res2[2], rtol=1e-12)  # this just checks scipy result
    assert_allclose(res1[1], 0.042, atol=0.005)
