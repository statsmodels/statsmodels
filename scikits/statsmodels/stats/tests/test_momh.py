# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 17:33:56 2011

@author: josef
"""
from scikits.statsmodels.stats.moment_helpers import cov2corr

import numpy as np
from numpy.testing import assert_almost_equal, assert_, assert_equal

def test_cov2corr():
    cov_a = np.ones((3,3))+np.diag(np.arange(1,4)**2 - 1)
    corr_a = np.array([[1, 1/2., 1/3.],[1/2., 1, 1/2./3.],[1/3., 1/2./3., 1]])

    corr = cov2corr(cov_a)
    assert_almost_equal(corr, corr_a, decimal=15)

    cov_mat = np.matrix(cov_a)
    corr_mat = cov2corr(cov_mat)
    assert_(isinstance(corr_mat, np.matrixlib.defmatrix.matrix))
    assert_equal(corr_mat, corr)

    cov_ma = np.ma.array(cov_a)
    corr_ma = cov2corr(cov_ma)
    assert_equal(corr_mat, corr)

    assert_(isinstance(corr_ma, np.ma.core.MaskedArray))

    cov_ma2 = np.ma.array(cov_a, mask = [[False, True, False],
                                         [True, False, False],
                                         [False, False, False]])

    corr_ma2 = cov2corr(cov_ma2)
    assert_(np.ma.allclose(corr_ma, corr, atol=1e-15))
    assert_equal(corr_ma2.mask, cov_ma2.mask)

if __name__ == '__main__':
    test_cov2corr()
