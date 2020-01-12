# -*- coding: utf-8 -*-
"""
Created on Tue Nov 08 22:28:48 2011

@author: josef
"""
import pytest

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from statsmodels.tools.eval_measures import (
    maxabs, meanabs, medianabs, medianbias, mse, rmse, vare,
    aic, aic_sigma, aicc, aicc_sigma, bias, bic, bic_sigma,
    hqic, hqic_sigma, iqr)


def test_eval_measures():
    #mainly regression tests

    x = np.arange(20).reshape(4,5)
    y = np.ones((4,5))
    assert_equal(iqr(x, y), 5*np.ones(5))
    assert_equal(iqr(x, y, axis=1), 2*np.ones(4))
    assert_equal(iqr(x, y, axis=None), 9)

    assert_equal(mse(x, y),
                 np.array([  73.5,   87.5,  103.5,  121.5,  141.5]))
    assert_equal(mse(x, y, axis=1),
                 np.array([   3.,   38.,  123.,  258.]))

    assert_almost_equal(rmse(x, y),
                        np.array([  8.5732141 ,   9.35414347,  10.17349497,
                                   11.02270384,  11.89537725]))
    assert_almost_equal(rmse(x, y, axis=1),
                        np.array([  1.73205081,   6.164414,
                                   11.09053651,  16.0623784 ]))

    assert_equal(maxabs(x, y),
                 np.array([ 14.,  15.,  16.,  17.,  18.]))
    assert_equal(maxabs(x, y, axis=1),
                 np.array([  3.,   8.,  13.,  18.]))

    assert_equal(meanabs(x, y),
                 np.array([  7. ,   7.5,   8.5,   9.5,  10.5]))
    assert_equal(meanabs(x, y, axis=1),
                 np.array([  1.4,   6. ,  11. ,  16. ]))
    assert_equal(meanabs(x, y, axis=0),
                 np.array([  7. ,   7.5,   8.5,   9.5,  10.5]))

    assert_equal(medianabs(x, y),
                 np.array([  6.5,   7.5,   8.5,   9.5,  10.5]))
    assert_equal(medianabs(x, y, axis=1),
                 np.array([  1.,   6.,  11.,  16.]))

    assert_equal(bias(x, y),
                 np.array([  6.5,   7.5,   8.5,   9.5,  10.5]))
    assert_equal(bias(x, y, axis=1),
                 np.array([  1.,   6.,  11.,  16.]))

    assert_equal(medianbias(x, y),
                 np.array([  6.5,   7.5,   8.5,   9.5,  10.5]))
    assert_equal(medianbias(x, y, axis=1),
                 np.array([  1.,   6.,  11.,  16.]))

    assert_equal(vare(x, y),
                 np.array([ 31.25,  31.25,  31.25,  31.25,  31.25]))
    assert_equal(vare(x, y, axis=1),
                 np.array([ 2.,  2.,  2.,  2.]))


ics = [aic, aicc, bic, hqic]
ics_sig = [aic_sigma, aicc_sigma, bic_sigma, hqic_sigma]


@pytest.mark.parametrize('ic,ic_sig', zip(ics, ics_sig))
def test_ic_equivalence(ic, ic_sig):
    # consistency check

    assert ic(np.array(2), 10, 2).dtype == np.float
    assert ic_sig(np.array(2), 10, 2).dtype == np.float

    assert_almost_equal(ic(-10./2.*np.log(2.), 10, 2)/10,
                        ic_sig(2, 10, 2),
                        decimal=14)

    assert_almost_equal(ic_sig(np.log(2.), 10, 2, islog=True),
                        ic_sig(2, 10, 2),
                        decimal=14)


def test_ic():
    # test information criteria

    # examples penalty directly from formula
    n = 10
    k = 2
    assert_almost_equal(aic(0, 10, 2), 2*k, decimal=14)
    # next see Wikipedia
    assert_almost_equal(aicc(0, 10, 2),
                        aic(0, n, k) + 2*k*(k+1.)/(n-k-1.), decimal=14)
    assert_almost_equal(bic(0, 10, 2), np.log(n)*k, decimal=14)
    assert_almost_equal(hqic(0, 10, 2), 2*np.log(np.log(n))*k, decimal=14)


def test_iqr_axis(reset_randomstate):
    x1 = np.random.standard_normal((100, 100))
    x2 = np.random.standard_normal((100, 100))
    ax_none = iqr(x1, x2, axis=None)
    ax_none_direct = iqr(x1.ravel(), x2.ravel())
    assert_equal(ax_none, ax_none_direct)

    ax_0 = iqr(x1, x2, axis=0)
    assert ax_0.shape == (100,)
    ax_0_direct = [iqr(x1[:,i], x2[:,i]) for i in range(100)]
    assert_almost_equal(ax_0, np.array(ax_0_direct))

    ax_1 = iqr(x1, x2, axis=1)
    assert ax_1.shape == (100,)
    ax_1_direct = [iqr(x1[i,:], x2[i,:]) for i in range(100)]
    assert_almost_equal(ax_1, np.array(ax_1_direct))

    assert any(ax_0 != ax_1)
