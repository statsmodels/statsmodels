# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:33:33 2017

Author: Josef Perktold

"""
from __future__ import division

import numpy as np
from numpy.testing import assert_allclose, assert_equal

import statsmodels.nonparametric.binned_regression as smbr

def test_binned_regression_ex1():
    #  check that it works and basic properties
    nobs = 1000
    sig_e = 0.1
    np.random.seed(123987)
    x = np.sort(np.random.rand(nobs))
    y_true = x + np.sin(0.75 * 2 * np.pi * x**2)
    y = y_true + sig_e * np.random.randn(nobs)

    binn = smbr.Binner(x, n_bins=300, xmin=0, xmax=1)
    xb = binn.bin_center
    yb = binn.bin_data(y)
    w = binn.bin_data()
    local_poly = smbr.BinnedLocalPolynomialProjector(xb, window_length=9, weights=w)

    # test that we have a good fit
    res = local_poly.project(yb)
    ssr = ((yb - res.fittedvalues * w)**2).sum()
    scale = ssr / nobs
    assert_allclose(scale, sig_e**2, rtol=0.25)
    # local constant kernel regression
    ssr0 = ((yb - res.fitted_locpoly0 * w)**2).sum()
    scale0 = ssr0 / nobs
    assert_allclose(scale0, sig_e**2, rtol=0.2)

    # test properties of binner and local poly
    assert_equal(local_poly.bwi, 9)
    assert_allclose(w.sum(), nobs, rtol=1e-13)
    assert_equal(binn.n_bins, 300)   # we have actually 301 grid points
    assert_equal(yb.shape, (301, ))
    assert_equal(xb.shape, (301, ))
    assert_allclose(yb.sum(), y.sum(), rtol=1e-13)

    res2 = smbr.fit_loclin_cvbw(y, x, weights=1., n_bins=300)
    assert_equal(res2.projector.bwi, 45)
    ssr = ((yb - res2.fittedvalues * w)**2).sum()
    scale = ssr / nobs
    assert_allclose(scale, sig_e**2, rtol=0.1)
