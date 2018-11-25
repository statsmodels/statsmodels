# -*- coding: utf-8 -*-
"""
unit test for spline and other smoother classes

Author: Luca Puggini

"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.gam.smooth_basis import (UnivariatePolynomialSmoother,
                                          PolynomialSmoother)


def test_univariate_polynomial_smoother():
    x = np.linspace(0, 1, 5)
    # test_univariate_polynomial_smoother()
    # test_make_basis()
    pol = UnivariatePolynomialSmoother(x, degree=3)
    assert_equal(pol.basis_.shape, (5, 3))
    assert_allclose(pol.basis_[:, 2], x.ravel() ** 3)


def test_multivariate_polynomial_basis():
    x = np.random.normal(0, 1, (10, 2))
    degrees = [3, 4]
    mps = PolynomialSmoother(x, degrees)
    for i, deg in enumerate(degrees):
        uv_basis = UnivariatePolynomialSmoother(x[:, i], degree=deg).basis_
        assert_allclose(mps.smoothers_[i].basis_, uv_basis)
