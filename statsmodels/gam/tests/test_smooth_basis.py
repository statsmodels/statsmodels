__author__ = 'Luca Puggini: <lucapuggio@gmail.com>'

import numpy as np
from patsy.state import stateful_transform
from statsmodels.gam.smooth_basis import (make_bsplines_basis, BS,
                                          UnivariatePolynomialSmoother,
                                          PolynomialSmoother)
from numpy.testing import assert_allclose, assert_equal


def test_make_basis():
    bs = stateful_transform(BS)
    df = 10
    degree = 4
    x = np.logspace(-1, 1, 100)
    result = bs(x, df=df, degree=degree, include_intercept=True)
    basis, der1, der2 = result
    basis_old, der1_old, der2_old = make_bsplines_basis(x, df, degree)
    assert_equal(basis, basis_old)
    assert_equal(der1, der1_old)
    assert_equal(der2, der2_old)


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
