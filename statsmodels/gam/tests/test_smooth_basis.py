__author__ = 'Luca Puggini: <lucapuggio@gmail.com>'

import numpy as np
from patsy.state import stateful_transform
from statsmodels.gam.smooth_basis import (make_bsplines_basis, BS, UnivariatePolynomialSmoother,
                                          PolynomialSmoother)
from numpy.testing import assert_allclose


def test_make_basis():
    bs = stateful_transform(BS)
    df = 10
    degree = 4
    x = np.logspace(-1, 1, 100)
    result = bs(x, df=df, degree=degree, include_intercept=True)
    basis, der1, der2 = result
    basis_old, der1_old, der2_old = make_bsplines_basis(x, df, degree)
    assert ((basis == basis_old).all())
    assert ((der1 == der1_old).all())
    assert ((der2 == der2_old).all())
    return


def test_univariate_polynomial_smoother():
    x = np.linspace(0, 1, 5)
    # test_univariate_polynomial_smoother()
    # test_make_basis()
    pol = UnivariatePolynomialSmoother(x, degree=3)
    assert pol.basis_.shape == (5, 3)
    assert_allclose(pol.basis_[:, 2], x.ravel() ** 3)


def test_multivariate_polynomial_basis():
    x = np.random.normal(0, 1, (10, 2))
    degrees = [3, 4]
    mps = PolynomialSmoother(x, degrees)
    for i, deg in enumerate(degrees):
        assert_allclose(mps.smoothers_[i].basis_, UnivariatePolynomialSmoother(x[:, i], degree=deg).basis_)
