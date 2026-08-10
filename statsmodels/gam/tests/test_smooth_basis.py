"""
unit test for spline and other smoother classes

Author: Luca Puggini

"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

from statsmodels.gam.smooth_basis import (
    BSplines,
    CubicSplines,
    PolynomialSmoother,
    UnivariatePolynomialSmoother,
)


def test_univariate_polynomial_smoother():
    x = np.linspace(0, 1, 5)
    pol = UnivariatePolynomialSmoother(x, degree=3)
    assert_equal(pol.basis.shape, (5, 3))
    assert_allclose(pol.basis[:, 2], x.ravel() ** 3)


def test_multivariate_polynomial_basis():
    rs = np.random.RandomState(1)
    x = rs.normal(0, 1, (10, 2))
    degrees = [3, 4]
    mps = PolynomialSmoother(x, degrees)
    for i, deg in enumerate(degrees):
        uv_basis = UnivariatePolynomialSmoother(x[:, i], degree=deg).basis
        assert_allclose(mps.smoothers[i].basis, uv_basis)


@pytest.mark.parametrize(
    "x, df, degree",
    [
        (np.c_[np.linspace(0, 1, 100), np.linspace(0, 10, 100)], [5, 6], [3, 5]),
        (np.linspace(0, 1, 100), 6, 3),
    ],
)
def test_bsplines(x, df, degree):
    bspline = BSplines(x, df, degree)
    bspline.transform(x)


def test_cubic_splines_transform_not_shadowed():
    # GH: CubicSplines.__init__ stored its `transform` constructor argument
    # (a string like "domain") as `self.transform`, shadowing the inherited
    # AdditiveGamSmoother.transform(x_new) method needed for out-of-sample
    # basis construction (e.g., GAM prediction on new data). Before the fix,
    # `cs.transform` was the literal string "domain", not callable.
    x = np.linspace(0, 1, 50)
    cs = CubicSplines(x, df=[5], transform="domain")
    assert cs.transform_arg == "domain"
    assert callable(cs.transform)
