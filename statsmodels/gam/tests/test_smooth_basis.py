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
    get_knots_bsplines,
)


def test_get_knots_bsplines_spacing():
    rs = np.random.RandomState(0)
    x = rs.uniform(size=100)

    # df=8, degree=3 -> n_inner_knots=4, avoiding the pre-existing,
    # unrelated IndexError that spacing="equal" hits when there is only
    # 1 inner knot (e.g. df=5, degree=3)
    knots_q = get_knots_bsplines(x, df=8, degree=3, spacing="quantile")
    knots_e = get_knots_bsplines(x, df=8, degree=3, spacing="equal")
    assert knots_q.ndim == 1
    assert knots_e.ndim == 1

    with pytest.raises(ValueError, match="spacing"):
        get_knots_bsplines(x, df=8, degree=3, spacing="not-a-spacing")

    # previously silent: an invalid `spacing` was never checked at all when
    # `knots` was given directly instead of `df`, so this used to succeed
    # (identically to spacing="quantile") instead of raising.
    with pytest.raises(ValueError, match="spacing"):
        get_knots_bsplines(x, knots=[0.25, 0.5, 0.75], degree=3, spacing="bogus")


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
