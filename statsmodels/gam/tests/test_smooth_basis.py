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

    # df=8, degree=3 -> n_inner_knots=4
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


@pytest.mark.parametrize(
    "df,degree,n_inner_knots",
    [
        (4, 3, 0),  # cubic spline, no inner knots at all
        (5, 3, 1),  # cubic spline, exactly one inner knot
        (3, 2, 0),  # quadratic spline, no inner knots
        (4, 2, 1),  # quadratic spline, exactly one inner knot
    ],
)
def test_get_knots_bsplines_spacing_equal_few_inner_knots(df, degree, n_inner_knots):
    # GH: spacing="equal" raised IndexError ("index 1 is out of bounds")
    # whenever n_inner_knots = df - (degree + 1) was 0 or 1, because
    # diff_knots was computed as inner_knots[1] - inner_knots[0], which
    # needs at least 2 inner knots to exist.
    rs = np.random.RandomState(0)
    x = rs.uniform(size=100)
    x_min, x_max = x.min(), x.max()
    order = degree + 1

    all_knots = get_knots_bsplines(x, df=df, degree=degree, spacing="equal")

    assert all_knots.ndim == 1
    assert len(all_knots) == n_inner_knots + 2 * order
    assert np.all(np.diff(all_knots) > 0)

    # Independent check, not just "it didn't raise": spacing="equal" means
    # every gap between consecutive knots equals the width of one of the
    # n_inner_knots + 1 equal segments spanning [x_min, x_max].
    expected_step = (x_max - x_min) / (n_inner_knots + 1)
    assert_allclose(np.diff(all_knots), expected_step)

    expected_inner = x_min + expected_step * np.arange(1, n_inner_knots + 1)
    assert_allclose(all_knots[order:order + n_inner_knots], expected_inner)


def test_get_knots_bsplines_spacing_equal_unchanged_for_multiple_inner_knots():
    # regression check for the fix above: n_inner_knots >= 2 must still
    # match the pre-fix inner_knots[1] - inner_knots[0] formula, since that
    # difference is well-defined there and was never broken.
    rs = np.random.RandomState(0)
    x = rs.uniform(size=100)
    x_min, x_max = x.min(), x.max()

    for df, degree in [(6, 3), (12, 3), (9, 2)]:
        order = degree + 1
        n_inner_knots = df - order
        grid = np.linspace(0, 1, n_inner_knots + 2)[1:-1]
        inner_knots = x_min + grid * (x_max - x_min)
        diff_knots = inner_knots[1] - inner_knots[0]
        diffs = np.arange(1, order + 1) * diff_knots
        expected = np.sort(np.concatenate((
            inner_knots[0] - diffs[::-1], inner_knots, inner_knots[-1] + diffs,
        )))

        actual = get_knots_bsplines(x, df=df, degree=degree, spacing="equal")
        assert_allclose(actual, expected)


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
