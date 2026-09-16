"""
Tests corresponding to sandbox.tsa.movstat

The moving-mean/variance/moment regression values below are ported from the
hand-verified `if __name__ == "__main__":` self-test block that used to be
the only check this module had.
"""
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
import pytest

from statsmodels.sandbox.tsa.movstat import (
    expandarr,
    movmean,
    movmoment,
    movorder,
    movvar,
)


def test_expandarr_1d():
    x = np.array([1.0, 2.0, 3.0])
    result = expandarr(x, 2)
    assert_array_equal(result, [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0])


def test_expandarr_2d():
    x = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    result = expandarr(x, 2)
    expected = np.array(
        [
            [1.0, 10.0],
            [1.0, 10.0],
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [3.0, 30.0],
            [3.0, 30.0],
        ]
    )
    assert_array_equal(result, expected)


def test_expandarr_zero_padding_is_noop():
    x = np.array([1.0, 2.0, 3.0])
    assert_array_equal(expandarr(x, 0), x)


def test_movorder_max_lagged_matches_increasing_series():
    # a trailing (lagged) moving max of a strictly increasing series, with
    # edge-padding by the boundary value, always equals the series itself
    x = np.arange(1, 10, dtype=float)
    result = movorder(x, order=2, windsize=3, lag="lagged")
    assert_array_equal(result, x)


def test_movorder_min_lagged_matches_decreasing_series():
    x = np.arange(10, 1, -1, dtype=float)
    result = movorder(x, order=0, windsize=3, lag="lagged")
    assert_array_equal(result, x)


def test_movorder_min_centered():
    x = np.arange(10, 1, -1, dtype=float)
    result = movorder(x, order=0, windsize=3, lag="centered")
    assert_array_equal(result[:-1], x[1:])


def test_movorder_invalid_lag_raises():
    x = np.arange(1, 10, dtype=float)
    with pytest.raises(ValueError):
        movorder(x, order=1, windsize=3, lag="bogus")


def test_movorder_invalid_order_raises_for_numeric_path():
    # a numeric order outside the valid range is not itself validated, but
    # a non-finite order (nan/inf) also fails to match any branch
    x = np.arange(1, 10, dtype=float)
    with pytest.raises(ValueError):
        movorder(x, order=np.nan, windsize=3, lag="lagged")


@pytest.mark.xfail(
    reason=(
        "BUG: movorder's order='min'/'max'/'med' string API (its documented "
        "primary interface) is completely broken. `np.isfinite(order)` is "
        "called unconditionally before checking `order in ('med', 'min', "
        "'max')`, and raises TypeError for any string order before the "
        "string branches are ever reached. Even the module's own "
        "check_movorder() self-test crashes on its first call for this "
        "reason. This test documents the *intended* behavior from the "
        "docstring/check_movorder(); it should start passing once the "
        "order dispatch is fixed to check string values before "
        "np.isfinite()."
    ),
    raises=TypeError,
    strict=True,
)
def test_movorder_string_order_max():
    x = np.arange(1, 10, dtype=float)
    result = movorder(x, order="max", windsize=3, lag="lagged")
    assert_array_equal(result, x)


@pytest.mark.xfail(
    reason="BUG: see test_movorder_string_order_max; same np.isfinite(order) crash.",
    raises=TypeError,
    strict=True,
)
def test_movorder_string_order_min():
    x = np.arange(10, 1, -1, dtype=float)
    result = movorder(x, order="min", windsize=3, lag="lagged")
    assert_array_equal(result, x)


@pytest.mark.xfail(
    reason="BUG: see test_movorder_string_order_max; same np.isfinite(order) crash.",
    raises=TypeError,
    strict=True,
)
def test_movorder_string_order_med():
    x = np.arange(1, 10, dtype=float)
    # should not raise for the documented "med" order
    movorder(x, order="med", windsize=3, lag="lagged")


# regression values for movmean/movvar/movmoment, windowsize=3, from the
# module's own former __main__ self-test block
_WS = 3
_VAR_REGRESSION = np.array(
    [
        [0.0, 0.0],
        [0.22222222, 0.88888889],
        [0.66666667, 2.66666667],
        [0.66666667, 2.66666667],
        [0.66666667, 2.66666667],
        [0.66666667, 2.66666667],
        [0.66666667, 2.66666667],
        [0.66666667, 2.66666667],
        [0.66666667, 2.66666667],
        [0.66666667, 2.66666667],
        [0.22222222, 0.88888889],
        [0.0, 0.0],
    ]
)
_MEAN_REGRESSION_1D = np.array(
    [0.0, 1 / 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 26 / 3.0, 9]
)


@pytest.mark.parametrize(
    "lag,expected_slice",
    [
        ("leading", slice(_WS - 1, None)),
        ("centered", slice(_WS // 2, -_WS // 2 + 1)),
        ("lagged", slice(None, -_WS + 1)),
    ],
)
def test_movvar_1d(lag, expected_slice):
    x = np.arange(10)
    result = movvar(x, windowsize=_WS, lag=lag)
    assert_array_almost_equal(_VAR_REGRESSION[expected_slice, 0], result)


@pytest.mark.parametrize(
    "lag,expected_slice",
    [
        ("leading", slice(_WS - 1, None)),
        ("centered", slice(_WS // 2, -_WS // 2 + 1)),
        ("lagged", slice(None, -_WS + 1)),
    ],
)
def test_movvar_2d_is_columnwise(lag, expected_slice):
    x = np.arange(10)
    x2d = np.c_[x, 2 * x]
    result = movvar(x2d, windowsize=_WS, lag=lag)
    assert_array_almost_equal(_VAR_REGRESSION[expected_slice, :], result)


def test_movmean_equals_movmoment_order_1():
    x = np.arange(10)
    assert_array_almost_equal(
        movmean(x, windowsize=_WS, lag="lagged"),
        movmoment(x, 1, windowsize=_WS, lag="lagged"),
    )


def test_movvar_equals_second_minus_first_squared_moment():
    x = np.arange(10)
    m1 = movmoment(x, 1, windowsize=_WS, lag="lagged")
    m2 = movmoment(x, 2, windowsize=_WS, lag="lagged")
    assert_array_almost_equal(movvar(x, windowsize=_WS, lag="lagged"), m2 - m1 * m1)


def test_movmean_1d_leading_matches_regression_values():
    x = np.arange(10)
    result = movmean(x, windowsize=_WS, lag="leading")
    assert_array_almost_equal(_MEAN_REGRESSION_1D[_WS - 1 :], result)


def test_movmean_1d_centered_matches_regression_values():
    x = np.arange(10)
    result = movmean(x, windowsize=_WS, lag="centered")
    assert_array_almost_equal(
        _MEAN_REGRESSION_1D[_WS // 2 : -_WS // 2 + 1], result
    )


def test_movmean_1d_lagged_matches_regression_values():
    x = np.arange(10)
    result = movmean(x, windowsize=_WS, lag="lagged")
    assert_array_almost_equal(_MEAN_REGRESSION_1D[: -_WS + 1], result)


def test_movmean_2d_matches_1d_broadcast():
    # the module docs claim moving moments are computed per-column for 2d
    # input; check that a (n, 2) array with a scaled second column gives
    # column-wise results consistent with the 1d computation
    x = np.arange(10)
    x2d = np.c_[x, 2 * x]
    for lag in ("leading", "centered", "lagged"):
        result_2d = movmean(x2d, windowsize=_WS, lag=lag)
        result_1d = movmean(x, windowsize=_WS, lag=lag)
        assert_array_almost_equal(result_2d[:, 0], result_1d)
        assert_array_almost_equal(result_2d[:, 1], 2 * result_1d)


_LAGGED_100_REGRESSION = np.array(
    [0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.1, 2.8, 3.6]
    + list(np.arange(9, 100) - 4.5)
)
# fmt: off
_LEADING_100_REGRESSION = np.array([
    0.3, 0.6, 1.0, 1.5, 2.1, 2.8, 3.6, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5,
    11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5,
    23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5,
    35.5, 36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5,
    47.5, 48.5, 49.5, 50.5, 51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5,
    59.5, 60.5, 61.5, 62.5, 63.5, 64.5, 65.5, 66.5, 67.5, 68.5, 69.5, 70.5,
    71.5, 72.5, 73.5, 74.5, 75.5, 76.5, 77.5, 78.5, 79.5, 80.5, 81.5, 82.5,
    83.5, 84.5, 85.5, 86.5, 87.5, 88.5, 89.5, 90.5, 91.5, 92.5, 93.5, 94.5,
    95.4, 96.2, 96.9, 97.5, 98.0, 98.4, 98.7, 98.9, 99.0,
])
# fmt: on
_CENTERED_100_REGRESSION_HEAD = np.array(
    [1.36363636, 1.90909091, 2.54545455, 3.27272727, 4.09090909]
)
_CENTERED_100_REGRESSION_TAIL = np.array(
    [94.90909091, 95.72727273, 96.45454545, 97.09090909, 97.63636364]
)


def test_movmean_lagged_window_100_matches_regression_values():
    # windowsize=10 over a longer series; verifies boundary handling beyond
    # the small windowsize=3 cases above. Reference values ported (and
    # numerically re-verified) from this module's former __main__ self-test.
    x = np.arange(100)
    result = movmean(x, 10, "lagged")
    assert_equal(len(result), 100)
    assert_array_almost_equal(result, _LAGGED_100_REGRESSION)
    # clean interior, unaffected by boundary padding: mean of [i-9, i] == i - 4.5
    assert_array_almost_equal(result[9:], np.arange(9, 100) - 4.5)


def test_movmean_leading_window_100_matches_regression_values():
    # unlike the lagged case, a leading window is boundary-affected at
    # *both* ends, and the output is longer than the input (107 vs 100)
    # since the trailing padded region is also returned. Compare against
    # the full pre-verified reference array (no closed-form formula).
    x = np.arange(100)
    result = movmean(x, 10, "leading")
    assert_equal(len(result), 107)
    assert_array_almost_equal(result, _LEADING_100_REGRESSION)


def test_movmean_centered_window_101_matches_regression_values():
    x = np.arange(100)
    result = movmean(x, 11, "centered")
    assert_equal(len(result), 100)
    assert_array_almost_equal(result[:5], _CENTERED_100_REGRESSION_HEAD)
    assert_array_almost_equal(result[-5:], _CENTERED_100_REGRESSION_TAIL)
    # interior of a centered, odd-length window is just the center value
    assert_array_almost_equal(result[5:95], np.arange(5, 95, dtype=float))


def test_movmoment_invalid_lag_raises():
    x = np.arange(10)
    with pytest.raises(ValueError):
        movmoment(x, 1, windowsize=_WS, lag="bogus")


def test_movmoment_prints_debug_output(capsys):
    # BUG (minor): movmoment() has a couple of unconditional, un-gated
    # `print()` calls left over from development (`print(sl)` always, plus
    # `print(xext.shape)` / `print(avgkern[:, None].shape)` for 2d input).
    # These are not behind any debug flag and fire on every call. This test
    # documents the current behavior rather than silently tolerating it;
    # if the prints are removed, this test should be updated/removed too.
    x = np.arange(10)
    capsys.readouterr()  # clear anything already buffered
    movmoment(x, 1, windowsize=_WS, lag="lagged")
    captured = capsys.readouterr()
    assert captured.out.strip() != ""

    x2d = np.c_[x, 2 * x]
    movmoment(x2d, 1, windowsize=_WS, lag="lagged")
    captured = capsys.readouterr()
    # the 2d branch prints two additional shape lines
    assert len(captured.out.strip().splitlines()) >= 3
