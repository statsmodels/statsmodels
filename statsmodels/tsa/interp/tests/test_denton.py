import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.linalg import null_space

from statsmodels.tsa.interp import dentonm
from statsmodels.tsa.tests._contract import SERIES_FORMS, series_forms


def test_denton_quarterly():
    # Data and results taken from IMF paper
    indicator = np.array([
        98.2, 100.8, 102.2, 100.8, 99.0, 101.6,
        102.7, 101.5, 100.5, 103.0, 103.5, 101.5
    ])
    benchmark = np.array([4000., 4161.4])
    x_imf = dentonm(indicator, benchmark, freq="aq")
    imf_stata = np.array([
        969.8, 998.4, 1018.3, 1013.4, 1007.2, 1042.9,
        1060.3, 1051.0, 1040.6, 1066.5, 1071.7, 1051.0
    ])
    np.testing.assert_almost_equal(imf_stata, x_imf, 1)


def test_denton_quarterly2():
    # Test denton vs stata. Higher precision than other test.
    zQ = np.array([50, 100, 150, 100] * 5)
    Y = np.array([500, 400, 300, 400, 500])
    x_denton = dentonm(zQ, Y, freq="aq")
    x_stata = np.array([
        64.334796, 127.80616, 187.82379, 120.03526, 56.563894,
        105.97568, 147.50144, 89.958987, 40.547201, 74.445963,
        108.34473, 76.66211, 42.763347, 94.14664, 153.41596,
        109.67405, 58.290761, 122.62556, 190.41409, 128.66959
    ])
    np.testing.assert_almost_equal(x_denton, x_stata, 5)


def test_denton_freq_qm_matches_benchmark_sums():
    # Denton benchmarking preserves the low-frequency totals: each block of
    # k consecutive high-frequency values must sum to its benchmark value.
    indicator = np.array([10.0, 12.0, 11.0, 9.0, 13.0, 14.0])
    benchmark = np.array([300.0, 360.0])
    x = dentonm(indicator, benchmark, freq="qm")
    assert_allclose(x[0:3].sum(), benchmark[0])
    assert_allclose(x[3:6].sum(), benchmark[1])


def test_denton_freq_other_matches_benchmark_sums():
    indicator = np.array([10.0, 12.0, 11.0, 9.0, 13.0, 14.0, 15.0, 16.0])
    benchmark = np.array([300.0, 360.0])
    x = dentonm(indicator, benchmark, freq="other", k=4)
    assert_allclose(x[0:4].sum(), benchmark[0])
    assert_allclose(x[4:8].sum(), benchmark[1])


def test_denton_freq_other_requires_k():
    indicator = np.array([10.0, 12.0, 11.0, 9.0])
    benchmark = np.array([50.0])
    with pytest.raises(ValueError, match="k must be supplied"):
        dentonm(indicator, benchmark, freq="other")


def test_denton_invalid_freq_raises():
    indicator = np.array([10.0, 12.0, 11.0, 9.0])
    benchmark = np.array([50.0])
    with pytest.raises(ValueError, match="freq"):
        dentonm(indicator, benchmark, freq="not-a-freq")


def test_denton_too_few_indicators():
    # an indicator shorter than k * len(benchmark) used to crash deep in the
    # linalg ("shapes (3,3) and (6,6) not aligned"); it must report the
    # shortfall instead
    with pytest.raises(ValueError, match="requires at least 6"):
        dentonm([1.0, 2.0, 3.0], [10.0, 20.0], freq="qm")
    with pytest.raises(ValueError, match="requires at least 8"):
        dentonm([1.0], [10.0, 20.0], freq="aq")
    # the boundary, exactly k * len(benchmark) observations, still works
    res = dentonm(np.ones(6), [10.0, 20.0], freq="qm")
    assert_allclose(res[:3].sum(axis=0), [10.0])
    assert_allclose(res[3:].sum(axis=0), [20.0])


def _denton_by_minimisation(indicator, benchmark, k):
    """
    Minimise the proportionate first-difference objective directly

    With r = X / I, minimise sum(diff(r) ** 2) subject to each block of k
    values of X summing to its benchmark, by solving the least-squares
    problem in the null space of the constraints. The values after the last
    benchmarked block are extrapolated with the last ratio X / I.
    """
    indicator = np.asarray(indicator, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)
    m = benchmark.shape[0]
    n = k * m
    ind = indicator[:n]
    constraints = np.kron(np.eye(m), np.ones((1, k))) * ind
    r0 = np.linalg.lstsq(constraints, benchmark, rcond=None)[0]
    null = null_space(constraints)
    diff = np.diff(np.eye(n), axis=0)
    z = np.linalg.lstsq(diff @ null, -diff @ r0, rcond=None)[0]
    x = ind * (r0 + null @ z)
    return np.r_[x, x[-1] / ind[-1] * indicator[n:]]


@pytest.mark.parametrize(
    "freq,kwargs,k",
    [("aq", {}, 4), ("qm", {}, 3), ("other", {"k": 2}, 2), ("other", {"k": 5}, 5)],
)
@pytest.mark.parametrize("extra", [0, 3])
@pytest.mark.parametrize("two_dim", [False, True])
def test_denton_minimises_objective(freq, kwargs, k, extra, two_dim):
    rng = np.random.default_rng([20261120, 6])
    m = 4
    indicator = 100 + np.cumsum(rng.standard_normal(k * m + extra))
    benchmark = k * 110 + 10 * rng.standard_normal(m)
    expected = _denton_by_minimisation(indicator, benchmark, k)
    if two_dim:
        # (n, 1) and (m, 1) columns give the same result as 1-d input
        indicator = indicator[:, None]
        benchmark = benchmark[:, None]
    x = dentonm(indicator, benchmark, freq=freq, **kwargs)
    assert x.shape == (k * m + extra,)
    assert_allclose(x, expected, rtol=1e-9)
    block_sums = x[: k * m].reshape(m, k).sum(1)
    assert_allclose(block_sums, np.squeeze(benchmark), rtol=1e-12)


INDICATOR = np.array(
    [98.2, 100.8, 102.2, 100.8, 99.0, 101.6, 102.7, 101.5, 100.5, 103.0, 103.5,
     101.5, 102.0, 104.0]
)


@pytest.mark.parametrize("form", SERIES_FORMS + ("int",))
def test_denton_indicator_forms(form):
    indicator = np.round(INDICATOR)
    benchmark = np.array([4000.0, 4161.4])
    expected = dentonm(indicator, benchmark)
    if form == "int":
        x = indicator.astype(np.int64)
    else:
        x = series_forms(indicator)[form]
    res = dentonm(x, benchmark)
    assert isinstance(res, np.ndarray)
    assert res.shape == (INDICATOR.shape[0],)
    assert_allclose(res, expected, rtol=1e-12)


@pytest.mark.parametrize(
    "form", ["list", "tuple", "series", "column", "dataframe", "int"]
)
def test_denton_benchmark_forms(form):
    benchmark = np.array([4000.0, 4161.0])
    expected = dentonm(INDICATOR, benchmark)
    index = pd.date_range("2000-01-01", periods=2, freq="YS")
    forms = {
        "list": benchmark.tolist(),
        "tuple": tuple(benchmark.tolist()),
        "series": pd.Series(benchmark, index=index),
        "column": benchmark[:, None],
        "dataframe": pd.DataFrame({"benchmark": benchmark}, index=index),
        "int": benchmark.astype(np.int64),
    }
    res = dentonm(INDICATOR, forms[form])
    assert isinstance(res, np.ndarray)
    assert_allclose(res, expected, rtol=1e-12)


def _denton_null_space_solution(indicator, benchmark, k):
    # Denton's proportionate first-difference problem solved directly: with
    # x = indicator * ratio, minimise sum(diff(ratio) ** 2) subject to each
    # block of k values of x summing to its benchmark. The ratios are a
    # particular solution of the constraints plus a combination of the null
    # space of the constraints (from the SVD), with the coefficients from an
    # unconstrained least-squares problem. The values after the last block are
    # extrapolated with the last ratio.
    indicator = np.asarray(indicator, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)
    m = benchmark.shape[0]
    n = k * m
    constraints = np.kron(np.eye(m), np.ones((1, k))) * indicator[:n]
    ratio0 = np.linalg.lstsq(constraints, benchmark, rcond=None)[0]
    basis = np.linalg.svd(constraints)[2][m:].T
    diff = np.diff(np.eye(n), axis=0)
    coef = np.linalg.lstsq(diff @ basis, -diff @ ratio0, rcond=None)[0]
    ratio = ratio0 + basis @ coef
    return indicator * np.r_[ratio, np.full(indicator.shape[0] - n, ratio[-1])]


@pytest.mark.parametrize(
    "k",
    [4, np.int64(4), 4.0, np.float64(4.0)],
    ids=["int", "int64", "float", "float64"],
)
def test_denton_k_integer_like(k):
    # k is converted with int_like, so an integer-valued float, which raised a
    # TypeError from numpy, now works
    indicator = np.array([10.0, 12.0, 11.0, 9.0, 13.0, 14.0, 15.0, 16.0, 12.0, 11.0])
    benchmark = np.array([40.0, 50.0])
    res = dentonm(indicator, benchmark, freq="other", k=k)
    expected = _denton_null_space_solution(indicator, benchmark, 4)
    assert_allclose(res, expected, rtol=1e-10)


@pytest.mark.parametrize(
    "k, error, match",
    [
        (2.5, TypeError, "k must be integer_like"),
        ("4", TypeError, "k must be integer_like"),
        (0, ValueError, "k must be a positive integer, got 0"),
        (-1, ValueError, "k must be a positive integer, got -1"),
    ],
)
def test_denton_invalid_k(k, error, match):
    # A non-integer k raised a TypeError from numpy or a comparison, 0 the
    # error that k is missing, and a negative k "negative dimensions are not
    # allowed"
    with pytest.raises(error, match=match):
        dentonm(np.arange(1.0, 9.0), [10.0, 26.0], freq="other", k=k)


@pytest.mark.parametrize(
    "benchmark",
    [np.ones((2, 2)), np.ones((1, 2)), np.ones((2, 1, 1)), np.float64(10.0)],
    ids=["two-columns", "row", "3-d", "scalar"],
)
def test_denton_benchmark_not_single_series(benchmark):
    # Only one series is benchmarked. Several columns, a row or 3-d input
    # raised a numpy broadcasting error, and a scalar "len() of unsized object"
    with pytest.raises(ValueError, match="benchmark must be a single series"):
        dentonm(np.arange(1.0, 9.0), benchmark)


@pytest.mark.parametrize(
    "form", [np.asarray, lambda b: np.asarray(b)[:, None]], ids=["1-d", "column"]
)
def test_denton_benchmark_single_series(form):
    indicator = np.array([10.0, 12.0, 11.0, 9.0, 13.0, 14.0, 15.0, 16.0, 12.0, 11.0])
    benchmark = [40.0, 50.0]
    res = dentonm(indicator, form(benchmark), freq="other", k=4)
    expected = _denton_null_space_solution(indicator, benchmark, 4)
    assert_allclose(res, expected, rtol=1e-10)


@pytest.mark.parametrize(
    "kwargs, names",
    [({}, None), ({"penalty": "D4"}, "'penalty'"), ({"z": 1, "a": 2}, "'a', 'z'")],
    ids=["none", "one", "two"],
)
def test_denton_unknown_keywords_deprecated(kwargs, names):
    # Keyword arguments other than k were silently ignored. They still are,
    # with a FutureWarning, and the result does not change.
    indicator = np.array([10.0, 12.0, 11.0, 9.0, 13.0, 14.0, 15.0, 16.0, 12.0, 11.0])
    benchmark = np.array([40.0, 50.0])
    if names is None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            res = dentonm(indicator, benchmark, freq="other", k=4)
    else:
        match = f"Unknown keyword arguments are ignored: {names}"
        with pytest.warns(FutureWarning, match=match) as record:
            res = dentonm(indicator, benchmark, freq="other", k=4, **kwargs)
        assert len(record) == 1
        # The warning points to the caller
        assert record[0].filename == __file__
    expected = _denton_null_space_solution(indicator, benchmark, 4)
    assert_allclose(res, expected, rtol=1e-10)
