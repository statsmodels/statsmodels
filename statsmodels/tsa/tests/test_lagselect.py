from fractions import Fraction

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

from statsmodels.tsa._lagselect import nested_lag_ols

EPS = np.finfo(float).eps


def _exact_ols(x, y):
    # Least squares in exact rational arithmetic. Returns the sum of squared
    # residuals and the last coefficient as Fractions.
    nobs, k = x.shape
    x = [[Fraction(v) for v in row] for row in x]
    y = [Fraction(v) for v in y]
    xtx = [
        [sum(x[t][i] * x[t][j] for t in range(nobs)) for j in range(k)]
        for i in range(k)
    ]
    xty = [sum(x[t][i] * y[t] for t in range(nobs)) for i in range(k)]
    # Gauss-Jordan elimination of [X'X, X'y]
    aug = [xtx[i] + [xty[i]] for i in range(k)]
    for i in range(k):
        pivot = next(r for r in range(i, k) if aug[r][i] != 0)
        aug[i], aug[pivot] = aug[pivot], aug[i]
        aug[i] = [v / aug[i][i] for v in aug[i]]
        for r in range(k):
            if r != i:
                factor = aug[r][i]
                aug[r] = [a - factor * b for a, b in zip(aug[r], aug[i], strict=True)]
    beta = [aug[i][k] for i in range(k)]
    ssr = sum(v * v for v in y) - sum(b * c for b, c in zip(xty, beta, strict=True))
    return ssr, beta[-1]


def _ols_loop(endog, blocks, exog, start=None):
    # One least squares fit per order
    nobs = endog.shape[0]
    nlags = len(blocks)
    if start is None:
        start = np.zeros(nlags + 1, dtype=int)
    exog = np.empty((nobs, 0)) if exog is None else exog
    ssr, params, tvalues = [], [], []
    for p in range(nlags + 1):
        x = np.column_stack([exog] + list(blocks[:p]))[start[p] :]
        y = endog[start[p] :]
        if x.shape[1]:
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            resid = y - x @ beta
            cov = np.linalg.inv(x.T @ x)[-1, -1]
            s2 = np.atleast_1d(np.sum(resid**2, 0)) / (y.shape[0] - x.shape[1])
            params.append(beta[-1])
            tvalues.append(beta[-1] / np.sqrt(cov * s2))
        else:
            resid = y
            params.append(np.full(endog.shape[1:], np.nan))
            tvalues.append(np.full(np.atleast_1d(endog[0]).shape, np.nan))
        ssr.append(resid.T @ resid)
    return np.array(ssr), np.array(params), np.squeeze(np.array(tvalues))


def _ar_data(nobs, neqs, rng):
    e = rng.standard_normal((nobs, neqs))
    y = np.zeros((nobs, neqs))
    for t in range(2, nobs):
        y[t] = 0.5 * y[t - 1] - 0.3 * y[t - 2] + e[t]
    return y


def _lag_blocks(y, nlags):
    # Lag j of all columns of y over the observations nlags, ..., nobs - 1
    nobs = y.shape[0]
    return [y[nlags - j : nobs - j] for j in range(1, nlags + 1)]


@pytest.mark.parametrize("nobs", [50, 30000])
@pytest.mark.parametrize("use_exog", [True, False])
@pytest.mark.parametrize("as_array", [True, False])
def test_univariate(nobs, use_exog, as_array):
    rng = np.random.default_rng(0)
    y = _ar_data(nobs, 1, rng)[:, 0]
    nlags = 8
    blocks = [b[:, None] for b in _lag_blocks(y, nlags)]
    exog = None
    if use_exog:
        exog = np.column_stack(
            [np.ones(nobs - nlags), rng.standard_normal(nobs - nlags)]
        )
    lags = np.column_stack(blocks) if as_array else blocks
    res = nested_lag_ols(y[nlags:], lags, exog, rcond=nobs * EPS)
    ssr, params, tvalues = _ols_loop(y[nlags:], blocks, exog)
    assert_allclose(res.ssr, ssr, rtol=1e-10)
    assert_allclose(res.params_last, np.squeeze(params), rtol=1e-10)
    assert_allclose(res.tvalues_last, tvalues, rtol=1e-8)
    k_exog = 0 if exog is None else 2
    assert_equal(res.df_model, k_exog + np.arange(nlags + 1))
    assert_equal(res.nobs, np.full(nlags + 1, nobs - nlags))
    assert res.ssr.shape == (nlags + 1,)
    if not use_exog:
        assert np.isnan(res.params_last[0])


@pytest.mark.parametrize("nobs", [60, 20000])
def test_multivariate(nobs):
    rng = np.random.default_rng(1)
    y = _ar_data(nobs, 3, rng)
    nlags = 4
    blocks = _lag_blocks(y, nlags)
    exog = np.column_stack([np.ones(nobs - nlags), np.arange(nobs - nlags)])
    res = nested_lag_ols(y[nlags:], blocks, exog, rcond=nobs * EPS)
    ssr, params, tvalues = _ols_loop(y[nlags:], blocks, exog)
    assert res.ssr.shape == (nlags + 1, 3, 3)
    assert_allclose(res.ssr, ssr, rtol=1e-10)
    assert_allclose(res.params_last, params, rtol=1e-9)
    assert_allclose(res.tvalues_last, tvalues, rtol=1e-8)
    assert_equal(res.df_model, 2 + 3 * np.arange(nlags + 1))


def test_block_widths():
    rng = np.random.default_rng(2)
    nobs = 200
    y = rng.standard_normal(nobs)
    blocks = [rng.standard_normal((nobs, w)) for w in (2, 1, 3)]
    res = nested_lag_ols(y, blocks, rcond=nobs * EPS)
    ssr, params, _ = _ols_loop(y, blocks, None)
    assert_allclose(res.ssr, ssr, rtol=1e-10)
    assert_allclose(res.params_last[1:], params[1:], rtol=1e-10)
    assert_equal(res.df_model, [0, 2, 3, 6])


@pytest.mark.parametrize("neqs", [1, 2])
@pytest.mark.parametrize("nobs", [40, 20000])
def test_start(neqs, nobs):
    # Each order uses its own sample through the indicator columns
    rng = np.random.default_rng(3)
    y = _ar_data(nobs, neqs, rng)
    if neqs == 1:
        y = y[:, 0]
    blocks = [rng.standard_normal((nobs, w)) for w in (1, 2, 1, 1, 3, 1)]
    exog = np.ones((nobs, 1))
    start = np.array([0, 1, 1, 3, 4, 4, 7])
    res = nested_lag_ols(y, blocks, exog, rcond=nobs * EPS, start=start)
    ssr, params, tvalues = _ols_loop(y, blocks, exog, start)
    assert_allclose(res.ssr, ssr, rtol=1e-10)
    assert_allclose(res.params_last, np.squeeze(params), rtol=1e-9)
    assert_allclose(res.tvalues_last, tvalues, rtol=1e-8)
    assert_equal(res.nobs, nobs - start)
    assert_equal(res.df_model, 1 + np.cumsum([0, 1, 2, 1, 1, 3, 1]))
    # The values of a lag block before its sample do not matter. The large
    # values limit the accuracy, which is relative to their size.
    for j, block in enumerate(blocks):
        block[: start[j + 1]] = 1e6 * rng.standard_normal(block[: start[j + 1]].shape)
    res2 = nested_lag_ols(y, blocks, exog, rcond=nobs * EPS, start=start)
    assert_allclose(res2.ssr, ssr, rtol=1e-8)


def test_triangular_rows():
    # The rows of the triangular factor give the same fits as the data
    rng = np.random.default_rng(4)
    nobs = 500
    y = rng.standard_normal(nobs)
    x = rng.standard_normal((nobs, 5))
    exog = np.ones((nobs, 1))
    res = nested_lag_ols(y, x, exog, rcond=nobs * EPS)
    r = res.r[1:, 1:]
    cols = [3, 0]
    sub = nested_lag_ols(r[:, -1], r[:, [1, 4]], r[:, cols], rcond=None)
    sub_exog = np.column_stack([exog, x[:, cols]])
    ssr, _, _ = _ols_loop(y, [x[:, [1]], x[:, [4]]], sub_exog)
    assert_allclose(sub.ssr, ssr, rtol=1e-10)


def test_rank_check():
    rng = np.random.default_rng(5)
    nobs = 100
    y = rng.standard_normal(nobs)
    x = rng.standard_normal((nobs, 3))
    exog = np.ones((nobs, 1))
    # a constant lag column
    x_const = x.copy()
    x_const[:, 2] = 2.0
    assert nested_lag_ols(y, x_const, exog, rcond=nobs * EPS) is None
    # collinear lag columns
    x_col = x.copy()
    x_col[:, 2] = x[:, 0] - x[:, 1]
    assert nested_lag_ols(y, x_col, exog, rcond=nobs * EPS) is None
    # almost collinear, the cutoff decides
    x_col[:, 2] += 1e-9 * rng.standard_normal(nobs)
    assert nested_lag_ols(y, x_col, exog, rcond=nobs * EPS) is not None
    assert nested_lag_ols(y, x_col, exog, rcond=1e-8) is None
    # a lag column that is zero in the sample of the orders that use it
    x_head = x.copy()
    x_head[:, 1] = 0.0
    x_head[:3, 1] = rng.standard_normal(3)
    start = [0, 0, 5, 5]
    assert nested_lag_ols(y, x_head, exog, rcond=nobs * EPS, start=start) is None
    # fewer observations than regressors
    assert nested_lag_ols(y[:3], x[:3], exog[:3], rcond=EPS) is None


def test_nonfinite():
    rng = np.random.default_rng(6)
    nobs = 100
    y = rng.standard_normal(nobs)
    x = rng.standard_normal((nobs, 3))
    y_nan = y.copy()
    y_nan[50] = np.nan
    assert nested_lag_ols(y_nan, x, rcond=nobs * EPS) is None
    x_inf = x.copy()
    x_inf[2, 1] = np.inf
    assert nested_lag_ols(y, x_inf, rcond=nobs * EPS) is None
    assert nested_lag_ols(y, x_inf, rcond=nobs * EPS, start=[0, 1, 4, 4]) is None


def test_check_fit():
    rng = np.random.default_rng(7)
    nobs = 100
    x = rng.standard_normal((nobs, 3))
    y = x @ [1.0, -2.0, 0.5] + 1e-7 * rng.standard_normal(nobs)
    assert nested_lag_ols(y, x, rcond=nobs * EPS) is None
    res = nested_lag_ols(y, x, rcond=nobs * EPS, check_fit=False)
    assert_allclose(res.params_last[-1], 0.5, rtol=1e-6)
    # Collinear dependent variables
    y2 = np.column_stack([y + rng.standard_normal(nobs), x[:, 0]])
    y2 = np.column_stack([y2, y2[:, 0] - y2[:, 1]])
    assert nested_lag_ols(y2, x[:, 1:], rcond=nobs * EPS) is None


def test_exceptions():
    y = np.zeros(10)
    x = np.ones((10, 2))
    with pytest.raises(ValueError, match="lags must"):
        nested_lag_ols(y, np.ones((9, 2)), rcond=None)
    with pytest.raises(ValueError, match="each block"):
        nested_lag_ols(y, [np.ones((10, 0))], rcond=None)
    with pytest.raises(ValueError, match="exog must"):
        nested_lag_ols(y, x, np.ones(10), rcond=None)
    for start in ([0, 1], [1, 1, 1], [0, 2, 1], [0, 1, 11]):
        with pytest.raises(ValueError, match="start must"):
            nested_lag_ols(y, x, rcond=None, start=start)


@pytest.mark.parametrize("seed", [0, 1])
def test_ill_conditioned_exact(seed):
    # Compare with exact rational arithmetic on a design with a condition
    # number of about 1e8 to 1e9. The errors are about 1e-9 (ssr) and 1e-7
    # (coefficient), as for lstsq, so the tolerances are not tight.
    rng = np.random.default_rng(seed)
    nobs = 40
    z = rng.standard_normal((nobs, 3))
    x = np.column_stack(
        [
            z[:, 0],
            z[:, 0] + 1e-8 * z[:, 1],
            z[:, 2],
            z[:, 0] - z[:, 2] + 3e-8 * rng.standard_normal(nobs),
        ]
    )
    exog = np.ones((nobs, 1))
    y = x @ [1.0, -0.5, 2.0, 0.3] + rng.standard_normal(nobs)
    assert 1e8 < np.linalg.cond(np.column_stack([exog, x])) < 1e10
    res = nested_lag_ols(y, x, exog, rcond=nobs * EPS)
    assert res is not None
    for p in range(x.shape[1] + 1):
        ssr, last = _exact_ols(np.column_stack([exog, x[:, :p]]), y)
        assert_allclose(res.ssr[p], float(ssr), rtol=1e-7)
        assert_allclose(res.params_last[p], float(last), rtol=1e-6)


@pytest.mark.parametrize("const_last", [True, False])
@pytest.mark.parametrize("neqs", [1, 2])
@pytest.mark.parametrize("level", [0.0, 1e3, 1e5])
def test_level(level, neqs, const_last):
    # A large level of the data is removed before the decomposition if exog
    # has a constant. The results are accurate to rounding of the data in
    # exact arithmetic, much better than lstsq.
    rng = np.random.default_rng(8)
    nobs = 40
    x = rng.standard_normal((nobs, 3)) + level
    y = rng.standard_normal((nobs, neqs)) + 0.5 * x[:, [0]] + level
    extra = rng.standard_normal((nobs, 1))
    ones = np.ones((nobs, 1))
    exog = np.column_stack([extra, ones] if const_last else [ones, extra])
    endog = y if neqs > 1 else y[:, 0]
    res = nested_lag_ols(endog, x, exog, rcond=nobs * EPS)
    assert res is not None
    for j in range(neqs):
        for p in range(4):
            ssr, last = _exact_ols(np.column_stack([exog, x[:, :p]]), y[:, j])
            expected_ssr = res.ssr[p] if neqs == 1 else res.ssr[p, j, j]
            expected_last = res.params_last[p] if neqs == 1 else res.params_last[p, j]
            assert_allclose(expected_ssr, float(ssr), rtol=1e-12)
            assert_allclose(expected_last, float(last), rtol=1e-10)
    # The t-values, including that of the constant of order 0, are found
    # from the shifted data
    blocks = [x[:, [j]] for j in range(3)]
    _, _, tvalues = _ols_loop(endog, blocks, exog)
    assert_allclose(res.tvalues_last, tvalues, rtol=1e-5)


@pytest.mark.parametrize("with_const", [True, False])
def test_offset(with_const):
    # r is the factor of the data after subtracting offset, which is the
    # mean of the lags and endog if exog has a constant and zero otherwise
    rng = np.random.default_rng(10)
    nobs = 50
    x = rng.standard_normal((nobs, 3)) + 100.0
    y = rng.standard_normal(nobs) + 0.5 * x[:, 0]
    extra = rng.standard_normal((nobs, 1))
    exog = np.column_stack([np.ones(nobs), extra]) if with_const else extra
    res = nested_lag_ols(y, x, exog, rcond=nobs * EPS)
    assert res is not None
    data = np.column_stack([exog, x, y])
    expected = np.zeros(data.shape[1])
    if with_const:
        expected[exog.shape[1] :] = data[:, exog.shape[1] :].mean(0)
    assert_allclose(res.offset, expected, rtol=1e-14)
    shifted = data - res.offset
    assert_allclose(res.r.T @ res.r, shifted.T @ shifted, rtol=1e-10, atol=1e-9)


def test_level_declines():
    # The checks apply to the data with the level. A large level does not
    # hide an exact fit or collinear regressors.
    rng = np.random.default_rng(9)
    nobs = 100
    level = 1e4
    x = rng.standard_normal((nobs, 3)) + level
    exog = np.ones((nobs, 1))
    y = level + x @ [1.0, -2.0, 0.5] + 1e-7 * rng.standard_normal(nobs)
    assert nested_lag_ols(y, x, exog, rcond=nobs * EPS) is None
    y = level + x @ [1.0, -2.0, 0.5] + rng.standard_normal(nobs)
    assert nested_lag_ols(y, x, exog, rcond=nobs * EPS) is not None
    # collinear with the constant, as the level is much larger than the
    # variation
    x_const = x.copy()
    x_const[:, 2] = level + 1e-10 * rng.standard_normal(nobs)
    assert nested_lag_ols(y, x_const, exog, rcond=nobs * EPS) is None
    # the cutoff for the original columns
    x_big = x - level + 1e9
    assert nested_lag_ols(y, x_big, exog, rcond=nobs * EPS) is None
    assert nested_lag_ols(y, x_big, exog, rcond=1e-20) is not None
