"""Tests for rolling regression models."""

from statsmodels.compat.python import PYTHON_IMPL_WASM

from io import BytesIO
from itertools import product
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest

from statsmodels import tools
from statsmodels.regression.linear_model import WLS
from statsmodels.regression.rolling import RollingOLS, RollingWLS


def gen_data(nobs, nvar, const, pandas=False, missing=0.0, weights=False):
    rs = np.random.RandomState(987499302)
    x = rs.standard_normal((nobs, nvar))
    cols = [f"x{i}" for i in range(nvar)]
    if const:
        x = tools.add_constant(x)
        cols = ["const"] + cols
    if missing > 0.0:
        mask = rs.random_sample(x.shape) < missing
        x[mask] = np.nan
    if x.shape[1] > 1:
        y = x[:, :-1].sum(1) + rs.standard_normal(nobs)
    else:
        y = x.sum(1) + rs.standard_normal(nobs)
    w = rs.chisquare(5, y.shape[0]) / 5
    if pandas:
        idx = pd.date_range("12-31-1999", periods=nobs)
        x = pd.DataFrame(x, index=idx, columns=cols)
        y = pd.Series(y, index=idx, name="y")
        w = pd.Series(w, index=idx, name="weights")
    if not weights:
        w = None

    return y, x, w


nobs = (250,)
nvar = (3, 0)
tf = (True, False)
missing = (0, 0.1)
params = list(product(nobs, nvar, tf, tf, missing))
params = [param for param in params if param[1] + param[2] > 0]
ids = ["-".join(map(str, param)) for param in params]

basic_params = [param for param in params if params[2] and params[4]]
weighted_params = [param + (tf,) for param in params for tf in (True, False)]
weighted_ids = ["-".join(map(str, param)) for param in weighted_params]


@pytest.fixture(scope="module", params=params, ids=ids)
def data(request):
    return gen_data(*request.param)


@pytest.fixture(scope="module", params=basic_params, ids=ids)
def basic_data(request):
    return gen_data(*request.param)


@pytest.fixture(scope="module", params=weighted_params, ids=weighted_ids)
def weighted_data(request):
    return gen_data(*request.param)


def get_single(x, idx):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.iloc[idx]
    return x[idx]


def get_sub(x, idx, window):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        out = x.iloc[idx - window : idx]
        return np.asarray(out)
    return x[idx - window : idx]


def test_has_nan(data):
    y, x, w = data
    mod = RollingWLS(y, x, window=100, weights=w)
    has_nan = np.zeros(y.shape[0], dtype=bool)
    for i in range(100, y.shape[0] + 1):
        _y = get_sub(y, i, 100)
        _x = get_sub(x, i, 100)
        has_nan[i - 1] = np.squeeze(np.any(np.isnan(_y)) or np.any(np.isnan(_x)))
    assert_array_equal(mod._has_nan, has_nan)


def test_weighted_against_wls(weighted_data):
    y, x, w = weighted_data
    mod = RollingWLS(y, x, weights=w, window=100)
    res = mod.fit(use_t=True)
    for i in range(100, y.shape[0]):
        _y = get_sub(y, i, 100)
        _x = get_sub(x, i, 100)
        if w is not None:
            _w = get_sub(w, i, 100)
        else:
            _w = np.ones_like(_y)
        wls = WLS(_y, _x, weights=_w, missing="drop").fit()
        rolling_params = get_single(res.params, i - 1)
        rolling_nobs = get_single(res.nobs, i - 1)
        assert_allclose(rolling_params, wls.params)
        assert_allclose(rolling_nobs, wls.nobs)
        assert_allclose(get_single(res.ssr, i - 1), wls.ssr)
        assert_allclose(get_single(res.llf, i - 1), wls.llf)
        assert_allclose(get_single(res.aic, i - 1), wls.aic)
        assert_allclose(get_single(res.bic, i - 1), wls.bic)
        assert_allclose(get_single(res.centered_tss, i - 1), wls.centered_tss)
        assert_allclose(res.df_model, wls.df_model)
        assert_allclose(get_single(res.df_resid, i - 1), wls.df_resid)
        assert_allclose(get_single(res.ess, i - 1), wls.ess, atol=1e-8)
        assert_allclose(res.k_constant, wls.k_constant)
        assert_allclose(get_single(res.mse_model, i - 1), wls.mse_model)
        assert_allclose(get_single(res.mse_resid, i - 1), wls.mse_resid)
        assert_allclose(get_single(res.mse_total, i - 1), wls.mse_total)
        assert_allclose(get_single(res.rsquared, i - 1), wls.rsquared, atol=1e-8)
        assert_allclose(
            get_single(res.rsquared_adj, i - 1), wls.rsquared_adj, atol=1e-8
        )
        assert_allclose(get_single(res.uncentered_tss, i - 1), wls.uncentered_tss)


@pytest.mark.parametrize("cov_type", ["nonrobust", "HC0"])
@pytest.mark.parametrize("use_t", [None, True, False])
def test_against_wls_inference(data, use_t, cov_type):
    y, x, w = data
    mod = RollingWLS(y, x, window=100, weights=w)
    res = mod.fit(use_t=use_t, cov_type=cov_type)
    ci = res.conf_int()

    # This is a smoke test of cov_params to make sure it works
    res.cov_params()

    # Skip to improve performance
    for i in range(100, y.shape[0]):
        _y = get_sub(y, i, 100)
        _x = get_sub(x, i, 100)
        wls = WLS(_y, _x, missing="drop").fit(use_t=use_t, cov_type=cov_type)
        assert_allclose(get_single(res.tvalues, i - 1), wls.tvalues)
        assert_allclose(get_single(res.bse, i - 1), wls.bse)
        assert_allclose(get_single(res.pvalues, i - 1), wls.pvalues, atol=1e-8)
        assert_allclose(get_single(res.fvalue, i - 1), wls.fvalue)
        with np.errstate(invalid="ignore"):
            assert_allclose(get_single(res.f_pvalue, i - 1), wls.f_pvalue, atol=1e-8)
        assert res.cov_type == wls.cov_type
        assert res.use_t == wls.use_t
        wls_ci = wls.conf_int()
        if isinstance(ci, pd.DataFrame):
            ci_val = ci.iloc[i - 1]
            ci_val = np.asarray(ci_val).reshape((-1, 2))
        else:
            ci_val = ci[i - 1].T
        assert_allclose(ci_val, wls_ci)


def test_raise(data):
    y, x, w = data

    mod = RollingWLS(y, x, window=100, missing="drop", weights=w)
    res = mod.fit()
    params = np.asarray(res.params)
    assert np.all(np.isfinite(params[99:]))

    if not np.any(np.isnan(y)):
        return
    mod = RollingWLS(y, x, window=100, missing="skip")
    res = mod.fit()
    params = np.asarray(res.params)
    assert np.any(np.isnan(params[100:]))


def test_error():
    y, x, _ = gen_data(250, 2, True)
    with pytest.raises(ValueError, match="reset must be a positive integer"):
        RollingWLS(
            y,
            x,
        ).fit(reset=-1)
    with pytest.raises(ValueError):
        RollingWLS(y, x).fit(method="unknown")
    with pytest.raises(ValueError):
        RollingWLS(y, x).fit(cov_type="unknown")
    with pytest.raises(ValueError, match="min_nobs must be larger"):
        RollingWLS(y, x, min_nobs=1)
    with pytest.raises(ValueError, match="min_nobs must be larger"):
        RollingWLS(y, x, window=60, min_nobs=100)


def test_save_load(data):
    y, x, w = data
    res = RollingOLS(y, x, window=60).fit()
    fh = BytesIO()
    # test wrapped results load save pickle
    res.save(fh)
    fh.seek(0, 0)
    res_unpickled = res.__class__.load(fh)
    assert type(res_unpickled) is type(res)

    fh = BytesIO()
    # test wrapped results load save pickle
    res.save(fh, remove_data=True)
    fh.seek(0, 0)
    res_unpickled = res.__class__.load(fh)
    assert type(res_unpickled) is type(res)


def test_formula():
    y, x, w = gen_data(250, 3, True, pandas=True)
    fmla = "y ~ 1 + x0 + x1 + x2"
    data = pd.concat([y, x], axis=1)
    mod = RollingWLS.from_formula(fmla, window=100, data=data, weights=w)
    res = mod.fit()
    alt = RollingWLS(y, x, window=100)
    alt_res = alt.fit()
    assert_allclose(res.params, alt_res.params)
    ols_mod = RollingOLS.from_formula(fmla, window=100, data=data)
    ols_mod.fit()


@pytest.mark.thread_unsafe(reason="uses matplotlib")
@pytest.mark.matplotlib
def test_plot(close_figures):
    import matplotlib.pyplot as plt

    y, x, w = gen_data(250, 3, True, pandas=True)
    fmla = "y ~ 1 + x0 + x1 + x2"
    data = pd.concat([y, x], axis=1)
    mod = RollingWLS.from_formula(fmla, window=100, data=data, weights=w)
    res = mod.fit()
    fig = res.plot_recursive_coefficient()
    assert isinstance(fig, plt.Figure)
    res.plot_recursive_coefficient(variables=2, alpha=None, figsize=(30, 7))
    res.plot_recursive_coefficient(variables="x0", alpha=None, figsize=(30, 7))
    res.plot_recursive_coefficient(variables=[0, 2], alpha=None, figsize=(30, 7))
    plt.close("all")
    res.plot_recursive_coefficient(variables=["x0"], alpha=None, figsize=(30, 7))
    res.plot_recursive_coefficient(
        variables=["x0", "x1", "x2"], alpha=None, figsize=(30, 7)
    )
    plt.close("all")
    with pytest.raises(ValueError, match="variable x4 is not an integer"):
        res.plot_recursive_coefficient(variables="x4")

    fig = plt.Figure()
    # Just silence the warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = res.plot_recursive_coefficient(fig=fig)
    assert out is fig
    res.plot_recursive_coefficient(alpha=None, figsize=(30, 7))


@pytest.mark.parametrize("params_only", [True, False])
def test_methods(basic_data, params_only):
    y, x, _ = basic_data
    mod = RollingOLS(y, x, 150)
    res_inv = mod.fit(method="inv", params_only=params_only)
    res_lstsq = mod.fit(method="lstsq", params_only=params_only)
    res_pinv = mod.fit(method="pinv", params_only=params_only)
    assert_allclose(res_inv.params, res_lstsq.params)
    assert_allclose(res_inv.params, res_pinv.params)


@pytest.mark.parametrize("method", ["inv", "lstsq", "pinv"])
def test_params_only(basic_data, method):
    y, x, _ = basic_data
    mod = RollingOLS(y, x, 150)
    res = mod.fit(method=method, params_only=False)
    res_params_only = mod.fit(method=method, params_only=True)
    # use assert_allclose to incorporate for numerical errors on x86 platforms
    assert_allclose(res_params_only.params, res.params)


def test_min_nobs(basic_data):
    y, x, w = basic_data
    if not np.any(np.isnan(np.asarray(x))):
        return
    mod = RollingOLS(y, x, 150)
    res = mod.fit()
    # Ensures that the constraint binds
    min_nobs = res.nobs[res.nobs != 0].min() + 1
    mod = RollingOLS(y, x, 150, min_nobs=min_nobs)
    res = mod.fit()
    assert np.all(res.nobs[res.nobs != 0] >= min_nobs)


@pytest.mark.parametrize("model", [RollingOLS, RollingWLS])
def test_window_larger_than_nobs(model):
    # window > nobs has no valid window for the non-expanding recursion and
    # used to fail with an IndexError from inside fit; require a clean
    # ValueError at construction. The expanding=True path is covered by
    # test_expanding_window_larger_than_nobs.
    rs = np.random.RandomState(2948739)
    n, w = 10, 20
    x = tools.add_constant(rs.standard_normal((n, 1)))
    y = rs.standard_normal(n)
    kwargs = {"window": w, "min_nobs": 2}
    if model is RollingWLS:
        kwargs["weights"] = np.ones(n)
    with pytest.raises(ValueError, match="window must be less than or equal"):
        model(y, x, **kwargs)


def test_expanding(basic_data):
    y, x, w = basic_data
    xa = np.asarray(x)
    mod = RollingOLS(y, x, 150, min_nobs=50, expanding=True)
    res = mod.fit()
    params = np.asarray(res.params)
    assert np.all(np.isnan(params[:49]))
    first = np.where(np.cumsum(np.all(np.isfinite(xa), axis=1)) >= 50)[0][0]
    assert np.all(np.isfinite(params[first:]))


def test_expanding_window_larger_than_nobs():
    # GH 9287: expanding with window > nobs must not raise IndexError
    rs = np.random.RandomState(987499302)
    n, w = 10, 20
    x = tools.add_constant(rs.standard_normal((n, 1)))
    y = rs.standard_normal(n)
    mod = RollingOLS(y, x, window=w, min_nobs=2, expanding=True)
    res = mod.fit()
    assert res.params.shape == (n, x.shape[1])
    assert np.all(np.isnan(res.params[:1]))
    assert np.all(np.isfinite(res.params[1:]))
    assert_array_equal(res.nobs[1:], np.arange(2, n + 1))


def test_has_nan_first_obs_in_window():
    # A window is missing if its first observation is missing
    y, x, _ = gen_data(30, 2, True)
    y = y.copy()
    y[10] = np.nan
    mod = RollingOLS(y, x, window=5, missing="skip")
    expected = np.zeros(30, dtype=bool)
    expected[10:15] = True
    assert_array_equal(mod._has_nan, expected)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("expanding", [False, True])
@pytest.mark.parametrize("window", [5, 20])
def test_skip_against_wls(window, expanding, weighted):
    # Windows after skipped windows must use the observations in the window
    y, x, w = gen_data(150, 2, True, weights=weighted)
    y = y.copy()
    x = x.copy()
    y[[30, 31, 90]] = np.nan
    x[60, 1] = np.nan
    mod = RollingWLS(
        y,
        x,
        window=window,
        weights=w,
        min_nobs=5,
        missing="skip",
        expanding=expanding,
    )
    params = np.asarray(mod.fit().params)
    w = np.ones_like(y) if w is None else w
    n_valid = 0
    for t in range(4 if expanding else window - 1, 150):
        start = max(t - window + 1, 0)
        loc = slice(start, t + 1)
        if np.any(np.isnan(y[loc])) or np.any(np.isnan(x[loc])):
            assert np.all(np.isnan(params[t]))
            continue
        wls = WLS(y[loc], x[loc], weights=w[loc]).fit()
        assert_allclose(params[t], wls.params)
        n_valid += 1
    assert n_valid > 50


def test_get_resid_against_wls(weighted_data):
    y, x, w = weighted_data
    window = 100
    res = RollingWLS(y, x, weights=w, window=window).fit()
    resid = res.get_resid()
    resid_oos = res.get_resid(out_of_sample=True)
    if isinstance(y, pd.Series):
        assert isinstance(resid, pd.Series)
        assert resid.index.equals(y.index)
    resid = np.asarray(resid)
    resid_oos = np.asarray(resid_oos)
    assert np.all(np.isnan(resid[: window - 1]))
    assert np.all(np.isnan(resid_oos[:window]))
    for i in range(window, y.shape[0] + 1):
        _y = get_sub(y, i, window)
        _x = get_sub(x, i, window)
        _w = np.ones_like(_y) if w is None else get_sub(w, i, window)
        wls = WLS(_y, _x, weights=_w, missing="drop").fit()
        last_x = np.asarray(get_single(x, i - 1))
        expected = get_single(y, i - 1) - last_x @ wls.params
        assert_allclose(resid[i - 1], expected)
        if i < y.shape[0]:
            next_x = np.asarray(get_single(x, i))
            expected = get_single(y, i) - next_x @ wls.params
            assert_allclose(resid_oos[i], expected)


def test_get_resid_expanding(basic_data):
    y, x, _ = basic_data
    res = RollingOLS(y, x, window=150, min_nobs=50, expanding=True).fit()
    resid = np.asarray(res.get_resid())
    params = np.asarray(res.params)
    for i in (49, 50, 100, 149, 150, 249):
        if np.any(np.isnan(params[i])):
            assert np.isnan(resid[i])
            continue
        start = max(0, i + 1 - 150)
        ols = WLS(
            get_sub(y, i + 1, i + 1 - start),
            get_sub(x, i + 1, i + 1 - start),
            missing="drop",
        ).fit()
        expected = get_single(y, i) - np.asarray(get_single(x, i)) @ ols.params
        assert_allclose(resid[i], expected)


@pytest.mark.parametrize("weighted", [False, True])
def test_get_resid_r(weighted):
    # R 4.5.3, first 30 rows of macrodata, weights 1 / pop
    # d <- read.csv("macro30.csv"); y <- d$realinv; x <- d$realgdp
    # wt <- if (weighted) 1 / d$pop else rep(1, 30); w <- 20; n <- 30
    # ins <- rep(NA, n); oos <- rep(NA, n)
    # for (t in w:n) {
    #   idx <- (t - w + 1):t
    #   f <- lm(y[idx] ~ x[idx], weights = wt[idx])
    #   ins[t] <- resid(f)[w]
    #   if (t < n) oos[t + 1] <- y[t + 1] - sum(c(1, x[t + 1]) * coef(f))
    # }
    # format(ins[w:n], digits = 15); format(oos[(w + 1):n], digits = 15)
    from statsmodels.datasets import macrodata

    data = macrodata.load_pandas().data.iloc[:30]
    y = data["realinv"].to_numpy()
    x = tools.add_constant(data["realgdp"].to_numpy())
    if weighted:
        weights = 1 / data["pop"].to_numpy()
        ins = [
            3.502547513971217,
            4.866898435252920,
            -3.891154177769110,
            -2.289164948737150,
            -1.609271399899344,
            16.111857522805064,
            4.631445232096264,
            3.445982316207549,
            -7.770754990082807,
            11.150628809861876,
            0.501349720738392,
        ]
        oos = [
            7.615610750216547,
            -2.870326165827862,
            -2.386145537590551,
            -0.685636369204701,
            23.427417096348336,
            7.090939753941143,
            5.832317934405410,
            -12.181983310725343,
            12.397314904706491,
            -0.126742035725044,
        ]
    else:
        weights = None
        ins = [
            3.407228809857275,
            4.761188630243743,
            -3.923839030286847,
            -2.307282838576138,
            -1.586919090261549,
            16.131396492004882,
            4.661023661953397,
            3.493047646696471,
            -7.746744190372824,
            11.061156072834121,
            0.395245579764173,
        ]
        oos = [
            7.480292664902379,
            -2.995532905775804,
            -2.429986428834809,
            -0.705514405558176,
            23.444632445393950,
            7.110521859563846,
            5.864958655891257,
            -12.126982086050816,
            12.424791667555837,
            -0.218668756771365,
        ]
    res = RollingWLS(y, x, window=20, weights=weights).fit()
    resid = res.get_resid()
    resid_oos = res.get_resid(out_of_sample=True)
    assert np.all(np.isnan(resid[:19]))
    assert np.all(np.isnan(resid_oos[:20]))
    assert_allclose(resid[19:], ins, rtol=1e-8)
    assert_allclose(resid_oos[20:], oos, rtol=1e-8)


def test_get_resid_formula():
    y, x, _ = gen_data(250, 2, False, pandas=True)
    rs = np.random.RandomState(1234)
    data = pd.concat([y, x], axis=1)
    data["g"] = rs.choice(["a", "b", "c"], size=250)
    res = RollingOLS.from_formula("y ~ x0 + C(g)", data=data, window=60).fit()
    resid = res.get_resid()
    assert isinstance(resid, pd.Series)
    assert resid.index.equals(data.index)
    exog = pd.get_dummies(data["g"], drop_first=True, dtype=float)
    exog = tools.add_constant(pd.concat([data[["x0"]], exog], axis=1))
    alt = RollingOLS(data["y"], exog, window=60).fit()
    assert_allclose(resid, alt.get_resid())
    assert_allclose(
        res.get_resid(out_of_sample=True), alt.get_resid(out_of_sample=True)
    )


@pytest.mark.parametrize("pandas", [False, True])
@pytest.mark.parametrize("out_of_sample", [False, True])
def test_get_resid_missing_skip(pandas, out_of_sample):
    y, x, _ = gen_data(100, 2, True, pandas=pandas)
    y = y.copy()
    x = x.copy()
    if pandas:
        y.iloc[[30, 70]] = np.nan
        x.iloc[50, 1] = np.nan
    else:
        y[[30, 70]] = np.nan
        x[50, 1] = np.nan
    window = 20
    res = RollingOLS(y, x, window=window, missing="skip").fit()
    resid = res.get_resid(out_of_sample=out_of_sample)
    assert resid.shape == (100,)
    if pandas:
        assert isinstance(resid, pd.Series)
        assert resid.index.equals(y.index)
    resid = np.asarray(resid)
    ya = np.asarray(y)
    xa = np.asarray(x)
    params = np.asarray(res.params)
    if out_of_sample:
        # residual at t uses the parameters of the window ending at t - 1
        params = np.vstack((np.full((1, 3), np.nan), params[:-1]))
    skipped = np.any(np.isnan(params), axis=1)
    missing = np.isnan(ya) | np.any(np.isnan(xa), axis=1)
    expected_nan = skipped | missing
    assert_array_equal(np.isnan(resid), expected_nan)
    assert 0 < expected_nan.sum() < 100
    valid = ~expected_nan
    expected = ya[valid] - np.sum(xa[valid] * params[valid], axis=1)
    assert_allclose(resid[valid], expected)


def fit_loop(mod, **kwargs):
    # Estimate one window at a time using the rank-one update loop
    mod._use_batched = lambda first, method: False
    try:
        return mod.fit(**kwargs)
    finally:
        del mod._use_batched


def assert_results_equal(res, expected, rtol=1e-10, robust=False):
    assert_array_equal(np.isnan(res.params), np.isnan(expected.params))
    assert_allclose(res.params, expected.params, rtol=rtol)
    assert_array_equal(res.nobs, expected.nobs)
    for attr in (
        "bse",
        "ssr",
        "llf",
        "centered_tss",
        "uncentered_tss",
        "rsquared",
        "fvalue",
    ):
        assert_allclose(getattr(res, attr), getattr(expected, attr), rtol=rtol)
    assert_allclose(res.cov_params(), expected.cov_params(), rtol=rtol, atol=1e-14)
    if not robust:
        assert np.all(np.isnan(res._xepxe))


def batched_data(weighted, missing):
    rs = np.random.default_rng(84171)
    n = 300
    x = np.column_stack([np.ones(n), rs.standard_normal((n, 2))])
    y = x @ np.array([1.0, 2.0, -1.0]) + rs.standard_normal(n)
    w = rs.chisquare(5, n) / 5 if weighted else None
    if missing:
        y[[5, 40, 41, 150]] = np.nan
        x[[100, 220], 1] = np.nan
    return y, x, w


@pytest.mark.parametrize("method", ["inv", "lstsq", "pinv"])
@pytest.mark.parametrize("cov_type", ["nonrobust", "HC0"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("expanding", [False, True])
@pytest.mark.parametrize("missing", ["none", "drop", "skip"])
def test_batched_against_loop(method, cov_type, weighted, expanding, missing):
    y, x, w = batched_data(weighted, missing != "none")
    mod = RollingWLS(
        y,
        x,
        window=40,
        weights=w,
        min_nobs=10,
        expanding=expanding,
        missing="drop" if missing == "none" else missing,
    )
    res = mod.fit(method=method, cov_type=cov_type)
    expected = fit_loop(mod, method=method, cov_type=cov_type, reset=1)
    assert np.isfinite(res.params[-1]).all()
    assert_results_equal(res, expected, robust=cov_type != "nonrobust")
    res_params = mod.fit(method=method, params_only=True)
    assert_allclose(res_params.params, expected.params, rtol=1e-10)
    assert np.all(np.isnan(res_params._xpxi))
    assert_array_equal(res_params.nobs, 0)


@pytest.mark.thread_unsafe(reason="modifies a module constant")
@pytest.mark.parametrize("method", ["inv", "lstsq"])
@pytest.mark.parametrize("reset", [None, 7])
def test_batched_small_batches(monkeypatch, method, reset):
    # Many batches, a partial last batch, batches without any window and,
    # when reset is None, inner products carried across batches
    from statsmodels.regression import rolling

    y, x, w = batched_data(True, False)
    y[160:200] = np.nan
    mod = RollingWLS(y, x, window=25, weights=w, missing="skip")
    expected = mod.fit(method=method, cov_type="HC0", reset=reset)
    monkeypatch.setattr(rolling, "_BATCH_SIZE", 160)
    res = mod.fit(method=method, cov_type="HC0", reset=reset)
    assert_results_equal(res, expected, robust=True)
    assert np.all(np.isnan(res.params[160:224]))
    assert np.all(np.isfinite(res.params[224:]))


def test_batched_no_valid_window():
    y, x, _ = batched_data(False, False)
    y[::10] = np.nan
    res = RollingOLS(y, x, window=20, missing="skip").fit()
    assert np.all(np.isnan(res.params))
    assert_array_equal(res.nobs, 0)


@pytest.mark.parametrize("method", ["inv", "lstsq", "pinv"])
@pytest.mark.parametrize("params_only", [True, False])
def test_batched_singular_windows(method, params_only):
    if PYTHON_IMPL_WASM and not params_only and method != "inv":
        pytest.skip("LinAlgError not raised on WASM")
    # An event dummy makes the windows that do not contain the event singular
    rs = np.random.default_rng(2410)
    n, window = 120, 20
    x = np.column_stack([np.ones(n), rs.standard_normal(n), np.zeros(n)])
    x[60, 2] = 1.0
    y = x @ np.array([1.0, 2.0, 3.0]) + rs.standard_normal(n)
    mod = RollingOLS(y, x, window=window)
    res = mod.fit(method=method, params_only=params_only)
    expected = fit_loop(mod, method=method, params_only=params_only, reset=1)
    assert_array_equal(np.isnan(res.params), np.isnan(expected.params))
    assert_allclose(res.params, expected.params, rtol=1e-10, atol=1e-12)
    contains_event = np.zeros(n, dtype=bool)
    contains_event[60 : 60 + window] = True
    assert np.all(np.isfinite(res.params[contains_event]))
    if not params_only:
        assert_allclose(res.bse, expected.bse, rtol=1e-10)
        assert_allclose(res.ssr, expected.ssr, rtol=1e-10)
    if method == "inv" or not params_only:
        # inv fails for the singular windows, as in the loop
        assert np.all(np.isnan(res.params[window - 1 : 60]))
    else:
        # lstsq and pinv give the minimum norm solution
        assert np.all(np.isfinite(res.params[window - 1 :]))
        assert_allclose(res.params[window - 1 : 60, 2], 0.0, atol=1e-12)


@pytest.mark.parametrize("weighted", [False, True])
def test_batched_large_offset(weighted):
    # x + 1e4 is ill-conditioned. The exact estimates are a linear
    # transformation of the estimates from the well-conditioned design.
    rs = np.random.default_rng(97531)
    n, window, offset = 2000, 60, 1e4
    z = rs.standard_normal((n, 2))
    x0 = np.column_stack([np.ones(n), z])
    x = np.column_stack([np.ones(n), z + offset])
    y = x0 @ np.array([1.0, 2.0, -1.0]) + 0.1 * rs.standard_normal(n)
    w = rs.chisquare(5, n) / 5 if weighted else None
    ref = RollingWLS(y, x0, window=window, weights=w).fit(method="lstsq")
    ref_params = np.asarray(ref.params).copy()
    ref_params[:, 0] -= offset * ref_params[:, 1:].sum(1)
    valid = slice(window - 1, None)

    def max_rel_err(res):
        params = np.asarray(res.params)[valid]
        diff = np.abs(params - ref_params[valid])
        return np.max(diff / np.abs(ref_params[valid]))

    mod = RollingWLS(y, x, window=window, weights=w)
    for method in ("lstsq", "pinv"):
        res = mod.fit(method=method)
        assert max_rel_err(res) < 1e-7
        assert_allclose(res.ssr, ref.ssr, rtol=1e-6)
        loop = fit_loop(mod, method=method)
        assert max_rel_err(res) <= 10 * max_rel_err(loop)
    # The normal equations lose about twice as many digits
    res = mod.fit(method="inv")
    assert max_rel_err(res) < 1e-2
    loop = fit_loop(mod, method="inv")
    assert max_rel_err(res) <= 10 * max_rel_err(loop)


@pytest.mark.parametrize("method", ["inv", "lstsq"])
@pytest.mark.parametrize("expanding", [False, True])
def test_batched_full_sample_window(method, expanding):
    # window=None uses the full sample, and with expanding=True gives
    # recursive estimates
    y, x, w = batched_data(True, True)
    mod = RollingWLS(y, x, weights=w, min_nobs=5, expanding=expanding)
    res = mod.fit(method=method, cov_type="HC0")
    expected = fit_loop(mod, method=method, cov_type="HC0")
    assert_results_equal(res, expected, rtol=1e-9, robust=True)
    assert np.isfinite(res.params[-1]).all()
    assert np.isfinite(res.params[5]).all() == expanding
