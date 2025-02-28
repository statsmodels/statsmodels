from statsmodels.compat.pandas import MONTH_END

import os
import pickle

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from statsmodels.datasets import co2
from statsmodels.tsa.seasonal import STL, DecomposeResult

cur_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(cur_dir, "results", "stl_test_results.csv")
results = pd.read_csv(file_path)
results.columns = [c.strip() for c in results.columns]
results.scenario = results.scenario.apply(str.strip)
results = results.set_index(["scenario", "idx"])


@pytest.fixture(scope="module", params=[True, False])
def robust(request):
    return request.param


def default_kwargs_base():
    file_path = os.path.join(cur_dir, "results", "stl_co2.csv")
    co2 = np.asarray(pd.read_csv(file_path, header=None).iloc[:, 0])
    y = co2
    nobs = y.shape[0]
    nperiod = 12
    work = np.zeros((nobs + 2 * nperiod, 7))
    rw = np.ones(nobs)
    trend = np.zeros(nobs)
    season = np.zeros(nobs)
    return dict(
        y=y,
        n=y.shape[0],
        np=nperiod,
        ns=35,
        nt=19,
        nl=13,
        no=2,
        ni=1,
        nsjump=4,
        ntjump=2,
        nljump=2,
        isdeg=1,
        itdeg=1,
        ildeg=1,
        rw=rw,
        trend=trend,
        season=season,
        work=work,
    )


@pytest.fixture(scope="function")
def default_kwargs():
    return default_kwargs_base()


@pytest.fixture(scope="function")
def default_kwargs_short():
    kwargs = default_kwargs_base()
    y = kwargs["y"][:-1]
    nobs = y.shape[0]
    work = np.zeros((nobs + 2 * kwargs["np"], 7))
    rw = np.ones(nobs)
    trend = np.zeros(nobs)
    season = np.zeros(nobs)
    kwargs.update(
        dict(y=y, n=nobs, rw=rw, trend=trend, season=season, work=work)
    )
    return kwargs


def _to_class_kwargs(kwargs, robust=False):
    endog = kwargs["y"]
    np = kwargs["np"]
    ns = kwargs["ns"]
    nt = kwargs["nt"]
    nl = kwargs["nl"]
    isdeg = kwargs["isdeg"]
    itdeg = kwargs["itdeg"]
    ildeg = kwargs["ildeg"]
    nsjump = kwargs["nsjump"]
    ntjump = kwargs["ntjump"]
    nljump = kwargs["nljump"]
    outer_iter = kwargs["no"]
    inner_iter = kwargs["ni"]
    class_kwargs = dict(
        endog=endog,
        period=np,
        seasonal=ns,
        trend=nt,
        low_pass=nl,
        seasonal_deg=isdeg,
        trend_deg=itdeg,
        low_pass_deg=ildeg,
        robust=robust,
        seasonal_jump=nsjump,
        trend_jump=ntjump,
        low_pass_jump=nljump,
    )
    return class_kwargs, outer_iter, inner_iter


def test_baseline_class(default_kwargs):
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs)
    mod = STL(**class_kwargs)
    res = mod.fit(outer_iter=outer, inner_iter=inner)

    expected = results.loc["baseline"].sort_index()
    assert_allclose(res.trend, expected.trend)
    assert_allclose(res.seasonal, expected.season)
    assert_allclose(res.weights, expected.rw)
    resid = class_kwargs["endog"] - expected.trend - expected.season
    assert_allclose(res.resid, resid)


def test_short_class(default_kwargs_short):
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs_short)
    mod = STL(**class_kwargs)
    res = mod.fit(outer_iter=outer, inner_iter=inner)

    expected = results.loc["short"].sort_index()
    assert_allclose(res.seasonal, expected.season)
    assert_allclose(res.trend, expected.trend)
    assert_allclose(res.weights, expected.rw)


def test_nljump_1_class(default_kwargs):
    default_kwargs["nljump"] = 1
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs)
    mod = STL(**class_kwargs)
    res = mod.fit(outer_iter=outer, inner_iter=inner)

    expected = results.loc["nljump-1"].sort_index()
    assert_allclose(res.seasonal, expected.season)
    assert_allclose(res.trend, expected.trend)
    assert_allclose(res.weights, expected.rw)


def test_ntjump_1_class(default_kwargs):
    default_kwargs["ntjump"] = 1
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs)
    mod = STL(**class_kwargs)
    res = mod.fit(outer_iter=outer, inner_iter=inner)

    expected = results.loc["ntjump-1"].sort_index()
    assert_allclose(res.seasonal, expected.season)
    assert_allclose(res.trend, expected.trend)
    assert_allclose(res.weights, expected.rw)


def test_nljump_1_ntjump_1_class(default_kwargs):
    default_kwargs["nljump"] = 1
    default_kwargs["ntjump"] = 1
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs)
    mod = STL(**class_kwargs)
    res = mod.fit(outer_iter=outer, inner_iter=inner)

    expected = results.loc["nljump-1-ntjump-1"].sort_index()
    assert_allclose(res.seasonal, expected.season)
    assert_allclose(res.trend, expected.trend)
    assert_allclose(res.weights, expected.rw)


def test_parameter_checks_period(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    endog = class_kwargs["endog"]
    endog2 = np.hstack((endog[:, None], endog[:, None]))
    period = class_kwargs["period"]
    with pytest.raises(ValueError, match="endog is required to have ndim 1"):
        STL(endog=endog2, period=period)
    match = "period must be a positive integer >= 2"
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=1)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=-12)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=4.0)


def test_parameter_checks_seasonal(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    endog = class_kwargs["endog"]
    period = class_kwargs["period"]
    match = "seasonal must be an odd positive integer >= 3"
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, seasonal=2)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, seasonal=-7)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, seasonal=13.0)


def test_parameter_checks_trend(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    endog = class_kwargs["endog"]
    period = class_kwargs["period"]
    match = "trend must be an odd positive integer >= 3 where trend > period"
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, trend=14)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, trend=11)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, trend=-19)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, trend=19.0)


def test_parameter_checks_low_pass(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    endog = class_kwargs["endog"]
    period = class_kwargs["period"]

    match = (
        "low_pass must be an odd positive integer >= 3 where"
        " low_pass > period"
    )
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, low_pass=14)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, low_pass=7)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, low_pass=-19)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, low_pass=19.0)


def test_jump_errors(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    endog = class_kwargs["endog"]
    period = class_kwargs["period"]
    with pytest.raises(ValueError, match="low_pass_jump must be a positive"):
        STL(endog=endog, period=period, low_pass_jump=0)
    with pytest.raises(ValueError, match="low_pass_jump must be a positive"):
        STL(endog=endog, period=period, low_pass_jump=1.0)
    with pytest.raises(ValueError, match="seasonal_jump must be a positive"):
        STL(endog=endog, period=period, seasonal_jump=0)
    with pytest.raises(ValueError, match="seasonal_jump must be a positive"):
        STL(endog=endog, period=period, seasonal_jump=1.0)
    with pytest.raises(ValueError, match="trend_jump must be a positive"):
        STL(endog=endog, period=period, trend_jump=0)
    with pytest.raises(ValueError, match="trend_jump must be a positive"):
        STL(endog=endog, period=period, trend_jump=1.0)


def test_defaults_smoke(default_kwargs, robust):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs, robust)
    endog = class_kwargs["endog"]
    period = class_kwargs["period"]
    mod = STL(endog=endog, period=period)
    mod.fit()


def test_pandas(default_kwargs, robust):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs, robust)
    endog = pd.Series(class_kwargs["endog"], name="y")
    period = class_kwargs["period"]
    mod = STL(endog=endog, period=period)
    res = mod.fit()
    assert isinstance(res.trend, pd.Series)
    assert isinstance(res.seasonal, pd.Series)
    assert isinstance(res.resid, pd.Series)
    assert isinstance(res.weights, pd.Series)


def test_period_detection(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    mod = STL(**class_kwargs)
    res = mod.fit()

    del class_kwargs["period"]
    endog = class_kwargs["endog"]
    index = pd.date_range("1-1-1959", periods=348, freq=MONTH_END)
    class_kwargs["endog"] = pd.Series(endog, index=index)
    mod = STL(**class_kwargs)

    res_implicit_period = mod.fit()
    assert_allclose(res.seasonal, res_implicit_period.seasonal)


def test_no_period(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    del class_kwargs["period"]
    class_kwargs["endog"] = pd.Series(class_kwargs["endog"])
    with pytest.raises(ValueError, match="Unable to determine period from"):
        STL(**class_kwargs)


@pytest.mark.matplotlib
def test_plot(default_kwargs, close_figures):
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs)
    res = STL(**class_kwargs).fit(outer_iter=outer, inner_iter=inner)
    res.plot()

    class_kwargs["endog"] = pd.Series(class_kwargs["endog"], name="CO2")
    res = STL(**class_kwargs).fit()
    res.plot()


def estimate(y_, x, X_):
    X = X_[~np.isnan(y_)]
    y = y_[~np.isnan(y_)]
    x0, xmin, xmax = X_[x, 1], X_[0, 1], X_[-1, 1]
    h = max(x0-xmin, xmax-x0)
    wi = [(1-(np.abs(xi-x0)/h)**3)**3 for xi in X[:, 1].A1]
    W = np.diag(wi)
    inv = X.T.dot(W).dot(X).I
    H = X_[x].dot(inv).dot(X.T)
    return H.dot(W).dot(y).item()


def test_est(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    res = STL(**class_kwargs)

    y = np.array([1, 2, 4, 8, 16])
    X = np.matrix([np.ones(5), [0, 1, 2, 3, 4]]).T
    ys_expect = [estimate(y, xs, X) for xs in range(5)]
    ys = [res._estimate(y, xs, 0, 5) for xs in range(5)]
    assert_allclose(ys, ys_expect)


def test_est_nans(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    res = STL(**class_kwargs)

    y = np.array([1, 2, 4, np.nan, 16, 32])
    X = np.matrix([np.ones(6), [0, 1, 2, 3, 4, 5]]).T
    ys_expect = np.array([estimate(y, xs, X) for xs in range(6)])
    ys = [res._estimate(y, xs, 0, 6) for xs in range(6)]
    assert_allclose(ys, ys_expect)


def test_get_maxmin_0_nan(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    res = STL(**class_kwargs)

    X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
         20, 21, 22, 23, 24, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tests = [
        (1, 1, 5), (2, 1, 5), (3, 1, 5), (4, 2, 6),
        (11, 9, 13), (12, 10, 14), (13, 11, 15), (14, 12, 16), (15, 13, 17),
        (22, 20, 24), (23, 21, 25), (24, 21, 25), (25, 21, 25), (26, 21, 25)
    ]

    for (xs, xmin_exp, xmax_exp) in tests:
        xmin, xmax = res._get_maxmin(X, 25, xs, 5)
        assert (xmin, xmax) == (xmin_exp, xmax_exp)

    xmin, xmax = res._get_maxmin(X, 25, 1, 35)
    assert (xmin, xmax) == (1, 25)


def test_get_maxmin_1_nan(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    res = STL(**class_kwargs)

    X = [1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10, 11, np.nan, 13, 14, 15, 16, 17,
         18, 19, 20, np.nan, 22, 23, 24, 25, 0, 0, 0]
    tests = [
        (1, 1, 6), (2, 1, 6), (3, 1, 6), (4, 2, 7), (5, 2, 7),
        (11, 9, 14), (12, 10, 15), (13, 10, 15), (14, 11, 16), (15, 13, 17),
        (25, 20, 25), (24, 20, 25), (23, 20, 25), (22, 19, 24), (21, 19, 24)
    ]

    for (xs, xmin_exp, xmax_exp) in tests:
        xmin, xmax = res._get_maxmin(X, 25, xs, 5)
        assert (xmin, xmax) == (xmin_exp, xmax_exp)


def test_get_maxmin_more_nan(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    res = STL(**class_kwargs)

    X = [1, 2, 3, np.nan, np.nan, 6, 7, 8, 9, np.nan, 11, np.nan, 13, 14,
         np.nan, np.nan, 17, np.nan, 19, 20, np.nan, 22, 23, 24, np.nan]
    tests = [
        (-1, 1, 7), (0, 1, 7),
        (1, 1, 7), (2, 1, 7), (3, 1, 7), (4, 1, 7), (5, 2, 8),
        (6, 3, 9), (7, 3, 9), (8, 6, 11), (9, 6, 11), (10, 7, 13),
        (11, 8, 14), (12, 8, 14), (13, 9, 17), (14, 11, 19), (15, 11, 19),
        (16, 13, 20), (17, 13, 20), (18, 14, 22), (19, 17, 23), (20, 17, 23),
        (21, 19, 24), (22, 19, 24), (23, 19, 24), (24, 19, 24), (25, 19, 24),
        (26, 19, 24), (27, 19, 24)
    ]

    for (xs, xmin_exp, xmax_exp) in tests:
        xmin, xmax = res._get_maxmin(X, 25, xs, 5)
        assert (xmin, xmax) == (xmin_exp, xmax_exp)


def test_endog_with_nulls(default_kwargs):
    class_kwargs, inner, outer = _to_class_kwargs(default_kwargs)

    len_ = class_kwargs['trend']
    Y = class_kwargs['endog']
    for i in [2, 11, 16, 90, 104, 107, 124, 125, 174, 216, 287, 340, ]:
        Y[i] = np.nan

    mod = STL(**class_kwargs)

    nsh = (len_ + 2) // 2
    nleft = 0
    nright = len_
    res = []
    for i in range(len(Y)):
        if (i + 1) > nsh and nright != len(Y):
            nleft += 1
            nright += 1
        res.append(mod._estimate(Y, i, nleft, nright))
    assert not any(np.isnan(res))


def test_decomp_with_nulls(default_kwargs):
    class_kwargs, outer_iter, inner_iter = _to_class_kwargs(default_kwargs)

    for i in [2, 11, 16, 90, 104, 107, 124, 125, 174, 216, 287, 340, ]:
        class_kwargs['endog'][i] = np.nan

    mod = STL(**class_kwargs)
    res = mod.fit(inner_iter=5, outer_iter=0)

    assert not any(np.isnan(res.seasonal))


def test_decomp_with_nulls_is_close(default_kwargs):
    class_kwargs, outer_iter, inner_iter = _to_class_kwargs(default_kwargs)

    res = STL(**class_kwargs).fit(inner_iter, outer_iter)
    for i in [2, 11, 16, 90, 104, 107, 124, 125, 174, 216, 287, 340, ]:
        class_kwargs['endog'][i] = np.nan

    res_nulls = STL(**class_kwargs).fit(inner_iter, outer_iter)

    # Because the input series are not equal, we only expect the output
    # decomposition to be roughly equal. Here we test whether at each
    # point, `x1 > 0.1 + (1.001 * x2)` and the other way around. Because
    # res.seasonal has some values very close to 0, we need atol as well.
    assert_allclose(res.seasonal, res_nulls.seasonal, rtol=0.001, atol=0.1)
    assert_allclose(res.trend, res_nulls.trend, rtol=0.001)


# Values produced by R standard function STL().
def test_r_is_close(default_kwargs):
    r_res_path = os.path.join(cur_dir, "results", "stl_co2_r_result.csv")
    r_res = pd.read_csv(r_res_path, sep=r'\s+', header=6,
                        names=['m', 'y', 'seasonal', 'trend', 'remainder'],
                        usecols=['seasonal', 'trend', 'remainder'])
    class_kwargs, outer_iter, inner_iter = _to_class_kwargs(default_kwargs)
    res = STL(**class_kwargs).fit(inner_iter=inner_iter, outer_iter=outer_iter)

    # Values produced by R are very close but don't match exactly.
    assert_allclose(res.seasonal, r_res['seasonal'], rtol=1e-6, atol=1e-5)
    assert_allclose(res.trend, r_res['trend'], rtol=1e-6, atol=1e-5)


# The R stlplus package performs STL decomposition allowing missing values.
def test_r_stlplus_is_close(default_kwargs):
    r_res_path = os.path.join(cur_dir, "results", "stl_co2_weekly_r_result.csv")

    r_res = pd.read_csv(r_res_path, sep=r'\s+', header=1,
                        nrows=2276,
                        names=['rowno', 'raw', 'seasonal', 'trend', 'resid',
                               'weights', 'ss', 'subseries'])

    data = (co2.load().data
            .pipe(lambda df: df[df.index.isocalendar().week < 53])
            )
    res = STL(data, period=52, seasonal=35).fit()

    # Values produced by R are very close but don't match exactly.
    # Note that for seasonal, we are using rtol 10 times bigger.
    assert_allclose(res.seasonal, r_res['seasonal'], rtol=1e-5, atol=1e-5)
    assert_allclose(res.trend, r_res['trend'], rtol=1e-6, atol=1e-5)


def test_default_trend(default_kwargs):
    # GH 6686
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    class_kwargs["seasonal"] = 17
    class_kwargs["trend"] = None
    mod = STL(**class_kwargs)
    period = class_kwargs["period"]
    seasonal = class_kwargs["seasonal"]
    expected = int(np.ceil(1.5 * period / (1 - 1.5 / seasonal)))
    expected += 1 if expected % 2 == 0 else 0
    assert mod.config["trend"] == expected

    class_kwargs["seasonal"] = 7
    mod = STL(**class_kwargs)
    period = class_kwargs["period"]
    seasonal = class_kwargs["seasonal"]
    expected = int(np.ceil(1.5 * period / (1 - 1.5 / seasonal)))
    expected += 1 if expected % 2 == 0 else 0
    assert mod.config["trend"] == expected


def test_pickle(default_kwargs):
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs)
    mod = STL(**class_kwargs)
    res = mod.fit()
    pkl = pickle.dumps(mod)
    reloaded = pickle.loads(pkl)
    res2 = reloaded.fit()
    assert_allclose(res.trend, res2.trend)
    assert_allclose(res.seasonal, res2.seasonal)
    assert mod.config == reloaded.config


def test_squezable_to_1d():
    data = co2.load().data
    data = data.resample(MONTH_END).mean().ffill()
    res = STL(data).fit()
    assert isinstance(res, DecomposeResult)
