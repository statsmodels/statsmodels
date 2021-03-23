"""
Test AR Model
"""
from itertools import product

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
from pandas import Index, Series, date_range, period_range
from pandas.testing import assert_series_equal
import pytest

import statsmodels.api as sm
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.tools import Bunch
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.deterministic import (
    DeterministicProcess,
    Seasonality,
    TimeTrend,
)
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.tests.results import results_ar

DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4


def gen_ar_data(nobs):
    rs = np.random.RandomState(982739)
    idx = pd.date_range("1-1-1900", freq="M", periods=nobs)
    return pd.Series(rs.standard_normal(nobs), index=idx), rs


def gen_ols_regressors(ar, seasonal, trend, exog):
    nobs = 500
    y, rs = gen_ar_data(nobs)
    maxlag = ar if isinstance(ar, int) else max(ar)
    reg = []
    if "c" in trend:
        const = pd.Series(np.ones(nobs), index=y.index, name="const")
        reg.append(const)
    if "t" in trend:
        time = np.arange(1, nobs + 1)
        time = pd.Series(time, index=y.index, name="time")
        reg.append(time)
    if isinstance(ar, int) and ar:
        lags = np.arange(1, ar + 1)
    elif ar == 0:
        lags = None
    else:
        lags = ar
    if seasonal:
        seasons = np.zeros((500, 12))
        for i in range(12):
            seasons[i::12, i] = 1
        cols = ["s.{0}".format(i) for i in range(12)]
        seasons = pd.DataFrame(seasons, columns=cols, index=y.index)
        if "c" in trend:
            seasons = seasons.iloc[:, 1:]
        reg.append(seasons)
    if maxlag:
        for lag in lags:
            reg.append(y.shift(lag))
    if exog:
        x = rs.standard_normal((nobs, exog))
        cols = ["x.{0}".format(i) for i in range(exog)]
        x = pd.DataFrame(x, columns=cols, index=y.index)
        reg.append(x)
    else:
        x = None
    reg.insert(0, y)
    df = pd.concat(reg, 1).dropna()
    endog = df.iloc[:, 0]
    exog = df.iloc[:, 1:]
    return y, x, endog, exog


ar = [0, 3, [1, 3], [3]]
seasonal = [True, False]
trend = ["n", "c", "t", "ct"]
exog = [None, 2]
covs = ["nonrobust", "HC0"]
params = list(product(ar, seasonal, trend, exog, covs))
final = []
for param in params:
    if param[0] != 0 or param[1] or param[2] != "n" or param[3]:
        final.append(param)
params = final
names = ("AR", "Seasonal", "Trend", "Exog", "Cov Type")
ids = [
    ", ".join([n + ": " + str(p) for n, p in zip(names, param)])
    for param in params
]


@pytest.fixture(scope="module", params=params, ids=ids)
def ols_autoreg_result(request):
    ar, seasonal, trend, exog, cov_type = request.param
    y, x, endog, exog = gen_ols_regressors(ar, seasonal, trend, exog)
    ar_mod = AutoReg(
        y, ar, seasonal=seasonal, trend=trend, exog=x, old_names=False
    )
    ar_res = ar_mod.fit(cov_type=cov_type)
    ols = OLS(endog, exog)
    ols_res = ols.fit(cov_type=cov_type, use_t=False)
    return ar_res, ols_res


attributes = [
    "bse",
    "cov_params",
    "df_model",
    "df_resid",
    "fittedvalues",
    "llf",
    "nobs",
    "params",
    "resid",
    "scale",
    "tvalues",
    "use_t",
]


def fix_ols_attribute(val, attrib, res):
    """
    fixes to correct for df adjustment b/t OLS and AutoReg with nonrobust cov
    """
    nparam = res.k_constant + res.df_model
    nobs = nparam + res.df_resid
    df_correction = (nobs - nparam) / nobs
    if attrib in ("scale",):
        return val * df_correction
    elif attrib == "df_model":
        return val + res.k_constant
    elif res.cov_type != "nonrobust":
        return val
    elif attrib in ("bse", "conf_int"):
        return val * np.sqrt(df_correction)
    elif attrib in ("cov_params", "scale"):
        return val * df_correction
    elif attrib in ("f_test",):
        return val / df_correction
    elif attrib in ("tvalues",):
        return val / np.sqrt(df_correction)

    return val


@pytest.mark.parametrize("attribute", attributes)
def test_equiv_ols_autoreg(ols_autoreg_result, attribute):
    a, o = ols_autoreg_result
    ols_a = getattr(o, attribute)
    ar_a = getattr(a, attribute)
    if callable(ols_a):
        ols_a = ols_a()
        ar_a = ar_a()
    ols_a = fix_ols_attribute(ols_a, attribute, o)
    assert_allclose(ols_a, ar_a)


def test_conf_int_ols_autoreg(ols_autoreg_result):
    a, o = ols_autoreg_result
    a_ci = a.conf_int()
    o_ci = o.conf_int()
    if o.cov_type == "nonrobust":
        spread = o_ci.T - o.params
        spread = fix_ols_attribute(spread, "conf_int", o)
        o_ci = (spread + o.params).T

    assert_allclose(a_ci, o_ci)


def test_f_test_ols_autoreg(ols_autoreg_result):
    a, o = ols_autoreg_result
    r = np.eye(a.params.shape[0])
    a_f = a.f_test(r).fvalue
    o_f = o.f_test(r).fvalue
    o_f = fix_ols_attribute(o_f, "f_test", o)

    assert_allclose(a_f, o_f)


@pytest.mark.smoke
def test_other_tests_autoreg(ols_autoreg_result):
    a, _ = ols_autoreg_result
    r = np.ones_like(a.params)
    a.t_test(r)
    r = np.eye(a.params.shape[0])
    a.wald_test(r)


# TODO: test likelihood for ARX model?


class CheckARMixin(object):
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_6)

    def test_bse(self):
        bse = np.sqrt(np.diag(self.res1.cov_params()))
        # no dof correction for compatability with Stata
        assert_almost_equal(bse, self.res2.bse_stata, DECIMAL_6)
        assert_almost_equal(self.res1.bse, self.res2.bse_gretl, DECIMAL_5)

    def test_llf(self):
        assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL_6)

    def test_fpe(self):
        assert_almost_equal(self.res1.fpe, self.res2.fpe, DECIMAL_6)

    def test_pickle(self):
        from io import BytesIO

        fh = BytesIO()
        # test wrapped results load save pickle
        self.res1.save(fh)
        fh.seek(0, 0)
        res_unpickled = self.res1.__class__.load(fh)
        assert type(res_unpickled) is type(self.res1)  # noqa: E721

    @pytest.mark.smoke
    def test_summary(self):
        assert isinstance(self.res1.summary().as_text(), str)

    @pytest.mark.smoke
    def test_pvalues(self):
        assert isinstance(self.res1.pvalues, (np.ndarray, pd.Series))


params = product(
    [0, 1, 3, [1, 3]],
    ["n", "c", "t", "ct"],
    [True, False],
    [0, 2],
    [None, 11],
    ["none", "drop"],
    [True, False],
    [None, 12],
)
params = list(params)
params = [
    param
    for param in params
    if (param[0] or param[1] != "n" or param[2] or param[3])
]
params = [
    param
    for param in params
    if not param[2] or (param[2] and (param[4] or param[6]))
]
param_fmt = """\
lags: {0}, trend: {1}, seasonal: {2}, nexog: {3}, periods: {4}, \
missing: {5}, pandas: {6}, hold_back{7}"""

ids = [param_fmt.format(*param) for param in params]


def gen_data(nobs, nexog, pandas, seed=92874765):
    rs = np.random.RandomState(seed)
    endog = rs.standard_normal((nobs))
    exog = rs.standard_normal((nobs, nexog)) if nexog else None
    if pandas:
        index = pd.date_range("31-12-1999", periods=nobs, freq="M")
        endog = pd.Series(endog, name="endog", index=index)
        if nexog:
            cols = ["exog.{0}".format(i) for i in range(exog.shape[1])]
            exog = pd.DataFrame(exog, columns=cols, index=index)
    from collections import namedtuple

    DataSet = namedtuple("DataSet", ["endog", "exog"])
    return DataSet(endog=endog, exog=exog)


@pytest.fixture(scope="module", params=params, ids=ids)
def ar_data(request):
    lags, trend, seasonal = request.param[:3]
    nexog, period, missing, use_pandas, hold_back = request.param[3:]
    data = gen_data(250, nexog, use_pandas)
    return Bunch(
        trend=trend,
        lags=lags,
        seasonal=seasonal,
        period=period,
        endog=data.endog,
        exog=data.exog,
        missing=missing,
        hold_back=hold_back,
    )


params = product(
    [0, 3, [1, 3]],
    ["c"],
    [True, False],
    [0],
    [None, 11],
    ["drop"],
    [True, False],
    [None, 12],
)
params = list(params)
params = [
    param
    for param in params
    if (param[0] or param[1] != "n" or param[2] or param[3])
]
params = [
    param
    for param in params
    if not param[2] or (param[2] and (param[4] or param[6]))
]
param_fmt = """\
lags: {0}, trend: {1}, seasonal: {2}, nexog: {3}, periods: {4}, \
missing: {5}, pandas: {6}, hold_back: {7}"""

ids = [param_fmt.format(*param) for param in params]


# Only test 1/3 to save time
@pytest.fixture(scope="module", params=params[::3], ids=ids[::3])
def plot_data(request):
    lags, trend, seasonal = request.param[:3]
    nexog, period, missing, use_pandas, hold_back = request.param[3:]
    data = gen_data(250, nexog, use_pandas)
    return Bunch(
        trend=trend,
        lags=lags,
        seasonal=seasonal,
        period=period,
        endog=data.endog,
        exog=data.exog,
        missing=missing,
        hold_back=hold_back,
    )


@pytest.mark.matplotlib
@pytest.mark.smoke
def test_autoreg_smoke_plots(plot_data, close_figures):
    from matplotlib.figure import Figure

    mod = AutoReg(
        plot_data.endog,
        plot_data.lags,
        trend=plot_data.trend,
        seasonal=plot_data.seasonal,
        exog=plot_data.exog,
        hold_back=plot_data.hold_back,
        period=plot_data.period,
        missing=plot_data.missing,
        old_names=False,
    )
    res = mod.fit()
    fig = res.plot_diagnostics()
    assert isinstance(fig, Figure)
    if plot_data.exog is None:
        fig = res.plot_predict(end=300)
        assert isinstance(fig, Figure)
        fig = res.plot_predict(end=300, alpha=None, in_sample=False)
        assert isinstance(fig, Figure)
    assert isinstance(res.summary(), Summary)


@pytest.mark.smoke
def test_autoreg_predict_smoke(ar_data):
    mod = AutoReg(
        ar_data.endog,
        ar_data.lags,
        trend=ar_data.trend,
        seasonal=ar_data.seasonal,
        exog=ar_data.exog,
        hold_back=ar_data.hold_back,
        period=ar_data.period,
        missing=ar_data.missing,
        old_names=False,
    )
    res = mod.fit()
    exog_oos = None
    if ar_data.exog is not None:
        exog_oos = np.empty((1, ar_data.exog.shape[1]))
    mod.predict(res.params, 0, 250, exog_oos=exog_oos)
    if ar_data.lags == 0 and ar_data.exog is None:
        mod.predict(res.params, 0, 350, exog_oos=exog_oos)
    if isinstance(ar_data.endog, pd.Series) and (
        not ar_data.seasonal or ar_data.period is not None
    ):
        ar_data.endog.index = list(range(ar_data.endog.shape[0]))
        if ar_data.exog is not None:
            ar_data.exog.index = list(range(ar_data.endog.shape[0]))
        mod = AutoReg(
            ar_data.endog,
            ar_data.lags,
            trend=ar_data.trend,
            seasonal=ar_data.seasonal,
            exog=ar_data.exog,
            period=ar_data.period,
            missing=ar_data.missing,
            old_names=False,
        )
        mod.predict(res.params, 0, 250, exog_oos=exog_oos)


@pytest.mark.matplotlib
def test_parameterless_autoreg():
    data = gen_data(250, 0, False)
    mod = AutoReg(
        data.endog, 0, trend="n", seasonal=False, exog=None, old_names=False
    )
    res = mod.fit()
    for attr in dir(res):
        if attr.startswith("_"):
            continue

        # TODO
        if attr in (
            "predict",
            "f_test",
            "t_test",
            "initialize",
            "load",
            "remove_data",
            "save",
            "t_test",
            "t_test_pairwise",
            "wald_test",
            "wald_test_terms",
        ):
            continue
        warning = None if attr != "test_serial_correlation" else FutureWarning
        attr = getattr(res, attr)
        if callable(attr):
            with pytest.warns(warning):
                attr()
        else:
            assert isinstance(attr, object)


def test_predict_errors():
    data = gen_data(250, 2, True)
    mod = AutoReg(data.endog, 3, old_names=False)
    res = mod.fit()
    with pytest.raises(ValueError, match="exog and exog_oos cannot be used"):
        mod.predict(res.params, exog=data.exog)
    with pytest.raises(ValueError, match="exog and exog_oos cannot be used"):
        mod.predict(res.params, exog_oos=data.exog)
    with pytest.raises(ValueError, match="hold_back must be >= lags"):
        AutoReg(data.endog, 3, hold_back=1, old_names=False)
    with pytest.raises(ValueError, match="freq cannot be inferred"):
        AutoReg(data.endog.values, 3, seasonal=True, old_names=False)

    mod = AutoReg(data.endog, 3, exog=data.exog, old_names=False)
    res = mod.fit()
    with pytest.raises(ValueError, match=r"The shape of exog \(200, 2\)"):
        mod.predict(res.params, exog=data.exog.iloc[:200])
    with pytest.raises(ValueError, match="The number of columns in exog_oos"):
        mod.predict(res.params, exog_oos=data.exog.iloc[:, :1])
    with pytest.raises(ValueError, match="Prediction must have `end` after"):
        mod.predict(res.params, start=200, end=199)
    with pytest.raises(ValueError, match="exog_oos must be provided"):
        mod.predict(res.params, end=250, exog_oos=None)

    mod = AutoReg(data.endog, 0, exog=data.exog, old_names=False)
    res = mod.fit()
    with pytest.raises(ValueError, match="start and end indicate that 10"):
        mod.predict(res.params, end=259, exog_oos=data.exog.iloc[:5])


def test_spec_errors():
    data = gen_data(250, 2, True)
    with pytest.raises(ValueError, match="lags must be a positive scalar"):
        AutoReg(data.endog, -1, old_names=False)
    with pytest.raises(ValueError, match="All values in lags must be pos"):
        AutoReg(data.endog, [1, 1, 1], old_names=False)
    with pytest.raises(ValueError, match="All values in lags must be pos"):
        AutoReg(data.endog, [1, -2, 3], old_names=False)


@pytest.mark.smoke
def test_dynamic_forecast_smoke(ar_data):
    mod = AutoReg(
        ar_data.endog,
        ar_data.lags,
        trend=ar_data.trend,
        seasonal=ar_data.seasonal,
        exog=ar_data.exog,
        hold_back=ar_data.hold_back,
        period=ar_data.period,
        missing=ar_data.missing,
        old_names=False,
    )
    res = mod.fit()
    res.predict(dynamic=True)
    if ar_data.exog is None:
        res.predict(end=260, dynamic=True)


@pytest.mark.smoke
def test_ar_select_order_smoke():
    data = sm.datasets.sunspots.load(as_pandas=True).data["SUNACTIVITY"]
    ar_select_order(data, 4, glob=True, trend="n", old_names=False)
    ar_select_order(data, 4, glob=False, trend="n", old_names=False)
    ar_select_order(data, 4, seasonal=True, period=12, old_names=False)
    ar_select_order(data, 4, seasonal=False, old_names=False)
    ar_select_order(data, 4, glob=True, old_names=False)
    ar_select_order(
        data, 4, glob=True, seasonal=True, period=12, old_names=False
    )


class CheckAutoRegMixin(CheckARMixin):
    def test_bse(self):
        assert_almost_equal(self.res1.bse, self.res2.bse_stata, DECIMAL_6)


class TestAutoRegOLSConstant(CheckAutoRegMixin):
    """
    Test AutoReg fit by OLS with a constant.
    """

    @classmethod
    def setup_class(cls):
        data = sm.datasets.sunspots.load(as_pandas=True)
        data.endog.index = list(range(len(data.endog)))
        cls.res1 = AutoReg(data.endog, lags=9, old_names=False).fit()
        cls.res2 = results_ar.ARResultsOLS(constant=True)

    def test_predict(self):
        model = self.res1.model
        params = self.res1.params
        assert_almost_equal(
            model.predict(params), self.res2.FVOLSnneg1start0, DECIMAL_4
        )
        assert_almost_equal(
            model.predict(params), self.res2.FVOLSnneg1start9, DECIMAL_4
        )
        assert_almost_equal(
            model.predict(params, start=100),
            self.res2.FVOLSnneg1start100,
            DECIMAL_4,
        )
        assert_almost_equal(
            model.predict(params, start=9, end=200),
            self.res2.FVOLSn200start0,
            DECIMAL_4,
        )
        assert_almost_equal(
            model.predict(params), self.res2.FVOLSdefault, DECIMAL_4
        )
        assert_almost_equal(
            model.predict(params, start=200, end=400),
            self.res2.FVOLSn200start200,
            DECIMAL_4,
        )
        assert_almost_equal(
            model.predict(params, start=308, end=424),
            self.res2.FVOLSn100start325,
            DECIMAL_4,
        )
        assert_almost_equal(
            model.predict(params, start=9, end=310),
            self.res2.FVOLSn301start9,
            DECIMAL_4,
        )
        assert_almost_equal(
            model.predict(params, start=308, end=316),
            self.res2.FVOLSn4start312,
            DECIMAL_4,
        )
        assert_almost_equal(
            model.predict(params, start=308, end=327),
            self.res2.FVOLSn15start312,
            DECIMAL_4,
        )


class TestAutoRegOLSNoConstant(CheckAutoRegMixin):
    """f
    Test AR fit by OLS without a constant.
    """

    @classmethod
    def setup_class(cls):
        data = sm.datasets.sunspots.load(as_pandas=False)
        cls.res1 = AutoReg(
            data.endog, lags=9, trend="n", old_names=False
        ).fit()
        cls.res2 = results_ar.ARResultsOLS(constant=False)

    def test_predict(self):
        model = self.res1.model
        params = self.res1.params
        assert_almost_equal(
            model.predict(params), self.res2.FVOLSnneg1start0, DECIMAL_4
        )
        assert_almost_equal(
            model.predict(params), self.res2.FVOLSnneg1start9, DECIMAL_4
        )
        assert_almost_equal(
            model.predict(params, start=100),
            self.res2.FVOLSnneg1start100,
            DECIMAL_4,
        )
        assert_almost_equal(
            model.predict(params, start=9, end=200),
            self.res2.FVOLSn200start0,
            DECIMAL_4,
        )
        assert_almost_equal(
            model.predict(params), self.res2.FVOLSdefault, DECIMAL_4
        )
        assert_almost_equal(
            model.predict(params, start=200, end=400),
            self.res2.FVOLSn200start200,
            DECIMAL_4,
        )
        assert_almost_equal(
            model.predict(params, start=308, end=424),
            self.res2.FVOLSn100start325,
            DECIMAL_4,
        )
        assert_almost_equal(
            model.predict(params, start=9, end=310),
            self.res2.FVOLSn301start9,
            DECIMAL_4,
        )
        assert_almost_equal(
            model.predict(params, start=308, end=316),
            self.res2.FVOLSn4start312,
            DECIMAL_4,
        )
        assert_almost_equal(
            model.predict(params, start=308, end=327),
            self.res2.FVOLSn15start312,
            DECIMAL_4,
        )


@pytest.mark.parametrize("lag", list(np.arange(1, 16 + 1)))
def test_autoreg_info_criterion(lag):
    data = sm.datasets.sunspots.load(as_pandas=False)
    endog = data.endog
    endog_tmp = endog[16 - lag :]
    r = AutoReg(endog_tmp, lags=lag, old_names=False).fit()
    # See issue #324 for the corrections vs. R
    k_ar = len(r.model.ar_lags)
    k_trend = 1
    log_sigma2 = np.log(r.sigma2)
    aic = r.aic
    aic = (aic - log_sigma2) * (1 + k_ar) / (1 + k_ar + k_trend)
    aic += log_sigma2

    hqic = r.hqic
    hqic = (hqic - log_sigma2) * (1 + k_ar) / (1 + k_ar + k_trend)
    hqic += log_sigma2

    bic = r.bic
    bic = (bic - log_sigma2) * (1 + k_ar) / (1 + k_ar + k_trend)
    bic += log_sigma2

    res1 = np.array([aic, hqic, bic, r.fpe])
    # aic correction to match R
    res2 = results_ar.ARLagResults("const").ic.T

    assert_almost_equal(res1, res2[lag - 1, :], DECIMAL_6)

    r2 = AutoReg(endog, lags=lag, hold_back=16, old_names=False).fit()
    assert_allclose(r.aic, r2.aic)
    assert_allclose(r.bic, r2.bic)
    assert_allclose(r.hqic, r2.hqic)
    assert_allclose(r.fpe, r2.fpe)


@pytest.mark.parametrize("old_names", [True, False])
def test_autoreg_named_series(reset_randomstate, old_names):
    dates = period_range(start="2011-1", periods=72, freq="M")
    y = Series(np.random.randn(72), name="foobar", index=dates)
    results = AutoReg(y, lags=2, old_names=old_names).fit()
    if old_names:
        idx = Index(["intercept", "foobar.L1", "foobar.L2"])
    else:
        idx = Index(["const", "foobar.L1", "foobar.L2"])
    assert results.params.index.equals(idx)


@pytest.mark.smoke
def test_autoreg_series():
    # GH#773
    dta = sm.datasets.macrodata.load_pandas().data["cpi"].diff().dropna()
    dates = period_range(start="1959Q1", periods=len(dta), freq="Q")
    dta.index = dates
    ar = AutoReg(dta, lags=15, old_names=False).fit()
    ar.bse


def test_ar_order_select():
    # GH#2118
    np.random.seed(12345)
    y = sm.tsa.arma_generate_sample([1, -0.75, 0.3], [1], 100)
    ts = Series(y, index=date_range(start="1/1/1990", periods=100, freq="M"))
    res = ar_select_order(ts, maxlag=12, ic="aic", old_names=False)
    assert tuple(res.ar_lags) == (1, 2)
    assert isinstance(res.aic, dict)
    assert isinstance(res.bic, dict)
    assert isinstance(res.hqic, dict)
    assert isinstance(res.model, AutoReg)
    assert not res.seasonal
    assert res.trend == "c"
    assert res.period is None


def test_autoreg_constant_column_trend():
    sample = np.array(
        [
            0.46341460943222046,
            0.46341460943222046,
            0.39024388790130615,
            0.4146341383457184,
            0.4146341383457184,
            0.4146341383457184,
            0.3414634168148041,
            0.4390243887901306,
            0.46341460943222046,
            0.4390243887901306,
        ]
    )

    with pytest.raises(ValueError, match="The model specification cannot"):
        AutoReg(sample, lags=7, old_names=False)
    with pytest.raises(ValueError, match="The model specification cannot"):
        AutoReg(sample, lags=7, trend="n", old_names=False)


@pytest.mark.parametrize("old_names", [True, False])
def test_autoreg_summary_corner(old_names):
    data = sm.datasets.macrodata.load_pandas().data["cpi"].diff().dropna()
    dates = period_range(start="1959Q1", periods=len(data), freq="Q")
    data.index = dates
    res = AutoReg(data, lags=4, old_names=old_names).fit()
    summ = res.summary().as_text()
    assert "AutoReg(4)" in summ
    assert "cpi.L4" in summ
    assert "03-31-1960" in summ
    res = AutoReg(data, lags=0, old_names=old_names).fit()
    summ = res.summary().as_text()
    if old_names:
        assert "intercept" in summ
    else:
        assert "const" in summ
    assert "AutoReg(0)" in summ


@pytest.mark.smoke
def test_autoreg_score():
    data = sm.datasets.sunspots.load_pandas()
    ar = AutoReg(np.asarray(data.endog), 3, old_names=False)
    res = ar.fit()
    score = ar.score(res.params)
    assert isinstance(score, np.ndarray)
    assert score.shape == (4,)
    assert ar.information(res.params).shape == (4, 4)


def test_autoreg_roots():
    data = sm.datasets.sunspots.load_pandas()
    ar = AutoReg(np.asarray(data.endog), lags=1, old_names=False)
    res = ar.fit()
    assert_almost_equal(res.roots, np.array([1.0 / res.params[-1]]))


def test_equiv_dynamic(reset_randomstate):
    e = np.random.standard_normal(1001)
    y = np.empty(1001)
    y[0] = e[0] * np.sqrt(1.0 / (1 - 0.9 ** 2))
    for i in range(1, 1001):
        y[i] = 0.9 * y[i - 1] + e[i]
    mod = AutoReg(y, 1, old_names=False)
    res = mod.fit()
    pred0 = res.predict(500, 800, dynamic=0)
    pred1 = res.predict(500, 800, dynamic=True)
    idx = pd.date_range("31-01-2000", periods=1001, freq="M")
    y = pd.Series(y, index=idx)
    mod = AutoReg(y, 1, old_names=False)
    res = mod.fit()
    pred2 = res.predict(idx[500], idx[800], dynamic=idx[500])
    pred3 = res.predict(idx[500], idx[800], dynamic=0)
    pred4 = res.predict(idx[500], idx[800], dynamic=True)
    assert_allclose(pred0, pred1)
    assert_allclose(pred0, pred2)
    assert_allclose(pred0, pred3)
    assert_allclose(pred0, pred4)


def test_dynamic_against_sarimax():
    rs = np.random.RandomState(12345678)
    e = rs.standard_normal(1001)
    y = np.empty(1001)
    y[0] = e[0] * np.sqrt(1.0 / (1 - 0.9 ** 2))
    for i in range(1, 1001):
        y[i] = 0.9 * y[i - 1] + e[i]
    smod = SARIMAX(y, order=(1, 0, 0), trend="c")
    sres = smod.fit(disp=False)
    mod = AutoReg(y, 1, old_names=False)
    spred = sres.predict(900, 1100)
    pred = mod.predict(sres.params[:2], 900, 1100)
    assert_allclose(spred, pred)

    spred = sres.predict(900, 1100, dynamic=True)
    pred = mod.predict(sres.params[:2], 900, 1100, dynamic=True)
    assert_allclose(spred, pred)

    spred = sres.predict(900, 1100, dynamic=50)
    pred = mod.predict(sres.params[:2], 900, 1100, dynamic=50)
    assert_allclose(spred, pred)


def test_predict_seasonal():
    rs = np.random.RandomState(12345678)
    e = rs.standard_normal(1001)
    y = np.empty(1001)
    y[0] = e[0] * np.sqrt(1.0 / (1 - 0.9 ** 2))
    effects = 10 * np.cos(np.arange(12) / 11 * 2 * np.pi)
    for i in range(1, 1001):
        y[i] = 10 + 0.9 * y[i - 1] + e[i] + effects[i % 12]
    ys = pd.Series(y, index=pd.date_range("1-1-1950", periods=1001, freq="M"))
    mod = AutoReg(ys, 1, seasonal=True, old_names=False)
    res = mod.fit()
    c = res.params.iloc[0]
    seasons = np.zeros(12)
    seasons[1:] = res.params.iloc[1:-1]
    ar = res.params.iloc[-1]
    pred = res.predict(900, 1100, True)
    direct = np.zeros(201)
    direct[0] = y[899] * ar + c + seasons[900 % 12]
    for i in range(1, 201):
        direct[i] = direct[i - 1] * ar + c + seasons[(900 + i) % 12]
    direct = pd.Series(
        direct, index=pd.date_range(ys.index[900], periods=201, freq="M")
    )
    assert_series_equal(pred, direct)

    pred = res.predict(900, dynamic=False)
    direct = y[899:-1] * ar + c + seasons[np.arange(900, 1001) % 12]
    direct = pd.Series(
        direct, index=pd.date_range(ys.index[900], periods=101, freq="M")
    )
    assert_series_equal(pred, direct)


def test_predict_exog():
    rs = np.random.RandomState(12345678)
    e = rs.standard_normal(1001)
    y = np.empty(1001)
    x = rs.standard_normal((1001, 2))
    y[:3] = e[:3] * np.sqrt(1.0 / (1 - 0.9 ** 2)) + x[:3].sum(1)
    for i in range(3, 1001):
        y[i] = 10 + 0.9 * y[i - 1] - 0.5 * y[i - 3] + e[i] + x[i].sum()
    ys = pd.Series(y, index=pd.date_range("1-1-1950", periods=1001, freq="M"))
    xdf = pd.DataFrame(x, columns=["x0", "x1"], index=ys.index)
    mod = AutoReg(ys, [1, 3], trend="c", exog=xdf, old_names=False)
    res = mod.fit()

    pred = res.predict(900)
    c = res.params.iloc[0]
    ar = res.params.iloc[1:3]
    ex = np.asarray(res.params.iloc[3:])
    direct = c + ar[0] * y[899:-1] + ar[1] * y[897:-3]
    direct += ex[0] * x[900:, 0] + ex[1] * x[900:, 1]
    idx = pd.date_range(ys.index[900], periods=101, freq="M")
    direct = pd.Series(direct, index=idx)
    assert_series_equal(pred, direct)
    exog_oos = rs.standard_normal((100, 2))

    pred = res.predict(900, 1100, dynamic=True, exog_oos=exog_oos)
    direct = np.zeros(201)
    direct[0] = c + ar[0] * y[899] + ar[1] * y[897] + x[900] @ ex
    direct[1] = c + ar[0] * direct[0] + ar[1] * y[898] + x[901] @ ex
    direct[2] = c + ar[0] * direct[1] + ar[1] * y[899] + x[902] @ ex
    for i in range(3, 201):
        direct[i] = c + ar[0] * direct[i - 1] + ar[1] * direct[i - 3]
        if 900 + i < x.shape[0]:
            direct[i] += x[900 + i] @ ex
        else:
            direct[i] += exog_oos[i - 101] @ ex

    direct = pd.Series(
        direct, index=pd.date_range(ys.index[900], periods=201, freq="M")
    )
    assert_series_equal(pred, direct)


def test_predict_irregular_ar():
    rs = np.random.RandomState(12345678)
    e = rs.standard_normal(1001)
    y = np.empty(1001)
    y[:3] = e[:3] * np.sqrt(1.0 / (1 - 0.9 ** 2))
    for i in range(3, 1001):
        y[i] = 10 + 0.9 * y[i - 1] - 0.5 * y[i - 3] + e[i]
    ys = pd.Series(y, index=pd.date_range("1-1-1950", periods=1001, freq="M"))
    mod = AutoReg(ys, [1, 3], trend="ct", old_names=False)
    res = mod.fit()
    c = res.params.iloc[0]
    t = res.params.iloc[1]
    ar = np.asarray(res.params.iloc[2:])

    pred = res.predict(900, 1100, True)
    direct = np.zeros(201)
    direct[0] = c + t * 901 + ar[0] * y[899] + ar[1] * y[897]
    direct[1] = c + t * 902 + ar[0] * direct[0] + ar[1] * y[898]
    direct[2] = c + t * 903 + ar[0] * direct[1] + ar[1] * y[899]
    for i in range(3, 201):
        direct[i] = (
            c + t * (901 + i) + ar[0] * direct[i - 1] + ar[1] * direct[i - 3]
        )
    direct = pd.Series(
        direct, index=pd.date_range(ys.index[900], periods=201, freq="M")
    )
    assert_series_equal(pred, direct)

    pred = res.predict(900)
    direct = (
        c
        + t * np.arange(901, 901 + 101)
        + ar[0] * y[899:-1]
        + ar[1] * y[897:-3]
    )
    idx = pd.date_range(ys.index[900], periods=101, freq="M")
    direct = pd.Series(direct, index=idx)
    assert_series_equal(pred, direct)


@pytest.mark.parametrize("dynamic", [True, False])
def test_forecast_start_end_equiv(dynamic):
    rs = np.random.RandomState(12345678)
    e = rs.standard_normal(1001)
    y = np.empty(1001)
    y[0] = e[0] * np.sqrt(1.0 / (1 - 0.9 ** 2))
    effects = 10 * np.cos(np.arange(12) / 11 * 2 * np.pi)
    for i in range(1, 1001):
        y[i] = 10 + 0.9 * y[i - 1] + e[i] + effects[i % 12]
    ys = pd.Series(y, index=pd.date_range("1-1-1950", periods=1001, freq="M"))
    mod = AutoReg(ys, 1, seasonal=True, old_names=False)
    res = mod.fit()
    pred_int = res.predict(1000, 1020, dynamic=dynamic)
    dates = pd.date_range("1-1-1950", periods=1021, freq="M")
    pred_dates = res.predict(dates[1000], dates[1020], dynamic=dynamic)
    assert_series_equal(pred_int, pred_dates)


@pytest.mark.parametrize("start", [21, 25])
def test_autoreg_start(start):
    y_train = pd.Series(np.random.normal(size=20))
    m = AutoReg(y_train, lags=2, old_names=False)
    mf = m.fit()
    end = start + 5
    pred = mf.predict(start=start, end=end)
    assert pred.shape[0] == end - start + 1


def test_deterministic(reset_randomstate):
    y = pd.Series(np.random.normal(size=200))
    terms = [TimeTrend(constant=True, order=1), Seasonality(12)]
    dp = DeterministicProcess(y.index, additional_terms=terms)
    m = AutoReg(y, trend="n", seasonal=False, lags=2, deterministic=dp)
    res = m.fit()
    m2 = AutoReg(
        y, trend="ct", seasonal=True, lags=2, period=12, old_names=False
    )
    res2 = m2.fit()
    assert_almost_equal(np.asarray(res.params), np.asarray(res2.params))
    with pytest.warns(RuntimeWarning, match="When using deterministic, trend"):
        AutoReg(y, trend="ct", seasonal=False, lags=2, deterministic=dp)


def test_autoreg_predict_forecast_equiv(reset_randomstate):
    e = np.random.normal(size=1000)
    nobs = e.shape[0]
    idx = pd.date_range("2020-1-1", freq="D", periods=nobs)
    for i in range(1, nobs):
        e[i] = 0.95 * e[i - 1] + e[i]
    y = pd.Series(e, index=idx)
    m = AutoReg(y, trend="c", lags=1, old_names=False)
    res = m.fit()
    a = res.forecast(12)
    b = res.predict(nobs, nobs + 11)
    c = res.forecast("2022-10-08")
    assert_series_equal(a, b)
    assert_series_equal(a, c)
    sarimax_res = SARIMAX(y, order=(1, 0, 0), trend="c").fit(disp=False)
    d = sarimax_res.forecast(12)
    pd.testing.assert_index_equal(a.index, d.index)


def test_autoreg_forecast_period_index():
    pi = pd.period_range("1990-1-1", periods=524, freq="M")
    y = np.random.RandomState(0).standard_normal(500)
    ys = pd.Series(y, index=pi[:500], name="y")
    mod = AutoReg(ys, 3, seasonal=True, old_names=False)
    res = mod.fit()
    fcast = res.forecast(24)
    assert isinstance(fcast.index, pd.PeriodIndex)
    pd.testing.assert_index_equal(fcast.index, pi[-24:])


@pytest.mark.matplotlib
def test_autoreg_plot_err():
    y = np.random.standard_normal(100)
    mod = AutoReg(y, lags=[1, 3], old_names=False)
    res = mod.fit()
    with pytest.raises(ValueError):
        res.plot_predict(0, end=50, in_sample=False)


def test_autoreg_resids():
    idx = pd.date_range("1900-01-01", periods=250, freq="M")
    rs = np.random.RandomState(0)
    idx_dates = sorted(rs.choice(idx, size=100, replace=False))
    e = rs.standard_normal(250)
    y = np.zeros(250)
    y[:2] = e[:2]
    for i in range(2, 250):
        y[i] = 2 + 1.8 * y[i - 1] - 0.95 * y[i - 2] + e[i]
    ys = pd.Series(y[-100:], index=idx_dates, name="y")
    with pytest.warns(ValueWarning):
        res = AutoReg(ys, lags=2, old_names=False).fit()
    assert np.all(np.isfinite(res.resid))
