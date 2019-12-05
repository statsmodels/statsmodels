"""
Test AR Model
"""
from itertools import product

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_allclose
from pandas import Series, Index, date_range, period_range
from pandas.testing import assert_series_equal

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.testing import assert_equal
from statsmodels.tools.tools import Bunch
from statsmodels.tsa.ar_model import AR, AutoReg, ar_select_order
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.tests.results import results_ar

DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4


def gen_ar_data(nobs):
    rs = np.random.RandomState(982739)
    idx = pd.date_range('1-1-1900', freq="M", periods=nobs)
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
        cols = ['x.{0}'.format(i) for i in range(exog)]
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
ids = [', '.join([n + ": " + str(p) for n, p in zip(names, param)])
       for param in params]


@pytest.fixture(scope="module", params=params, ids=ids)
def ols_autoreg_result(request):
    ar, seasonal, trend, exog, cov_type = request.param
    y, x, endog, exog = gen_ols_regressors(ar, seasonal, trend, exog)
    ar_mod = AutoReg(y, ar, seasonal=seasonal, trend=trend, exog=x)
    ar_res = ar_mod.fit(cov_type=cov_type)
    ols = OLS(endog, exog)
    ols_res = ols.fit(cov_type=cov_type, use_t=False)
    return ar_res, ols_res


attributes = ['bse', 'cov_params', 'df_model', 'df_resid', 'fittedvalues',
              'llf', 'nobs', 'params', 'resid', 'scale',
              'tvalues', 'use_t']


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


@pytest.mark.parametrize('attribute', attributes)
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


class CheckAutoRegMixin(CheckARMixin):
    def test_bse(self):
        assert_almost_equal(self.res1.bse, self.res2.bse_stata, DECIMAL_6)


class TestAROLSConstant(CheckARMixin):
    """
    Test AR fit by OLS with a constant.
    """

    @classmethod
    def setup_class(cls):
        data = sm.datasets.sunspots.load(as_pandas=False)
        with pytest.warns(FutureWarning):
            cls.res1 = AR(data.endog).fit(maxlag=9, method='cmle')
        cls.res2 = results_ar.ARResultsOLS(constant=True)

    def test_predict(self):
        model = self.res1.model
        params = self.res1.params
        assert_almost_equal(model.predict(params), self.res2.FVOLSnneg1start0,
                            DECIMAL_4)
        assert_almost_equal(model.predict(params), self.res2.FVOLSnneg1start9,
                            DECIMAL_4)
        assert_almost_equal(model.predict(params, start=100),
                            self.res2.FVOLSnneg1start100, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=9, end=200),
                            self.res2.FVOLSn200start0, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=200, end=400),
                            self.res2.FVOLSn200start200, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=308, end=424),
                            self.res2.FVOLSn100start325, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=9, end=310),
                            self.res2.FVOLSn301start9, DECIMAL_4)
        assert_almost_equal(model.predict(params),
                            self.res2.FVOLSdefault, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=308, end=316),
                            self.res2.FVOLSn4start312, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=308, end=327),
                            self.res2.FVOLSn15start312, DECIMAL_4)


class TestAROLSNoConstant(CheckARMixin):
    """f
    Test AR fit by OLS without a constant.
    """

    @classmethod
    def setup_class(cls):
        data = sm.datasets.sunspots.load(as_pandas=False)
        with pytest.warns(FutureWarning):
            cls.res1 = AR(data.endog).fit(maxlag=9, method='cmle', trend='nc')
        cls.res2 = results_ar.ARResultsOLS(constant=False)

    def test_predict(self):
        model = self.res1.model
        params = self.res1.params
        assert_almost_equal(model.predict(params), self.res2.FVOLSnneg1start0,
                            DECIMAL_4)
        assert_almost_equal(model.predict(params), self.res2.FVOLSnneg1start9,
                            DECIMAL_4)
        assert_almost_equal(model.predict(params, start=100),
                            self.res2.FVOLSnneg1start100, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=9, end=200),
                            self.res2.FVOLSn200start0, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=200, end=400),
                            self.res2.FVOLSn200start200, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=308, end=424),
                            self.res2.FVOLSn100start325, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=9, end=310),
                            self.res2.FVOLSn301start9, DECIMAL_4)
        assert_almost_equal(model.predict(params),
                            self.res2.FVOLSdefault, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=308, end=316),
                            self.res2.FVOLSn4start312, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=308, end=327),
                            self.res2.FVOLSn15start312, DECIMAL_4)

    def test_mle(self):
        # check predict with no constant, #3945
        res1 = self.res1
        endog = res1.model.endog
        with pytest.warns(FutureWarning):
            res0 = AR(endog).fit(maxlag=9, method='mle', trend='nc', disp=0)
        assert_allclose(res0.fittedvalues[-10:], res0.fittedvalues[-10:],
                        rtol=0.015)

        res_arma = ARMA(endog, (9, 0)).fit(method='mle', trend='nc', disp=0)
        assert_allclose(res0.params, res_arma.params, atol=5e-6)
        assert_allclose(res0.fittedvalues[-10:], res_arma.fittedvalues[-10:],
                        rtol=1e-4)


class TestARMLEConstant(object):
    @classmethod
    def setup_class(cls):
        data = sm.datasets.sunspots.load(as_pandas=False)
        with pytest.warns(FutureWarning):
            cls.res1 = AR(data.endog).fit(maxlag=9, method="mle", disp=-1)
        cls.res2 = results_ar.ARResultsMLE(constant=True)

    def test_predict(self):
        model = self.res1.model
        # for some reason convergence is off in 1 out of 10 runs on
        # some platforms. i've never been able to replicate. see #910
        params = np.array([5.66817602, 1.16071069, -0.39538222,
                           -0.16634055, 0.15044614, -0.09439266,
                           0.00906289, 0.05205291, -0.08584362,
                           0.25239198])
        assert_almost_equal(model.predict(params), self.res2.FVMLEdefault,
                            DECIMAL_4)
        assert_almost_equal(model.predict(params, start=9, end=308),
                            self.res2.FVMLEstart9end308, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=100, end=308),
                            self.res2.FVMLEstart100end308, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=0, end=200),
                            self.res2.FVMLEstart0end200, DECIMAL_4)

        # Note: factor 0.5 in below two tests needed to meet precision on OS X.
        assert_almost_equal(0.5 * model.predict(params, start=200, end=333),
                            0.5 * self.res2.FVMLEstart200end334, DECIMAL_4)
        assert_almost_equal(0.5 * model.predict(params, start=308, end=333),
                            0.5 * self.res2.FVMLEstart308end334, DECIMAL_4)

        assert_almost_equal(model.predict(params, start=9, end=309),
                            self.res2.FVMLEstart9end309, DECIMAL_4)
        assert_almost_equal(model.predict(params, end=301),
                            self.res2.FVMLEstart0end301, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=4, end=312),
                            self.res2.FVMLEstart4end312, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=2, end=7),
                            self.res2.FVMLEstart2end7, DECIMAL_4)

    def test_dynamic_predict(self):
        # for some reason convergence is off in 1 out of 10 runs on
        # some platforms. i've never been able to replicate. see #910
        params = np.array([5.66817602, 1.16071069, -0.39538222,
                           -0.16634055, 0.15044614, -0.09439266,
                           0.00906289, 0.05205291, -0.08584362,
                           0.25239198])
        res1 = self.res1
        res2 = self.res2

        rtol = 8e-6
        # assert_raises pre-sample

        start, end = 9, 51
        fv = res1.model.predict(params, start, end, dynamic=True)
        assert_allclose(fv, res2.fcdyn[start:end + 1], rtol=rtol)

        start, end = 9, 308
        fv = res1.model.predict(params, start, end, dynamic=True)
        assert_allclose(fv, res2.fcdyn[start:end + 1], rtol=rtol)

        start, end = 9, 333
        fv = res1.model.predict(params, start, end, dynamic=True)
        assert_allclose(fv, res2.fcdyn[start:end + 1], rtol=rtol)

        start, end = 100, 151
        fv = res1.model.predict(params, start, end, dynamic=True)
        assert_allclose(fv, res2.fcdyn2[start:end + 1], rtol=rtol)

        start, end = 100, 308
        fv = res1.model.predict(params, start, end, dynamic=True)
        assert_allclose(fv, res2.fcdyn2[start:end + 1], rtol=rtol)

        start, end = 100, 333
        fv = res1.model.predict(params, start, end, dynamic=True)
        assert_allclose(fv, res2.fcdyn2[start:end + 1], rtol=rtol)

        start, end = 308, 308
        fv = res1.model.predict(params, start, end, dynamic=True)
        assert_allclose(fv, res2.fcdyn3[start:end + 1], rtol=rtol)

        start, end = 308, 333
        fv = res1.model.predict(params, start, end, dynamic=True)
        assert_allclose(fv, res2.fcdyn3[start:end + 1], rtol=rtol)

        start, end = 309, 333
        fv = res1.model.predict(params, start, end, dynamic=True)
        assert_allclose(fv, res2.fcdyn4[start:end + 1], rtol=rtol)

        # start, end = None, None
        fv = res1.model.predict(params, dynamic=True)
        assert_allclose(fv, res2.fcdyn[9:309], rtol=rtol)


class TestAutolagAR(object):
    @classmethod
    def setup_class(cls):
        data = sm.datasets.sunspots.load(as_pandas=False)
        endog = data.endog
        results = []
        for lag in range(1, 16 + 1):
            endog_tmp = endog[16 - lag:]
            with pytest.warns(FutureWarning):
                r = AR(endog_tmp).fit(maxlag=lag)
            # See issue #324 for why we're doing these corrections vs. R
            # results
            k_ar = r.k_ar
            k_trend = r.k_trend
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

            results.append([aic, hqic, bic, r.fpe])
        res1 = np.asarray(results).T.reshape(4, -1, order='C')
        # aic correction to match R
        cls.res1 = res1
        cls.res2 = results_ar.ARLagResults("const").ic

    def test_ic(self):
        npt.assert_almost_equal(self.res1, self.res2, DECIMAL_6)


def test_ar_dates():
    # just make sure they work
    data = sm.datasets.sunspots.load(as_pandas=False)
    dates = date_range(start='1700', periods=len(data.endog), freq='A')
    endog = Series(data.endog, index=dates)
    with pytest.warns(FutureWarning):
        ar_model = AR(endog, freq='A').fit(maxlag=9, method='mle', disp=-1)
    pred = ar_model.predict(start='2005', end='2015')
    predict_dates = date_range(start='2005', end='2016', freq='A')[:11]

    assert_equal(ar_model.data.predict_dates, predict_dates)
    assert_equal(pred.index, predict_dates)


def test_ar_named_series(reset_randomstate):
    dates = period_range(start="2011-1", periods=72, freq='M')
    y = Series(np.random.randn(72), name="foobar", index=dates)
    with pytest.warns(FutureWarning):
        results = AR(y).fit(2)
    idx = Index(["const", "L1.foobar", "L2.foobar"])
    assert results.params.index.equals(idx)


@pytest.mark.smoke
def test_ar_start_params():
    # fix GH#236
    data = sm.datasets.sunspots.load(as_pandas=False)
    with pytest.warns(FutureWarning):
        AR(data.endog).fit(maxlag=9, start_params=0.1 * np.ones(10),
                           method="mle", disp=-1, maxiter=100)


@pytest.mark.smoke
def test_ar_series():
    # GH#773
    dta = sm.datasets.macrodata.load_pandas().data["cpi"].diff().dropna()
    dates = period_range(start='1959Q1', periods=len(dta), freq='Q')
    dta.index = dates
    with pytest.warns(FutureWarning):
        ar = AR(dta).fit(maxlags=15)
    ar.bse


def test_ar_select_order():
    # GH#2118
    np.random.seed(12345)
    y = sm.tsa.arma_generate_sample([1, -.75, .3], [1], 100)
    ts = Series(y, index=date_range(start='1/1/1990', periods=100,
                                    freq='M'))
    with pytest.warns(FutureWarning):
        ar = AR(ts)
    with pytest.warns(FutureWarning):
        res = ar.select_order(maxlag=12, ic='aic')
    assert res == 2


# GH 2658
def test_ar_select_order_tstat():
    rs = np.random.RandomState(123)
    tau = 25
    y = rs.randn(tau)
    ts = Series(y, index=date_range(start='1/1/1990', periods=tau,
                                    freq='M'))
    with pytest.warns(FutureWarning):
        ar = AR(ts)
    with pytest.warns(FutureWarning):
        res = ar.select_order(maxlag=5, ic='t-stat')
    assert_equal(res, 0)


def test_constant_column_trend():
    # GH#5258, after calling lagmat, the sample below has a constant column,
    #  which used to cause the result.k_trend attribute to be set incorrectly
    # See also GH#5538

    sample = np.array([
        0.46341460943222046, 0.46341460943222046, 0.39024388790130615,
        0.4146341383457184, 0.4146341383457184, 0.4146341383457184,
        0.3414634168148041, 0.4390243887901306, 0.46341460943222046,
        0.4390243887901306])
    with pytest.warns(FutureWarning):
        model = AR(sample)

    # Fitting with a constant and maxlag=7 raises because of regressor
    #  collinearity.
    with pytest.raises(ValueError, match="trend='c' is not allowed"):
        model.fit(trend="c")

    res = model.fit(trend="nc")
    assert res.k_trend == 0
    assert res.k_ar == 7
    assert len(res.params) == 7
    pred = res.predict(start=10, end=12)
    # expected numbers are regression-test
    expected = np.array([0.44687422, 0.45608137, 0.47046381])
    assert_allclose(pred, expected)


def test_summary_corner():
    data = sm.datasets.macrodata.load_pandas().data["cpi"].diff().dropna()
    dates = period_range(start='1959Q1', periods=len(data), freq='Q')
    data.index = dates
    with pytest.warns(FutureWarning):
        res = AR(data).fit(maxlag=4)
    summ = res.summary().as_text()
    assert 'AR(4)' in summ
    assert 'L4.cpi' in summ
    assert '03-31-1959' in summ
    with pytest.warns(FutureWarning):
        res = AR(data).fit(maxlag=0)
    summ = res.summary().as_text()
    assert 'const' in summ
    assert 'AR(0)' in summ


def test_at_repeated_fit():
    data = sm.datasets.sunspots.load_pandas()
    with pytest.warns(FutureWarning):
        ar = AR(data.endog)
    ar.fit()
    with pytest.raises(RuntimeError):
        ar.fit(maxlag=1)
    with pytest.raises(RuntimeError):
        ar.fit(method='mle')
    with pytest.raises(RuntimeError):
        ar.fit(ic='aic')
    with pytest.raises(RuntimeError):
        ar.fit(trend='nc')


def test_ar_errors(reset_randomstate):
    with pytest.raises(ValueError, match='Only the univariate case'):
        with pytest.warns(FutureWarning):
            AR(np.empty((1000, 2)))
    with pytest.warns(FutureWarning):
        ar = AR(np.random.standard_normal(1000))
    with pytest.raises(ValueError, match='Method yw not'):
        ar.fit(method='yw')
    with pytest.raises(ValueError, match='ic option fpic not'):
        ar.fit(ic='fpic')
    res = ar.fit()
    with pytest.raises(ValueError, match='Start must be >= k_ar'):
        res.predict(start=0)
    with pytest.raises(ValueError, match='Prediction must have `end` after'):
        res.predict(start=100, end=99)
    with pytest.raises(ValueError, match='Length of start params'):
        with pytest.warns(FutureWarning):
            AR(np.random.standard_normal(1000)).fit(maxlag=2, method='mle',
                                                    start_params=[1, 1])


@pytest.mark.smoke
def test_ar_score():
    data = sm.datasets.sunspots.load_pandas()
    with pytest.warns(FutureWarning):
        ar = AR(np.asarray(data.endog))
    res = ar.fit(3)
    score = ar.score(res.params)
    assert isinstance(score, np.ndarray)
    assert score.shape == (4,)
    assert ar.information(res.params) is None


@pytest.mark.smoke
def test_bestlag_stop():
    data = sm.datasets.sunspots.load_pandas()
    with pytest.warns(FutureWarning):
        ar = AR(np.asarray(data.endog))
    with pytest.warns(FutureWarning):
        ar.fit(2, ic='t-stat')
    assert ar.k_ar == 2


def test_roots():
    data = sm.datasets.sunspots.load_pandas()
    with pytest.warns(FutureWarning):
        ar = AR(np.asarray(data.endog))
    res = ar.fit(1)
    assert_almost_equal(res.roots, np.array([1. / res.params[-1]]))


def test_resids_mle():
    data = sm.datasets.sunspots.load_pandas()
    with pytest.warns(FutureWarning):
        ar = AR(np.asarray(data.endog))
    res = ar.fit(1, method='mle', disp=-1)
    assert res.resid.shape[0] == data.endog.shape[0]


def test_ar_repeated_fit():
    data = sm.datasets.sunspots.load(as_pandas=False)
    with pytest.warns(FutureWarning):
        mod = AR(data.endog)
    res = mod.fit(maxlag=2, method="cmle")
    repeat = mod.fit(maxlag=2, method="cmle")
    assert_allclose(res.params, repeat.params)
    assert isinstance(res.summary().as_text(), str)
    assert isinstance(repeat.summary().as_text(), str)


def test_ar_predict_no_fit():
    data = sm.datasets.sunspots.load(as_pandas=False)
    with pytest.warns(FutureWarning):
        mod = AR(data.endog)
    with pytest.raises(RuntimeError, match='Model must be fit'):
        mod.predict([.1])


params = product([0, 1, 3, [1, 3]],
                 ['n', 'c', 't', 'ct'],
                 [True, False],
                 [0, 2],
                 [None, 11],
                 ['none', 'drop'],
                 [True, False],
                 [None, 12])
params = list(params)
params = [param for param in params if
          (param[0] or param[1] != 'n' or param[2] or param[3])]
params = [param for param in params if
          not param[2] or (param[2] and (param[4] or param[6]))]
param_fmt = """\
lags: {0}, trend: {1}, seasonal: {2}, nexog: {3}, periods: {4}, \
missing: {5}, pandas: {6}, hold_back{7}"""

ids = [param_fmt.format(*param) for param in params]


def gen_data(nobs, nexog, pandas, seed=92874765):
    rs = np.random.RandomState(seed)
    endog = rs.standard_normal((nobs))
    exog = rs.standard_normal((nobs, nexog)) if nexog else None
    if pandas:
        index = pd.date_range('31-12-1999', periods=nobs, freq='M')
        endog = pd.Series(endog, name='endog', index=index)
        if nexog:
            cols = ['exog.{0}'.format(i) for i in range(exog.shape[1])]
            exog = pd.DataFrame(exog, columns=cols, index=index)
    from collections import namedtuple
    DataSet = namedtuple('DataSet', ['endog', 'exog'])
    return DataSet(endog=endog, exog=exog)


@pytest.fixture(scope='module', params=params, ids=ids)
def ar_data(request):
    lags, trend, seasonal = request.param[:3]
    nexog, period, missing, use_pandas, hold_back = request.param[3:]
    data = gen_data(250, nexog, use_pandas)
    return Bunch(trend=trend, lags=lags, seasonal=seasonal, period=period,
                 endog=data.endog, exog=data.exog, missing=missing,
                 hold_back=hold_back)


params = product([0, 3, [1, 3]],
                 ['c'],
                 [True, False],
                 [0],
                 [None, 11],
                 ['drop'],
                 [True, False],
                 [None, 12])
params = list(params)
params = [param for param in params if
          (param[0] or param[1] != 'n' or param[2] or param[3])]
params = [param for param in params if
          not param[2] or (param[2] and (param[4] or param[6]))]
param_fmt = """\
lags: {0}, trend: {1}, seasonal: {2}, nexog: {3}, periods: {4}, \
missing: {5}, pandas: {6}, hold_back: {7}"""

ids = [param_fmt.format(*param) for param in params]


# Only test 1/3 to save time
@pytest.fixture(scope='module', params=params[::3], ids=ids[::3])
def plot_data(request):
    lags, trend, seasonal = request.param[:3]
    nexog, period, missing, use_pandas, hold_back = request.param[3:]
    data = gen_data(250, nexog, use_pandas)
    return Bunch(trend=trend, lags=lags, seasonal=seasonal, period=period,
                 endog=data.endog, exog=data.exog, missing=missing,
                 hold_back=hold_back)


@pytest.mark.matplotlib
@pytest.mark.smoke
def test_autoreg_smoke_plots(plot_data, close_figures):
    from matplotlib.figure import Figure
    mod = AutoReg(plot_data.endog, plot_data.lags, trend=plot_data.trend,
                  seasonal=plot_data.seasonal, exog=plot_data.exog,
                  hold_back=plot_data.hold_back, period=plot_data.period,
                  missing=plot_data.missing)
    res = mod.fit()
    fig = res.plot_diagnostics()
    assert isinstance(fig, Figure)
    if plot_data.exog is None:
        fig = res.plot_predict(end=300)
        assert isinstance(fig, Figure)
        fig = res.plot_predict(end=300, alpha=None, in_sample=False)
        assert isinstance(fig, Figure)


@pytest.mark.smoke
def test_autoreg_predict_smoke(ar_data):
    mod = AutoReg(ar_data.endog, ar_data.lags, trend=ar_data.trend,
                  seasonal=ar_data.seasonal, exog=ar_data.exog,
                  hold_back=ar_data.hold_back, period=ar_data.period,
                  missing=ar_data.missing)
    res = mod.fit()
    exog_oos = None
    if ar_data.exog is not None:
        exog_oos = np.empty((1, ar_data.exog.shape[1]))
    mod.predict(res.params, 0, 250, exog_oos=exog_oos)
    if ar_data.lags == 0 and ar_data.exog is None:
        mod.predict(res.params, 0, 350, exog_oos=exog_oos)
    if isinstance(ar_data.endog, pd.Series) and \
            (not ar_data.seasonal or ar_data.period is not None):
        ar_data.endog.index = list(range(ar_data.endog.shape[0]))
        if ar_data.exog is not None:
            ar_data.exog.index = list(range(ar_data.endog.shape[0]))
        mod = AutoReg(ar_data.endog, ar_data.lags, trend=ar_data.trend,
                      seasonal=ar_data.seasonal, exog=ar_data.exog,
                      period=ar_data.period, missing=ar_data.missing)
        mod.predict(res.params, 0, 250, exog_oos=exog_oos)


@pytest.mark.matplotlib
def test_parameterless_autoreg():
    data = gen_data(250, 0, False)
    mod = AutoReg(data.endog, 0, trend='n', seasonal=False, exog=None)
    res = mod.fit()
    for attr in dir(res):
        if attr.startswith('_'):
            continue

        # TODO
        if attr in ('predict', 'f_test', 't_test', 'initialize', 'load',
                    'remove_data', 'save', 't_test', 't_test_pairwise',
                    'wald_test', 'wald_test_terms'):
            continue
        attr = getattr(res, attr)
        if callable(attr):
            attr()
        else:
            assert isinstance(attr, object)


def test_predict_errors():
    data = gen_data(250, 2, True)
    mod = AutoReg(data.endog, 3)
    res = mod.fit()
    with pytest.raises(ValueError, match='exog and exog_oos cannot be used'):
        mod.predict(res.params, exog=data.exog)
    with pytest.raises(ValueError, match='exog and exog_oos cannot be used'):
        mod.predict(res.params, exog_oos=data.exog)
    with pytest.raises(ValueError, match='hold_back must be >= lags'):
        AutoReg(data.endog, 3, hold_back=1)
    with pytest.raises(ValueError, match='freq cannot be inferred'):
        AutoReg(data.endog.values, 3, seasonal=True)

    mod = AutoReg(data.endog, 3, exog=data.exog)
    res = mod.fit()
    with pytest.raises(ValueError, match=r'The shape of exog \(200, 2\)'):
        mod.predict(res.params, exog=data.exog.iloc[:200])
    with pytest.raises(ValueError, match='The number of columns in exog_oos'):
        mod.predict(res.params, exog_oos=data.exog.iloc[:, :1])
    with pytest.raises(ValueError, match='Prediction must have `end` after'):
        mod.predict(res.params, start=200, end=199)
    with pytest.raises(ValueError, match='exog_oos must be provided'):
        mod.predict(res.params, end=250, exog_oos=None)

    mod = AutoReg(data.endog, 0, exog=data.exog)
    res = mod.fit()
    with pytest.raises(ValueError, match='start and end indicate that 10'):
        mod.predict(res.params, end=259, exog_oos=data.exog.iloc[:5])


def test_spec_errors():
    data = gen_data(250, 2, True)
    with pytest.raises(ValueError, match='lags must be a positive scalar'):
        AutoReg(data.endog, -1)
    with pytest.raises(ValueError, match='All values in lags must be pos'):
        AutoReg(data.endog, [1, 1, 1])
    with pytest.raises(ValueError, match='All values in lags must be pos'):
        AutoReg(data.endog, [1, -2, 3])


@pytest.mark.smoke
def test_dynamic_forecast_smoke(ar_data):
    mod = AutoReg(ar_data.endog, ar_data.lags, trend=ar_data.trend,
                  seasonal=ar_data.seasonal, exog=ar_data.exog,
                  hold_back=ar_data.hold_back, period=ar_data.period,
                  missing=ar_data.missing)
    res = mod.fit()
    res.predict(dynamic=True)
    if ar_data.exog is None:
        res.predict(end=260, dynamic=True)


@pytest.mark.smoke
def test_ar_select_order_smoke():
    data = sm.datasets.sunspots.load(as_pandas=True).data['SUNACTIVITY']
    ar_select_order(data, 4, glob=True, trend='n')
    ar_select_order(data, 4, glob=False, trend='n')
    ar_select_order(data, 4, seasonal=True, period=12)
    ar_select_order(data, 4, seasonal=False)
    ar_select_order(data, 4, glob=True)
    ar_select_order(data, 4, glob=True, seasonal=True, period=12)


class TestAutoRegOLSConstant(CheckAutoRegMixin):
    """
    Test AutoReg fit by OLS with a constant.
    """

    @classmethod
    def setup_class(cls):
        data = sm.datasets.sunspots.load(as_pandas=True)
        data.endog.index = list(range(len(data.endog)))
        cls.res1 = AutoReg(data.endog, lags=9).fit()
        cls.res2 = results_ar.ARResultsOLS(constant=True)

    def test_predict(self):
        model = self.res1.model
        params = self.res1.params
        assert_almost_equal(model.predict(params), self.res2.FVOLSnneg1start0,
                            DECIMAL_4)
        assert_almost_equal(model.predict(params), self.res2.FVOLSnneg1start9,
                            DECIMAL_4)
        assert_almost_equal(model.predict(params, start=100),
                            self.res2.FVOLSnneg1start100, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=9, end=200),
                            self.res2.FVOLSn200start0, DECIMAL_4)
        assert_almost_equal(model.predict(params),
                            self.res2.FVOLSdefault, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=200, end=400),
                            self.res2.FVOLSn200start200, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=308, end=424),
                            self.res2.FVOLSn100start325, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=9, end=310),
                            self.res2.FVOLSn301start9, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=308, end=316),
                            self.res2.FVOLSn4start312, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=308, end=327),
                            self.res2.FVOLSn15start312, DECIMAL_4)


class TestAutoRegOLSNoConstant(CheckAutoRegMixin):
    """f
    Test AR fit by OLS without a constant.
    """

    @classmethod
    def setup_class(cls):
        data = sm.datasets.sunspots.load(as_pandas=False)
        cls.res1 = AutoReg(data.endog, lags=9, trend='n').fit()
        cls.res2 = results_ar.ARResultsOLS(constant=False)

    def test_predict(self):
        model = self.res1.model
        params = self.res1.params
        assert_almost_equal(model.predict(params), self.res2.FVOLSnneg1start0,
                            DECIMAL_4)
        assert_almost_equal(model.predict(params), self.res2.FVOLSnneg1start9,
                            DECIMAL_4)
        assert_almost_equal(model.predict(params, start=100),
                            self.res2.FVOLSnneg1start100, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=9, end=200),
                            self.res2.FVOLSn200start0, DECIMAL_4)
        assert_almost_equal(model.predict(params),
                            self.res2.FVOLSdefault, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=200, end=400),
                            self.res2.FVOLSn200start200, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=308, end=424),
                            self.res2.FVOLSn100start325, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=9, end=310),
                            self.res2.FVOLSn301start9, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=308, end=316),
                            self.res2.FVOLSn4start312, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=308, end=327),
                            self.res2.FVOLSn15start312, DECIMAL_4)


@pytest.mark.parametrize('lag', np.arange(1, 16 + 1))
def test_autoreg_info_criterion(lag):
    data = sm.datasets.sunspots.load(as_pandas=False)
    endog = data.endog
    endog_tmp = endog[16 - lag:]
    r = AutoReg(endog_tmp, lags=lag).fit()
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

    r2 = AutoReg(endog, lags=lag, hold_back=16).fit()
    assert_allclose(r.aic, r2.aic)
    assert_allclose(r.bic, r2.bic)
    assert_allclose(r.hqic, r2.hqic)
    assert_allclose(r.fpe, r2.fpe)


def test_autoreg_named_series(reset_randomstate):
    dates = period_range(start="2011-1", periods=72, freq='M')
    y = Series(np.random.randn(72), name="foobar", index=dates)
    results = AutoReg(y, lags=2).fit()
    idx = Index(["intercept", "foobar.L1", "foobar.L2"])
    assert results.params.index.equals(idx)


@pytest.mark.smoke
def test_autoreg_series():
    # GH#773
    dta = sm.datasets.macrodata.load_pandas().data["cpi"].diff().dropna()
    dates = period_range(start='1959Q1', periods=len(dta), freq='Q')
    dta.index = dates
    ar = AutoReg(dta, lags=15).fit()
    ar.bse


def test_ar_order_select():
    # GH#2118
    np.random.seed(12345)
    y = sm.tsa.arma_generate_sample([1, -.75, .3], [1], 100)
    ts = Series(y, index=date_range(start='1/1/1990', periods=100,
                                    freq='M'))
    res = ar_select_order(ts, maxlag=12, ic='aic')
    assert tuple(res.ar_lags) == (1, 2)
    assert isinstance(res.aic, dict)
    assert isinstance(res.bic, dict)
    assert isinstance(res.hqic, dict)
    assert isinstance(res.model, AutoReg)
    assert not res.seasonal
    assert res.trend == 'c'
    assert res.period is None


def test_autoreg_constant_column_trend():
    sample = np.array([
        0.46341460943222046, 0.46341460943222046, 0.39024388790130615,
        0.4146341383457184, 0.4146341383457184, 0.4146341383457184,
        0.3414634168148041, 0.4390243887901306, 0.46341460943222046,
        0.4390243887901306])

    with pytest.raises(ValueError, match='The model specification cannot'):
        AutoReg(sample, lags=7)
    with pytest.raises(ValueError, match='The model specification cannot'):
        AutoReg(sample, lags=7, trend='n')


def test_autoreg_summary_corner():
    data = sm.datasets.macrodata.load_pandas().data["cpi"].diff().dropna()
    dates = period_range(start='1959Q1', periods=len(data), freq='Q')
    data.index = dates
    res = AutoReg(data, lags=4).fit()
    summ = res.summary().as_text()
    assert 'AutoReg(4)' in summ
    assert 'cpi.L4' in summ
    assert '03-31-1960' in summ
    res = AutoReg(data, lags=0).fit()
    summ = res.summary().as_text()
    assert 'intercept' in summ
    assert 'AutoReg(0)' in summ


@pytest.mark.smoke
def test_autoreg_score():
    data = sm.datasets.sunspots.load_pandas()
    ar = AutoReg(np.asarray(data.endog), 3)
    res = ar.fit()
    score = ar.score(res.params)
    assert isinstance(score, np.ndarray)
    assert score.shape == (4,)
    assert ar.information(res.params).shape == (4, 4)


def test_autoreg_roots():
    data = sm.datasets.sunspots.load_pandas()
    ar = AutoReg(np.asarray(data.endog), lags=1)
    res = ar.fit()
    assert_almost_equal(res.roots, np.array([1. / res.params[-1]]))


def test_equiv_dynamic(reset_randomstate):
    e = np.random.standard_normal(1001)
    y = np.empty(1001)
    y[0] = e[0] * np.sqrt(1.0 / (1 - 0.9 ** 2))
    for i in range(1, 1001):
        y[i] = 0.9 * y[i - 1] + e[i]
    mod = AutoReg(y, 1)
    res = mod.fit()
    pred0 = res.predict(500, 800, dynamic=0)
    pred1 = res.predict(500, 800, dynamic=True)
    idx = pd.date_range('31-01-2000', periods=1001, freq='M')
    y = pd.Series(y, index=idx)
    mod = AutoReg(y, 1)
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
    smod = SARIMAX(y, order=(1, 0, 0), trend='c')
    sres = smod.fit(disp=False)
    mod = AutoReg(y, 1)
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
    mod = AutoReg(ys, 1, seasonal=True)
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
    direct = pd.Series(direct, index=pd.date_range(ys.index[900], periods=201,
                                                   freq="M"))
    assert_series_equal(pred, direct)

    pred = res.predict(900, dynamic=False)
    direct = y[899:-1] * ar + c + seasons[np.arange(900, 1001) % 12]
    direct = pd.Series(direct, index=pd.date_range(ys.index[900], periods=101,
                                                   freq="M"))
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
    mod = AutoReg(ys, [1, 3], trend="c", exog=xdf)
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

    direct = pd.Series(direct, index=pd.date_range(ys.index[900], periods=201,
                                                   freq="M"))
    assert_series_equal(pred, direct)


def test_predict_irregular_ar():
    rs = np.random.RandomState(12345678)
    e = rs.standard_normal(1001)
    y = np.empty(1001)
    y[:3] = e[:3] * np.sqrt(1.0 / (1 - 0.9 ** 2))
    for i in range(3, 1001):
        y[i] = 10 + 0.9 * y[i - 1] - 0.5 * y[i - 3] + e[i]
    ys = pd.Series(y, index=pd.date_range("1-1-1950", periods=1001, freq="M"))
    mod = AutoReg(ys, [1, 3], trend="ct")
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
        direct[i] = c + t * (901 + i) + ar[0] * direct[i - 1] + ar[1] * direct[
            i - 3]
    direct = pd.Series(direct, index=pd.date_range(ys.index[900], periods=201,
                                                   freq="M"))
    assert_series_equal(pred, direct)

    pred = res.predict(900)
    direct = (c + t * np.arange(901, 901+101) + ar[0] * y[899:-1]
              + ar[1] * y[897:-3])
    idx = pd.date_range(ys.index[900], periods=101, freq="M")
    direct = pd.Series(direct, index=idx)
    assert_series_equal(pred, direct)


@pytest.mark.parametrize('dynamic', [True, False])
def test_forecast_start_end_equiv(dynamic):
    rs = np.random.RandomState(12345678)
    e = rs.standard_normal(1001)
    y = np.empty(1001)
    y[0] = e[0] * np.sqrt(1.0 / (1 - 0.9 ** 2))
    effects = 10 * np.cos(np.arange(12) / 11 * 2 * np.pi)
    for i in range(1, 1001):
        y[i] = 10 + 0.9 * y[i - 1] + e[i] + effects[i % 12]
    ys = pd.Series(y, index=pd.date_range("1-1-1950", periods=1001, freq="M"))
    mod = AutoReg(ys, 1, seasonal=True)
    res = mod.fit()
    pred_int = res.predict(1000, 1020, dynamic=dynamic)
    dates = pd.date_range("1-1-1950", periods=1021, freq="M")
    pred_dates = res.predict(dates[1000], dates[1020], dynamic=dynamic)
    assert_series_equal(pred_int, pred_dates)
