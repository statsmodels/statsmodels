"""
Test AR Model
"""
import statsmodels.api as sm
from statsmodels.compat.python import range
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
from numpy.testing import (assert_almost_equal, assert_allclose, assert_)
from statsmodels.tools.testing import assert_equal
from .results import results_ar

import pytest
import numpy as np
import numpy.testing as npt
from pandas import Series, Index, date_range, period_range


DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4


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
        from statsmodels.compat.python import BytesIO
        fh = BytesIO()
        #test wrapped results load save pickle
        self.res1.save(fh)
        fh.seek(0,0)
        res_unpickled = self.res1.__class__.load(fh)
        assert type(res_unpickled) is type(self.res1)  # noqa: E721


class TestAROLSConstant(CheckARMixin):
    """
    Test AR fit by OLS with a constant.
    """
    @classmethod
    def setup_class(cls):
        data = sm.datasets.sunspots.load(as_pandas=False)
        cls.res1 = AR(data.endog).fit(maxlag=9, method='cmle')
        cls.res2 = results_ar.ARResultsOLS(constant=True)

    def test_predict(self):
        model = self.res1.model
        params = self.res1.params
        assert_almost_equal(model.predict(params),self.res2.FVOLSnneg1start0,
                DECIMAL_4)
        assert_almost_equal(model.predict(params),self.res2.FVOLSnneg1start9,
                DECIMAL_4)
        assert_almost_equal(model.predict(params, start=100),
                self.res2.FVOLSnneg1start100, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=9, end=200),
                self.res2.FVOLSn200start0, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=200, end=400),
                self.res2.FVOLSn200start200, DECIMAL_4)
        #assert_almost_equal(model.predict(params, n=200,start=-109),
        #        self.res2.FVOLSn200startneg109, DECIMAL_4)
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
        cls.res1 = AR(data.endog).fit(maxlag=9, method='cmle', trend='nc')
        cls.res2 = results_ar.ARResultsOLS(constant=False)

    def test_predict(self):
        model = self.res1.model
        params = self.res1.params
        assert_almost_equal(model.predict(params),self.res2.FVOLSnneg1start0,
                DECIMAL_4)
        assert_almost_equal(model.predict(params),self.res2.FVOLSnneg1start9,
                DECIMAL_4)
        assert_almost_equal(model.predict(params, start=100),
                self.res2.FVOLSnneg1start100, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=9, end=200),
                self.res2.FVOLSn200start0, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=200, end=400),
                self.res2.FVOLSn200start200, DECIMAL_4)
        #assert_almost_equal(model.predict(params, n=200,start=-109),
        #        self.res2.FVOLSn200startneg109, DECIMAL_4)
        assert_almost_equal(model.predict(params, start=308,end=424),
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
        cls.res1 = AR(data.endog).fit(maxlag=9,method="mle", disp=-1)
        cls.res2 = results_ar.ARResultsMLE(constant=True)

    def test_predict(self):
        model = self.res1.model
        # for some reason convergence is off in 1 out of 10 runs on
        # some platforms. i've never been able to replicate. see #910
        params = np.array([ 5.66817602,  1.16071069, -0.39538222,
                           -0.16634055,  0.15044614, -0.09439266,
                           0.00906289,  0.05205291, -0.08584362,
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

        assert_almost_equal(model.predict(params, start=9,end=309),
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
        params = np.array([ 5.66817602,  1.16071069, -0.39538222,
                           -0.16634055,  0.15044614, -0.09439266,
                           0.00906289,  0.05205291, -0.08584362,
                           0.25239198])
        res1 = self.res1
        res2 = self.res2

        rtol = 8e-6
        # assert_raises pre-sample

        # 9, 51
        start, end = 9, 51
        fv = res1.model.predict(params, start, end, dynamic=True)
        assert_allclose(fv, res2.fcdyn[start:end+1], rtol=rtol)

        # 9, 308
        start, end = 9, 308
        fv = res1.model.predict(params, start, end, dynamic=True)
        assert_allclose(fv, res2.fcdyn[start:end+1], rtol=rtol)

        # 9, 333
        start, end = 9, 333
        fv = res1.model.predict(params, start, end, dynamic=True)
        assert_allclose(fv, res2.fcdyn[start:end+1], rtol=rtol)

        # 100, 151
        start, end = 100, 151
        fv = res1.model.predict(params, start, end, dynamic=True)
        assert_allclose(fv, res2.fcdyn2[start:end+1], rtol=rtol)

        # 100, 308
        start, end = 100, 308
        fv = res1.model.predict(params, start, end, dynamic=True)
        assert_allclose(fv, res2.fcdyn2[start:end+1], rtol=rtol)

        # 100, 333
        start, end = 100, 333
        fv = res1.model.predict(params, start, end, dynamic=True)
        assert_allclose(fv, res2.fcdyn2[start:end+1], rtol=rtol)

        # 308, 308
        start, end = 308, 308
        fv = res1.model.predict(params, start, end, dynamic=True)
        assert_allclose(fv, res2.fcdyn3[start:end+1], rtol=rtol)

        # 308, 333
        start, end = 308, 333
        fv = res1.model.predict(params, start, end, dynamic=True)
        assert_allclose(fv, res2.fcdyn3[start:end+1], rtol=rtol)

        # 309, 333
        start, end = 309, 333
        fv = res1.model.predict(params, start, end, dynamic=True)
        assert_allclose(fv, res2.fcdyn4[start:end+1], rtol=rtol)

        # None, None
        start, end = None, None
        fv = res1.model.predict(params, dynamic=True)
        assert_allclose(fv, res2.fcdyn[9:309], rtol=rtol)


class TestAutolagAR(object):
    @classmethod
    def setup_class(cls):
        data = sm.datasets.sunspots.load(as_pandas=False)
        endog = data.endog
        results = []
        for lag in range(1,16+1):
            endog_tmp = endog[16-lag:]
            r = AR(endog_tmp).fit(maxlag=lag)
            # See issue #324 for why we're doing these corrections vs. R
            # results
            k_ar = r.k_ar
            k_trend = r.k_trend
            log_sigma2 = np.log(r.sigma2)
            aic = r.aic
            aic = (aic - log_sigma2) * (1 + k_ar)/(1 + k_ar + k_trend)
            aic += log_sigma2

            hqic = r.hqic
            hqic = (hqic - log_sigma2) * (1 + k_ar)/(1 + k_ar + k_trend)
            hqic += log_sigma2

            bic = r.bic
            bic = (bic - log_sigma2) * (1 + k_ar)/(1 + k_ar + k_trend)
            bic += log_sigma2

            results.append([aic, hqic, bic, r.fpe])
        res1 = np.asarray(results).T.reshape(4,-1, order='C')
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
    ar_model = sm.tsa.AR(endog, freq='A').fit(maxlag=9, method='mle', disp=-1)
    pred = ar_model.predict(start='2005', end='2015')
    predict_dates = date_range(start='2005', end='2016', freq='A')[:11]

    assert_equal(ar_model.data.predict_dates, predict_dates)
    assert_equal(pred.index, predict_dates)


def test_ar_named_series(reset_randomstate):
    dates = period_range(start="2011-1", periods=72, freq='M')
    y = Series(np.random.randn(72), name="foobar", index=dates)
    results = sm.tsa.AR(y).fit(2)
    assert_(results.params.index.equals(Index(["const", "L1.foobar",
                                               "L2.foobar"])))


@pytest.mark.smoke
def test_ar_start_params():
    # fix GH#236
    data = sm.datasets.sunspots.load(as_pandas=False)
    res = AR(data.endog).fit(maxlag=9, start_params=0.1*np.ones(10),
                             method="mle", disp=-1, maxiter=100)


@pytest.mark.smoke
def test_ar_series():
    # GH#773
    dta = sm.datasets.macrodata.load_pandas().data["cpi"].diff().dropna()
    dates = period_range(start='1959Q1', periods=len(dta), freq='Q')
    dta.index = dates
    ar = AR(dta).fit(maxlags=15)
    ar.bse


def test_ar_select_order():
    # GH#2118
    np.random.seed(12345)
    y = sm.tsa.arma_generate_sample([1, -.75, .3], [1], 100)
    ts = Series(y, index=date_range(start='1/1/1990', periods=100,
                                    freq='M'))
    ar = AR(ts)
    res = ar.select_order(maxlag=12, ic='aic')
    assert_(res == 2)


# GH 2658
def test_ar_select_order_tstat():
    rs = np.random.RandomState(123)
    tau = 25
    y = rs.randn(tau)
    ts = Series(y, index=date_range(start='1/1/1990', periods=tau,
                                    freq='M'))

    ar = AR(ts)
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


#TODO: likelihood for ARX model?
#class TestAutolagARX(object):
#    def setup(self):
#        data = sm.datasets.macrodata.load(as_pandas=False)
#        endog = data.data.realgdp
#        exog = data.data.realint
#        results = []
#        for lag in range(1, 26):
#            endog_tmp = endog[26-lag:]
#            exog_tmp = exog[26-lag:]
#            r = AR(endog_tmp, exog_tmp).fit(maxlag=lag, trend='ct')
#            results.append([r.aic, r.hqic, r.bic, r.fpe])
#        self.res1 = np.asarray(results).T.reshape(4,-1, order='C')
