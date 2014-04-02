"""
Test AR Model
"""
import statsmodels.api as sm
from statsmodels.compat.python import range
from statsmodels.tsa.ar_model import AR
from numpy.testing import (assert_almost_equal, assert_allclose, assert_)
from statsmodels.tools.testing import assert_equal
from .results import results_ar
import numpy as np
import numpy.testing as npt
from pandas import Series, Index, DatetimeIndex


DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4

class CheckARMixin(object):
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_6)

    def test_bse(self):
        bse = np.sqrt(np.diag(self.res1.cov_params())) # no dof correction
                                            # for compatability with Stata
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
        assert_(type(res_unpickled) is type(self.res1))

class TestAROLSConstant(CheckARMixin):
    """
    Test AR fit by OLS with a constant.
    """
    @classmethod
    def setupClass(cls):
        data = sm.datasets.sunspots.load()
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
    def setupClass(cls):
        data = sm.datasets.sunspots.load()
        cls.res1 = AR(data.endog).fit(maxlag=9,method='cmle',trend='nc')
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

        #class TestARMLEConstant(CheckAR):


class TestARMLEConstant(object):
    @classmethod
    def setupClass(cls):
        data = sm.datasets.sunspots.load()
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
    def setupClass(cls):
        data = sm.datasets.sunspots.load()
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
    data = sm.datasets.sunspots.load()
    dates = sm.tsa.datetools.dates_from_range('1700', length=len(data.endog))
    endog = Series(data.endog, index=dates)
    ar_model = sm.tsa.AR(endog, freq='A').fit(maxlag=9, method='mle', disp=-1)
    pred = ar_model.predict(start='2005', end='2015')
    predict_dates = sm.tsa.datetools.dates_from_range('2005', '2015')
    predict_dates = DatetimeIndex(predict_dates, freq='infer')

    assert_equal(ar_model.data.predict_dates, predict_dates)
    assert_equal(pred.index, predict_dates)

def test_ar_named_series():
    dates = sm.tsa.datetools.dates_from_range("2011m1", length=72)
    y = Series(np.random.randn(72), name="foobar", index=dates)
    results = sm.tsa.AR(y).fit(2)
    assert_(results.params.index.equals(Index(["const", "L1.foobar",
                                               "L2.foobar"])))

def test_ar_start_params():
    # fix 236
    # smoke test
    data = sm.datasets.sunspots.load()
    res = AR(data.endog).fit(maxlag=9, start_params=0.1*np.ones(10),
                             method="mle", disp=-1, maxiter=100)

def test_ar_series():
    # smoke test for 773
    dta = sm.datasets.macrodata.load_pandas().data["cpi"].diff().dropna()
    dates = sm.tsa.datetools.dates_from_range("1959Q1", length=len(dta))
    dta.index = dates
    ar = AR(dta).fit(maxlags=15)
    ar.bse


def test_ar_select_order():
    # 2118
    np.random.seed(12345)
    y = sm.tsa.arma_generate_sample([1, -.75, .3], [1], 100)
    ts = Series(y, index=DatetimeIndex(start='1/1/1990', periods=100,
                                           freq='M'))
    ar = AR(ts)
    res = ar.select_order(maxlag=12, ic='aic')
    assert_(res == 2)

# GH 2658
def test_ar_select_order_tstat():
    rs = np.random.RandomState(123)
    tau = 25
    y = rs.randn(tau)
    ts = Series(y, index=DatetimeIndex(start='1/1/1990', periods=tau,
                                   freq='M'))

    ar = AR(ts)
    res = ar.select_order(maxlag=5, ic='t-stat')
    assert_equal(res, 0)



#TODO: likelihood for ARX model?
#class TestAutolagARX(object):
#    def setup(self):
#        data = sm.datasets.macrodata.load()
#        endog = data.data.realgdp
#        exog = data.data.realint
#        results = []
#        for lag in range(1, 26):
#            endog_tmp = endog[26-lag:]
#            exog_tmp = exog[26-lag:]
#            r = AR(endog_tmp, exog_tmp).fit(maxlag=lag, trend='ct')
#            results.append([r.aic, r.hqic, r.bic, r.fpe])
#        self.res1 = np.asarray(results).T.reshape(4,-1, order='C')



