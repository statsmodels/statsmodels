"""
Test AR Model
"""
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR
from numpy.testing import (assert_almost_equal, assert_equal, #assert_allclose,
                           assert_)
from results import results_ar
import numpy as np
import numpy.testing as npt

DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4

class CheckAR(object):
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
        from statsmodels.compatnp.py3k import BytesIO
        fh = BytesIO()
        #test wrapped results load save pickle
        self.res1.save(fh)
        fh.seek(0,0)
        res_unpickled = self.res1.__class__.load(fh)
        assert_(type(res_unpickled) is type(self.res1))

class TestAROLSConstant(CheckAR):
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


class TestAROLSNoConstant(CheckAR):
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
        params = self.res1.params
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
        res1 = self.res1
        res2 = self.res2

        # assert_raises pre-sample

        # 9, 51
        start, end = 9, 51
        fv = res1.predict(start, end, dynamic=True)
        assert_almost_equal(fv, res2.fcdyn[start:end+1], DECIMAL_4)

        # 9, 308
        start, end = 9, 308
        fv = res1.predict(start, end, dynamic=True)
        assert_almost_equal(fv, res2.fcdyn[start:end+1], DECIMAL_4)

        # 9, 333
        start, end = 9, 333
        fv = res1.predict(start, end, dynamic=True)
        assert_almost_equal(fv, res2.fcdyn[start:end+1], DECIMAL_4)

        # 100, 151
        start, end = 100, 151
        fv = res1.predict(start, end, dynamic=True)
        assert_almost_equal(fv, res2.fcdyn2[start:end+1], DECIMAL_4)

        # 100, 308
        start, end = 100, 308
        fv = res1.predict(start, end, dynamic=True)
        assert_almost_equal(fv, res2.fcdyn2[start:end+1], DECIMAL_4)

        # 100, 333
        start, end = 100, 333
        fv = res1.predict(start, end, dynamic=True)
        assert_almost_equal(fv, res2.fcdyn2[start:end+1], DECIMAL_4)

        # 308, 308
        start, end = 308, 308
        fv = res1.predict(start, end, dynamic=True)
        assert_almost_equal(fv, res2.fcdyn3[start:end+1], DECIMAL_4)

        # 308, 333
        start, end = 308, 333
        fv = res1.predict(start, end, dynamic=True)
        assert_almost_equal(fv, res2.fcdyn3[start:end+1], DECIMAL_4)

        # 309, 333
        start, end = 309, 333
        fv = res1.predict(start, end, dynamic=True)
        assert_almost_equal(fv, res2.fcdyn4[start:end+1], DECIMAL_4)

        # None, None
        start, end = None, None
        fv = res1.predict(dynamic=True)
        assert_almost_equal(fv, res2.fcdyn[9:309], DECIMAL_4)


class TestAutolagAR(object):
    @classmethod
    def setupClass(cls):
        data = sm.datasets.sunspots.load()
        endog = data.endog
        results = []
        for lag in range(1,16+1):
            endog_tmp = endog[16-lag:]
            r = AR(endog_tmp).fit(maxlag=lag)
            results.append([r.aic, r.hqic, r.bic, r.fpe])
        cls.res1 = np.asarray(results).T.reshape(4,-1, order='C')
        cls.res2 = results_ar.ARLagResults("const").ic

    def test_ic(self):
        npt.assert_almost_equal(self.res1, self.res2, DECIMAL_6)

def test_ar_dates():
    # just make sure they work
    data = sm.datasets.sunspots.load()
    dates = sm.tsa.datetools.dates_from_range('1700', length=len(data.endog))
    from pandas import Series
    endog = Series(data.endog, index=dates)
    ar_model = sm.tsa.AR(endog, freq='A').fit(maxlag=9, method='mle', disp=-1)
    pred = ar_model.predict(start='2005', end='2015')
    predict_dates = sm.tsa.datetools.dates_from_range('2005', '2015')
    try:
        from pandas import DatetimeIndex
        predict_dates = DatetimeIndex(predict_dates, freq='infer')
    except:
        pass
    assert_equal(ar_model.data.predict_dates, predict_dates)
    assert_equal(pred.index, predict_dates)

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



