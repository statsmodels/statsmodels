"""
Test AR Model
"""
import scikits.statsmodels.api as sm
from scikits.statsmodels.tsa.ar_model import AR
from numpy.testing import assert_almost_equal, assert_equal
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
        assert_almost_equal(model.predict(params, n=-1,start=100),
                self.res2.FVOLSnneg1start100, DECIMAL_4)
        assert_almost_equal(model.predict(params, n=200,start=0),
                self.res2.FVOLSn200start0, DECIMAL_4)
        assert_almost_equal(model.predict(params, n=200,start=200),
                self.res2.FVOLSn200start200, DECIMAL_4)
        assert_almost_equal(model.predict(params, n=200,start=-109),
                self.res2.FVOLSn200startneg109, DECIMAL_4)
        assert_almost_equal(model.predict(params, n=100,start=325),
                self.res2.FVOLSn100start325, DECIMAL_4)
        assert_almost_equal(model.predict(params, n=301,start=9),
                self.res2.FVOLSn301start9, DECIMAL_4)
        assert_almost_equal(model.predict(params, n=301,start=0),
                self.res2.FVOLSn301start0, DECIMAL_4)
        assert_almost_equal(model.predict(params, n=4,start=312),
                self.res2.FVOLSn4start312, DECIMAL_4)
        assert_almost_equal(model.predict(params, n=15,start=312),
                self.res2.FVOLSn15start312, DECIMAL_4)


class TestAROLSNoConstant(CheckAR):
    """
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
        assert_almost_equal(model.predict(params, n=-1,start=100),
                self.res2.FVOLSnneg1start100, DECIMAL_4)
        assert_almost_equal(model.predict(params, n=200,start=0),
                self.res2.FVOLSn200start0, DECIMAL_4)
        assert_almost_equal(model.predict(params, n=200,start=200),
                self.res2.FVOLSn200start200, DECIMAL_4)
        assert_almost_equal(model.predict(params, n=200,start=-109),
                self.res2.FVOLSn200startneg109, DECIMAL_4)
        assert_almost_equal(model.predict(params, n=100,start=325),
                self.res2.FVOLSn100start325, DECIMAL_4)
        assert_almost_equal(model.predict(params, n=301,start=9),
                self.res2.FVOLSn301start9, DECIMAL_4)
        assert_almost_equal(model.predict(params, n=301,start=0),
                self.res2.FVOLSn301start0, DECIMAL_4)
        assert_almost_equal(model.predict(params, n=4,start=312),
                self.res2.FVOLSn4start312, DECIMAL_4)
        assert_almost_equal(model.predict(params, n=15,start=312),
                self.res2.FVOLSn15start312, DECIMAL_4)

#class TestARMLEConstant(CheckAR):
#    @classmethod
#    def setupClass(cls):
#        data = sm.datasets.sunspots.load()
#        cls.res1 = AR(data.endog).fit(maxlag=9,method="mle", disp=0)

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



