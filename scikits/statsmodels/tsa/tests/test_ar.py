"""
Test AR Model
"""
import scikits.statsmodels as sm
from scikits.statsmodels.tsa import AR
from numpy.testing import assert_almost_equal, assert_equal
from results import results_ar
import numpy as np
import numpy.testing as npt

DECIMAL_6 = 6

class CheckAR(object):
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_6)

    def test_bse(self):
        bse = np.sqrt(np.diag(self.res1.cov_params())) # no dof correction
                                            # for compatability with Stata
        assert_almost_equal(bse, self.res2.bse_stata, DECIMAL_6)
        assert_almost_equal(self.res1.bse, self.res2.bse_gretl, DECIMAL_6)

    def test_llf(self):
        assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL_6)

    def test_fpe(self):
        assert_almost_equal(self.res1.fpe, self.res2.fpe, DECIMAL_6)

class TestAR(object):
    def __init__(self):
        self.data = sm.datasets.sunspots.load()

class TestAROLSConstant(TestAR, CheckAR):
    """
    Test AR fit by OLS with a constant.
    """
    def setup(self):
        self.res1 = AR(self.data.endog).fit(maxlag=9, method='cmle')
        self.res2 = results_ar.ARResultsOLS(constant=True)

class TestAROLSNoConstant(TestAR, CheckAR):
    """
    Test AR fit by OLS without a constant.
    """
    def setup(self):
        self.res1 = AR(self.data.endog).fit(maxlag=9,method='cmle',trend='nc')
        self.res2 = results_ar.ARResultsOLS(constant=False)

class TestARMLEConstant(TestAR, CheckAR):
    def setup(self)
        self.res1 = AR(self.data.endog).fit(maxlag=9,method="mle")

class TestAutolagAR(object):
    def setup(self):
        data = sm.datasets.sunspots.load()
        endog = data.endog
        results = []
        for lag in range(1,16+1):
            endog_tmp = endog[16-lag:]
            r = AR(endog_tmp).fit(maxlag=lag)
            results.append([r.aic, r.hqic, r.bic, r.fpe])
        self.res1 = np.asarray(results).T.reshape(4,-1, order='C')
        self.res2 = results_ar.ARLagResults("const").ic

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



