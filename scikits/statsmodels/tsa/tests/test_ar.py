"""
Test AR Model
"""
import scikits.statsmodels as sm
from scikits.statsmodels.sandbox.tsa.var import AR
from numpy.testing import assert_almost_equal, assert_equal
from results import results_ar
import numpy as np
import numpy.testing as npt

DECIMAL_6 = 6

class CheckAR(object):
    def test_params(self):


    def test_llf(self):
        pass

class TestAR(CheckAR):
    def __init__(self):
        self.data = sm.datasets.sunspots.load()

class TestAROLS(TestAR):
    def setup(self):
        self.res1 = AR(self.data.endog).fit(maxlag=9, method='ols', trend='c',
                        demean=False)

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


