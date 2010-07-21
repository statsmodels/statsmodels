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
        pass

    def test_llf(self):
        pass

class TestAROLS(object):
    pass

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


