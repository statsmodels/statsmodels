"""
Test VAR Model
"""

import scikits.statsmodels as sm
from numpy.testing import assert_almost_equal, assert_equal

DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4

class CheckVAR(object):
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_6)

    def test_neqs(self):
        assert_equal(self.res1.neqs, self.res2.neqs)

    def test_nobs(self):
        assert_equal(self.res1.nobs, self.res2.nobs)

    def test_df_eq(self):
        assert_equal(self.res1.df_eq, self.res2.df_eq)

    def test_rmse(self):
        results = self.results
        for i in range(len(results)):
            assert_almost_equal(results[i].mse_resid**.5,
                    eval('self.res2.rmse_'+str(i)), DECIMAL_6)

    def test_rsquared(self):
        results = self.results
            assert_almost_equal(results[i].rsquared,
                    eval('self.res2.rsquared_'+str(i)), DECIMAL_6)

    def test_llf(self):
        assert_almost_equal(self.res1.llf, self.res2.llf)
        for i in range(len(results)):
            assert_almost_equal(results[i].llf,
                    eval('self.res2.llf_'+str(i)), DECIMAL_6)

    def test_aic(self):
        assert_almost_equal(self.res1.aic, self.res2.aic)

    def test_bic(self):
        assert_almost_equal(self.res1.bic, self.res2.bic)

    def test_hqic(self):
        assert_almost_equal(self.res1.hqic, self.res2.hqic)

    def test_fpe(self):
        assert_almost_equal(self.res1.fpe, self.res2.fpe)

    def test_detsig(self):
        assert_almost_equal(self.res1.detomega, self.res1.detsig)

    def test_bse(self)
        assert_almost_equal(self.res1.bse, self.res2.bse)


class TestVAR(CheckVAR):
    def __init__(self):
        data = sm.macrodata.load()
        data = data.data[['realinv','realgdp','realcons']].view((float,3))
        data = np.diff(np.log(XX),axis=0)
        self.res1 = VAR2(endog=data, laglen=2).fit()


