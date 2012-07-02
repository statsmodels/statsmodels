import numpy as np
from numpy.testing import *

import statsmodels.api as sm
from statsmodels.sysreg.sysmodel import *

class CheckSysregResults(object):
    # TODO : adjust better rtol/atol
    rtol = 1e-2
    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params, rtol=self.rtol)

    def test_fittedvalues(self):
        assert_allclose(self.res1.predict(), self.res2.fittedvalues, 
                rtol=self.rtol)

    def test_normalized_cov_params(self):
        assert_allclose(self.res1.normalized_cov_params, 
                self.res2.cov_params, rtol=self.rtol)

    def test_cov_resids_est(self):
        assert_allclose(self.res1.cov_resids_est, self.res2.cov_resids_est,
                rtol=self.rtol)

    def test_cov_resids(self):
        assert_allclose(self.res1.cov_resids, self.res2.cov_resids, 
                rtol=self.rtol)


# Build system
grun_data = sm.datasets.grunfeld.load()
firms = ['Chrysler', 'General Electric', 'General Motors',
         'US Steel', 'Westinghouse']
grun_exog = grun_data.exog
grun_endog = grun_data.endog
sys = []
for f in firms:
    eq_f = {}
    index_f = grun_exog['firm'] == f
    eq_f['endog'] = grun_endog[index_f]
    exog = (grun_exog[index_f][var] for var in ['value', 'capital'])
    eq_f['exog'] = np.column_stack(exog)
    eq_f['exog'] = sm.add_constant(eq_f['exog'], prepend=True)
    sys.append(eq_f)

class TestSUR(CheckSysregResults):
    @classmethod
    def setupClass(cls):
        from results.results_sysreg import RSUR
        res2 = RSUR
        res1 = SysSUR(sys).fit()
        cls.res1 = res1
        cls.res2 = res2

class TestSURI(CheckSysregResults):
    @classmethod
    def setupClass(cls):
        from results.results_sysreg import RSURI
        res2 = RSURI
        res1 = SysSUR(sys).fit(igls=True)
        cls.res1 = res1
        cls.res2 = res2

class TestSURR(CheckSysregResults):
    @classmethod
    def setupClass(cls):
        from results.results_sysreg import RSURR
        res2 = RSURR

        R = np.array([[1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
                      [2, 0, 1, 1, 0, 0, 0, 3, 1, 1, 1, 0, 0, 1, 0]])
        q = np.array([0, 0])
        res1 = SysSUR(sys, restrictMatrix=R, restrictVect=q).fit()

        cls.res1 = res1
        cls.res2 = res2

class TestSURIR(CheckSysregResults):
    @classmethod
    def setupClass(cls):
        from results.results_sysreg import RSURIR
        res2 = RSURIR

        R = np.array([[1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
                      [2, 0, 1, 1, 0, 0, 0, 3, 1, 1, 1, 0, 0, 1, 0]])
        q = np.array([0, 0])
        res1 = SysSUR(sys, restrictMatrix=R, restrictVect=q).fit(igls=True)

        cls.res1 = res1
        cls.res2 = res2

