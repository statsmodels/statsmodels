import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import statsmodels.api as sm
from statsmodels.sysreg.sysmodel import SysSUR 
from statsmodels.sysreg.syssem import Sys2SLS, Sys3SLS

class CheckSysregResults(object):
    # TODO : adjust better rtol/atol
    rtol = 1e-2
    atol = 1e-4
    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params, rtol=self.rtol)

    def test_fittedvalues(self):
        assert_allclose(self.res1.predict(), self.res2.fittedvalues, 
                rtol=self.rtol)

    def test_normalized_cov_params(self):
        assert_allclose(self.res1.normalized_cov_params, 
                self.res2.cov_params, rtol=self.rtol, atol=self.atol)

    def test_cov_resids_est(self):
        assert_allclose(self.res1.cov_resids_est, self.res2.cov_resids_est,
                rtol=self.rtol)

    def test_cov_resids(self):
        assert_allclose(self.res1.cov_resids, self.res2.cov_resids, 
                rtol=self.rtol)

    def test_df_model(self):
        assert_array_equal(self.res1.df_model, self.res2.df_model)

    def test_df_resid(self):
        assert_array_equal(self.res1.df_resid, self.res2.df_resid)


## SUR tests
# Build system
grun_data = sm.datasets.grunfeld.load()
firms = ['Chrysler', 'General Electric', 'General Motors',
         'US Steel', 'Westinghouse']
grun_exog = grun_data.exog
grun_endog = grun_data.endog
grun_sys = []
for f in firms:
    eq_f = {}
    index_f = grun_exog['firm'] == f
    eq_f['endog'] = grun_endog[index_f]
    exog = (grun_exog[index_f][var] for var in ['value', 'capital'])
    eq_f['exog'] = np.column_stack(exog)
    eq_f['exog'] = sm.add_constant(eq_f['exog'], prepend=True)
    grun_sys.append(eq_f)

class TestSUR(CheckSysregResults):
    @classmethod
    def setupClass(cls):
        from results.results_sysreg import RSUR
        res2 = RSUR
        res1 = SysSUR(grun_sys).fit()
        cls.res1 = res1
        cls.res2 = res2

class TestSURI(CheckSysregResults):
    @classmethod
    def setupClass(cls):
        from results.results_sysreg import RSURI
        res2 = RSURI
        res1 = SysSUR(grun_sys).fit(iterative=True)
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
        res1 = SysSUR(grun_sys, restrict_matrix=R, restrict_vect=q).fit()

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
        res1 = SysSUR(grun_sys, restrict_matrix=R, restrict_vect=q).fit(iterative=True)

        cls.res1 = res1
        cls.res2 = res2

## 2SLS tests based on example/example_sysreg3.py
# Build system
# See sysreg/tests/results/kmenta.R

kmenta_data = sm.datasets.kmenta.load().data

y = kmenta_data['consump']
x1 = np.column_stack((kmenta_data['price'], kmenta_data['income']))
x2 = np.column_stack((kmenta_data['price'], kmenta_data['farmPrice'], kmenta_data['trend']))
x1 = sm.add_constant(x1, prepend=True)
x2 = sm.add_constant(x2, prepend=True)

kmenta_eq1 = {'endog' : y, 'exog' : x1, 'indep_endog' : [1]}
kmenta_eq2 = {'endog' : y, 'exog' : x2, 'indep_endog' : [1]}
kmenta_sys = [kmenta_eq1, kmenta_eq2]

class Test2SLS(CheckSysregResults):
    @classmethod
    def setupClass(cls):
        from results.results_sysreg import R2SLS
        res2 = R2SLS
        res1 = Sys2SLS(kmenta_sys).fit()
        cls.res1 = res1
        cls.res2 = res2

class Test3SLS(CheckSysregResults):
    @classmethod
    def setupClass(cls):
        from results.results_sysreg import R3SLS
        res2 = R3SLS
        res1 = Sys3SLS(kmenta_sys).fit()
        cls.res1 = res1
        cls.res2 = res2

class TestI3SLS(CheckSysregResults):
    @classmethod
    def setupClass(cls):
        from results.results_sysreg import RI3SLS
        res2 = RI3SLS
        res1 = Sys3SLS(kmenta_sys).fit(iterative=True)
        cls.res1 = res1
        cls.res2 = res2

