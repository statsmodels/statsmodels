import numpy as np
from numpy.testing import *

import statsmodels.api as sm
from statsmodels.sysreg.sysmodel import *

class CheckSysregResults(object):
    decimal_params = 3
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, self.decimal_params)
    def test_fittedvalues(self):
        assert_almost_equal(self.res1.predict(), self.res2.fittedvalues, 3) # fail with decimal error >= 4
    def test_normalized_cov_params(self):
        assert_almost_equal(self.res1.normalized_cov_params, self.res2.cov_params, 2) # fail with decimal error >= 3

class TestSUR(CheckSysregResults):
    @classmethod
    def setupClass(cls):
        from results.results_sysreg import sur
        res2 = sur
      
        # Redundant code with example_sysreg.py. How to avoid this?
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
        
        res1 = SysSUR(sys).fit()
        
        cls.res1 = res1
        cls.res2 = res2

