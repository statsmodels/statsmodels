import numpy as np
from numpy.testing import *

import statsmodels.api as sm
from statsmodels.sandbox.sysreg import *

class CheckSysregResults(object):
   decimal_params = 4
   def test_params(self):
      assert_almost_equal(self.res1.params, self.res2.params, self.decimal_params)

class TestSUR(CheckSysregResults):
   @classmethod
   def setupClass(cls):
      from results.results_sysreg import GrunfeldSUR
      res2 = GrunfeldSUR()
      
      # No Python 3 compat (see example_sysreg.py if needed)
      grun_data = sm.datasets.grunfeld.load()
      firms = ['Chrysler', 'General Electric', 'General Motors',
        'US Steel', 'Westinghouse']
      grun_exog = grun_data.exog
      grun_endog = grun_data.endog
      # Right now takes SUR takes a list of arrays
      # The array alternates between the LHS of an equation and RHS side of an
      # equation
      # This is very likely to change
      grun_sys = []
      for i in firms:
         index = grun_exog['firm'] == i
         grun_sys.append(grun_endog[index])
         exog = grun_exog[index][['value','capital']].view(float).reshape(-1,2)
         exog = sm.add_constant(exog, prepend=True)
         grun_sys.append(exog)
      grun_mod = SUR(grun_sys)
      res1 = grun_mod.fit()
   
      cls.res1 = res1
      cls.res2 = res2
