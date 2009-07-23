"""
Test functions for models.rlm
"""

import numpy.random as R
from numpy.testing import *
import models
from rmodelwrap import RModel
from rpy import r
import rpy # for hampel test...ugh
import numpy as np # ditto
from models.rlm import RLM

DECIMAL = 4

class check_rlm_results(object):
    '''
    res2 contains results from Rmodelwrap or were obtained from a statistical
    packages such as R or Stata and written to model_results
    '''
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL)

    def test_standarderrors(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL)

    @dec.knownfailureif(True, "Not given by RModelwrap")
    def test_confidenceintervals(self):
        assert_almost_equal(self.res1.conf_int(), self.res2.conf_int, DECIMAL)

    def test_scale(self):
        assert_almost_equal(self.res1.scale, self.res2.scale, DECIMAL-1)
# off by ~2e-04

#    def test_k2(self):
#        assert_almost_equal(self.res1.k2, self.res2.k2, DECIMAL)
# The tuning constant for Huber 2?  This is the tuning constant for
# MAD scale estimate?  Check references

    def test_weights(self):
        assert_almost_equal(self.res1.weights, self.res2.weights, DECIMAL)

    @dec.knownfailureif(True, "Not implemented")
    def test_stddev(self):
        assert_almost_equal(self.res1.stddev, self.res2.stddev, DECIMAL)

    def test_residuals(self):
        assert_almost_equal(self.res1.resid, self.res2.resid, DECIMAL)

    def test_degrees(self):
        assert_almost_equal(self.res1.df_model, self.res2.df_model, DECIMAL)
        assert_almost_equal(self.res1.df_resid, self.res2.df_resid, DECIMAL)

    def test_bcov_unscaled(self):
        assert_almost_equal(self.res1.bcov_unscaled, self.res2.bcov_unscaled,
                    DECIMAL)

class test_rlm(check_rlm_results):
    from models.datasets.stackloss.data import load
    data = load()
    data.exog = models.functions.add_constant(data.exog)
    def __init__(self):
#        from models.datasets.stackloss.data import load
#        self.data = load()
#        self.data.exog = models.functions.add_constant(self.data.exog)
        results = RLM(self.data.endog, self.data.exog,\
                    M=models.robust.norms.HuberT()).fit()   # default M
        self.res1 = results
        r.library('MASS')
        self.res2 = RModel(self.data.endog, self.data.exog,
                        r.rlm, psi="psi.huber")

    def test_hampel(self):
#        d = rpy.as_list(r('stackloss'))
#        y = d[0]['stack.loss']
#        x = np.column_stack(np.array(d[0][name]) for name in d[0].keys()[0:-1])
#        x = np.column_stack((x,np.ones((len(x),1))))
#        x = np.column_stack((x[:,2],x[:,1],x[:,0],x[:,3]))
# why in the world the above works and just passing data.endog and data.exog does not is
# completely beyond me
        results = RLM(self.data.endog, self.data.exog,
                    M=models.robust.norms.Hampel()).fit()

        self.res1 = results
        y = self.data.endog
        x = self.data.exog.copy()
        self.res2 = RModel(y, x,
                        r.rlm, psi="psi.hampel")

class test_rlm_bisquare(test_rlm):
    def __init__(self):
        results = RLM(self.data.endog, self.data.exog,
                    M=models.robust.norms.TukeyBiweight()).fit()
        self.res1 = results
        self.res2 = RModel(self.data.endog, self.data.exog,
                        r.rlm, psi="psi.bisquare")

if __name__=="__main__":
    run_module_suite()



