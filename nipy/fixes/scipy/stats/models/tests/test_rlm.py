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
import model_results
import nose

DECIMAL = 4
DECIMAL_less = 3
DECIMAL_lesser = 2
DECIMAL_least = 1

class check_rlm_results(object):
    '''
    res2 contains  results from Rmodelwrap or were obtained from a statistical
    packages such as R, Stata, or SAS and written to model_results.

    Covariance matrices were obtained from SAS and are imported from
    rlm_results
    '''
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL)

    def test_standarderrors(self):
        if isinstance(self.res2, RModel):
            raise nose.SkipTest("R bse from different cov matrix")
        else:
            assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL)

    def test_confidenceintervals(self):
        if isinstance(self.res2, RModel):
            raise nose.SkipTest("Results from RModel wrapper")
        else:
            assert_almost_equal(self.res1.conf_int(), self.res2.conf_int,
            DECIMAL)

    def test_scale(self):
        assert_almost_equal(self.res1.scale, self.res2.scale, DECIMAL_less)
        # off by ~2e-04

    def test_weights(self):
        assert_almost_equal(self.res1.weights, self.res2.weights, DECIMAL)

#    def test_stddev(self):
#        assert_almost_equal(self.res1.stddev, self.res2.stddev, DECIMAL)
#   don't know how R calculates this

    def test_residuals(self):
        assert_almost_equal(self.res1.resid, self.res2.resid, DECIMAL)

    def test_degrees(self):
        assert_almost_equal(self.res1.df_model, self.res2.df_model, DECIMAL)
        assert_almost_equal(self.res1.df_resid, self.res2.df_resid, DECIMAL)

    def test_bcov_unscaled(self):
        if self.res2.__module__ == 'model_results':
            raise nose.SkipTest("No unscaled cov matrix from SAS")
        else:
            assert_almost_equal(self.res1.bcov_unscaled,
                    self.res2.bcov_unscaled, DECIMAL)

    def test_bcov_scaled(self):
        assert_almost_equal(self.res1.bcov_scaled, self.res2.h1, DECIMAL_least)
        assert_almost_equal(self.res1.h2, self.res2.h2, DECIMAL_least)
        assert_almost_equal(self.res1.h3, self.res2.h3, DECIMAL_least)
        # rounding errors in Andrew's make it necessary to use least vs. lesser

class test_rlm(check_rlm_results):
    from models.datasets.stackloss.data import load
    data = load()
    data.exog = models.tools.add_constant(data.exog)
    r.library('MASS')
    def __init__(self):
        results = RLM(self.data.endog, self.data.exog,\
                    M=models.robust.norms.HuberT()).fit()   # default M
        h2 = RLM(self.data.endog, self.data.exog,\
                    M=models.robust.norms.HuberT()).fit(cov="H2").bcov_scaled
        h3 = RLM(self.data.endog, self.data.exog,\
                    M=models.robust.norms.HuberT()).fit(cov="H3").bcov_scaled
        self.res1 = results
        self.res1.h2 = h2
        self.res1.h3 = h3
        self.res2 = RModel(self.data.endog, self.data.exog,
                        r.rlm, psi="psi.huber")
        self.res2.h1 = model_results.huber.h1
        self.res2.h2 = model_results.huber.h2
        self.res2.h3 = model_results.huber.h3

class test_hampel(test_rlm):
    def __init__(self):
        results = RLM(self.data.endog, self.data.exog,
                    M=models.robust.norms.Hampel()).fit()
        h2 = RLM(self.data.endog, self.data.exog,\
                    M=models.robust.norms.Hampel()).fit(cov="H2").bcov_scaled
        h3 = RLM(self.data.endog, self.data.exog,\
                    M=models.robust.norms.Hampel()).fit(cov="H3").bcov_scaled
        self.res1 = results
        self.res1.h2 = h2
        self.res1.h3 = h3
        self.res2 = RModel(self.data.endog[:,None], self.data.exog,
                    r.rlm, psi="psi.hampel") #, init="lts")
        self.res2.h1 = model_results.hampel.h1
        self.res2.h2 = model_results.hampel.h2
        self.res2.h3 = model_results.hampel.h3

class test_rlm_bisquare(test_rlm):
    def __init__(self):
        results = RLM(self.data.endog, self.data.exog,
                    M=models.robust.norms.TukeyBiweight()).fit()
        h2 = RLM(self.data.endog, self.data.exog,\
                    M=models.robust.norms.TukeyBiweight()).fit(cov=\
                    "H2").bcov_scaled
        h3 = RLM(self.data.endog, self.data.exog,\
                    M=models.robust.norms.TukeyBiweight()).fit(cov=\
                    "H3").bcov_scaled
        self.res1 = results
        self.res1.h2 = h2
        self.res1.h3 = h3
        self.res2 = RModel(self.data.endog, self.data.exog,
                        r.rlm, psi="psi.bisquare")
        self.res2.h1 = model_results.bisquare.h1
        self.res2.h2 = model_results.bisquare.h2
        self.res2.h3 = model_results.bisquare.h3

class test_rlm_andrews(test_rlm):
    def __init__(self):
        results = RLM(self.data.endog, self.data.exog,
                    M=models.robust.norms.AndrewWave()).fit()
        h2 = RLM(self.data.endog, self.data.exog,
                    M=models.robust.norms.AndrewWave()).fit(cov=\
                    "H2").bcov_scaled
        h3 = RLM(self.data.endog, self.data.exog,
                    M=models.robust.norms.AndrewWave()).fit(cov=\
                    "H3").bcov_scaled
        self.res1 = results
        self.res1.h2 = h2
        self.res1.h3 = h3
        self.res2 = model_results.andrews()

### tests with Huber scaling

class test_rlm_huber(check_rlm_results):
    from models.datasets.stackloss.data import load
    data = load()
    data.exog = models.tools.add_constant(data.exog)
    def __init__(self):
        results = RLM(self.data.endog, self.data.exog,\
                    M=models.robust.norms.HuberT()).fit(scale_est="Huber")
        h2 = RLM(self.data.endog, self.data.exog,\
                    M=models.robust.norms.HuberT()).fit(cov="H2",
                    scale_est="Huber").bcov_scaled
        h3 = RLM(self.data.endog, self.data.exog,\
                    M=models.robust.norms.HuberT()).fit(cov="H3",
                    scale_est="Huber").bcov_scaled
        self.res1 = results
        self.res1.h2 = h2
        self.res1.h3 = h3
        self.res2 = model_results.huber_huber()

class test_hampel_huber(test_rlm):
    def __init__(self):
        results = RLM(self.data.endog, self.data.exog,
                    M=models.robust.norms.Hampel()).fit(scale_est="Huber")
        h2 = RLM(self.data.endog, self.data.exog,\
                    M=models.robust.norms.Hampel()).fit(cov="H2",
                    scale_est="Huber").bcov_scaled
        h3 = RLM(self.data.endog, self.data.exog,\
                    M=models.robust.norms.Hampel()).fit(cov="H3",
                    scale_est="Huber").bcov_scaled
        self.res1 = results
        self.res1.h2 = h2
        self.res1.h3 = h3
        self.res2 = model_results.hampel_huber()

class test_rlm_bisquare_huber(test_rlm):
    def __init__(self):
        results = RLM(self.data.endog, self.data.exog,
                    M=models.robust.norms.TukeyBiweight()).fit(\
                    scale_est="Huber")
        h2 = RLM(self.data.endog, self.data.exog,\
                    M=models.robust.norms.TukeyBiweight()).fit(cov=\
                    "H2", scale_est="Huber").bcov_scaled
        h3 = RLM(self.data.endog, self.data.exog,\
                    M=models.robust.norms.TukeyBiweight()).fit(cov=\
                    "H3", scale_est="Huber").bcov_scaled
        self.res1 = results
        self.res1.h2 = h2
        self.res1.h3 = h3
        self.res2 = model_results.bisquare_huber()

class test_rlm_andrews_huber(test_rlm):
    def __init__(self):
        results = RLM(self.data.endog, self.data.exog,
                    M=models.robust.norms.AndrewWave()).fit(scale_est="Huber")
        h2 = RLM(self.data.endog, self.data.exog,
                    M=models.robust.norms.AndrewWave()).fit(cov=\
                    "H2", scale_est="Huber").bcov_scaled
        h3 = RLM(self.data.endog, self.data.exog,
                    M=models.robust.norms.AndrewWave()).fit(cov=\
                    "H3", scale_est="Huber").bcov_scaled
        self.res1 = results
        self.res1.h2 = h2
        self.res1.h3 = h3
        self.res2 = model_results.andrews_huber()

if __name__=="__main__":
    run_module_suite()



