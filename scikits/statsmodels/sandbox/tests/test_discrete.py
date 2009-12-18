"""
Tests for discrete models
"""

import numpy as np
from numpy.testing import *
from scikits.statsmodels.sandbox.discretemod import *
import scikits.statsmodels as sm
import model_results
from nose import SkipTest

DECIMAL = 4
DECIMAL_less = 3
DECIMAL_lesser = 2
DECIMAL_least = 1
DECIMAL_none = 0

class CheckModelResults(object):
    """
    res2 should be the test results from RModelWrap
    or the results as defined in model_results_data
    """
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL)

    def test_conf_int(self):
        pass

    def test_cov_params(self):
        assert_almost_equal(self.res1.cov_params(), self.res2.cov_params,
                DECIMAL)

    def test_llf(self):
        assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL)

    def test_llnull(self):
        assert_almost_equal(self.res1.llnull, self.res2.llnull, DECIMAL)

    def test_llr(self):
        assert_almost_equal(self.res1.llr, self.res2.llr, DECIMAL)

    def test_llr_pvalue(self):
        assert_almost_equal(self.res1.llr_pvalue, self.res2.llr_pvalue, DECIMAL)

    def test_margeff(self):
        pass
    # this probably needs it's own test class?

    def test_normalized_cov_params(self):
        pass

    def test_bse(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL)

    def test_t(self):
        pass
#TODO: these need to be z scores.  this stuff should be a mixin class

    def test_t_test(self):
        pass

    def test_dof(self):
        assert_equal(self.res1.df_model, self.res2.df_model)
        assert_equal(self.res1.df_resid, self.res2.df_resid)

    def test_aic(self):
        assert_almost_equal(self.res1.aic, self.res2.aic)

    def test_bic(self):
        assert_almost_equal(self.res1.bic, self.res2.bic)

class TestProbitNewton(CheckModelResults):
    def __init__(self):
        data = sm.datasets.spector.Load()
        data.exog = sm.add_constant(data.exog)
        self.data = data
        self.res1 = Probit(data.endog, data.exog).fit(method="newton")
        res2 = model_results.Spector()
        res2.probit()
        self.res2 = res2

class TestLogitNewton(CheckModelResults):
    def __init__(self):
        data = sm.datasets.spector.Load()
        data.exog = sm.add_constant(data.exog)
        self.data = data
        self.res1 = Logit(data.endog, data.exog).fit(method="newton")
        res2 = model_results.Spector()
        res2.logit()
        self.res2 = res2

#@dec.skipif(True, "Test not written yet")
#class TestPoisson(CheckModelResults):
#    raise SkipTest("Test not written yet")

#@dec.skipif(True, "Test not written yet")
#class testMNLogit(CheckModelResults):
#    raise SkipTest("Test not written yet")


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'],
            exit=False)
