"""
Tests for discrete models

Notes
-----
DECIMAL_less is used because it seems that there is a loss of precision
in the Stata *.dta -> *.csv output, NOT the estimator for the Poisson
tests.
"""

import numpy as np
from numpy.testing import *
from scikits.statsmodels.discretemod import *
import scikits.statsmodels as sm
import model_results

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
        assert_almost_equal(self.res1.conf_int(), self.res2.conf_int, DECIMAL)

    def test_zstat(self):
        assert_almost_equal(self.res1.t(), self.res2.z, DECIMAL)

    def pvalues(self):
        assert_almost_equal(self.res1.pvalues, self.res2.pvalues, DECIMAL)

#    def test_cov_params(self):
#        assert_almost_equal(self.res1.cov_params(), self.res2.cov_params,
#                DECIMAL)

    def test_llf(self):
        assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL)

    def test_llnull(self):
        assert_almost_equal(self.res1.llnull, self.res2.llnull, DECIMAL)

    def test_llr(self):
        assert_almost_equal(self.res1.llr, self.res2.llr, DECIMAL_less)

    def test_llr_pvalue(self):
        assert_almost_equal(self.res1.llr_pvalue, self.res2.llr_pvalue, DECIMAL)

    def test_margeff(self):
        pass
    # this probably needs it's own test class?

    def test_normalized_cov_params(self):
        pass

    def test_bse(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL)

    def test_dof(self):
        assert_equal(self.res1.df_model, self.res2.df_model)
        assert_equal(self.res1.df_resid, self.res2.df_resid)

    def test_aic(self):
        assert_almost_equal(self.res1.aic, self.res2.aic, DECIMAL_less)

    def test_bic(self):
        assert_almost_equal(self.res1.bic, self.res2.bic, DECIMAL_less)

class CheckMargEff():
#    def test_defaults(self):
#        assert_almost_equal(self.res1.margeff_default, self.res2.margeff_default)

    def test_nodiscrete_overall(self):
        pass

    def test_nodiscrete_mean(self):
        pass

    def test_nodiscrete_median(self):
        pass

    def test_nodiscrete_zero(self):
        pass

#    def test_eform(self):

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

class TestPoissonNewton(CheckModelResults):
    def __init__(self):
        data = sm.datasets.randhie.Load()
        nobs = len(data.endog)
        exog = sm.add_constant(data.exog.view(float).reshape(nobs,-1))
        self.res1 = Poisson(data.endog, exog).fit(method='newton')
        res2 = model_results.RandHIE()
        res2.poisson()
        self.res2 = res2

class TestMNLogitNewtonBaseZero(CheckModelResults):
    def __init__(self):
        data = sm.datasets.anes96.Load()
        exog = data.exog
        exog[:,0] = np.log(exog[:,0] + .1)
        exog = np.column_stack((exog[:,0],exog[:,2],
            exog[:,5:8]))
        exog = sm.add_constant(exog)
        self.res1 = MNLogit(data.endog, exog).fit(method="newton")
        res2 = model_results.Anes()
        res2.mnlogit_basezero()
        self.res2 = res2

    def test_j(self):
        assert_equal(self.res1.model.J, self.res2.J)

    def test_k(self):
        assert_equal(self.res1.model.K, self.res2.K)




if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'],
            exit=False)
