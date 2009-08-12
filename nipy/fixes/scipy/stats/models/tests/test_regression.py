"""
Test functions for models.regression
"""
import numpy as np
from numpy.random import standard_normal
from numpy.testing import *
from scipy.linalg import toeplitz
from models.tools import add_constant
from models.regression import OLS, AR, WLS, GLS, yule_walker
import models

W = standard_normal
DECIMAL = 4
DECIMAL_less = 3
DECIMAL_lesser = 2
DECIMAL_least = 1
DECIMAL_sig = 7


class check_regression_results(object):
    '''
    res2 contains results from Rmodelwrap or were obtained from a statistical
    packages such as R or Stata and written to model_results
    '''

    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL)

    def test_standarderrors(self):
        assert_almost_equal(self.res1.bse,self.res2.bse, DECIMAL)

    def test_confidenceintervals(self):
        self.check_confidenceintervals(self.res1.conf_int(),
                self.res2.conf_int)

    def test_scale(self):
        assert_almost_equal(self.res1.scale, self.res2.scale, DECIMAL)

    def test_RSquared(self):
        assert_almost_equal(self.res1.Rsq, self.res2.Rsq,DECIMAL)

    def test_AdjRSquared(self):
        assert_almost_equal(self.res1.adjRsq, self.res2.adjRsq, DECIMAL)

    def test_degrees(self):
        assert_almost_equal(self.res1.df_model, self.res2.df_model, DECIMAL)
        assert_almost_equal(self.res1.df_resid, self.res2.df_resid, DECIMAL)

    def test_ExplainedSumofSquares(self):
        assert_almost_equal(self.res1.ESS, self.res2.ESS, DECIMAL)

    def test_SumofSquaredResiduals(self):
        assert_almost_equal(self.res1.SSR, self.res2.SSR,DECIMAL)

    def test_MeanSquaredError(self):
        assert_almost_equal(self.res1.MSE_model, self.res2.MSE_model, DECIMAL)
        assert_almost_equal(self.res1.MSE_resid, self.res2.MSE_resid, DECIMAL)

    def test_FStatistic(self):
        assert_almost_equal(self.res1.F, self.res2.F, DECIMAL)

    def test_loglike(self):
        assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL)

    def test_AIC(self):
        aic = self.res1.information_criteria()['aic']
        assert_almost_equal(aic, self.res2.AIC, DECIMAL)

    def test_BIC(self):
        bic = self.res1.information_criteria()['bic']
        assert_almost_equal(bic, self.res2.BIC, DECIMAL)

    @dec.skipif(True, "Results not included yet")
    def test_resids(self):
        pass

class test_ols(check_regression_results):
    def __init__(self):
        from models.datasets.longley.data import load
        from model_results import longley
        data = load()
        data.exog = add_constant(data.exog)
        results = OLS(data.endog, data.exog).fit()
        self.res1 = results
        self.res2 = longley()

    def check_confidenceintervals(self, conf1, conf2):
        for i in range(len(conf1)):
            assert_approx_equal(conf1[i][0], conf2[i][0], 6)
            assert_approx_equal(conf1[i][1], conf2[i][1], 6)
            # stata rounds big residuals to significant digits

    def check_params(self, params1, params2):
        assert_almost_equal(params1, params2, DECIMAL)

class test_gls(object):
    '''
    These test results were obtained by replication with R.
    '''
    def __init__(self):
        from models.datasets.longley.data import load
        from model_results import longley_gls

        data = load()
        exog = add_constant(np.column_stack(\
                (data.exog[:,1],data.exog[:,4])))
        tmp_results = OLS(data.endog, exog).fit()
        rho = np.corrcoef(tmp_results.resid[1:],
                tmp_results.resid[:-1])[0][1] # by assumption
        order = toeplitz(np.arange(16))
        sigma = rho**order
        GLS_results = GLS(data.endog, exog, sigma=sigma).fit()
        self.res1 = GLS_results
        self.res2 = longley_gls()

    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_least)
        # rounding vs. stata

    def test_standarderrors(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_lesser)
        # rounding vs. stata

class test_gls_scalar(check_regression_results):
    '''
    Test that GLS with no argument is equivalent to OLS.
    '''
    def __init__(self):
        from models.datasets.longley.data import load
        data = load()
        data.exog = add_constant(data.exog)
        ols_res = OLS(data.endog, data.exog).fit()
        gls_res = GLS(data.endog, data.exog).fit()
        self.res1 = gls_res
        self.res2 = ols_res
        self.res2.conf_int = self.res2.conf_int()
        self.res2.BIC = self.res2.information_criteria()['bic']
        self.res2.AIC = self.res2.information_criteria()['aic']

    def check_confidenceintervals(self, conf1, conf2):
        assert_almost_equal(conf1, conf2, DECIMAL)


#class test_wls(object):
#    '''
#    GLM results are an implicit test of WLS
#    '''
#TODO: Make sure no argument given is the same as OLS
#    pass

#TODO: test AR
# why the two-stage in AR?
#class test_ar(object):
#    from models.datasets.sunspots.data import load
#    data = load()
#    model = AR(data.endog, rho=4).fit()
#    R_res = RModel(data.endog, aic="FALSE", order_max=4)

#    def test_params(self):
#        assert_almost_equal(self.model.rho,
#        pass

#    def test_order(self):
# In R this can be defined or chosen by minimizing the AIC if aic=True
#        pass

class test_yule_walker(object):
    def __init__(self):
        from models.datasets.sunspots.data import load
        from rpy import r
        data = load()
        self.rho, self.sigma = yule_walker(data.endog, order=4, method="mle")
        R_results = r.ar(data.endog, aic="FALSE", order_max=4)
        self.R_params = R_results['ar']

    def test_params(self):
        R_params = np.array([ 1.28310031, -0.45240924, -0.20770299,
                    0.04794365])
        assert_almost_equal(self.rho, self.R_params, DECIMAL)

if __name__=="__main__":
    run_module_suite()

#  Robust error tests.  Compare values computed with SAS
#    res0 = SSM.regression.OLS(x).fit(y, HCC='HC0')
#    nptest.assert_almost_equal(res0.bse, sas_bse_HC0, 4)
#    res1 = SSM.regression.OLS(x).fit(y, HCC='HC1')
#    nptest.assert_almost_equal(res1.bse, sas_bse_HC1, 4)
#    res2 = SSM.regression.OLS(x).fit(y, HCC='HC2')
#    nptest.assert_almost_equal(res2.bse, sas_bse_HC2, 4)
#    res3 = SSM.regression.OLS(x).fit(y, HCC='HC3')
#    nptest.assert_almost_equal(res3.bse, sas_bse_HC3, 4)


