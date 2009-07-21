"""
Test functions for models.regression
"""

from numpy.random import standard_normal
from numpy.testing import *

from models.regression import OLS, AR, WLS, GLS
import models

W = standard_normal
DECIMAL = 4


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
        assert_almost_equal(self.res1.conf_int(), self.res2.conf_int, DECIMAL)

    def test_scale(self):
        assert_almost_equal(self.res1.scale, self.res2.scale, DECIMAL)

    def test_RSquared(self):
        assert_almost_equal(self.res1.Rsq, self.res2.Rsq,DECIMAL)

    def test_AdjRSquared(self):
        assert_almost_equal(self.res1.adjRsq, self.res2.adjRSq, DECIMAL)

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
        data.exog = models.functions.add_constant(data.exog)
        results = OLS(data.endog, data.exog).fit()
        self.res1 = results
        self.res2 = longley()

class test_gls(check_regression_results):
    '''
    These test results were obtained by replication with R.
    '''
    def __init__(self):
        from models.datasets.longley.data import load
        from model_results import longley_gls

        data = load()
        exog = models.functions.add_constant(np.column_stack(\
                (data.exog[:,1],data.exog[:,4])))
        tmp_results = OLS(data.endog, exog).fit()
        rho = np.corrcoef(tmp_results.resid[1:],
                tmp_results.resid[:-1])[0][1] # by assumption
        rows = np.arange(1,17).reshape(16,1)*np.ones((16,16))
        cols = np.arange(1,17)*np.ones((16,16))
        sigma = rho**np.fabs((rows-cols))   # matrix of exponents for
                                            # correlation structure
        GLS_results = GLS(data.endog, exog, sigma=sigma).fit()
        self.res1 = GLS_results
        self.res2 = longley_gls()

class test_gls_scalar(self):
    '''
    Test that GLS with no argument is equivalent to OLS.
    '''
    from models.datasets.longley.data import load
    data = load()
    data.exog = models.functions.add_constant(data.exog)
    ols_res = OLS(data.endog, data.exog).fit()
    gls_res = GLS(data.endog, data.exog).fit()
    self.res1 = gls_res
    self.res2 = ols_res

class test_wls(self):
    '''
    GLM results are an implicit test of WLS
    '''
#TODO: Make sure no argument given is the same as OLS


class test_ar(self):
    pass
#TODO: Come back to this



class TestRegression(TestCase):

    def testAR(self):
        X = W((40,10))
        Y = W((40,))
        model = AR(design=X, rho=0.4)
        results = model.fit(Y)
        self.assertEquals(results.df_resid, 30)

    def testOLSdegenerate(self):
        X = W((40,10))
        X[:,0] = X[:,1] + X[:,2]
        Y = W((40,))
        model = OLS(Y,X)
        results = model.fit()
        self.assertEquals(results.df_resid, 31)

    def testARdegenerate(self):
        X = W((40,10))
        X[:,0] = X[:,1] + X[:,2]
        Y = W((40,))
        model = AR(design=X, rho=0.9)
        results = model.fit(Y)
        self.assertEquals(results.df_resid, 31)


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


