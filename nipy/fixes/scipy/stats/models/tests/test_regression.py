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
from models import tools
from check_for_rpy import skip_rpy
from nose import SkipTest
from scipy.stats import t

W = standard_normal
DECIMAL = 4
DECIMAL_less = 3
DECIMAL_lesser = 2
DECIMAL_least = 1
DECIMAL_sig = 7
skipR = skip_rpy()
if not skipR:
    from rpy import r
    from rmodelwrap import RModel


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

    def test_conf_int_subset(self):
        ci1 = self.res1.conf_int(cols=(1,2))
        ci2 = self.res1.conf_int()[1:3]
        assert_almost_equal(ci1, ci2, DECIMAL)

    def test_scale(self):
        assert_almost_equal(self.res1.scale, self.res2.scale, DECIMAL)

    def test_RSquared(self):
        assert_almost_equal(self.res1.rsquared, self.res2.rsquared,DECIMAL)

    def test_AdjRSquared(self):
        assert_almost_equal(self.res1.rsquared_adj, self.res2.rsquared_adj,
                DECIMAL)

    def test_degrees(self):
        assert_almost_equal(self.res1.model.df_model, self.res2.df_model, DECIMAL)
        assert_almost_equal(self.res1.model.df_resid, self.res2.df_resid, DECIMAL)

    def test_ExplainedSumofSquares(self):
        assert_almost_equal(self.res1.ess, self.res2.ess, DECIMAL)

    def test_SumofSquaredResiduals(self):
        assert_almost_equal(self.res1.ssr, self.res2.ssr,DECIMAL)

    def test_MeanSquaredError(self):
        assert_almost_equal(self.res1.mse_model, self.res2.mse_model, DECIMAL)
        assert_almost_equal(self.res1.mse_resid, self.res2.mse_resid, DECIMAL)

    def test_FStatistic(self):
        assert_almost_equal(self.res1.fvalue, self.res2.fvalue, DECIMAL)

    def test_loglike(self):
        assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL)

    def test_AIC(self):
        assert_almost_equal(self.res1.aic, self.res2.aic, DECIMAL)

    def test_BIC(self):
        assert_almost_equal(self.res1.bic, self.res2.bic, DECIMAL)

    def test_pvalues(self):
        assert_almost_equal(self.res1.pvalues, self.res2.pvalues, DECIMAL)

    def test_wresid(self):
        if not hasattr(self.res2, 'wresid'):
            raise SkipTest('Comparison results (res2) has no wresid')
        else:
            assert_almost_equal(self.res1.wresid, self.res2.wresid, DECIMAL)

    def test_resids(self):
        assert_almost_equal(self.res1.resid, self.res2.resid, DECIMAL)

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

class TestFtest(object):
    def __init__(self):
        from models.datasets.longley.data import load
        data = load()
        self.data = data
        self.data.exog = add_constant(self.data.exog)
        self.res1 = OLS(data.endog, data.exog).fit()
        self.R = np.identity(7)[:-1,:]
        self.Ftest = self.res1.f_test(self.R)

    def test_F(self):
        assert_almost_equal(self.Ftest.fvalue, self.res1.fvalue, DECIMAL)

    def test_p(self):
        assert_almost_equal(self.Ftest.pvalue, self.res1.f_pvalue, DECIMAL)

    def test_Df_denom(self):
        assert_equal(self.Ftest.df_denom, self.res1.model.df_resid)

    def test_Df_num(self):
        assert_equal(self.Ftest.df_num, tools.rank(self.R))

class TestFTest2(TestFtest):
    '''
    A joint test that the coefficient on
    GNP = the coefficient on UNEMP  and that the coefficient on
    POP = the coefficient on YEAR for the Longley dataset.
    '''

    def setup(self):
        if skipR:
            raise SkipTest, "Rpy not installed"
        try:
            r.library('car')
        except RPy_RException:
            raise SkipTest, "car library not installed for R"
        self.R2 = [[0,1,-1,0,0,0,0],[0, 0, 0, 0, 1, -1, 0]]
        self.Ftest2 = self.res1.f_test(self.R2)
        self.R_Results = RModel(self.data.endog, self.data.exog, r.lm).robj
        self.F = r.linear_hypothesis(self.R_Results,
                r.c('x.2 = x.3', 'x.5 = x.6'))


    def test_F(self):
        assert_almost_equal(self.Ftest2.fvalue, self.F['F'][1], DECIMAL)

    def test_p(self):
        assert_almost_equal(self.Ftest2.pvalue, self.F['Pr(>F)'][1], DECIMAL)

    def test_Df_denom(self):
        assert_equal(self.Ftest2.df_denom, self.F['Res.Df'][0])

    def test_Df_num(self):
        self.F['Res.Df'].reverse()
        assert_equal(self.Ftest2.df_num, np.subtract.reduce(self.F['Res.Df']))

class TestTtest(object):
    '''
    Test individual t-tests.  Ie., are the coefficients significantly
    different than zero.
    '''
    def __init__(self):
        from models.datasets.longley.data import load
        data = load()
        self.data = data
        data.exog = add_constant(data.exog)
        self.res1 = OLS(data.endog, data.exog).fit()

    def setup(self):
        if skipR:
            raise SkipTest, "Rpy not installed"
        else:
            self.R = np.identity(len(self.res1.params))
            self.Ttest = self.res1.t_test(self.R)
            self.R_Results = RModel(self.data.endog, self.data.exog, r.lm).robj

    def test_T(self):
        assert_almost_equal(np.diag(self.Ttest.tvalue), self.res1.t(), DECIMAL)

    def test_sd(self):
        assert_almost_equal(np.diag(self.Ttest.sd), self.res1.bse, DECIMAL)

    def test_p(self):
        assert_almost_equal(np.diag(self.Ttest.pvalue),
                t.sf(np.abs(self.res1.t()),self.res1.model.df_resid), DECIMAL)

    def test_Df_denom(self):
        assert_equal(self.Ttest.df_denom, self.res1.model.df_resid)

    def test_effect(self):
        assert_almost_equal(self.Ttest.effect, self.res1.params)

#this test is broken wrong arguments in linear hypothesis
class TestTtest2(TestTtest):
    '''
    Tests the hypothesis that the coefficients on POP and YEAR
    are equal.
    '''
    def setup(self):
        if skipR:
            raise SkipTest, "Rpy not installed"
        try:
            r.library('car')
        except RPy_RException:
            raise SkipTest, "car library not installed for R"
        R = np.zeros(len(self.res1.params))
        R[4:6] = [1,-1]
        self.R = R
        self.Ttest1 = self.res1.t_test(self.R)
        self.R_Results = RModel(self.data.endog, self.data.exog, r.lm).robj
        self.Ttest2 = r.linear_hypothesis(self.R_Results['F'], r.c('x.5 = x.6',))
        t = np.inner(self.R, self.res1.params)/\
            (np.sign(np.inner(self.R, self.res1.params))*\
            np.sqrt(self.Ttest2['F'][1]))
        self.t = t
        effect = np.inner(self.R, self.res1.params)
        self.effect = effect


    def test_T(self):
        assert_equal(self.Ttest1.t, self.t, DECIMAL)

    def test_sd(self):
        assert_almost_equal(self.Ttest1.sd, effect/t, DECIMAL)

    def test_p(self):
        assert_almost_equal(self.Ftest2.pvalue, t.sf(self.t, self.F['Res.Df'][0]),
            DECIMAL)

    def test_Df_denom(self):
        assert_equal(self.Ftest2.df_denom, self.F['Res.Df'][0])

    def test_effect(self):
        assert_equal(self.Ttest1.effect, self.effect, DECIMAL)


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

#FIXME: I know this fails.  Need to get results from a GLS regression
# or WLS that allows n x n weights
    def test_standarderrors(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_lesser)

class test_gls_nosigma(check_regression_results):
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

    def check_confidenceintervals(self, conf1, conf2):
        assert_almost_equal(conf1, conf2, DECIMAL)


class test_wls(check_regression_results):
    '''
    GLM results are an implicit test of WLS
    '''
    def __init__(self):
        from models.datasets.ccard.data import load
        data = load()
        self.res1 = WLS(data.endog, data.exog, weights=1/data.exog[:,2]).fit()
        self.res2 = RModel(data.endog, data.exog, r.lm,
                weights=1/data.exog[:,2])
        self.res2.wresid = self.res2.rsum['residuals']
        self.res2.scale = self.res2.scale**2 # R has sigma not sigma**2
#FIXME: triaged results
        self.res1.ess = self.res1.uncentered_tss - self.res1.ssr
        self.res1.rsquared = self.res1.ess/self.res1.uncentered_tss
        self.res1.mse_model = self.res1.ess/(self.res1.df_model + 1)
        self.res1.fvalue = self.res1.mse_model/self.res1.mse_resid
        self.res1.rsquared_adj = 1 -(self.res1.nobs)/(self.res1.df_resid)*\
                (1-self.res1.rsquared)


    def setup(self):
        if skipR:
            raise SkipTest, "Rpy not installed"

    def check_confidenceintervals(self, conf1, conf2):
        assert_almost_equal(conf1, conf2, DECIMAL)

#NOTE: R reports the whitened residuals


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
        data = load()
        self.rho, self.sigma = yule_walker(data.endog, order=4, method="mle")
        R_results = r.ar(data.endog, aic="FALSE", order_max=4)
        self.R_params = R_results['ar']

    def setup(self):
        if skipR:
            raise SkipTest, "Rpy not installed."

    def test_params(self):
        assert_almost_equal(self.rho, self.R_params, DECIMAL)

if __name__=="__main__":
    #run_module_suite()
    import nose
    nose.runmodule(argv=[__file__,'-vvs','-x'], exit=False) #, '--pdb'

#  Robust error tests.  Compare values computed with SAS
#    res0 = SSM.regression.OLS(x).fit(y, HCC='HC0')
#    nptest.assert_almost_equal(res0.bse, sas_bse_HC0, 4)
#    res1 = SSM.regression.OLS(x).fit(y, HCC='HC1')
#    nptest.assert_almost_equal(res1.bse, sas_bse_HC1, 4)
#    res2 = SSM.regression.OLS(x).fit(y, HCC='HC2')
#    nptest.assert_almost_equal(res2.bse, sas_bse_HC2, 4)
#    res3 = SSM.regression.OLS(x).fit(y, HCC='HC3')
#    nptest.assert_almost_equal(res3.bse, sas_bse_HC3, 4)


