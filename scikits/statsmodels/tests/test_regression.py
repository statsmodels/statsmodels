"""
Test functions for models.regression
"""
import numpy as np
from numpy.testing import *
from scipy.linalg import toeplitz
from scikits.statsmodels.tools import add_constant, rank
from scikits.statsmodels.regression import OLS, GLSAR, WLS, GLS, yule_walker
from check_for_rpy import skip_rpy
from nose import SkipTest
from scipy.stats import t as student_t

DECIMAL = 4
DECIMAL_less = 3
DECIMAL_lesser = 2
DECIMAL_least = 1
DECIMAL_sig = 7
DECIMAL_none = 0
skipR = skip_rpy()
if not skipR:
    from rpy import r, RPyRException
    from rmodelwrap import RModel


class CheckRegressionResults(object):
    '''
    res2 contains results from Rmodelwrap or were obtained from a statistical
    packages such as R, Stata, or SAS and were written to model_results
    '''
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL)

    def test_standarderrors(self):
        assert_almost_equal(self.res1.bse,self.res2.bse, DECIMAL)

    def test_confidenceintervals(self):
        if hasattr(self.res2, 'conf_int'):
            self.check_confidenceintervals(self.res1.conf_int(),
                self.res2.conf_int)
        else:
            raise SkipTest, "Results from Rpy"

    def test_conf_int_subset(self):
        ci1 = self.res1.conf_int(cols=(1,2))
        ci2 = self.res1.conf_int()[1:3]
        assert_almost_equal(ci1, ci2, DECIMAL)

    def test_scale(self):
        assert_almost_equal(self.res1.scale, self.res2.scale, DECIMAL)

    def test_rsquared(self):
        if hasattr(self.res2, 'rsquared'):
            assert_almost_equal(self.res1.rsquared, self.res2.rsquared,DECIMAL)
        else:
            raise SkipTest, "Results from R"

    def test_rsquared_adju(self):
        if hasattr(self.res2, 'rsquared_adj'):
            assert_almost_equal(self.res1.rsquared_adj, self.res2.rsquared_adj,
                    DECIMAL)
        else:
            raise SkipTest, "Results from R"

    def test_degrees(self):
        if hasattr(self.res2, 'df_resid') and hasattr(self.res2, 'df_model'):
            assert_almost_equal(self.res1.model.df_model, self.res2.df_model, DECIMAL)
            assert_almost_equal(self.res1.model.df_resid, self.res2.df_resid, DECIMAL)
        else:
            raise SkipTest, "Results from R"

    def test_explained_sumof_squares(self):
        if hasattr(self.res2, 'ess'):
            assert_almost_equal(self.res1.ess, self.res2.ess, DECIMAL)
        else:
            raise SkipTest, "Results from Rpy"

    def test_sumof_squaredresids(self):
        if hasattr(self.res2, 'ssr'):
            assert_almost_equal(self.res1.ssr, self.res2.ssr,DECIMAL)
        else:
            raise SkipTest, "Results from Rpy"

    def test_mean_squared_error(self):
        if hasattr(self.res2, "mse_model") and hasattr(self.res2, "mse_resid"):
            assert_almost_equal(self.res1.mse_model, self.res2.mse_model, DECIMAL)
            assert_almost_equal(self.res1.mse_resid, self.res2.mse_resid, DECIMAL)
        else:
            raise SkipTest, "Results from Rpy"

    def test_fvalue(self):
        if hasattr(self.res2, 'fvalue'):
            assert_almost_equal(self.res1.fvalue, self.res2.fvalue, DECIMAL)
        else:
            raise SkipTest, "Results from R"

    def test_loglike(self):
        assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL)

    def test_aic(self):
        if hasattr(self.res2, 'aic'):
            assert_almost_equal(self.res1.aic, self.res2.aic, DECIMAL)
        else:
            raise SkipTest, "Results from Rpy"

    def test_bic(self):
        if hasattr(self.res2, 'bic'):
            assert_almost_equal(self.res1.bic, self.res2.bic, DECIMAL)
        else:
            raise SkipTest, "Results from Rpy"

    def test_pvalues(self):
        assert_almost_equal(self.res1.pvalues, self.res2.pvalues, DECIMAL)

    def test_wresid(self):
        if not hasattr(self.res2, 'wresid'):
            raise SkipTest('Comparison results (res2) has no wresid')
        else:
            assert_almost_equal(self.res1.wresid, self.res2.wresid, DECIMAL)

    def test_resids(self):
        assert_almost_equal(self.res1.resid, self.res2.resid, DECIMAL)

class TestOLS(CheckRegressionResults):
    def __init__(self):
        from scikits.statsmodels.datasets.longley import Load
        from model_results import Longley
        data = Load()
        data.exog = add_constant(data.exog)
        results = OLS(data.endog, data.exog).fit()
        self.res1 = results
        self.res2 = Longley()

    def check_confidenceintervals(self, conf1, conf2):
        for i in range(len(conf1)):
            assert_approx_equal(conf1[i][0], conf2[i][0], 6)
            assert_approx_equal(conf1[i][1], conf2[i][1], 6)
            # stata rounds big residuals to significant digits

    def check_params(self, params1, params2):
        assert_almost_equal(params1, params2, DECIMAL)

#  Robust error tests.  Compare values computed with SAS
    def test_HC0_errors(self):
        '''
        They are split up because the copied results do not have any decimal
        places for the last place.
        '''
        assert_almost_equal(self.res1.HC0_se[:-1],
                self.res2.HC0_se[:-1], DECIMAL)
        assert_approx_equal(np.round(self.res1.HC0_se[-1]), self.res2.HC0_se[-1])

    def test_HC1_errors(self):
        assert_almost_equal(self.res1.HC1_se[:-1],
                self.res2.HC1_se[:-1], DECIMAL)
        assert_approx_equal(self.res1.HC1_se[-1], self.res2.HC1_se[-1])

    def test_HC2_errors(self):
        assert_almost_equal(self.res1.HC2_se[:-1],
                self.res2.HC2_se[:-1], DECIMAL)
        assert_approx_equal(self.res1.HC2_se[-1], self.res2.HC2_se[-1])

    def test_HC3_errors(self):
        assert_almost_equal(self.res1.HC3_se[:-1],
                self.res2.HC3_se[:-1], DECIMAL)
        assert_approx_equal(self.res1.HC3_se[-1], self.res2.HC3_se[-1])

class TestFtest(object):
    def __init__(self):
        from scikits.statsmodels.datasets.longley import Load
        data = Load()
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
        assert_equal(self.Ftest.df_num, rank(self.R))

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
        except RPyRException:
            raise SkipTest, "car library not installed for R"
        self.R2 = [[0,1,-1,0,0,0,0],[0, 0, 0, 0, 1, -1, 0]]
        self.Ftest2 = self.res1.f_test(self.R2)
        self.R_Results = RModel(self.data.endog, self.data.exog, r.lm).robj
        self.F = r.linear_hypothesis(self.R_Results,
                r.c('x.2 = x.3', 'x.5 = x.6'))


    def test_fvalue(self):
        assert_almost_equal(self.Ftest2.fvalue, self.F['F'][1], DECIMAL)

    def test_pvalue(self):
        assert_almost_equal(self.Ftest2.pvalue, self.F['Pr(>F)'][1], DECIMAL)

    def test_df_denom(self):
        assert_equal(self.Ftest2.df_denom, self.F['Res.Df'][0])

    def test_df_num(self):
        self.F['Res.Df'].reverse()
        assert_equal(self.Ftest2.df_num, np.subtract.reduce(self.F['Res.Df']))

class TestTtest(object):
    '''
    Test individual t-tests.  Ie., are the coefficients significantly
    different than zero.
    '''
    def __init__(self):
        from scikits.statsmodels.datasets.longley import Load
        data = Load()
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

    def test_tvalue(self):
        assert_almost_equal(np.diag(self.Ttest.tvalue), self.res1.t(), DECIMAL)

    def test_sd(self):
        assert_almost_equal(np.diag(self.Ttest.sd), self.res1.bse, DECIMAL)

    def test_pvalue(self):
        assert_almost_equal(np.diag(self.Ttest.pvalue),
                student_t.sf(np.abs(self.res1.t()),self.res1.model.df_resid),
                    DECIMAL)

    def test_df_denom(self):
        assert_equal(self.Ttest.df_denom, self.res1.model.df_resid)

    def test_effect(self):
        assert_almost_equal(self.Ttest.effect, self.res1.params)

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
        except RPyRException:
            raise SkipTest, "car library not installed for R"
        R = np.zeros(len(self.res1.params))
        R[4:6] = [1,-1]
        self.R = R
        self.Ttest1 = self.res1.t_test(self.R)
        self.R_Results = RModel(self.data.endog, self.data.exog, r.lm).robj
        self.Ttest2 = r.linear_hypothesis(self.R_Results, 'x.5 = x.6')
        t = np.sign(np.inner(self.R, self.res1.params))*\
            np.sqrt(self.Ttest2['F'][1])
        self.t = t
        effect = np.inner(self.R, self.res1.params)
        self.effect = effect

    def test_tvalue(self):
        assert_almost_equal(self.Ttest1.tvalue, self.t, DECIMAL)

    def test_sd(self):
        assert_almost_equal(self.Ttest1.sd, self.effect/self.t, DECIMAL)

    def test_pvalue(self):
        assert_almost_equal(self.Ttest1.pvalue, student_t.sf(np.abs(self.t),
            self.Ttest2['Res.Df'][0]),
            DECIMAL)

    def test_df_denom(self):
        assert_equal(self.Ttest1.df_denom, self.Ttest2['Res.Df'][0])

    def test_effect(self):
        assert_almost_equal(self.Ttest1.effect, self.effect, DECIMAL)

class TestGLS(object):
    '''
    These test results were obtained by replication with R.
    '''
    def __init__(self):
        from scikits.statsmodels.datasets.longley import Load
        from model_results import LongleyGls

        data = Load()
        exog = add_constant(np.column_stack(\
                (data.exog[:,1],data.exog[:,4])))
        tmp_results = OLS(data.endog, exog).fit()
        rho = np.corrcoef(tmp_results.resid[1:],
                tmp_results.resid[:-1])[0][1] # by assumption
        order = toeplitz(np.arange(16))
        sigma = rho**order
        GLS_results = GLS(data.endog, exog, sigma=sigma).fit()
        self.res1 = GLS_results
        self.res2 = LongleyGls()

    def test_aic(self):
        assert_approx_equal(self.res1.aic+2, self.res2.aic, 3)

    def test_bic(self):
        assert_approx_equal(self.res1.bic, self.res2.bic, 2)

    def test_loglike(self):
        assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL_none)

    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_least)

    def test_resid(self):
        assert_almost_equal(self.res1.resid, self.res2.resid, DECIMAL)

    def test_scale(self):
        assert_almost_equal(self.res1.scale, self.res2.scale, DECIMAL)

    def test_tvalues(self):
        assert_almost_equal(self.res1.t(), self.res2.t, DECIMAL)

    def test_standarderrors(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL)

    def test_fittedvalues(self):
        assert_almost_equal(self.res1.fittedvalues, self.res2.fittedvalues, DECIMAL)

    def test_pvalues(self):
        assert_almost_equal(self.res1.pvalues, self.res2.pvalues, DECIMAL)

class TestGLS_nosigma(CheckRegressionResults):
    '''
    Test that GLS with no argument is equivalent to OLS.
    '''
    def __init__(self):
        from scikits.statsmodels.datasets.longley import Load
        data = Load()
        data.exog = add_constant(data.exog)
        ols_res = OLS(data.endog, data.exog).fit()
        gls_res = GLS(data.endog, data.exog).fit()
        self.res1 = gls_res
        self.res2 = ols_res
        self.res2.conf_int = self.res2.conf_int()

    def check_confidenceintervals(self, conf1, conf2):
        assert_almost_equal(conf1, conf2, DECIMAL)

class TestWLS(CheckRegressionResults):
    '''
    Test WLS with Greene's credit card data
    '''
    def __init__(self):
        from scikits.statsmodels.datasets.ccard import Load
        data = Load()
        self.res1 = WLS(data.endog, data.exog, weights=1/data.exog[:,2]).fit()
        self.res2 = RModel(data.endog, data.exog, r.lm,
                weights=1/data.exog[:,2])
        self.res2.wresid = self.res2.rsum['residuals']
        self.res2.scale = self.res2.scale**2 # R has sigma not sigma**2
#FIXME: triaged results for noconstant
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


class TestWLS_GLS(CheckRegressionResults):
    def __init__(self):
        from scikits.statsmodels.datasets.ccard import Load
        data = Load()
        self.res1 = WLS(data.endog, data.exog, weights = 1/data.exog[:,2]).fit()
        self.res2 = GLS(data.endog, data.exog, sigma = data.exog[:,2]).fit()

    def check_confidenceintervals(self, conf1, conf2):
        assert_almost_equal(conf1, conf2(), DECIMAL)

class TestWLS_OLS(CheckRegressionResults):
    def __init__(self):
        from scikits.statsmodels.datasets.longley import Load
        data = Load()
        data.exog = add_constant(data.exog)
        self.res1 = OLS(data.endog, data.exog).fit()
        self.res2 = WLS(data.endog, data.exog).fit()

    def check_confidenceintervals(self, conf1, conf2):
        assert_almost_equal(conf1, conf2(), DECIMAL)

class TestGLS_OLS(CheckRegressionResults):
    def __init__(self):
        from scikits.statsmodels.datasets.longley import Load
        data = Load()
        data.exog = add_constant(data.exog)
        self.res1 = GLS(data.endog, data.exog).fit()
        self.res2 = OLS(data.endog, data.exog).fit()

    def check_confidenceintervals(self, conf1, conf2):
        assert_almost_equal(conf1, conf2(), DECIMAL)

#TODO: test AR
# why the two-stage in AR?
#class test_ar(object):
#    from scikits.statsmodels.datasets.sunspots import Load
#    data = Load()
#    model = AR(data.endog, rho=4).fit()
#    R_res = RModel(data.endog, aic="FALSE", order_max=4)

#    def test_params(self):
#        assert_almost_equal(self.model.rho,
#        pass

#    def test_order(self):
# In R this can be defined or chosen by minimizing the AIC if aic=True
#        pass


class TestYuleWalker(object):
    def __init__(self):
        from scikits.statsmodels.datasets.sunspots import Load
        data = Load()
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




