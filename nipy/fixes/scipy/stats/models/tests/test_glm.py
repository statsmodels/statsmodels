"""
Test functions for models.GLM
"""

import numpy as np
import numpy.random as R
from numpy.testing import *

import models
from models.glm import GLMtwo as GLM
from models.functions import add_constant, xi
from scipy import stats
from rmodelwrap import RModel

W = R.standard_normal

DECIMAL = 4

class check_model_results(object):
    '''
    res2 should be either the results from RModelWrap
    or the results as defined in model_results_data
    '''
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL)

    def test_standard_errors(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL)

# can't figure out how to have this evaluated on a case by case basis
# it appears as though the decorator is evaluated as soon
# as the class is inherited
#    @dec.skipif(R_model, "Skipped because results are from Rmodelwrap")
    def test_residuals(self):
        resids = np.column_stack((self.res1.resid_pearson, self.res1.resid_dev,
                    self.res1.resid_working, self.res1.resid_anscombe,
                    self.res1.resid_response))
        assert_almost_equal(resids, self.res2.resids, DECIMAL)

    def test_aic_R(self):
        assert_almost_equal(self.res1.information_criteria()['aic'],
                self.res2.aic_R, DECIMAL)

    # R does not provide good diagnostic residuals in any library
    # that I know of, though this could be added to Rmodelwrap
#    @dec.skipif(R_model, "Skipped because results are from Rmodelwrap")
    def test_aic_Stata(self):
        aic = self.res1.information_criteria()['aic']/self.res1.nobs
        assert_almost_equal(aic, self.res2.aic_Stata, DECIMAL)

    def test_deviance(self):
        assert_almost_equal(self.res1.deviance, self.res2.deviance, DECIMAL)

    def test_scale(self):
        assert_almost_equal(self.res1.scale, self.res2.scale, DECIMAL)

    def test_loglike(self):
        assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL)

    def test_null_deviance(self):
        assert_almost_equal(self.res1.null_deviance, self.res2.null_deviance,
                    DECIMAL)

    def test_bic(self):
        assert_almost_equal(self.res1.information_criteria()['aic'],
                    self.res2.bic, DECIMAL)

    def test_degrees(self):
        assert_almost_equal(self.res1.df_model,self.res2.df_model, DECIMAL)
        assert_almost_equal(self.res1.df_resid,self.res2.df_resid, DECIMAL)

    def test_pearsonX2(self):
        assert_almost_equal(self.res1.pearsonX2,self.res2.pearsonX2, DECIMAL)


class test_glm_gaussian(test_model_results, initialize):
    def __init__(self):
        '''
        Test Gaussian family with canonical identity link
        '''
        pass

    def test_log(self):
        pass

    def test_power(self):
        pass

class test_glm_binomial(test_model_results, initialize):
    def __init__(self):
        '''
        Test Binomial family with canonical logit link
        '''
        from models.datasets.star98.data import load
        data = load()
        data.exog = add_constant(data.exog)
        trials = data.endog[:,:2].sum(axis=1)
        self.res1 = GLM(data.endog, data.exog, \
        family=models.family.Binomial()).fit(data_weights = trials)

    def test_log(self):
        pass

    def test_logit(self):
        pass

    def test_probit(self):
        pass

    def test_cloglog(self):
        pass

    def test_power(self):
        pass

    def test_loglog(self):
        pass

    def test_logc(self):
        pass

class test_glm_bernoulli(check_model_results):
    def __init__(self):
        from model_results import lbw
        self.res2 = lbw()
        self.res1 = GLM(self.res2.endog, self.res2.exog,
                family=models.family.Binomial()).fit()

    def test_identity(self):
        pass

    def test_log(self):
        pass

    def test_probit(self):
        pass

    def test_cloglog(self):
        pass

    def test_power(self):
        pass

    def test_loglog(self):
        pass

    def test_logc(self):
        pass

class test_glm_gamma(test_model_results, initialize):
    def __init__(self):
        '''
        Tests Gamma family with canonical inverse link (power -1)
        '''
        from models.datasets.scotland.data import load
        from model_results import scotvote
        data = load()
        data.exog = add_constant(data.exog)
        self.res1 = GLM(data.endog, data.exog, \
                    family=models.family.Gamma()).fit()
        self.res2 = scotvote()

    def test_log(self):
        pass

    def test_power(self):
        pass

class test_glm_poisson(test_model_results, initialize):
    def __init__(self):
        '''
        Tests Poisson family with canonical log link.

        Test results were obtained by R.
        '''
        from model_results import cpunish
        from models.datasets.cpunish.data import load
        data = load()
        data.exog[:,3] = np.log(data.exog[:,3])
        data.exog = add_constant(data.exog)
        self.res1 = GLM(data.endog, data.exog,
                    family=models.family.Poisson()).fit()
        self.res2 = cpunish()

    def test_identity(self):
        pass

    def test_power(self):
        pass

class test_glm_invgauss(test_model_results, initialize):
    def __init__(self):
        '''
        Test Inverse Gaussian family with canonical power -2 link
        '''
        pass

    def test_log(self):
        pass

    def test_power(self):
        pass

class test_glm_negbinomial(test_model_results, initialize):
    def __init__(self):
        '''
        Test Negative Binomial family with canonical log link
        '''
        pass

    def test_log(self):
        pass

    def test_power(self):
        pass

    def test_nbinom(self):
        pass

class test_glm_gausslog(Check_Glm_R):
    def __init__(self):
        nobs = 100
        x = np.arange(nobs)
        y = 1.0 + 2.0*x + x**2 + 0.1 * np.random.randn(nobs)
        X = np.c_[np.ones((nobs,1)),x,x**2]
        lny = np.exp(-(-1.0 + 0.02*x + 0.0001*x**2)) +\
                        0.001 * np.random.randn(nobs)

        GaussLog_Model = GLM(lny, X, \
                family=models.family.Gaussian(models.family.links.log))
        GaussLog_Res = GaussLog_Model.fit()
        GaussLogLink = r.gaussian(link = "log")
        GaussLog_Res_R = RModel(lny, X, r.glm, family=GaussLogLink)
        self.res1 = GaussLog_res
        self.res2 = GaussLog_res_R


#class TestRegression(TestCase):
class TestRegression(object):
# this should allow the decorators to work
# see http://thread.gmane.org/gmane.comp.python.scientific.devel/9118

    def test_Logistic(self):
        X = W((40,10))
        Y = np.greater(W((40,)), 0)
        cmodel = GLM(Y,X, family=models.family.Binomial())
        results = cmodel.fit()
#        self.assertEquals(results.df_resid, 30)
        assert_equal(results.df_resid, 30)

    def test_Logisticdegenerate(self):
        X = W((40,10))
        X[:,0] = X[:,1] + X[:,2]
        Y = np.greater(W((40,)), 0)
        cmodel = GLM(Y, X, family=models.family.Binomial())
        results = cmodel.fit()
#        self.assertEquals(results.df_resid, 31)
        assert_equal(results.df_resid, 31)

### Refactoring Incomplete ###
    @dec.knownfailureif(True)
    def test_gaussian(self):
        from models.datasets.longley.data import load
        data = load()
        data.exog = add_constant(data.exog)
        ols_res = models.regression.OLS(data.endog,data.exog).fit()
        glm_res = GLM(data.endog,data.exog,
                        family=models.family.Gaussian()).fit()
        assert_array_equal(glm_res.params,ols_res.params)
        assert_equal(glm_res.llf,ols_res.llf)
        ols_aic = ols_res.information_criteria()['aic']
        glm_aic = glm_res.information_criteria()['aic']
        assert_equal(glm_aic,glm_aic)
        assert_equal(glm_res.bse,ols_res.bse)
        assert_equal(glm_res.scale, ols_res.scale)
        assert_equal(glm_res.resid_dev, ols_res.resid)
        assert_equal(glm_res.resid_pearson, ols_res.resid)
        assert_equal(glm_res.resid_working, ols_res.resid)
        assert_equal(glm_res.resid_response, ols_res.resid)
        assert_equal(glm_res.resid_anscombe, ols_res.resid)

    @dec.slow   # useful to know but could make it smaller subset of the data
    def test_inverse_guassian(self):
        '''
        Tests the Inverse Gaussian family in GLM.

        Notes
        -----
        Used the rndivgx.ado file provided by Hardin and Hilbe to
        generate the data.
        '''
        from model_results import inv_gauss
        self.res2 = inv_gauss()
        results = GLM(Y, X, family=models.family.InverseGaussian()).fit()

#        np.random.seed(54321)
#        x1 = np.abs(stats.norm.ppf((np.random.random(5000))))
#        x2 = np.abs(stats.norm.ppf((np.random.random(5000))))
#        X = np.column_stack((x1,x2))
#        X = add_constant(X)
#        params = np.array([.5, -.25, 1])
#        eta = np.dot(X, params)
#        mu = 1/np.sqrt(eta)
#        sigma = .5
#       This isn't correct.  Errors need to be normally distributed
#       But Y needs to be Inverse Gaussian, so we could build it up
#       by throwing out data?
#       Refs: Lai (2009) Generating inverse Gaussian random variates by
#        approximation
# Atkinson (1982) The simulation of generalized inverse gaussian and
#        hyperbolic random variables seems to be the canonical ref
#        Y = np.dot(X,params) + np.random.wald(mu, sigma, 1000)
#        model = GLM(Y, X, family=models.family.InverseGaussian(link=\
#            models.family.links.identity))

        R_params = (0.4519770, -0.2508288, 1.0359574)
        R_bse = (0.03148291, 0.02237211, 0.03429943)
        R_null_dev = 1520.673165475461
        R_df_null = 4999
        R_deviance = 1423.943980407997
        R_df_resid = 4997
        R_AIC = 5059.41911646446
        R_dispersion = 0.2867266359127567
        assert_almost_equal(results.params, R_params,5)
        assert_almost_equal(results.bse, R_bse,5)
        assert_almost_equal(results.null_deviance, R_null_dev,5)
        assert_almost_equal(results.df_resid+results.df_model,R_df_null,5)
        assert_almost_equal(results.deviance, R_deviance,5)
        assert_almost_equal(results.df_resid, R_df_resid,5)
        aic = results.information_criteria()['aic']
        assert_almost_equal(aic, R_AIC, 5)
        assert_almost_equal(results.scale, R_dispersion, 5)

# Stata

        Stata_AIC = 1.55228
        Stata_BIC = -41136.47
        Stata_PearsonX2 = 1432.771536
        Stata_estat_AIC = 7761.401
        Stata_estat_BIC = 7780.952
        Stata_llf = -3877.700354
        R_llf = -2525.70955823223

        print 'Figure out LLF for Inverse Gaussian'

    def test_negativebinomial():
        from models.datasets.committee.data import load
        data = load()
        data.exog[:,2] = np.log(data.exog[:,2])
        interaction = data.exog[:,2]*data.exog[:,1]
        data.exog = np.column_stack((data.exog,interaction))
        data.exog = add_constant(data.exog)

# Stata
        Stata_AIC = None
        Stata_BIC = None
        Stata_PearsonX2 = None
        Stata_llf = None
        Stata_params = None

if __name__=="__main__":
    run_module_suite()







