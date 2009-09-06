"""

Test functions for models.GLM
"""

import numpy as np
from numpy.testing import *
import scikits.statsmodels as models
from scikits.statsmodels.glm import GLM
from scikits.statsmodels.tools import add_constant
from nose import SkipTest
from check_for_rpy import skip_rpy

DECIMAL = 4
DECIMAL_less = 3
DECIMAL_lesser = 2
DECIMAL_least = 1
DECIMAL_none = 0
skipR = skip_rpy()
if not skipR:
    from rpy import r
    from rmodelwrap import RModel

class CheckModelResults(object):
    '''
    res2 should be either the results from RModelWrap
    or the results as defined in model_results_data
    '''
    def test_params(self):
        self.check_params(self.res1.params, self.res2.params)

    def test_standard_errors(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL)

    def test_residuals(self):
        if 'rmodelwrap' in self.res2.__module__ and not hasattr(self.res2, 'resids'):
           assert_almost_equal(self.res1.resid_deviance, self.res2.resid_deviance,
                DECIMAL)
        else:
            resids = np.column_stack((self.res1.resid_pearson,
            self.res1.resid_deviance, self.res1.resid_working,
            self.res1.resid_anscombe, self.res1.resid_response))
            self.check_resids(resids, self.res2.resids)

    def test_aic_R(self):
        # R includes the estimation of the scale as a lost dof
        # Doesn't with Gamma though
        if self.res1.scale != 1:
            dof = 2
        else: dof = 0
        self.check_aic_R(self.res1.aic+dof,
                self.res2.aic_R)

    def test_aic_Stata(self):
        if 'rmodelwrap' in self.res2.__module__:
            raise SkipTest("Results are from RModel wrapper")
        aic = self.res1.aic/self.res1.nobs
        self.check_aic_Stata(aic, self.res2.aic_Stata)

    def test_deviance(self):
        assert_almost_equal(self.res1.deviance, self.res2.deviance, DECIMAL)

    def test_scale(self):
        assert_almost_equal(self.res1.scale, self.res2.scale, DECIMAL)

    def test_loglike(self):
        self.check_loglike(self.res1.llf, self.res2.llf)

    def test_null_deviance(self):
        assert_almost_equal(self.res1.null_deviance, self.res2.null_deviance,
                    DECIMAL)

    def test_bic(self):
        if 'rmodelwrap' in self.res2.__module__ and not hasattr(self.res2, 'bic'):
            raise SkipTest("Results are from RModel wrapper")
        self.check_bic(self.res1.bic,
            self.res2.bic)

    def test_degrees(self):
        if not 'rmodelwrap' in self.res2.__module__:
            assert_almost_equal(self.res1.model.df_model,self.res2.df_model, DECIMAL)
        assert_almost_equal(self.res1.model.df_resid,self.res2.df_resid, DECIMAL)

    def test_pearson_chi2(self):
        if 'rmodelwrap' in self.res2.__module__:
            raise SkipTest("Results are from RModel wrapper")
        self.check_pearson_chi2(self.res1.pearson_chi2, self.res2.pearson_chi2)

class TestGlmGaussian(CheckModelResults):
    def __init__(self):
        '''
        Test Gaussian family with canonical identity link
        '''

        from scikits.statsmodels.datasets.longley import Load
        self.data = Load()
        self.data.exog = add_constant(self.data.exog)
        self.res1 = GLM(self.data.endog, self.data.exog,
                        family=models.family.Gaussian()).fit()
                                            # I think this is a bug in Rpy

    def setup(self):
        if skipR:
            raise SkipTest, "Rpy not installed."
        Gauss = r.gaussian
        self.res2 = RModel(self.data.endog, self.data.exog, r.glm, family=Gauss)
        self.res2.resids = np.array(self.res2.resid)[:,None]*np.ones((1,5))
        self.res2.null_deviance = 185008826 # taken from R.


    def check_params(self, params1, params2):
        assert_almost_equal(params1, params2, DECIMAL)

    def check_resids(self, resids1, resids2):
        assert_almost_equal(resids1, resids2, DECIMAL)

    def check_aic_R(self, aic1, aic2):
        assert_almost_equal(aic1, aic2, DECIMAL)

    def check_aic_Stata(self, aic1, aic2):
        assert_almost_equal(aic1, aic2, DECIMAL)

    def check_loglike(self, llf1, llf2):
        assert_almost_equal(llf1, llf2, DECIMAL)

    def check_bic(self, bic1, bic2):
        assert_almost_equal(bic1, bic2, DECIMAL)

    def check_pearson_chi2(self, pearson_chi21, pearson_chi22):
        assert_almost_equal(pearson_chi21, pearson_chi22, DECIMAL)

class TestGaussianLog(CheckModelResults):
    def __init__(self):
        nobs = 100
        x = np.arange(nobs)
        np.random.seed(54321)
#        y = 1.0 - .02*x - .001*x**2 + 0.001 * np.random.randn(nobs)
        self.X = np.c_[np.ones((nobs,1)),x,x**2]
        self.lny = np.exp(-(-1.0 + 0.02*x + 0.0001*x**2)) +\
                        0.001 * np.random.randn(nobs)

        GaussLog_Model = GLM(self.lny, self.X, \
                family=models.family.Gaussian(models.family.links.log))
        GaussLog_Res = GaussLog_Model.fit()
        self.res1 = GaussLog_Res

    def setup(self):
        if skipR:
            raise SkipTest, "Rpy not installed"
        GaussLogLink = r.gaussian(link = "log")
        GaussLog_Res_R = RModel(self.lny, self.X, r.glm, family=GaussLogLink)
        self.res2 = GaussLog_Res_R


    def test_null_deviance(self):
        assert_almost_equal(self.res1.null_deviance, self.res2.null_deviance,
                    DECIMAL_least)

    def check_params(self, params1, params2):
        assert_almost_equal(params1, params2, DECIMAL)

    def check_loglike(self, llf1, llf2):
        assert_almost_equal(llf1, llf2, DECIMAL_none)

    def check_aic_R(self, aic1, aic2):
        assert_almost_equal(aic1, aic2, DECIMAL_none)
#TODO: is this a full test?

class TestGaussianInverse(CheckModelResults):
    def __init__(self):
        nobs = 100
        x = np.arange(nobs)
        np.random.seed(54321)
        y = 1.0 + 2.0 * x + x**2 + 0.1 * np.random.randn(nobs)
        self.X = np.c_[np.ones((nobs,1)),x,x**2]
        self.y_inv = (1. + .02*x + .001*x**2)**-1 + .001 * np.random.randn(nobs)
        InverseLink_Model = GLM(self.y_inv, self.X,
                family=models.family.Gaussian(models.family.links.inverse))
        InverseLink_Res = InverseLink_Model.fit()
        self.res1 = InverseLink_Res

    def setup(self):
        if skipR:
            raise SkipTest, "Rpy not installed."
        InverseLink = r.gaussian(link = "inverse")
        InverseLink_Res_R = RModel(self.y_inv, self.X, r.glm, family=InverseLink)
        self.res2 = InverseLink_Res_R

    def check_params(self, params1, params2):
        assert_almost_equal(params1, params2, DECIMAL)

    def check_loglike(self, llf1, llf2):
        assert_almost_equal(llf1, llf2, DECIMAL_least)

    def check_aic_R(self, aic1, aic2):
        assert_almost_equal(aic1, aic2, DECIMAL_least)
#TODO: is this a full test?

    @dec.knownfailureif(True, "This is a bug in Rpy")
    def test_null_deviance(self):
        assert_almost_equal(self.res1.null_deviance, self.res2.null_deviance,
                    DECIMAL_least)

class TestGlmBinomial(CheckModelResults):
    def __init__(self):
        '''
        Test Binomial family with canonical logit link
        '''
        from scikits.statsmodels.datasets.star98 import Load
        from model_results import Star98
        self.data = Load()
        self.data.exog = add_constant(self.data.exog)
        trials = self.data.endog[:,:2].sum(axis=1)
        self.res1 = GLM(self.data.endog, self.data.exog, \
        family=models.family.Binomial()).fit(data_weights = trials)
        self.res2 = Star98()

    def check_params(self, params1, params2):
        assert_almost_equal(params1, params2, DECIMAL)

    def check_resids(self, resids1, resids2):
        assert_almost_equal(resids1, resids2, DECIMAL_least)
        # rounding difference vs. stata

    def check_aic_R(self, aic1, aic2):
        assert_almost_equal(aic1, aic2, DECIMAL)

    def check_aic_Stata(self, aic1, aic2):
        assert_almost_equal(aic1, aic2, DECIMAL)

    def check_loglike(self, llf1, llf2):
        assert_almost_equal(llf1, llf2, DECIMAL_less)
        # precise up to 3 decimals

    def check_bic(self, bic1, bic2):
        assert_almost_equal(bic1, bic2, DECIMAL_lesser)
        # accurate to 1e-02

    def check_pearson_chi2(self, pearson_chi21, pearson_chi22):
        assert_almost_equal(pearson_chi21, pearson_chi22, DECIMAL_lesser)
        # Pearson's X2 sums residuals that are rounded differently in Stata
#TODO:
#Non-Canonical Links for the Binomial family require the algorithm to be
#slightly changed
#class TestGlmBinomialLog(CheckModelResults):
#    pass

#class TestGlmBinomialLogit(CheckModelResults):
#    pass

#class TestGlmBinomialProbit(CheckModelResults):
#    pass

#class TestGlmBinomialCloglog(CheckModelResults):
#    pass

#class TestGlmBinomialPower(CheckModelResults):
#    pass

#class TestGlmBinomialLoglog(CheckModelResults):
#    pass

#class TestGlmBinomialLogc(CheckModelResults):
#TODO: need include logc link
#    pass

class TestGlmBernoulli(CheckModelResults):
    def __init__(self):
        from model_results import Lbw
        self.res2 = Lbw()
        self.res1 = GLM(self.res2.endog, self.res2.exog,
                family=models.family.Binomial()).fit()

    def check_params(self, params1, params2):
        assert_almost_equal(params1, params2, DECIMAL)

    def check_resids(self, resids1, resids2):
        assert_almost_equal(resids1, resids2, DECIMAL)

    def check_aic_R(self, aic1, aic2):
        assert_almost_equal(aic1, aic2, DECIMAL)

    def check_aic_Stata(self, aic1, aic2):
        assert_almost_equal(aic1, aic2, DECIMAL)

    def check_loglike(self, llf1, llf2):
        assert_almost_equal(llf1, llf2, DECIMAL)

    def check_bic(self, bic1, bic2):
        assert_almost_equal(bic1, bic2, DECIMAL)

    def check_pearson_chi2(self, pearson_chi21, pearson_chi22):
        assert_almost_equal(pearson_chi21, pearson_chi22, DECIMAL)

#class TestGlmBernoulliIdentity(CheckModelResults):
#    pass

#class TestGlmBernoulliLog(CheckModelResults):
#    pass

#class TestGlmBernoulliProbit(CheckModelResults):
#    pass

#class TestGlmBernoulliCloglog(CheckModelResults):
#    pass

#class TestGlmBernoulliPower(CheckModelResults):
#    pass

#class TestGlmBernoulliLoglog(CheckModelResults):
#    pass

#class test_glm_bernoulli_logc(CheckModelResults):
#    pass

class TestGlmGamma(CheckModelResults):

    def __init__(self):
        '''
        Tests Gamma family with canonical inverse link (power -1)
        '''
        from scikits.statsmodels.datasets.scotland import Load
        from model_results import Scotvote
        self.data = Load()
        self.data.exog = add_constant(self.data.exog)
        self.res1 = GLM(self.data.endog, self.data.exog, \
                    family=models.family.Gamma()).fit()
        self.res2 = Scotvote()

    def check_params(self, params1, params2):
        assert_almost_equal(params1, params2, DECIMAL)

    def check_resids(self, resids1, resids2):
        assert_almost_equal(resids1, resids2, DECIMAL_lesser)

    def check_aic_R(self, aic1, aic2):
        assert_approx_equal(aic1-2, aic2, DECIMAL_less)
        # R includes another degree of freedom in calculation of AIC, but not with
        # gamma for some reason
        # There is also a precision issue due to a different implementation

    def check_aic_Stata(self, aic1, aic2):
        llf1 = self.res1.model.family.loglike(self.res1.model.endog,
                self.res1.mu, scale=1)
        aic1 = 2 *(self.res1.model.df_model + 1 - llf1)/self.res1.nobs
        assert_almost_equal(aic1, aic2, DECIMAL)

    def check_loglike(self, llf1, llf2):
        llf1 = self.res1.model.family.loglike(self.res1.model.endog,
                self.res1.mu, scale=1)
        assert_almost_equal(llf1, llf2, DECIMAL)

    def check_bic(self, bic1, bic2):
        assert_almost_equal(bic1, bic2, DECIMAL)

    def check_pearson_chi2(self, pearson_chi21, pearson_chi22):
        assert_almost_equal(pearson_chi21, pearson_chi22, DECIMAL)

class TestGlmGammaLog(CheckModelResults):
    def __init__(self):
        from model_results import Cancer
        self.data = Cancer()
        self.res1 = GLM(self.data.endog, self.data.exog,
            family=models.family.Gamma(link=models.family.links.log)).fit()

    def setup(self):
        if skipR:
            raise SkipTest, "Rpy not installed."
        self.res2 = RModel(self.data.endog, self.data.exog, r.glm,
            family=r.Gamma(link="log"))
        self.res2.null_deviance = 27.92207137420696 # From R (bug in rpy)
        self.res2.bic = -154.1582 # from Stata


    def check_params(self, params1, params2):
        assert_almost_equal(params1, params2, DECIMAL)

    def check_resids(self, resids1, resids2):
        assert_almost_equal(resids1, resids2, DECIMAL)

    def check_aic_R(self, aic1, aic2):
        assert_almost_equal(aic1, aic2, DECIMAL_none)

    def check_loglike(self, llf1, llf2):
        assert_almost_equal(llf1, llf2, DECIMAL_least)

    def check_bic(self, bic1, bic2):
        assert_almost_equal(bic1, bic2, DECIMAL)

class TestGlmGammaIdentity(CheckModelResults):
    def __init__(self):
        from model_results import Cancer
        self.data = Cancer()
        self.res1 = GLM(self.data.endog, self.data.exog,
            family=models.family.Gamma(link=models.family.links.identity)).fit()

    def setup(self):
        if skipR:
            raise SkipTest, "Rpy not installed."
        self.res2 = RModel(self.data.endog, self.data.exog, r.glm,
            family=r.Gamma(link="identity"))
        self.res2.null_deviance = 27.92207137420696 # from R, Rpy bug

    def check_params(self, params1, params2):
        assert_almost_equal(params1, params2, DECIMAL_lesser)

    def check_resids(self, resids1, resids2):
        assert_almost_equal(resids1, resids2, DECIMAL)

    def check_aic_R(self, aic1, aic2):
        assert_almost_equal(aic1, aic2, DECIMAL_none)

    def check_loglike(self, llf1, llf2):
        assert_almost_equal(llf1, llf2, DECIMAL_least)

    def check_bic(self, bic1, bic2):
        assert_almost_equal(bic1, bic2, DECIMAL)

class TestGlmPoisson(CheckModelResults):
    def __init__(self):
        '''
        Tests Poisson family with canonical log link.

        Test results were obtained by R.
        '''
        from model_results import Cpunish
        from scikits.statsmodels.datasets.cpunish import Load
        self.data = Load()
        self.data.exog[:,3] = np.log(self.data.exog[:,3])
        self.data.exog = add_constant(self.data.exog)
        self.res1 = GLM(self.data.endog, self.data.exog,
                    family=models.family.Poisson()).fit()
        self.res2 = Cpunish()

    def check_params(self, params1, params2):
        assert_almost_equal(params1, params2, DECIMAL)

    def check_resids(self, resids1, resids2):
        assert_almost_equal(resids1, resids2, DECIMAL)

    def check_aic_R(self, aic1, aic2):
        assert_almost_equal(aic1, aic2, DECIMAL)

    def check_aic_Stata(self, aic1, aic2):
        assert_almost_equal(aic1, aic2, DECIMAL)

    def check_loglike(self, llf1, llf2):
        assert_almost_equal(llf1, llf2, DECIMAL)

    def check_bic(self, bic1, bic2):
        assert_almost_equal(bic1, bic2, DECIMAL)

    def check_pearson_chi2(self, pearson_chi21, pearson_chi22):
        assert_almost_equal(pearson_chi21, pearson_chi22, DECIMAL)

#class TestGlmPoissonIdentity(CheckModelResults):
#    pass

#class TestGlmPoissonPower(CheckModelResults):
#    pass

class TestGlmInvgauss(CheckModelResults):
    def __init__(self):
        '''
        Tests the Inverse Gaussian family in GLM.

        Notes
        -----
        Used the rndivgx.ado file provided by Hardin and Hilbe to
        generate the data.  Results are read from model_results, which
        were obtained by running R_ig.s
        '''

        from model_results import InvGauss
        self.res2 = InvGauss()
        self.res1 = GLM(self.res2.endog, self.res2.exog, \
                family=models.family.InverseGaussian()).fit()

#    def setup(self):
#        if skipR:
#            raise nose.SkipTest('requires rpy')

    def check_params(self, params1, params2):
        assert_almost_equal(params1, params2, DECIMAL)

    def check_resids(self, resids1, resids2):
        assert_almost_equal(resids1, resids2, DECIMAL)

    def check_aic_R(self, aic1, aic2):
        assert_approx_equal(aic1, aic2, DECIMAL)
        # Off by 2e-1 due to implementation difference

    def check_aic_Stata(self, aic1, aic2):
        llf1 = self.res1.model.family.loglike(self.res1.model.endog, self.res1.mu,
                scale=1)
        aic1 = 2 * (self.res1.model.df_model + 1 - llf1)/self.res1.nobs
        assert_almost_equal(aic1, aic2, DECIMAL)

    def check_loglike(self, llf1, llf2):
        llf1 = self.res1.model.family.loglike(self.res1.model.endog, self.res1.mu,
                scale=1)    # Stata assumes scale = 1 in calc,
                            # which shouldn't be right...
        assert_almost_equal(llf1, llf2, DECIMAL_less)

    def check_bic(self, bic1, bic2):
        assert_almost_equal(bic1, bic2, DECIMAL_lesser) # precision in STATA

    def check_pearson_chi2(self, pearson_chi21, pearson_chi22):
        assert_almost_equal(pearson_chi21, pearson_chi22, DECIMAL_less)# summed resids

class TestGlmInvgaussLog(CheckModelResults):
    def __init__(self):
        from model_results import Medpar1
        self.data = Medpar1()
        self.res1 = GLM(self.data.endog, self.data.exog,
            family=models.family.InverseGaussian(link=\
            models.family.links.log)).fit()
                                     # common across Gamma implementation

    def setup(self):
        if skipR:
            raise SkipTest, "Rpy not installed."
        self.res2 = RModel(self.data.endog, self.data.exog, r.glm,
            family=r.inverse_gaussian(link="log"))
        self.res2.null_deviance = 335.1539777981053 # from R, Rpy bug
        self.res2.llf = -12162.72308 # from Stata, R's has big rounding diff

    def check_params(self, params1, params2):
        assert_almost_equal(params1, params2, DECIMAL)

    def check_resids(self, resids1, resids2):
        assert_almost_equal(resids1, resids2, DECIMAL)

    @dec.knownfailureif(True, "Big rounding difference vs. R")
    def check_aic_R(self, aic1, aic2):
        assert_almost_equal(aic1, aic2, DECIMAL)

    def check_loglike(self, llf1, llf2):
        llf1 = self.res1.model.family.loglike(self.res1.model.endog,
                self.res1.mu, scale=1)
        assert_almost_equal(llf1, llf2, DECIMAL)

class TestGlmInvgaussIdentity(CheckModelResults):
    def __init__(self):
        from model_results import Medpar1
        self.data = Medpar1()
        self.res1 = GLM(self.data.endog, self.data.exog,
            family=models.family.InverseGaussian(link=\
            models.family.links.identity)).fit()

    def setup(self):
        if skipR:
            raise SkipTest, "Rpy not installed."
        self.res2 = RModel(self.data.endog, self.data.exog, r.glm,
            family=r.inverse_gaussian(link="identity"))
        self.res2.null_deviance = 335.1539777981053 # from R, Rpy bug
        self.res2.llf = -12163.25545    # from Stata, big diff with R

    def check_params(self, params1, params2):
        assert_almost_equal(params1, params2, DECIMAL_less)

    def check_resids(self, resids1, resids2):
        assert_almost_equal(resids1, resids2, DECIMAL)

    @dec.knownfailureif(True, "Big rounding difference vs R")
    def check_aic_R(self, aic1, aic2):
        assert_almost_equal(aic1, aic2, DECIMAL)

    def check_loglike(self, llf1, llf2):
        llf1 = self.res1.model.family.loglike(self.res1.model.endog,
                self.res1.mu, scale=1)
        assert_almost_equal(llf1, llf2, DECIMAL)

class TestGlmNegbinomial(CheckModelResults):
    def __init__(self):
        '''
        Test Negative Binomial family with canonical log link
        '''
        from scikits.statsmodels.datasets.committee import Load
        self.data = Load()
        self.data.exog[:,2] = np.log(self.data.exog[:,2])
        interaction = self.data.exog[:,2]*self.data.exog[:,1]
        self.data.exog = np.column_stack((self.data.exog,interaction))
        self.data.exog = add_constant(self.data.exog)
        results = GLM(self.data.endog, self.data.exog,
                family=models.family.NegativeBinomial()).fit()
        self.res1 = results
        # Rpy does not return the same null deviance as R for some reason

    def setup(self):
        if skipR:
            raise SkipTest, "Rpy not installed"
        r.library('MASS')  # this doesn't work when done in rmodelwrap?
        self.res2 = RModel(self.data.endog, self.data.exog, r.glm,
                family=r.negative_binomial(1))
        self.res2.null_deviance = 27.8110469364343

    def check_params(self, params1, params2):
        assert_almost_equal(params1, params2, DECIMAL-1)    # precision issue

    def check_resids(self, resids1, resids2):
        assert_almost_equal(resids1, resids2, DECIMAL)

    def check_aic_R(self, aic1, aic2):
        assert_almost_equal(aic1-2, aic2, DECIMAL)
        # note that R subtracts an extra degree of freedom for estimating
        # the scale

    def check_aic_Stata(self, aic1, aic2):
        aic1 = aci1/self.res1.nobs
        assert_almost_equal(aic1, aic2, DECIMAL)

    def check_loglike(self, llf1, llf2):
        assert_almost_equal(llf1, llf2, DECIMAL)

    def check_bic(self, bic1, bic2):
        assert_almost_equal(bic1, bic2, DECIMAL)

    def check_pearson_chi2(self, pearson_chi21, pearson_chi22):
        assert_almost_equal(pearson_chi21, pearson_chi22, DECIMAL)

#class TestGlmNegbinomial_log(CheckModelResults):
#    pass

#class TestGlmNegbinomial_power(CheckModelResults):
#    pass

#class TestGlmNegbinomial_nbinom(CheckModelResults):
#    pass

if __name__=="__main__":
    #run_module_suite()
    #taken from Fernando Perez:
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb'],
                       exit=False)
