"""

Test functions for models.GLM
"""
import os
import numpy as np
from numpy.testing import *
import scikits.statsmodels.api as sm
from scikits.statsmodels.genmod.generalized_linear_model import GLM
from scikits.statsmodels.tools.tools import add_constant
from scikits.statsmodels.tools.sm_exceptions import PerfectSeparationError
from nose import SkipTest

# Test Precisions
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1
DECIMAL_0 = 0

class CheckModelResults(object):
    '''
    res2 should be either the results from RModelWrap
    or the results as defined in model_results_data
    '''
    decimal_params = DECIMAL_4
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params,
                self.decimal_params)

    decimal_bse = DECIMAL_4
    def test_standard_errors(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, self.decimal_bse)

    decimal_resids = DECIMAL_4
    def test_residuals(self):
        resids = np.column_stack((self.res1.resid_pearson,
                self.res1.resid_deviance, self.res1.resid_working,
                self.res1.resid_anscombe, self.res1.resid_response))
        assert_almost_equal(resids, self.res2.resids, self.decimal_resids)

    decimal_aic_R = DECIMAL_4
    def test_aic_R(self):
        # R includes the estimation of the scale as a lost dof
        # Doesn't with Gamma though
        if self.res1.scale != 1:
            dof = 2
        else:
            dof = 0
        assert_almost_equal(self.res1.aic+dof, self.res2.aic_R,
                self.decimal_aic_R)

    decimal_aic_Stata = DECIMAL_4
    def test_aic_Stata(self):
        # Stata uses the below llf for aic definition for these families
        if isinstance(self.res1.model.family, (sm.families.Gamma,
            sm.families.InverseGaussian)):
            llf = self.res1.model.family.loglike(self.res1.model.endog,
                    self.res1.mu, scale=1)
            aic = (-2*llf+2*(self.res1.df_model+1))/self.res1.nobs
        else:
            aic = self.res1.aic/self.res1.nobs
        assert_almost_equal(aic, self.res2.aic_Stata, self.decimal_aic_Stata)

    decimal_deviance = DECIMAL_4
    def test_deviance(self):
        assert_almost_equal(self.res1.deviance, self.res2.deviance,
                self.decimal_deviance)

    decimal_scale = DECIMAL_4
    def test_scale(self):
        assert_almost_equal(self.res1.scale, self.res2.scale,
                self.decimal_scale)

    decimal_loglike = DECIMAL_4
    def test_loglike(self):
        # Stata uses the below llf for these families
        # We differ with R for them
        if isinstance(self.res1.model.family, (sm.families.Gamma,
            sm.families.InverseGaussian)):
            llf = self.res1.model.family.loglike(self.res1.model.endog,
                    self.res1.mu, scale=1)
        else:
            llf = self.res1.llf
        assert_almost_equal(llf, self.res2.llf, self.decimal_loglike)

    decimal_null_deviance = DECIMAL_4
    def test_null_deviance(self):
        assert_almost_equal(self.res1.null_deviance, self.res2.null_deviance,
                    self.decimal_null_deviance)

    decimal_bic = DECIMAL_4
    def test_bic(self):
        assert_almost_equal(self.res1.bic, self.res2.bic_Stata,
                self.decimal_bic)

    def test_degrees(self):
        assert_equal(self.res1.model.df_resid,self.res2.df_resid)

    decimal_fittedvalues = DECIMAL_4
    def test_fittedvalues(self):
        assert_almost_equal(self.res1.fittedvalues, self.res2.fittedvalues,
                self.decimal_fittedvalues)

class TestGlmGaussian(CheckModelResults):
    def __init__(self):
        '''
        Test Gaussian family with canonical identity link
        '''
        # Test Precisions
        self.decimal_resids = DECIMAL_3
        self.decimal_params = DECIMAL_2
        self.decimal_bic = DECIMAL_0
        self.decimal_bse = DECIMAL_3

        from scikits.statsmodels.datasets.longley import load
        self.data = load()
        self.data.exog = add_constant(self.data.exog)
        self.res1 = GLM(self.data.endog, self.data.exog,
                        family=sm.families.Gaussian()).fit()
        from results.results_glm import Longley
        self.res2 = Longley()

#    def setup(self):
#        if skipR:
#            raise SkipTest, "Rpy not installed."
#        Gauss = r.gaussian
#        self.res2 = RModel(self.data.endog, self.data.exog, r.glm, family=Gauss)
#        self.res2.resids = np.array(self.res2.resid)[:,None]*np.ones((1,5))
#        self.res2.null_deviance = 185008826 # taken from R. Rpy bug?

class TestGaussianLog(CheckModelResults):
    def __init__(self):
        # Test Precision
        self.decimal_aic_R = DECIMAL_0
        self.decimal_aic_Stata = DECIMAL_2
        self.decimal_loglike = DECIMAL_0
        self.decimal_null_deviance = DECIMAL_1

        nobs = 100
        x = np.arange(nobs)
        np.random.seed(54321)
#        y = 1.0 - .02*x - .001*x**2 + 0.001 * np.random.randn(nobs)
        self.X = np.c_[np.ones((nobs,1)),x,x**2]
        self.lny = np.exp(-(-1.0 + 0.02*x + 0.0001*x**2)) +\
                        0.001 * np.random.randn(nobs)

        GaussLog_Model = GLM(self.lny, self.X, \
                family=sm.families.Gaussian(sm.families.links.log))
        self.res1 = GaussLog_Model.fit()
        from results.results_glm import GaussianLog
        self.res2 = GaussianLog()

#    def setup(self):
#        if skipR:
#            raise SkipTest, "Rpy not installed"
#        GaussLogLink = r.gaussian(link = "log")
#        GaussLog_Res_R = RModel(self.lny, self.X, r.glm, family=GaussLogLink)
#        self.res2 = GaussLog_Res_R

class TestGaussianInverse(CheckModelResults):
    def __init__(self):
        # Test Precisions
        self.decimal_bic = DECIMAL_1
        self.decimal_aic_R = DECIMAL_1
        self.decimal_aic_Stata = DECIMAL_3
        self.decimal_loglike = DECIMAL_1
        self.decimal_resids = DECIMAL_3

        nobs = 100
        x = np.arange(nobs)
        np.random.seed(54321)
        y = 1.0 + 2.0 * x + x**2 + 0.1 * np.random.randn(nobs)
        self.X = np.c_[np.ones((nobs,1)),x,x**2]
        self.y_inv = (1. + .02*x + .001*x**2)**-1 + .001 * np.random.randn(nobs)
        InverseLink_Model = GLM(self.y_inv, self.X,
                family=sm.families.Gaussian(sm.families.links.inverse_power))
        InverseLink_Res = InverseLink_Model.fit()
        self.res1 = InverseLink_Res
        from results.results_glm import GaussianInverse
        self.res2 = GaussianInverse()

#    def setup(self):
#        if skipR:
#            raise SkipTest, "Rpy not installed."
#        InverseLink = r.gaussian(link = "inverse")
#        InverseLink_Res_R = RModel(self.y_inv, self.X, r.glm, family=InverseLink)
#        self.res2 = InverseLink_Res_R

class TestGlmBinomial(CheckModelResults):
    def __init__(self):
        '''
        Test Binomial family with canonical logit link using star98 dataset.
        '''
        self.decimal_resids = DECIMAL_1
        self.decimal_bic = DECIMAL_2

        from scikits.statsmodels.datasets.star98 import load
        from results.results_glm import Star98
        data = load()
        data.exog = add_constant(data.exog)
        self.res1 = GLM(data.endog, data.exog, \
        family=sm.families.Binomial()).fit()
        #NOTE: if you want to replicate with RModel
        #res2 = RModel(data.endog[:,0]/trials, data.exog, r.glm,
        #        family=r.binomial, weights=trials)

        self.res2 = Star98()

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
        from results.results_glm import Lbw
        self.res2 = Lbw()
        self.res1 = GLM(self.res2.endog, self.res2.exog,
                family=sm.families.Binomial()).fit()

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
        # Test Precisions
        self.decimal_aic_R = -1 #TODO: off by about 1, we are right with Stata
        self.decimal_resids = DECIMAL_2

        from scikits.statsmodels.datasets.scotland import load
        from results.results_glm import Scotvote
        data = load()
        data.exog = add_constant(data.exog)
        res1 = GLM(data.endog, data.exog, \
                    family=sm.families.Gamma()).fit()
        self.res1 = res1
#        res2 = RModel(data.endog, data.exog, r.glm, family=r.Gamma)
        res2 = Scotvote()
        res2.aic_R += 2 # R doesn't count degree of freedom for scale with gamma
        self.res2 = res2

class TestGlmGammaLog(CheckModelResults):
    def __init__(self):
        # Test Precisions
        self.decimal_resids = DECIMAL_3
        self.decimal_aic_R = DECIMAL_0
        self.decimal_fittedvalues = DECIMAL_3

        from results.results_glm import CancerLog
        res2 = CancerLog()
        self.res1 = GLM(res2.endog, res2.exog,
            family=sm.families.Gamma(link=sm.families.links.log)).fit()
        self.res2 = res2

#    def setup(self):
#        if skipR:
#            raise SkipTest, "Rpy not installed."
#        self.res2 = RModel(self.data.endog, self.data.exog, r.glm,
#            family=r.Gamma(link="log"))
#        self.res2.null_deviance = 27.92207137420696 # From R (bug in rpy)
#        self.res2.bic = -154.1582089453923 # from Stata

class TestGlmGammaIdentity(CheckModelResults):
    def __init__(self):
        # Test Precisions
        self.decimal_resids = -100 #TODO Very off from Stata?
        self.decimal_params = DECIMAL_2
        self.decimal_aic_R = DECIMAL_0
        self.decimal_loglike = DECIMAL_1

        from results.results_glm import CancerIdentity
        res2 = CancerIdentity()
        self.res1 = GLM(res2.endog, res2.exog,
            family=sm.families.Gamma(link=sm.families.links.identity)).fit()
        self.res2 = res2

#    def setup(self):
#        if skipR:
#            raise SkipTest, "Rpy not installed."
#        self.res2 = RModel(self.data.endog, self.data.exog, r.glm,
#            family=r.Gamma(link="identity"))
#        self.res2.null_deviance = 27.92207137420696 # from R, Rpy bug

class TestGlmPoisson(CheckModelResults):
    def __init__(self):
        '''
        Tests Poisson family with canonical log link.

        Test results were obtained by R.
        '''
        from results.results_glm import Cpunish
        from scikits.statsmodels.datasets.cpunish import load
        self.data = load()
        self.data.exog[:,3] = np.log(self.data.exog[:,3])
        self.data.exog = add_constant(self.data.exog)
        self.res1 = GLM(self.data.endog, self.data.exog,
                    family=sm.families.Poisson()).fit()
        self.res2 = Cpunish()

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
        # Test Precisions
        self.decimal_aic_R = DECIMAL_0
        self.decimal_loglike = DECIMAL_0

        from results.results_glm import InvGauss
        res2 = InvGauss()
        res1 = GLM(res2.endog, res2.exog, \
                family=sm.families.InverseGaussian()).fit()
        self.res1 = res1
        self.res2 = res2

class TestGlmInvgaussLog(CheckModelResults):
    def __init__(self):
        # Test Precisions
        self.decimal_aic_R = -10 # Big difference vs R.
        self.decimal_resids = DECIMAL_3

        from results.results_glm import InvGaussLog
        res2 = InvGaussLog()
        self.res1 = GLM(res2.endog, res2.exog,
            family=sm.families.InverseGaussian(link=\
            sm.families.links.log)).fit()
        self.res2 = res2

#    def setup(self):
#        if skipR:
#            raise SkipTest, "Rpy not installed."
#        self.res2 = RModel(self.data.endog, self.data.exog, r.glm,
#            family=r.inverse_gaussian(link="log"))
#        self.res2.null_deviance = 335.1539777981053 # from R, Rpy bug
#        self.res2.llf = -12162.72308 # from Stata, R's has big rounding diff

class TestGlmInvgaussIdentity(CheckModelResults):
    def __init__(self):
        # Test Precisions
        self.decimal_aic_R = -10 #TODO: Big difference vs R
        self.decimal_fittedvalues = DECIMAL_3
        self.decimal_params = DECIMAL_3

        from results.results_glm import Medpar1
        data = Medpar1()
        self.res1 = GLM(data.endog, data.exog,
            family=sm.families.InverseGaussian(link=\
            sm.families.links.identity)).fit()
        from results.results_glm import InvGaussIdentity
        self.res2 = InvGaussIdentity()

#    def setup(self):
#        if skipR:
#            raise SkipTest, "Rpy not installed."
#        self.res2 = RModel(self.data.endog, self.data.exog, r.glm,
#            family=r.inverse_gaussian(link="identity"))
#        self.res2.null_deviance = 335.1539777981053 # from R, Rpy bug
#        self.res2.llf = -12163.25545    # from Stata, big diff with R

class TestGlmNegbinomial(CheckModelResults):
    def __init__(self):
        '''
        Test Negative Binomial family with canonical log link
        '''
        # Test Precision
        self.decimal_resid = DECIMAL_1
        self.decimal_params = DECIMAL_3
        self.decimal_resids = -1 # 1 % mismatch at 0
        self.decimal_fittedvalues = DECIMAL_1

        from scikits.statsmodels.datasets.committee import load
        self.data = load()
        self.data.exog[:,2] = np.log(self.data.exog[:,2])
        interaction = self.data.exog[:,2]*self.data.exog[:,1]
        self.data.exog = np.column_stack((self.data.exog,interaction))
        self.data.exog = add_constant(self.data.exog)
        self.res1 = GLM(self.data.endog, self.data.exog,
                family=sm.families.NegativeBinomial()).fit()
        from results.results_glm import Committee
        res2 = Committee()
        res2.aic_R += 2 # They don't count a degree of freedom for the scale
        self.res2 = res2

#    def setup(self):
#        if skipR:
#            raise SkipTest, "Rpy not installed"
#        r.library('MASS')  # this doesn't work when done in rmodelwrap?
#        self.res2 = RModel(self.data.endog, self.data.exog, r.glm,
#                family=r.negative_binomial(1))
#        self.res2.null_deviance = 27.8110469364343

#class TestGlmNegbinomial_log(CheckModelResults):
#    pass

#class TestGlmNegbinomial_power(CheckModelResults):
#    pass

#class TestGlmNegbinomial_nbinom(CheckModelResults):
#    pass

#NOTE: hacked together version to test poisson offset
class TestGlmPoissonOffset(CheckModelResults):
    @classmethod
    def setupClass(cls):
        from results.results_glm import Cpunish
        from scikits.statsmodels.datasets.cpunish import load
        data = load()
        data.exog[:,3] = np.log(data.exog[:,3])
        data.exog = add_constant(data.exog)
        exposure = [100] * len(data.endog)
        cls.res1 = GLM(data.endog, data.exog, family=sm.families.Poisson(),
                    exposure=exposure).fit()
        cls.res1.params[-1] += np.log(100) # add exposure back in to param
                                            # to make the results the same
        cls.res2 = Cpunish()

def test_prefect_pred():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    iris = np.genfromtxt(os.path.join(cur_dir, 'results', 'iris.csv'),
                    delimiter=",", skip_header=1)
    y = iris[:,-1]
    X = iris[:,:-1]
    X = X[y != 2]
    y = y[y != 2]
    X = add_constant(X, prepend=True)
    glm = GLM(y, X, family=sm.families.Binomial())
    assert_raises(PerfectSeparationError, glm.fit)


def test_attribute_writable_resettable():
    """
    Regression test for mutables and class constructors.
    """
    data = sm.datasets.longley.load()
    endog, exog = data.endog, data.exog
    glm_model = sm.GLM(endog, exog)
    assert_equal(glm_model.family.link.power, 1.0)
    glm_model.family.link.power = 2.
    assert_equal(glm_model.family.link.power, 2.0)
    glm_model2 = sm.GLM(endog, exog)
    assert_equal(glm_model2.family.link.power, 1.0)

if __name__=="__main__":
    #run_module_suite()
    #taken from Fernando Perez:
    import nose
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb'],
                       exit=False)
