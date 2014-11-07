"""
Test functions for models.GLM
"""

from statsmodels.compat import range

import os
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_raises,
                           assert_allclose, assert_, assert_array_less)
from scipy import stats
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.tools import add_constant
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from statsmodels.discrete import discrete_model as discrete
from nose import SkipTest

# Test Precisions
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1
DECIMAL_0 = 0

class CheckModelResultsMixin(object):
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

    def test_tpvalues(self):
        # test comparing tvalues and pvalues with normal implementation
        # make sure they use normal distribution (inherited in results class)
        params = self.res1.params
        tvalues = params / self.res1.bse
        pvalues = stats.norm.sf(np.abs(tvalues)) * 2
        half_width = stats.norm.isf(0.025) * self.res1.bse
        conf_int = np.column_stack((params - half_width, params + half_width))

        assert_almost_equal(self.res1.tvalues, tvalues)
        assert_almost_equal(self.res1.pvalues, pvalues)
        assert_almost_equal(self.res1.conf_int(), conf_int)


class CheckComparisonMixin(object):

    def test_compare_discrete(self):
        res1 = self.res1
        resd = self.resd

        assert_allclose(res1.llf, resd.llf, rtol=1e-10)
        score_obs1 = res1.model.score_obs(res1.params)
        score_obsd = resd.model.score_obs(resd.params)
        assert_allclose(score_obs1, score_obsd, rtol=1e-10)

        # score
        score1 = res1.model.score(res1.params)
        assert_allclose(score1, score_obs1.sum(0), atol=1e-20)
        assert_allclose(score1, np.zeros(score_obs1.shape[1]), atol=1e-7)

        hessian1 = res1.model.hessian(res1.params, observed=False)
        hessiand = resd.model.hessian(resd.params)
        assert_allclose(hessian1, hessiand, rtol=1e-10)

        hessian1 = res1.model.hessian(res1.params, observed=True)
        hessiand = resd.model.hessian(resd.params)
        assert_allclose(hessian1, hessiand, rtol=1e-9)

    def test_score_test(self):
        res1 = self.res1
        # fake example, should be zero, k_constraint should be 0
        st, pv, df = res1.model.score_test(res1.params, k_constraints=1)
        assert_allclose(st, 0, atol=1e-20)
        assert_allclose(pv, 1, atol=1e-10)
        assert_equal(df, 1)

        st, pv, df = res1.model.score_test(res1.params, k_constraints=0)
        assert_allclose(st, 0, atol=1e-20)
        assert_(np.isnan(pv), msg=repr(pv))
        assert_equal(df, 0)

        # TODO: no verified numbers largely SMOKE test
        exog_extra = res1.model.exog[:,1]**2
        st, pv, df = res1.model.score_test(res1.params, exog_extra=exog_extra)
        assert_array_less(0.1, st)
        assert_array_less(0.1, pv)
        assert_equal(df, 1)


class TestGlmGaussian(CheckModelResultsMixin):
    def __init__(self):
        '''
        Test Gaussian family with canonical identity link
        '''
        # Test Precisions
        self.decimal_resids = DECIMAL_3
        self.decimal_params = DECIMAL_2
        self.decimal_bic = DECIMAL_0
        self.decimal_bse = DECIMAL_3

        from statsmodels.datasets.longley import load
        self.data = load()
        self.data.exog = add_constant(self.data.exog, prepend=False)
        self.res1 = GLM(self.data.endog, self.data.exog,
                        family=sm.families.Gaussian()).fit()
        from .results.results_glm import Longley
        self.res2 = Longley()


    def test_compare_OLS(self):
        res1 = self.res1
        # OLS doesn't define score_obs
        from statsmodels.regression.linear_model import OLS
        resd = OLS(self.data.endog, self.data.exog).fit()
        self.resd = resd  # attach to access from the outside

        assert_allclose(res1.llf, resd.llf, rtol=1e-10)
        score_obs1 = res1.model.score_obs(res1.params, scale=None)
        score_obsd = resd.resid[:, None] / resd.scale * resd.model.exog
        # low precision because of badly scaled exog
        assert_allclose(score_obs1, score_obsd, rtol=1e-8)

        score_obs1 = res1.model.score_obs(res1.params, scale=1)
        score_obsd = resd.resid[:, None] * resd.model.exog
        assert_allclose(score_obs1, score_obsd, rtol=1e-8)

        hess_obs1 = res1.model.hessian(res1.params, scale=None)
        hess_obsd = -1. / resd.scale * resd.model.exog.T.dot(resd.model.exog)
        # low precision because of badly scaled exog
        assert_allclose(hess_obs1, hess_obsd, rtol=1e-8)

#    def setup(self):
#        if skipR:
#            raise SkipTest, "Rpy not installed."
#        Gauss = r.gaussian
#        self.res2 = RModel(self.data.endog, self.data.exog, r.glm, family=Gauss)
#        self.res2.resids = np.array(self.res2.resid)[:,None]*np.ones((1,5))
#        self.res2.null_deviance = 185008826 # taken from R. Rpy bug?

class TestGaussianLog(CheckModelResultsMixin):
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
        from .results.results_glm import GaussianLog
        self.res2 = GaussianLog()

#    def setup(self):
#        if skipR:
#            raise SkipTest, "Rpy not installed"
#        GaussLogLink = r.gaussian(link = "log")
#        GaussLog_Res_R = RModel(self.lny, self.X, r.glm, family=GaussLogLink)
#        self.res2 = GaussLog_Res_R

class TestGaussianInverse(CheckModelResultsMixin):
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
        from .results.results_glm import GaussianInverse
        self.res2 = GaussianInverse()

#    def setup(self):
#        if skipR:
#            raise SkipTest, "Rpy not installed."
#        InverseLink = r.gaussian(link = "inverse")
#        InverseLink_Res_R = RModel(self.y_inv, self.X, r.glm, family=InverseLink)
#        self.res2 = InverseLink_Res_R

class TestGlmBinomial(CheckModelResultsMixin):
    def __init__(self):
        '''
        Test Binomial family with canonical logit link using star98 dataset.
        '''
        self.decimal_resids = DECIMAL_1
        self.decimal_bic = DECIMAL_2

        from statsmodels.datasets.star98 import load
        from .results.results_glm import Star98
        data = load()
        data.exog = add_constant(data.exog, prepend=False)
        self.res1 = GLM(data.endog, data.exog, \
        family=sm.families.Binomial()).fit()
        #NOTE: if you want to replicate with RModel
        #res2 = RModel(data.endog[:,0]/trials, data.exog, r.glm,
        #        family=r.binomial, weights=trials)

        self.res2 = Star98()

#TODO:
#Non-Canonical Links for the Binomial family require the algorithm to be
#slightly changed
#class TestGlmBinomialLog(CheckModelResultsMixin):
#    pass

#class TestGlmBinomialLogit(CheckModelResultsMixin):
#    pass

#class TestGlmBinomialProbit(CheckModelResultsMixin):
#    pass

#class TestGlmBinomialCloglog(CheckModelResultsMixin):
#    pass

#class TestGlmBinomialPower(CheckModelResultsMixin):
#    pass

#class TestGlmBinomialLoglog(CheckModelResultsMixin):
#    pass

#class TestGlmBinomialLogc(CheckModelResultsMixin):
#TODO: need include logc link
#    pass

class TestGlmBernoulli(CheckModelResultsMixin, CheckComparisonMixin):
    def __init__(self):
        from .results.results_glm import Lbw
        self.res2 = Lbw()
        self.res1 = GLM(self.res2.endog, self.res2.exog,
                family=sm.families.Binomial()).fit()

        modd = discrete.Logit(self.res2.endog, self.res2.exog)
        self.resd = modd.fit(start_params=self.res1.params * 0.9, disp=False)


    def score_test_r(self):
        res1 = self.res1
        res2 = self.res2
        st, pv, df = res1.model.score_test(res1.params,
                                           exog_extra=res1.model.exog[:, 1]**2)
        st_res = 0.2837680293459376  # (-0.5326988167303712)**2
        assert_allclose(st, st_res, rtol=1e-4)

        st, pv, df = res1.model.score_test(res1.params,
                                          exog_extra=res1.model.exog[:, 0]**2)
        st_res = 0.6713492821514992  # (-0.8193590679009413)**2
        assert_allclose(st, st_res, rtol=1e-4)

        select = list(range(9))
        select.pop(7)

        res1b = GLM(res2.endog, res2.exog[:, select],
                    family=sm.families.Binomial()).fit()
        tres = res1b.model.score_test(res1b.params,
                                      exog_extra=res1.model.exog[:, -2])
        tres = np.asarray(tres[:2]).ravel()
        tres_r = (2.7864148487452, 0.0950667)
        assert_allclose(tres, tres_r, rtol=1e-4)

        cmd_r = """\
        data = read.csv("...statsmodels\\statsmodels\\genmod\\tests\\results\\stata_lbw_glm.csv")

        data["race_black"] = data["race"] == "black"
        data["race_other"] = data["race"] == "other"
        mod = glm(low ~ age + lwt + race_black + race_other + smoke + ptl + ht + ui, family=binomial, data=data)
        options(digits=16)
        anova(mod, test="Rao")

        library(statmod)
        s = glm.scoretest(mod, data["age"]**2)
        s**2
        s = glm.scoretest(mod, data["lwt"]**2)
        s**2
        """

#class TestGlmBernoulliIdentity(CheckModelResultsMixin):
#    pass

#class TestGlmBernoulliLog(CheckModelResultsMixin):
#    pass

#class TestGlmBernoulliProbit(CheckModelResultsMixin):
#    pass

#class TestGlmBernoulliCloglog(CheckModelResultsMixin):
#    pass

#class TestGlmBernoulliPower(CheckModelResultsMixin):
#    pass

#class TestGlmBernoulliLoglog(CheckModelResultsMixin):
#    pass

#class test_glm_bernoulli_logc(CheckModelResultsMixin):
#    pass

class TestGlmGamma(CheckModelResultsMixin):

    def __init__(self):
        '''
        Tests Gamma family with canonical inverse link (power -1)
        '''
        # Test Precisions
        self.decimal_aic_R = -1 #TODO: off by about 1, we are right with Stata
        self.decimal_resids = DECIMAL_2

        from statsmodels.datasets.scotland import load
        from .results.results_glm import Scotvote
        data = load()
        data.exog = add_constant(data.exog, prepend=False)
        res1 = GLM(data.endog, data.exog, \
                    family=sm.families.Gamma()).fit()
        self.res1 = res1
#        res2 = RModel(data.endog, data.exog, r.glm, family=r.Gamma)
        res2 = Scotvote()
        res2.aic_R += 2 # R doesn't count degree of freedom for scale with gamma
        self.res2 = res2

class TestGlmGammaLog(CheckModelResultsMixin):
    def __init__(self):
        # Test Precisions
        self.decimal_resids = DECIMAL_3
        self.decimal_aic_R = DECIMAL_0
        self.decimal_fittedvalues = DECIMAL_3

        from .results.results_glm import CancerLog
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

class TestGlmGammaIdentity(CheckModelResultsMixin):
    def __init__(self):
        # Test Precisions
        self.decimal_resids = -100 #TODO Very off from Stata?
        self.decimal_params = DECIMAL_2
        self.decimal_aic_R = DECIMAL_0
        self.decimal_loglike = DECIMAL_1

        from .results.results_glm import CancerIdentity
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

class TestGlmPoisson(CheckModelResultsMixin, CheckComparisonMixin):
    def __init__(self):
        '''
        Tests Poisson family with canonical log link.

        Test results were obtained by R.
        '''
        from .results.results_glm import Cpunish
        from statsmodels.datasets.cpunish import load
        self.data = load()
        self.data.exog[:,3] = np.log(self.data.exog[:,3])
        self.data.exog = add_constant(self.data.exog, prepend=False)
        self.res1 = GLM(self.data.endog, self.data.exog,
                    family=sm.families.Poisson()).fit()
        self.res2 = Cpunish()
        # compare with discrete, start close to save time
        modd = discrete.Poisson(self.data.endog, self.data.exog)
        self.resd = modd.fit(start_params=self.res1.params * 0.9, disp=False)

#class TestGlmPoissonIdentity(CheckModelResultsMixin):
#    pass

#class TestGlmPoissonPower(CheckModelResultsMixin):
#    pass

class TestGlmInvgauss(CheckModelResultsMixin):
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

        from .results.results_glm import InvGauss
        res2 = InvGauss()
        res1 = GLM(res2.endog, res2.exog, \
                family=sm.families.InverseGaussian()).fit()
        self.res1 = res1
        self.res2 = res2

class TestGlmInvgaussLog(CheckModelResultsMixin):
    def __init__(self):
        # Test Precisions
        self.decimal_aic_R = -10 # Big difference vs R.
        self.decimal_resids = DECIMAL_3

        from .results.results_glm import InvGaussLog
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

class TestGlmInvgaussIdentity(CheckModelResultsMixin):
    def __init__(self):
        # Test Precisions
        self.decimal_aic_R = -10 #TODO: Big difference vs R
        self.decimal_fittedvalues = DECIMAL_3
        self.decimal_params = DECIMAL_3

        from .results.results_glm import Medpar1
        data = Medpar1()
        self.res1 = GLM(data.endog, data.exog,
            family=sm.families.InverseGaussian(link=\
            sm.families.links.identity)).fit()
        from .results.results_glm import InvGaussIdentity
        self.res2 = InvGaussIdentity()

#    def setup(self):
#        if skipR:
#            raise SkipTest, "Rpy not installed."
#        self.res2 = RModel(self.data.endog, self.data.exog, r.glm,
#            family=r.inverse_gaussian(link="identity"))
#        self.res2.null_deviance = 335.1539777981053 # from R, Rpy bug
#        self.res2.llf = -12163.25545    # from Stata, big diff with R

class TestGlmNegbinomial(CheckModelResultsMixin):
    def __init__(self):
        '''
        Test Negative Binomial family with canonical log link
        '''
        # Test Precision
        self.decimal_resid = DECIMAL_1
        self.decimal_params = DECIMAL_3
        self.decimal_resids = -1 # 1 % mismatch at 0
        self.decimal_fittedvalues = DECIMAL_1

        from statsmodels.datasets.committee import load
        self.data = load()
        self.data.exog[:,2] = np.log(self.data.exog[:,2])
        interaction = self.data.exog[:,2]*self.data.exog[:,1]
        self.data.exog = np.column_stack((self.data.exog,interaction))
        self.data.exog = add_constant(self.data.exog, prepend=False)
        self.res1 = GLM(self.data.endog, self.data.exog,
                family=sm.families.NegativeBinomial()).fit()
        from .results.results_glm import Committee
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

#class TestGlmNegbinomial_log(CheckModelResultsMixin):
#    pass

#class TestGlmNegbinomial_power(CheckModelResultsMixin):
#    pass

#class TestGlmNegbinomial_nbinom(CheckModelResultsMixin):
#    pass

#NOTE: hacked together version to test poisson offset
class TestGlmPoissonOffset(CheckModelResultsMixin):
    @classmethod
    def setupClass(cls):
        from .results.results_glm import Cpunish
        from statsmodels.datasets.cpunish import load
        data = load()
        data.exog[:,3] = np.log(data.exog[:,3])
        data.exog = add_constant(data.exog, prepend=False)
        exposure = [100] * len(data.endog)
        cls.data = data
        cls.exposure = exposure
        cls.res1 = GLM(data.endog, data.exog, family=sm.families.Poisson(),
                    exposure=exposure).fit()
        cls.res1.params[-1] += np.log(100) # add exposure back in to param
                                            # to make the results the same
        cls.res2 = Cpunish()

    def test_missing(self):
        # make sure offset is dropped correctly
        endog = self.data.endog.copy()
        endog[[2,4,6,8]] = np.nan
        mod = GLM(endog, self.data.exog, family=sm.families.Poisson(),
                    exposure=self.exposure, missing='drop')
        assert_equal(mod.exposure.shape[0], 13)

    def test_offset_exposure(self):
        # exposure=x and offset=log(x) should have the same effect
        np.random.seed(382304)
        endog = np.random.randint(0, 10, 100)
        exog = np.random.normal(size=(100,3))
        exposure = np.random.uniform(1, 2, 100)
        offset = np.random.uniform(1, 2, 100)
        mod1 = GLM(endog, exog, family=sm.families.Poisson(),
                   offset=offset, exposure=exposure).fit()
        offset2 = offset + np.log(exposure)
        mod2 = GLM(endog, exog, family=sm.families.Poisson(),
                   offset=offset2).fit()
        assert_almost_equal(mod1.params, mod2.params)

    def test_predict(self):
        np.random.seed(382304)
        endog = np.random.randint(0, 10, 100)
        exog = np.random.normal(size=(100,3))
        exposure = np.random.uniform(1, 2, 100)
        mod1 = GLM(endog, exog, family=sm.families.Poisson(),
                   exposure=exposure).fit()
        exog1 = np.random.normal(size=(10,3))
        exposure1 = np.random.uniform(1, 2, 10)

        # Doubling exposure time should double expected response
        pred1 = mod1.predict(exog=exog1, exposure=exposure1)
        pred2 = mod1.predict(exog=exog1, exposure=2*exposure1)
        assert_almost_equal(pred2, 2*pred1)

        # Check exposure defaults
        pred3 = mod1.predict()
        pred4 = mod1.predict(exposure=exposure)
        pred5 = mod1.predict(exog=exog, exposure=exposure)
        assert_almost_equal(pred3, pred4)
        assert_almost_equal(pred4, pred5)

        # Check offset defaults
        offset = np.random.uniform(1, 2, 100)
        mod2 = GLM(endog, exog, offset=offset, family=sm.families.Poisson()).fit()
        pred1 = mod2.predict()
        pred2 = mod2.predict(offset=offset)
        pred3 = mod2.predict(exog=exog, offset=offset)
        assert_almost_equal(pred1, pred2)
        assert_almost_equal(pred2, pred3)

        # Check that offset shifts the linear predictor
        mod3 = GLM(endog, exog, family=sm.families.Poisson()).fit()
        offset = np.random.uniform(1, 2, 10)
        pred1 = mod3.predict(exog=exog1, offset=offset, linear=True)
        pred2 = mod3.predict(exog=exog1, offset=2*offset, linear=True)
        assert_almost_equal(pred2, pred1+offset)


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


def test_score_test_OLS():
    # nicer example than Longley
    from statsmodels.regression.linear_model import OLS
    np.random.seed(5)
    nobs = 100
    sige = 0.5
    x = np.random.uniform(0, 1, size=(nobs, 5))
    x[:, 0] = 1
    beta = 1. / np.arange(1., x.shape[1] + 1)
    y = x.dot(beta) + sige * np.random.randn(nobs)

    res_ols = OLS(y, x).fit()
    res_olsc = OLS(y, x[:, :-2]).fit()
    co = res_ols.compare_lm_test(res_olsc, demean=False)

    res_glm = GLM(y, x[:, :-2], family=sm.families.Gaussian()).fit()
    co2 = res_glm.model.score_test(res_glm.params, exog_extra=x[:, -2:])
    # difference in df_resid versus nobs in scale see #1786
    assert_allclose(co[0] * 97 / 100., co2[0], rtol=1e-13)


def test_attribute_writable_resettable():
    # Regression test for mutables and class constructors.
    data = sm.datasets.longley.load()
    endog, exog = data.endog, data.exog
    glm_model = sm.GLM(endog, exog)
    assert_equal(glm_model.family.link.power, 1.0)
    glm_model.family.link.power = 2.
    assert_equal(glm_model.family.link.power, 2.0)
    glm_model2 = sm.GLM(endog, exog)
    assert_equal(glm_model2.family.link.power, 1.0)


class Test_start_params(CheckModelResultsMixin):
    def __init__(self):
        '''
        Test Gaussian family with canonical identity link
        '''
        # Test Precisions
        self.decimal_resids = DECIMAL_3
        self.decimal_params = DECIMAL_2
        self.decimal_bic = DECIMAL_0
        self.decimal_bse = DECIMAL_3

        from statsmodels.datasets.longley import load
        self.data = load()
        self.data.exog = add_constant(self.data.exog, prepend=False)
        params = sm.OLS(self.data.endog, self.data.exog).fit().params
        self.res1 = GLM(self.data.endog, self.data.exog,
                        family=sm.families.Gaussian()).fit(start_params=params)
        from .results.results_glm import Longley
        self.res2 = Longley()


def test_glm_start_params():
    # see 1604
    y2 = np.array('0 1 0 0 0 1'.split(), int)
    wt = np.array([50,1,50,1,5,10])
    y2 = np.repeat(y2, wt)
    x2 = np.repeat([0,0,0.001,100,-1,-1], wt)
    mod = sm.GLM(y2, sm.add_constant(x2), family=sm.families.Binomial())
    res = mod.fit(start_params=[-4, -5])
    np.testing.assert_almost_equal(res.params, [-4.60305022, -5.29634545], 6)


def test_loglike_no_opt():
    # see 1728

    y = np.asarray([0, 1, 0, 0, 1, 1, 0, 1, 1, 1])
    x = np.arange(10, dtype=np.float64)

    def llf(params):
        lin_pred = params[0] + params[1]*x
        pr = 1 / (1 + np.exp(-lin_pred))
        return np.sum(y*np.log(pr) + (1-y)*np.log(1-pr))

    for params in [0,0], [0,1], [0.5,0.5]:
        mod = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial())
        res = mod.fit(start_params=params, maxiter=0)
        like = llf(params)
        assert_almost_equal(like, res.llf)


def test_formula_missing_exposure():
    # see 2083
    import statsmodels.formula.api as smf
    import pandas as pd

    d = {'Foo': [1, 2, 10, 149], 'Bar': [1, 2, 3, np.nan],
         'constant': [1] * 4, 'exposure' : np.random.uniform(size=4),
         'x': [1, 3, 2, 1.5]}
    df = pd.DataFrame(d)

    family = sm.families.Gaussian(link=sm.families.links.log)

    mod = smf.glm("Foo ~ Bar", data=df, exposure=df.exposure,
                  family=family)
    assert_(type(mod.exposure) is np.ndarray, msg='Exposure is not ndarray')

    exposure = pd.Series(np.random.uniform(size=5))
    assert_raises(ValueError, smf.glm, "Foo ~ Bar", data=df,
                  exposure=exposure, family=family)
    assert_raises(ValueError, GLM, df.Foo, df[['constant', 'Bar']],
                  exposure=exposure, family=family)


if __name__=="__main__":
    #run_module_suite()
    #taken from Fernando Perez:
    import nose
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb'],
                       exit=False)
