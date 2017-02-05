"""
Test functions for models.GLM
"""
from __future__ import division
from statsmodels.compat import range

import os
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_raises,
                           assert_allclose, assert_, assert_array_less, dec)
from scipy import stats
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.tools import add_constant
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from statsmodels.discrete import discrete_model as discrete
from nose import SkipTest
import warnings

# Test Precisions
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1
DECIMAL_0 = 0

try:
    import matplotlib.pyplot as plt  #makes plt available for test functions
    have_matplotlib = True
except:
    have_matplotlib = False

pdf_output = False

if pdf_output:
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages("test_glm.pdf")
else:
    pdf = None

def close_or_save(pdf, fig):
    if pdf_output:
        pdf.savefig(fig)
    plt.close(fig)

def teardown_module():
    if have_matplotlib:
        plt.close('all')
        if pdf_output:
            pdf.close()

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
        # fix incorrect numbers in resid_working results
        # residuals for Poisson are also tested in test_glm_weights.py
        import copy
        # new numpy would have copy method
        resid2 = copy.copy(self.res2.resids)
        resid2[:, 2] *= self.res1.family.link.deriv(self.res1.mu)**2

        atol = 10**(-self.decimal_resids)
        resids = np.column_stack((self.res1.resid_pearson,
                self.res1.resid_deviance, self.res1.resid_working,
                self.res1.resid_anscombe, self.res1.resid_response))
        assert_allclose(resids, resid2, rtol=1e-6, atol=atol)


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
                                                 self.res1.mu, self.res1.model.freq_weights, scale=1)
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
                                                 self.res1.mu, self.res1.model.freq_weights, scale=1)
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

    def test_summary(self):
        #SMOKE test
        self.res1.summary()
        self.res1.summary2()


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
                family=sm.families.Gaussian(sm.families.links.log()))
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
                family=sm.families.Gaussian(sm.families.links.inverse_power()))
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res1 = GLM(data.endog, data.exog,
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
            family=sm.families.Gamma(link=sm.families.links.log())).fit()
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res1 = GLM(res2.endog, res2.exog,
                            family=sm.families.Gamma(
                                link=sm.families.links.identity())
                            ).fit()
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
            family=sm.families.InverseGaussian(
                link=sm.families.links.log())).fit()
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res1 = GLM(data.endog, data.exog,
                            family=sm.families.InverseGaussian(
                                link=sm.families.links.identity())).fit()
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
        Test Negative Binomial family with log link
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


class TestGlmPoissonOffset(CheckModelResultsMixin):
    @classmethod
    def setupClass(cls):
        from .results.results_glm import Cpunish_offset
        from statsmodels.datasets.cpunish import load
        cls.decimal_params = DECIMAL_4
        cls.decimal_bse = DECIMAL_4
        cls.decimal_aic_R = 3
        data = load()
        data.exog[:,3] = np.log(data.exog[:,3])
        data.exog = add_constant(data.exog, prepend=True)
        exposure = [100] * len(data.endog)
        cls.data = data
        cls.exposure = exposure
        cls.res1 = GLM(data.endog, data.exog, family=sm.families.Poisson(),
                    exposure=exposure).fit()
        cls.res2 = Cpunish_offset()

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

        # test recreating model
        mod1_ = mod1.model
        kwds = mod1_._get_init_kwds()
        assert_allclose(kwds['exposure'], exposure, rtol=1e-14)
        assert_allclose(kwds['offset'], mod1_.offset, rtol=1e-14)
        mod3 = mod1_.__class__(mod1_.endog, mod1_.exog, **kwds)
        assert_allclose(mod3.exposure, mod1_.exposure, rtol=1e-14)
        assert_allclose(mod3.offset, mod1_.offset, rtol=1e-14)


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


def test_perfect_pred():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    iris = np.genfromtxt(os.path.join(cur_dir, 'results', 'iris.csv'),
                         delimiter=",", skip_header=1)
    y = iris[:, -1]
    X = iris[:, :-1]
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

    family = sm.families.Gaussian(link=sm.families.links.log())

    mod = smf.glm("Foo ~ Bar", data=df, exposure=df.exposure,
                  family=family)
    assert_(type(mod.exposure) is np.ndarray, msg='Exposure is not ndarray')

    exposure = pd.Series(np.random.uniform(size=5))
    assert_raises(ValueError, smf.glm, "Foo ~ Bar", data=df,
                  exposure=exposure, family=family)
    assert_raises(ValueError, GLM, df.Foo, df[['constant', 'Bar']],
                  exposure=exposure, family=family)


@dec.skipif(not have_matplotlib)
def test_plots():

    np.random.seed(378)
    n = 200
    exog = np.random.normal(size=(n, 2))
    lin_pred = exog[:, 0] + exog[:, 1]**2
    prob = 1 / (1 + np.exp(-lin_pred))
    endog = 1 * (np.random.uniform(size=n) < prob)

    model = sm.GLM(endog, exog, family=sm.families.Binomial())
    result = model.fit()

    import matplotlib.pyplot as plt
    import pandas as pd
    from statsmodels.graphics.regressionplots import add_lowess

    # array interface
    for j in 0,1:
        fig = result.plot_added_variable(j)
        add_lowess(fig.axes[0], frac=0.5)
        close_or_save(pdf, fig)
        fig = result.plot_partial_residuals(j)
        add_lowess(fig.axes[0], frac=0.5)
        close_or_save(pdf, fig)
        fig = result.plot_ceres_residuals(j)
        add_lowess(fig.axes[0], frac=0.5)
        close_or_save(pdf, fig)

    # formula interface
    data = pd.DataFrame({"y": endog, "x1": exog[:, 0], "x2": exog[:, 1]})
    model = sm.GLM.from_formula("y ~ x1 + x2", data, family=sm.families.Binomial())
    result = model.fit()
    for j in 0,1:
        xname = ["x1", "x2"][j]
        fig = result.plot_added_variable(xname)
        add_lowess(fig.axes[0], frac=0.5)
        close_or_save(pdf, fig)
        fig = result.plot_partial_residuals(xname)
        add_lowess(fig.axes[0], frac=0.5)
        close_or_save(pdf, fig)
        fig = result.plot_ceres_residuals(xname)
        add_lowess(fig.axes[0], frac=0.5)
        close_or_save(pdf, fig)

def gen_endog(lin_pred, family_class, link, binom_version=0):

    np.random.seed(872)

    fam = sm.families

    mu = link().inverse(lin_pred)

    if family_class == fam.Binomial:
        if binom_version == 0:
            endog = 1*(np.random.uniform(size=len(lin_pred)) < mu)
        else:
            endog = np.empty((len(lin_pred), 2))
            n = 10
            endog[:, 0] = (np.random.uniform(size=(len(lin_pred), n)) < mu[:, None]).sum(1)
            endog[:, 1] = n - endog[:, 0]
    elif family_class == fam.Poisson:
        endog = np.random.poisson(mu)
    elif family_class == fam.Gamma:
        endog = np.random.gamma(2, mu)
    elif family_class == fam.Gaussian:
        endog = mu + np.random.normal(size=len(lin_pred))
    elif family_class == fam.NegativeBinomial:
        from scipy.stats.distributions import nbinom
        endog = nbinom.rvs(mu, 0.5)
    elif family_class == fam.InverseGaussian:
        from scipy.stats.distributions import invgauss
        endog = invgauss.rvs(mu)
    else:
        raise ValueError

    return endog


def test_summary():
    """
    Smoke test for summary.
    """

    np.random.seed(4323)

    n = 100
    exog = np.random.normal(size=(n, 2))
    exog[:, 0] = 1
    endog = np.random.normal(size=n)

    for method in "irls", "cg":
        fa = sm.families.Gaussian()
        model = sm.GLM(endog, exog, family=fa)
        rslt = model.fit(method=method)
        s = rslt.summary()

def test_gradient_irls():
    # Compare the results when using gradient optimization and IRLS.

    # TODO: Find working examples for inverse_squared link

    np.random.seed(87342)

    fam = sm.families
    lnk = sm.families.links
    families = [(fam.Binomial, [lnk.logit, lnk.probit, lnk.cloglog, lnk.log, lnk.cauchy]),
                (fam.Poisson, [lnk.log, lnk.identity, lnk.sqrt]),
                (fam.Gamma, [lnk.log, lnk.identity, lnk.inverse_power]),
                (fam.Gaussian, [lnk.identity, lnk.log, lnk.inverse_power]),
                (fam.InverseGaussian, [lnk.log, lnk.identity, lnk.inverse_power, lnk.inverse_squared]),
                (fam.NegativeBinomial, [lnk.log, lnk.inverse_power, lnk.inverse_squared, lnk.identity])]

    n = 100
    p = 3
    exog = np.random.normal(size=(n, p))
    exog[:, 0] = 1

    skip_one = False
    for family_class, family_links in families:
       for link in family_links:
           for binom_version in 0,1:

               if family_class != fam.Binomial and binom_version == 1:
                   continue

               if (family_class, link) == (fam.Poisson, lnk.identity):
                   lin_pred = 20 + exog.sum(1)
               elif (family_class, link) == (fam.Binomial, lnk.log):
                   lin_pred = -1 + exog.sum(1) / 8
               elif (family_class, link) == (fam.Poisson, lnk.sqrt):
                   lin_pred = 2 + exog.sum(1)
               elif (family_class, link) == (fam.InverseGaussian, lnk.log):
                   #skip_zero = True
                   lin_pred = -1 + exog.sum(1)
               elif (family_class, link) == (fam.InverseGaussian, lnk.identity):
                   lin_pred = 20 + 5*exog.sum(1)
                   lin_pred = np.clip(lin_pred, 1e-4, np.inf)
               elif (family_class, link) == (fam.InverseGaussian, lnk.inverse_squared):
                   lin_pred = 0.5 + exog.sum(1) / 5
                   continue # skip due to non-convergence
               elif (family_class, link) == (fam.InverseGaussian, lnk.inverse_power):
                   lin_pred = 1 + exog.sum(1) / 5
               elif (family_class, link) == (fam.NegativeBinomial, lnk.identity):
                   lin_pred = 20 + 5*exog.sum(1)
                   lin_pred = np.clip(lin_pred, 1e-4, np.inf)
               elif (family_class, link) == (fam.NegativeBinomial, lnk.inverse_squared):
                   lin_pred = 0.1 + np.random.uniform(size=exog.shape[0])
                   continue # skip due to non-convergence
               elif (family_class, link) == (fam.NegativeBinomial, lnk.inverse_power):
                   lin_pred = 1 + exog.sum(1) / 5

               elif (family_class, link) == (fam.Gaussian, lnk.inverse_power):
                   # adding skip because of convergence failure
                   skip_one = True
               else:
                   lin_pred = np.random.uniform(size=exog.shape[0])

               endog = gen_endog(lin_pred, family_class, link, binom_version)

               with warnings.catch_warnings():
                   warnings.simplefilter("ignore")
                   mod_irls = sm.GLM(endog, exog, family=family_class(link=link()))
               rslt_irls = mod_irls.fit(method="IRLS")

               # Try with and without starting values.
               for max_start_irls, start_params in (0, rslt_irls.params), (3, None):
                   # TODO: skip convergence failures for now
                   if max_start_irls > 0 and skip_one:
                       continue
                   with warnings.catch_warnings():
                       warnings.simplefilter("ignore")
                       mod_gradient = sm.GLM(endog, exog, family=family_class(link=link()))
                   rslt_gradient = mod_gradient.fit(max_start_irls=max_start_irls,
                                                    start_params=start_params,
                                                    method="newton")

                   assert_allclose(rslt_gradient.params,
                                   rslt_irls.params, rtol=1e-6, atol=5e-5)

                   assert_allclose(rslt_gradient.llf, rslt_irls.llf,
                                   rtol=1e-6, atol=1e-6)

                   assert_allclose(rslt_gradient.scale, rslt_irls.scale,
                                   rtol=1e-6, atol=1e-6)

                   # Get the standard errors using expected information.
                   gradient_bse = rslt_gradient.bse
                   ehess = mod_gradient.hessian(rslt_gradient.params, observed=False)
                   gradient_bse = np.sqrt(-np.diag(np.linalg.inv(ehess)))

                   assert_allclose(gradient_bse, rslt_irls.bse, rtol=1e-6, atol=5e-5)


class CheckWtdDuplicationMixin(object):
    decimal_params = DECIMAL_4

    def __init__(self):
        from statsmodels.datasets.cpunish import load
        self.data = load()
        self.endog = self.data.endog
        self.exog = self.data.exog
        np.random.seed(1234)
        self.weight = np.random.randint(5, 100, len(self.endog))
        self.endog_big = np.repeat(self.endog, self.weight)
        self.exog_big = np.repeat(self.exog, self.weight, axis=0)

    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params,  atol=1e-6,
                        rtol=1e-6)

    decimal_bse = DECIMAL_4

    def test_standard_errors(self):
        assert_allclose(self.res1.bse, self.res2.bse, rtol=1e-5, atol=1e-6)

    decimal_resids = DECIMAL_4

    # TODO: This doesn't work... Arrays are of different shape.
    # Perhaps we use self.res1.model.family.resid_XXX()?
    """
    def test_residuals(self):
        resids1 = np.column_stack((self.res1.resid_pearson,
                                   self.res1.resid_deviance,
                                   self.res1.resid_working,
                                   self.res1.resid_anscombe,
                                   self.res1.resid_response))
        resids2 = np.column_stack((self.res1.resid_pearson,
                                   self.res2.resid_deviance,
                                   self.res2.resid_working,
                                   self.res2.resid_anscombe,
                                   self.res2.resid_response))
        assert_allclose(resids1, resids2, self.decimal_resids)
    """

    def test_aic(self):
        # R includes the estimation of the scale as a lost dof
        # Doesn't with Gamma though
        assert_allclose(self.res1.aic, self.res2.aic,  atol=1e-6, rtol=1e-6)

    def test_deviance(self):
        assert_allclose(self.res1.deviance, self.res2.deviance,  atol=1e-6,
                        rtol=1e-6)

    def test_scale(self):
        assert_allclose(self.res1.scale, self.res2.scale, atol=1e-6, rtol=1e-6)

    def test_loglike(self):
        # Stata uses the below llf for these families
        # We differ with R for them
        assert_allclose(self.res1.llf, self.res2.llf, 1e-6)

    decimal_null_deviance = DECIMAL_4

    def test_null_deviance(self):
        assert_allclose(self.res1.null_deviance, self.res2.null_deviance,
                        atol=1e-6, rtol=1e-6)

    decimal_bic = DECIMAL_4

    def test_bic(self):
        assert_allclose(self.res1.bic, self.res2.bic,  atol=1e-6, rtol=1e-6)

    decimal_fittedvalues = DECIMAL_4

    def test_fittedvalues(self):
        res2_fitted = self.res2.predict(self.res1.model.exog)
        assert_allclose(self.res1.fittedvalues, res2_fitted, atol=1e-5,
                        rtol=1e-5)

    decimal_tpvalues = DECIMAL_4

    def test_tpvalues(self):
        # test comparing tvalues and pvalues with normal implementation
        # make sure they use normal distribution (inherited in results class)
        assert_allclose(self.res1.tvalues, self.res2.tvalues, atol=1e-6,
                        rtol=2e-4)
        assert_allclose(self.res1.pvalues, self.res2.pvalues, atol=1e-6,
                        rtol=1e-6)
        assert_allclose(self.res1.conf_int(), self.res2.conf_int(), atol=1e-6,
                        rtol=1e-6)


class TestWtdGlmPoisson(CheckWtdDuplicationMixin):
    def __init__(self):
        '''
        Tests Poisson family with canonical log link.
        '''
        super(TestWtdGlmPoisson, self).__init__()
        self.res1 = GLM(self.endog, self.exog,
                        freq_weights=self.weight,
                        family=sm.families.Poisson()).fit()
        self.res2 = GLM(self.endog_big, self.exog_big,
                        family=sm.families.Poisson()).fit()


class TestWtdGlmPoissonNewton(CheckWtdDuplicationMixin):
    def __init__(self):
        '''
        Tests Poisson family with canonical log link.
        '''
        super(TestWtdGlmPoissonNewton, self).__init__()

        start_params = np.array([1.82794424e-04, -4.76785037e-02,
                                 -9.48249717e-02, -2.92293226e-04,
                                 2.63728909e+00, -2.05934384e+01])

        fit_kwds = dict(method='newton')
        self.res1 = GLM(self.endog, self.exog,
                        freq_weights=self.weight,
                        family=sm.families.Poisson()).fit(**fit_kwds)
        fit_kwds = dict(method='newton', start_params=start_params)
        self.res2 = GLM(self.endog_big, self.exog_big,
                        family=sm.families.Poisson()).fit(**fit_kwds)


class TestWtdGlmPoissonHC0(CheckWtdDuplicationMixin):
    def __init__(self):
        '''
        Tests Poisson family with canonical log link.
        '''
        super(TestWtdGlmPoissonHC0, self).__init__()

        start_params = np.array([1.82794424e-04, -4.76785037e-02,
                                 -9.48249717e-02, -2.92293226e-04,
                                 2.63728909e+00, -2.05934384e+01])

        fit_kwds = dict(cov_type='HC0')
        self.res1 = GLM(self.endog, self.exog,
                        freq_weights=self.weight,
                        family=sm.families.Poisson()).fit(**fit_kwds)
        fit_kwds = dict(cov_type='HC0', start_params=start_params)
        self.res2 = GLM(self.endog_big, self.exog_big,
                        family=sm.families.Poisson()).fit(**fit_kwds)


class TestWtdGlmPoissonClu(CheckWtdDuplicationMixin):
    def __init__(self):
        '''
        Tests Poisson family with canonical log link.
        '''
        super(TestWtdGlmPoissonClu, self).__init__()

        start_params = np.array([1.82794424e-04, -4.76785037e-02,
                                 -9.48249717e-02, -2.92293226e-04,
                                 2.63728909e+00, -2.05934384e+01])

        gid = np.arange(1, len(self.endog) + 1) // 2
        fit_kwds = dict(cov_type='cluster', cov_kwds={'groups': gid, 'use_correction':False})

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res1 = GLM(self.endog, self.exog,
                            freq_weights=self.weight,
                            family=sm.families.Poisson()).fit(**fit_kwds)
            gidr = np.repeat(gid, self.weight)
            fit_kwds = dict(cov_type='cluster', cov_kwds={'groups': gidr, 'use_correction':False})
            self.res2 = GLM(self.endog_big, self.exog_big,
                            family=sm.families.Poisson()).fit(start_params=start_params,
                                                              **fit_kwds)


class TestWtdGlmBinomial(CheckWtdDuplicationMixin):
    def __init__(self):
        '''
        Tests Binomial family with canonical logit link.
        '''
        super(TestWtdGlmBinomial, self).__init__()
        self.endog = self.endog / 100
        self.endog_big = self.endog_big / 100
        self.res1 = GLM(self.endog, self.exog,
                        freq_weights=self.weight,
                        family=sm.families.Binomial()).fit()
        self.res2 = GLM(self.endog_big, self.exog_big,
                        family=sm.families.Binomial()).fit()


class TestWtdGlmNegativeBinomial(CheckWtdDuplicationMixin):
    def __init__(self):
        '''
        Tests Negative Binomial family with canonical link
        g(p) = log(p/(p + 1/alpha))
        '''
        super(TestWtdGlmNegativeBinomial, self).__init__()
        alpha=1.
        family_link = sm.families.NegativeBinomial(
            link=sm.families.links.nbinom(alpha=alpha),
            alpha=alpha)
        self.res1 = GLM(self.endog, self.exog,
                        freq_weights=self.weight,
                        family=family_link).fit()
        self.res2 = GLM(self.endog_big, self.exog_big,
                        family=family_link).fit()


class TestWtdGlmGamma(CheckWtdDuplicationMixin):
    def __init__(self):
        '''
        Tests Gamma family with log link.
        '''
        super(TestWtdGlmGamma, self).__init__()
        family_link = sm.families.Gamma(sm.families.links.log())
        self.res1 = GLM(self.endog, self.exog,
                        freq_weights=self.weight,
                        family=family_link).fit()
        self.res2 = GLM(self.endog_big, self.exog_big,
                        family=family_link).fit()


class TestWtdGlmGaussian(CheckWtdDuplicationMixin):
    def __init__(self):
        '''
        Tests Gaussian family with log link.
        '''
        super(TestWtdGlmGaussian, self).__init__()
        family_link = sm.families.Gaussian(sm.families.links.log())
        self.res1 = GLM(self.endog, self.exog,
                        freq_weights=self.weight,
                        family=family_link).fit()
        self.res2 = GLM(self.endog_big, self.exog_big,
                        family=family_link).fit()


class TestWtdGlmInverseGaussian(CheckWtdDuplicationMixin):
    def __init__(self):
        '''
        Tests InverseGuassian family with log link.
        '''
        super(TestWtdGlmInverseGaussian, self).__init__()
        family_link = sm.families.InverseGaussian(sm.families.links.log())
        self.res1 = GLM(self.endog, self.exog,
                        freq_weights=self.weight,
                        family=family_link).fit()
        self.res2 = GLM(self.endog_big, self.exog_big,
                        family=family_link).fit()


class TestWtdGlmGammaNewton(CheckWtdDuplicationMixin):
    def __init__(self):
        '''
        Tests Gamma family with log link.
        '''
        super(TestWtdGlmGammaNewton, self).__init__()
        family_link = sm.families.Gamma(sm.families.links.log())
        self.res1 = GLM(self.endog, self.exog,
                        freq_weights=self.weight,
                        family=family_link,
                        method='newton').fit()
        self.res2 = GLM(self.endog_big, self.exog_big,
                        family=family_link,
                        method='newton').fit()


class TestWtdGlmGammaScale_X2(CheckWtdDuplicationMixin):
    def __init__(self):
        '''
        Tests Gamma family with log link.
        '''
        super(TestWtdGlmGammaScale_X2, self).__init__()
        family_link = sm.families.Gamma(sm.families.links.log())
        self.res1 = GLM(self.endog, self.exog,
                        freq_weights=self.weight,
                        family=family_link,
                        scale='X2').fit()
        self.res2 = GLM(self.endog_big, self.exog_big,
                        family=family_link,
                        scale='X2').fit()


class TestWtdGlmGammaScale_dev(CheckWtdDuplicationMixin):
    def __init__(self):
        '''
        Tests Gamma family with log link.
        '''
        super(TestWtdGlmGammaScale_dev, self).__init__()
        family_link = sm.families.Gamma(sm.families.links.log())
        self.res1 = GLM(self.endog, self.exog,
                        freq_weights=self.weight,
                        family=family_link,
                        scale='dev').fit()
        self.res2 = GLM(self.endog_big, self.exog_big,
                        family=family_link,
                        scale='dev').fit()

    def test_missing(self):
        endog = self.data.endog.copy()
        exog = self.data.exog.copy()
        exog[0, 0] = np.nan
        endog[[2, 4, 6, 8]] = np.nan
        freq_weights = self.weight
        mod_misisng = GLM(endog, exog, family=self.res1.model.family,
                          freq_weights=freq_weights, missing='drop')
        assert_equal(mod_misisng.freq_weights.shape[0],
                     mod_misisng.endog.shape[0])
        assert_equal(mod_misisng.freq_weights.shape[0],
                     mod_misisng.exog.shape[0])
        keep_idx = np.array([1,  3,  5,  7,  9, 10, 11, 12, 13, 14, 15, 16])
        assert_equal(mod_misisng.freq_weights, self.weight[keep_idx])


class TestWtdTweedieLog(CheckWtdDuplicationMixin):
    def __init__(self):
        '''
        Tests Tweedie family with log link and var_power=1.
        '''
        super(TestWtdTweedieLog, self).__init__()
        family_link = sm.families.Tweedie(link=sm.families.links.log(),
                                          var_power=1)
        self.res1 = GLM(self.endog, self.exog,
                        freq_weights=self.weight,
                        family=family_link).fit()
        self.res2 = GLM(self.endog_big, self.exog_big,
                        family=family_link).fit()


class TestWtdTweediePower2(CheckWtdDuplicationMixin):
    def __init__(self):
        '''
        Tests Tweedie family with Power(1) link and var_power=2.
        '''
        from statsmodels.datasets.cpunish import load_pandas
        self.data = load_pandas()
        self.endog = self.data.endog
        self.exog = self.data.exog[['INCOME', 'SOUTH']]
        np.random.seed(1234)
        self.weight = np.random.randint(5, 100, len(self.endog))
        self.endog_big = np.repeat(self.endog.values, self.weight)
        self.exog_big = np.repeat(self.exog.values, self.weight, axis=0)
        link = sm.families.links.Power(1)
        family_link = sm.families.Tweedie(link=link, var_power=2)
        self.res1 = GLM(self.endog, self.exog,
                        freq_weights=self.weight,
                        family=family_link).fit()
        self.res2 = GLM(self.endog_big, self.exog_big,
                        family=family_link).fit()


class TestWtdTweediePower15(CheckWtdDuplicationMixin):
    def __init__(self):
        '''
        Tests Tweedie family with Power(0.5) link and var_power=1.5.
        '''
        super(TestWtdTweediePower15, self).__init__()
        family_link = sm.families.Tweedie(link=sm.families.links.Power(0.5),
                                          var_power=1.5)
        self.res1 = GLM(self.endog, self.exog,
                        freq_weights=self.weight,
                        family=family_link).fit()
        self.res2 = GLM(self.endog_big, self.exog_big,
                        family=family_link).fit()


def test_wtd_patsy_missing():
    from statsmodels.datasets.cpunish import load
    import pandas as pd
    data = load()
    data.exog[0, 0] = np.nan
    data.endog[[2, 4, 6, 8]] = np.nan
    data.pandas = pd.DataFrame(data.exog, columns=data.exog_name)
    data.pandas['EXECUTIONS'] = data.endog
    weights = np.arange(1, len(data.endog)+1)
    formula = """EXECUTIONS ~ INCOME + PERPOVERTY + PERBLACK + VC100k96 +
                 SOUTH + DEGREE"""
    mod_misisng = GLM.from_formula(formula, data=data.pandas,
                                   freq_weights=weights)
    assert_equal(mod_misisng.freq_weights.shape[0],
                 mod_misisng.endog.shape[0])
    assert_equal(mod_misisng.freq_weights.shape[0],
                 mod_misisng.exog.shape[0])
    assert_equal(mod_misisng.freq_weights.shape[0], 12)
    keep_weights = np.array([2,  4,  6,  8, 10, 11, 12, 13, 14, 15, 16, 17])
    assert_equal(mod_misisng.freq_weights, keep_weights)


class CheckTweedie(object):
    def test_resid(self):
        l = len(self.res1.resid_response) - 1
        l2 = len(self.res2.resid_response) - 1
        assert_allclose(np.concatenate((self.res1.resid_response[:17],
                                        [self.res1.resid_response[l]])),
                        np.concatenate((self.res2.resid_response[:17],
                                        [self.res2.resid_response[l2]])),
                        rtol=1e-5, atol=1e-5)
        assert_allclose(np.concatenate((self.res1.resid_pearson[:17],
                                        [self.res1.resid_pearson[l]])),
                        np.concatenate((self.res2.resid_pearson[:17],
                                        [self.res2.resid_pearson[l2]])),
                        rtol=1e-5, atol=1e-5)
        assert_allclose(np.concatenate((self.res1.resid_deviance[:17],
                                        [self.res1.resid_deviance[l]])),
                        np.concatenate((self.res2.resid_deviance[:17],
                                        [self.res2.resid_deviance[l2]])),
                        rtol=1e-5, atol=1e-5)

        assert_allclose(np.concatenate((self.res1.resid_working[:17],
                                        [self.res1.resid_working[l]])),
                        np.concatenate((self.res2.resid_working[:17],
                                        [self.res2.resid_working[l2]])),
                        rtol=1e-5, atol=1e-5)


    def test_bse(self):
        assert_allclose(self.res1.bse, self.res2.bse, atol=1e-6, rtol=1e6)

    def test_params(self):
        assert_allclose(self.res1.params, self.res2.params, atol=1e-5,
                        rtol=1e-5)

    def test_deviance(self):
        assert_allclose(self.res1.deviance, self.res2.deviance, atol=1e-6,
                        rtol=1e-6)

    def test_df(self):
        assert_equal(self.res1.df_model, self.res2.df_model)
        assert_equal(self.res1.df_resid, self.res2.df_resid)

    def test_fittedvalues(self):
        l = len(self.res1.fittedvalues) - 1
        l2 = len(self.res2.resid_response) - 1
        assert_allclose(np.concatenate((self.res1.fittedvalues[:17],
                                        [self.res1.fittedvalues[l]])),
                        np.concatenate((self.res2.fittedvalues[:17],
                                        [self.res2.fittedvalues[l2]])),
                        atol=1e-4, rtol=1e-4)

    def test_summary(self):
        self.res1.summary()
        self.res1.summary2()


class TestTweediePower15(CheckTweedie):
    @classmethod
    def setupClass(self):
        from .results.results_glm import CpunishTweediePower15
        from statsmodels.datasets.cpunish import load_pandas
        self.data = load_pandas()
        self.exog = self.data.exog[['INCOME', 'SOUTH']]
        self.endog = self.data.endog
        family_link = sm.families.Tweedie(link=sm.families.links.Power(1),
                                          var_power=1.5)
        self.res1 = sm.GLM(endog=self.data.endog,
                           exog=self.data.exog[['INCOME', 'SOUTH']],
                           family=family_link).fit()
        self.res2 = CpunishTweediePower15()


class TestTweediePower2(CheckTweedie):
    @classmethod
    def setupClass(self):
        from .results.results_glm import CpunishTweediePower2
        from statsmodels.datasets.cpunish import load_pandas
        self.data = load_pandas()
        self.exog = self.data.exog[['INCOME', 'SOUTH']]
        self.endog = self.data.endog
        family_link = sm.families.Tweedie(link=sm.families.links.Power(1),
                                          var_power=2.)
        self.res1 = sm.GLM(endog=self.data.endog,
                           exog=self.data.exog[['INCOME', 'SOUTH']],
                           family=family_link).fit()
        self.res2 = CpunishTweediePower2()


class TestTweedieLog1(CheckTweedie):
    @classmethod
    def setupClass(self):
        from .results.results_glm import CpunishTweedieLog1
        from statsmodels.datasets.cpunish import load_pandas
        self.data = load_pandas()
        self.exog = self.data.exog[['INCOME', 'SOUTH']]
        self.endog = self.data.endog
        family_link = sm.families.Tweedie(link=sm.families.links.log(),
                                          var_power=1.)
        self.res1 = sm.GLM(endog=self.data.endog,
                           exog=self.data.exog[['INCOME', 'SOUTH']],
                           family=family_link).fit()
        self.res2 = CpunishTweedieLog1()


class TestTweedieLog15Fair(CheckTweedie):
    @classmethod
    def setupClass(self):
        from .results.results_glm import FairTweedieLog15
        from statsmodels.datasets.fair import load_pandas
        data = load_pandas()
        family_link = sm.families.Tweedie(link=sm.families.links.log(),
                                          var_power=1.5)
        self.res1 = sm.GLM(endog=data.endog,
                           exog=data.exog[['rate_marriage', 'age',
                                           'yrs_married']],
                           family=family_link).fit()
        self.res2 = FairTweedieLog15()


class CheckTweedieSpecial(object):
    def test_mu(self):
        assert_allclose(self.res1.mu, self.res2.mu, rtol=1e-5, atol=1e-5)

    def test_resid(self):
        assert_allclose(self.res1.resid_response, self.res2.resid_response,
                        rtol=1e-5, atol=1e-5)
        assert_allclose(self.res1.resid_pearson, self.res2.resid_pearson,
                        rtol=1e-5, atol=1e-5)
        assert_allclose(self.res1.resid_deviance, self.res2.resid_deviance,
                        rtol=1e-5, atol=1e-5)
        assert_allclose(self.res1.resid_working, self.res2.resid_working,
                        rtol=1e-5, atol=1e-5)
        assert_allclose(self.res1.resid_anscombe, self.res2.resid_anscombe,
                        rtol=1e-5, atol=1e-5)


class TestTweedieSpecialLog0(CheckTweedieSpecial):
    @classmethod
    def setupClass(self):
        from statsmodels.datasets.cpunish import load_pandas
        self.data = load_pandas()
        self.exog = self.data.exog[['INCOME', 'SOUTH']]
        self.endog = self.data.endog
        family1 = sm.families.Gaussian(link=sm.families.links.log())
        self.res1 = sm.GLM(endog=self.data.endog,
                           exog=self.data.exog[['INCOME', 'SOUTH']],
                           family=family1).fit()
        family2 = sm.families.Tweedie(link=sm.families.links.log(),
                                      var_power=0)
        self.res2 = sm.GLM(endog=self.data.endog,
                           exog=self.data.exog[['INCOME', 'SOUTH']],
                           family=family2).fit()


class TestTweedieSpecialLog1(CheckTweedieSpecial):
    @classmethod
    def setupClass(self):
        from statsmodels.datasets.cpunish import load_pandas
        self.data = load_pandas()
        self.exog = self.data.exog[['INCOME', 'SOUTH']]
        self.endog = self.data.endog
        family1 = sm.families.Poisson(link=sm.families.links.log())
        self.res1 = sm.GLM(endog=self.data.endog,
                           exog=self.data.exog[['INCOME', 'SOUTH']],
                           family=family1).fit()
        family2 = sm.families.Tweedie(link=sm.families.links.log(),
                                      var_power=1)
        self.res2 = sm.GLM(endog=self.data.endog,
                           exog=self.data.exog[['INCOME', 'SOUTH']],
                           family=family2).fit()


class TestTweedieSpecialLog2(CheckTweedieSpecial):
    @classmethod
    def setupClass(self):
        from statsmodels.datasets.cpunish import load_pandas
        self.data = load_pandas()
        self.exog = self.data.exog[['INCOME', 'SOUTH']]
        self.endog = self.data.endog
        family1 = sm.families.Gamma(link=sm.families.links.log())
        self.res1 = sm.GLM(endog=self.data.endog,
                           exog=self.data.exog[['INCOME', 'SOUTH']],
                           family=family1).fit()
        family2 = sm.families.Tweedie(link=sm.families.links.log(),
                                      var_power=2)
        self.res2 = sm.GLM(endog=self.data.endog,
                           exog=self.data.exog[['INCOME', 'SOUTH']],
                           family=family2).fit()


class TestTweedieSpecialLog3(CheckTweedieSpecial):
    @classmethod
    def setupClass(self):
        from statsmodels.datasets.cpunish import load_pandas
        self.data = load_pandas()
        self.exog = self.data.exog[['INCOME', 'SOUTH']]
        self.endog = self.data.endog
        family1 = sm.families.InverseGaussian(link=sm.families.links.log())
        self.res1 = sm.GLM(endog=self.data.endog,
                           exog=self.data.exog[['INCOME', 'SOUTH']],
                           family=family1).fit()
        family2 = sm.families.Tweedie(link=sm.families.links.log(),
                                      var_power=3)
        self.res2 = sm.GLM(endog=self.data.endog,
                           exog=self.data.exog[['INCOME', 'SOUTH']],
                           family=family2).fit()


def testTweediePowerEstimate():
    """
    Test the Pearson estimate of the Tweedie variance and scale parameters.

    Ideally, this would match the following R code, but I can't make it work...

    setwd('c:/workspace')
    data <- read.csv('cpunish.csv', sep=",")

    library(tweedie)

    y <- c(1.00113835e+05,   6.89668315e+03,   6.15726842e+03,
           1.41718806e+03,   5.11776456e+02,   2.55369154e+02,
           1.07147443e+01,   3.56874698e+00,   4.06797842e-02,
           7.06996731e-05,   2.10165106e-07,   4.34276938e-08,
           1.56354040e-09,   0.00000000e+00,   0.00000000e+00,
           0.00000000e+00,   0.00000000e+00)

    data$NewY <- y

    out <- tweedie.profile( NewY ~ INCOME + SOUTH - 1,
                            p.vec=c(1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
                                    1.9), link.power=0,
                            data=data,do.plot = TRUE)
    """
    data = sm.datasets.cpunish.load_pandas()
    y = [1.00113835e+05,   6.89668315e+03,   6.15726842e+03,
         1.41718806e+03,   5.11776456e+02,   2.55369154e+02,
         1.07147443e+01,   3.56874698e+00,   4.06797842e-02,
         7.06996731e-05,   2.10165106e-07,   4.34276938e-08,
         1.56354040e-09,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00]
    model1 = sm.GLM(y, data.exog[['INCOME', 'SOUTH']],
                    family=sm.families.Tweedie(link=sm.families.links.log(),
                                               var_power=1.5))
    res1 = model1.fit()
    model2 = sm.GLM((y - res1.mu) ** 2,
                    np.column_stack((np.ones(len(res1.mu)), np.log(res1.mu))),
                    family=sm.families.Gamma(sm.families.links.log()))
    res2 = model2.fit()
    # Sample may be too small for this...
    # assert_allclose(res1.scale, np.exp(res2.params[0]), rtol=0.25)
    p = model1.estimate_tweedie_power(res1.mu)
    assert_allclose(p, res2.params[1], rtol=0.25)

class TestRegularized(object):

    def test_regularized(self):

        import os
        from . import glmnet_r_results

        for dtype in "binomial", "poisson":

            cur_dir = os.path.dirname(os.path.abspath(__file__))
            data = np.loadtxt(os.path.join(cur_dir, "results", "enet_%s.csv" % dtype),
                              delimiter=",")

            endog = data[:, 0]
            exog = data[:, 1:]

            fam = {"binomial" : sm.families.Binomial,
                   "poisson" : sm.families.Poisson}[dtype]

            for j in range(9):

                vn = "rslt_%s_%d" % (dtype, j)
                r_result = getattr(glmnet_r_results, vn)
                L1_wt = r_result[0]
                alpha = r_result[1]
                params = r_result[2:]

                model = GLM(endog, exog, family=fam())
                sm_result = model.fit_regularized(L1_wt=L1_wt, alpha=alpha)

                # Agreement is OK, see below for further check
                assert_allclose(params, sm_result.params, atol=1e-2, rtol=0.3)

                # The penalized log-likelihood that we are maximizing.
                def plf(params):
                    llf = model.loglike(params) / len(endog)
                    llf = llf - alpha * ((1 - L1_wt)*np.sum(params**2) / 2 + L1_wt*np.sum(np.abs(params)))
                    return llf

                # Confirm that we are doing better than glmnet.
                from numpy.testing import assert_equal
                llf_r = plf(params)
                llf_sm = plf(sm_result.params)
                assert_equal(np.sign(llf_sm - llf_r), 1)


class TestConvergence(object):
    def __init__(self):
        '''
        Test Binomial family with canonical logit link using star98 dataset.
        '''
        from statsmodels.datasets.star98 import load
        data = load()
        data.exog = add_constant(data.exog, prepend=False)
        self.model = GLM(data.endog, data.exog,
                         family=sm.families.Binomial())

    def _when_converged(self, atol=1e-8, rtol=0, tol_criterion='deviance'):
        for i, dev in enumerate(self.res.fit_history[tol_criterion]):
            orig = self.res.fit_history[tol_criterion][i]
            new = self.res.fit_history[tol_criterion][i + 1]
            if np.allclose(orig, new, atol=atol, rtol=rtol):
                return i
        raise ValueError('CONVERGENCE CHECK: It seems this doens\'t converge!')

    def test_convergence_atol_only(self):
        atol = 1e-8
        rtol = 0
        self.res = self.model.fit(atol=atol, rtol=rtol)
        expected_iterations = self._when_converged(atol=atol, rtol=rtol)
        actual_iterations = self.res.fit_history['iteration']
        # Note the first value is the list is np.inf. The second value
        # is the initial guess based off of start_params or the
        # estimate thereof. The third value (index = 2) is the actual "first
        # iteration"
        assert_equal(expected_iterations, actual_iterations)
        assert_equal(len(self.res.fit_history['deviance']) - 2,
                     actual_iterations)

    def test_convergence_rtol_only(self):
        atol = 0
        rtol = 1e-8
        self.res = self.model.fit(atol=atol, rtol=rtol)
        expected_iterations = self._when_converged(atol=atol, rtol=rtol)
        actual_iterations = self.res.fit_history['iteration']
        # Note the first value is the list is np.inf. The second value
        # is the initial guess based off of start_params or the
        # estimate thereof. The third value (index = 2) is the actual "first
        # iteration"
        assert_equal(expected_iterations, actual_iterations)
        assert_equal(len(self.res.fit_history['deviance']) - 2,
                     actual_iterations)

    def test_convergence_atol_rtol(self):
        atol = 1e-8
        rtol = 1e-8
        self.res = self.model.fit(atol=atol, rtol=rtol)
        expected_iterations = self._when_converged(atol=atol, rtol=rtol)
        actual_iterations = self.res.fit_history['iteration']
        # Note the first value is the list is np.inf. The second value
        # is the initial guess based off of start_params or the
        # estimate thereof. The third value (index = 2) is the actual "first
        # iteration"
        assert_equal(expected_iterations, actual_iterations)
        assert_equal(len(self.res.fit_history['deviance']) - 2,
                     actual_iterations)

    def test_convergence_atol_only_params(self):
        atol = 1e-8
        rtol = 0
        self.res = self.model.fit(atol=atol, rtol=rtol, tol_criterion='params')
        expected_iterations = self._when_converged(atol=atol, rtol=rtol,
                                                   tol_criterion='params')
        actual_iterations = self.res.fit_history['iteration']
        # Note the first value is the list is np.inf. The second value
        # is the initial guess based off of start_params or the
        # estimate thereof. The third value (index = 2) is the actual "first
        # iteration"
        assert_equal(expected_iterations, actual_iterations)
        assert_equal(len(self.res.fit_history['deviance']) - 2,
                     actual_iterations)

    def test_convergence_rtol_only_params(self):
        atol = 0
        rtol = 1e-8
        self.res = self.model.fit(atol=atol, rtol=rtol, tol_criterion='params')
        expected_iterations = self._when_converged(atol=atol, rtol=rtol,
                                                   tol_criterion='params')
        actual_iterations = self.res.fit_history['iteration']
        # Note the first value is the list is np.inf. The second value
        # is the initial guess based off of start_params or the
        # estimate thereof. The third value (index = 2) is the actual "first
        # iteration"
        assert_equal(expected_iterations, actual_iterations)
        assert_equal(len(self.res.fit_history['deviance']) - 2,
                     actual_iterations)

    def test_convergence_atol_rtol_params(self):
        atol = 1e-8
        rtol = 1e-8
        self.res = self.model.fit(atol=atol, rtol=rtol, tol_criterion='params')
        expected_iterations = self._when_converged(atol=atol, rtol=rtol,
                                                   tol_criterion='params')
        actual_iterations = self.res.fit_history['iteration']
        # Note the first value is the list is np.inf. The second value
        # is the initial guess based off of start_params or the
        # estimate thereof. The third value (index = 2) is the actual "first
        # iteration"
        assert_equal(expected_iterations, actual_iterations)
        assert_equal(len(self.res.fit_history['deviance']) - 2,
                     actual_iterations)


def test_poisson_deviance():
    # see #3355 missing term in deviance if resid_response.sum() != 0
    np.random.seed(123987)
    nobs, k_vars = 50, 3-1
    x = sm.add_constant(np.random.randn(nobs, k_vars))

    mu_true = np.exp(x.sum(1))
    y = np.random.poisson(mu_true, size=nobs)

    mod = sm.GLM(y, x[:, :], family=sm.genmod.families.Poisson())
    res = mod.fit()

    d_i = res.resid_deviance
    d = res.deviance
    lr = (mod.family.loglike(y, y+1e-20) -
          mod.family.loglike(y, res.fittedvalues)) * 2

    assert_allclose(d, (d_i**2).sum(), rtol=1e-12)
    assert_allclose(d, lr, rtol=1e-12)

    # case without constant, resid_response.sum() != 0
    mod_nc = sm.GLM(y, x[:, 1:], family=sm.genmod.families.Poisson())
    res_nc = mod_nc.fit()

    d_i = res_nc.resid_deviance
    d = res_nc.deviance
    lr = (mod.family.loglike(y, y+1e-20) -
          mod.family.loglike(y, res_nc.fittedvalues)) * 2

    assert_allclose(d, (d_i**2).sum(), rtol=1e-12)
    assert_allclose(d, lr, rtol=1e-12)


if __name__ == "__main__":
    # run_module_suite()
    # taken from Fernando Perez:
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'],
                   exit=False)
