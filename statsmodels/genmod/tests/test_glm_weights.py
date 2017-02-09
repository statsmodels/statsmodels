"""Test for weights in GLM, Poisson and OLS/WLS, continuous test_glm.py

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

from .results import results_glm_poisson_weights as res_stata
from .results import res_R_var_weight as res_r

# load data into module namespace
from statsmodels.datasets.cpunish import load
cpunish_data = load()
cpunish_data.exog[:,3] = np.log(cpunish_data.exog[:,3])
cpunish_data.exog = add_constant(cpunish_data.exog, prepend=False)


class CheckWeight(object):

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2

        assert_allclose(res1.params, res2.params, atol= 1e-6, rtol=2e-6)
        corr_fact = getattr(self, 'corr_fact', 1)
        assert_allclose(res1.bse, corr_fact * res2.bse, atol= 1e-6, rtol=2e-6)
        if not isinstance(self, (TestGlmGaussianAwNr, TestGlmGammaAwNr)):
            # Matching R is hard
            assert_allclose(res1.llf, res2.ll, atol= 1e-6, rtol=1e-7)
        assert_allclose(res1.deviance, res2.deviance, atol= 1e-6, rtol=1e-7)

    def test_residuals(self):
        res1 = self.res1
        res2 = self.res2
        if not hasattr(res2, 'resids'):
            return None  # use SkipError instead
        resid_all = dict(zip(res2.resids_colnames, res2.resids.T))

        assert_allclose(res1.resid_response, resid_all['resid_response'], atol= 1e-6, rtol=2e-6)
        assert_allclose(res1.resid_pearson, resid_all['resid_pearson'], atol= 1e-6, rtol=2e-6)
        assert_allclose(res1.resid_deviance, resid_all['resid_deviance'], atol= 1e-6, rtol=2e-6)
        assert_allclose(res1.resid_working, resid_all['resid_working'], atol= 1e-6, rtol=2e-6)
        if resid_all.get('resid_anscombe') is None:
            return None
        assert_allclose(res1.resid_anscombe, resid_all['resid_anscombe'], atol= 1e-6, rtol=2e-6)

    def test_compare_bfgs(self):
        res1 = self.res1
        if isinstance(res1.model.family, sm.families.Tweedie):
            # Can't do this on Tweedie as loglikelihood is too complex
            return None
        res2 = self.res1.model.fit(method='bfgs')
        assert_allclose(res1.params, res2.params, atol=1e-3, rtol=2e-3)
        assert_allclose(res1.bse, res2.bse, atol=1e-3, rtol=1e-3)


class TestGlmPoissonPlain(CheckWeight):
    @classmethod
    def setupClass(cls):
        self = cls # alias

        self.res1 = GLM(cpunish_data.endog, cpunish_data.exog,
                    family=sm.families.Poisson()).fit()
        # compare with discrete, start close to save time
        modd = discrete.Poisson(cpunish_data.endog, cpunish_data.exog)
        self.res2 = res_stata.results_poisson_none_nonrobust


class TestGlmPoissonFwNr(CheckWeight):
    @classmethod
    def setupClass(cls):
        self = cls # alias

        fweights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3]
        fweights = np.array(fweights)

        self.res1 = GLM(cpunish_data.endog, cpunish_data.exog,
                    family=sm.families.Poisson(), freq_weights=fweights).fit()
        # compare with discrete, start close to save time
        modd = discrete.Poisson(cpunish_data.endog, cpunish_data.exog)
        self.res2 = res_stata.results_poisson_fweight_nonrobust


class TestGlmPoissonAwNr(CheckWeight):
    @classmethod
    def setupClass(cls):
        self = cls # alias

        fweights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3]
        # faking aweights by using normalized freq_weights
        fweights = np.array(fweights)
        wsum = fweights.sum()
        nobs = len(cpunish_data.endog)
        aweights = fweights / wsum * nobs

        self.res1 = GLM(cpunish_data.endog, cpunish_data.exog,
                    family=sm.families.Poisson(), var_weights=aweights).fit()
        # compare with discrete, start close to save time
        modd = discrete.Poisson(cpunish_data.endog, cpunish_data.exog)
        
        # Need to copy to avoid inplace adjustment
        from copy import copy
        self.res2 = copy(res_stata.results_poisson_aweight_nonrobust)
        self.res2.resids = self.res2.resids.copy()

        # Need to adjust resids for pearson and deviance to add weights
        self.res2.resids[:, 3:5] *= np.sqrt(aweights[:, np.newaxis])


# prob_weights fail with HC, not properly implemented yet
class T_estGlmPoissonPwNr(CheckWeight):
    @classmethod
    def setupClass(cls):
        self = cls # alias

        fweights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3]
        # faking aweights by using normalized freq_weights
        fweights = np.array(fweights)
        wsum = fweights.sum()
        nobs = len(cpunish_data.endog)
        aweights = fweights / wsum * nobs

        self.res1 = GLM(cpunish_data.endog, cpunish_data.exog,
                    family=sm.families.Poisson(), freq_weights=fweights).fit(cov_type='HC1')
        # compare with discrete, start close to save time
        #modd = discrete.Poisson(cpunish_data.endog, cpunish_data.exog)
        self.res2 = res_stata.results_poisson_pweight_nonrobust


class TestGlmPoissonFwHC(CheckWeight):
    @classmethod
    def setupClass(cls):
        self = cls # alias

        fweights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3]
        # faking aweights by using normalized freq_weights
        fweights = np.array(fweights)
        wsum = fweights.sum()
        nobs = len(cpunish_data.endog)
        aweights = fweights / wsum * nobs
        self.corr_fact = np.sqrt((wsum - 1.) / wsum)
        self.res1 = GLM(cpunish_data.endog, cpunish_data.exog,
                        family=sm.families.Poisson(), freq_weights=fweights
                        ).fit(cov_type='HC0') #, cov_kwds={'use_correction':False})
        # compare with discrete, start close to save time
        #modd = discrete.Poisson(cpunish_data.endog, cpunish_data.exog)
        self.res2 = res_stata.results_poisson_fweight_hc1

# var_weights (aweights fail with HC, not properly implemented yet
class TestGlmPoissonAwHC(CheckWeight):
    @classmethod
    def setupClass(cls):
        self = cls # alias

        fweights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3]
        # faking aweights by using normalized freq_weights
        fweights = np.array(fweights)
        wsum = fweights.sum()
        nobs = len(cpunish_data.endog)
        aweights = fweights / wsum * nobs

        # This is really close when corr_fact = (wsum - 1.) / wsum, but to
        # avoid having loosen precision of the assert_allclose, I'm doing this
        # manually. Its *possible* lowering the IRLS convergence criterion
        # in stata and here will make this less sketchy. 
        self.corr_fact = np.sqrt((wsum - 1.) / wsum) * 0.98518473599905609
        self.res1 = GLM(cpunish_data.endog, cpunish_data.exog,
                        family=sm.families.Poisson(), var_weights=aweights
                        ).fit(cov_type='HC0') #, cov_kwds={'use_correction':False})
        # compare with discrete, start close to save time
        # modd = discrete.Poisson(cpunish_data.endog, cpunish_data.exog)
        self.res2 = res_stata.results_poisson_aweight_hc1


class TestGlmPoissonFwClu(CheckWeight):
    @classmethod
    def setupClass(cls):
        self = cls # alias

        fweights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3]
        # faking aweights by using normalized freq_weights
        fweights = np.array(fweights)
        wsum = fweights.sum()
        nobs = len(cpunish_data.endog)
        aweights = fweights / wsum * nobs

        gid = np.arange(1, 17 + 1) // 2
        n_groups = len(np.unique(gid))

        # no wnobs yet in sandwich covariance calcualtion
        self.corr_fact = 1 / np.sqrt(n_groups / (n_groups - 1))   #np.sqrt((wsum - 1.) / wsum)
        cov_kwds = {'groups': gid, 'use_correction':False}
        self.res1 = GLM(cpunish_data.endog, cpunish_data.exog,
                        family=sm.families.Poisson(), freq_weights=fweights
                        ).fit(cov_type='cluster', cov_kwds=cov_kwds)
        # compare with discrete, start close to save time
        #modd = discrete.Poisson(cpunish_data.endog, cpunish_data.exog)
        self.res2 = res_stata.results_poisson_fweight_clu1


class TestGlmTweedieAwNr(CheckWeight):
    @classmethod
    def setupClass(cls):
        self = cls
        import statsmodels.formula.api as smf

        data = sm.datasets.fair.load_pandas()
        endog = data.endog
        data = data.exog
        data['fair'] = endog
        aweights = np.repeat(1, len(data.index))
        aweights[::5] = 5
        aweights[::13] = 3
        model = smf.glm(
                'fair ~ age + yrs_married',
                data=data,
                family=sm.families.Tweedie(
                    var_power=1.55,
                    link=sm.families.links.log()
                    ),
                var_weights=aweights
        )
        self.res1 = model.fit(rtol=1e-25, atol=0)
        self.res2 = res_r.results_tweedie_aweights_nonrobust


class TestGlmGammaAwNr(CheckWeight):
    @classmethod
    def setupClass(cls):
        self = cls
        from .results.results_glm import CancerLog
        res2 = CancerLog()
        endog = res2.endog
        exog = res2.exog[:, :-1]
        exog = sm.add_constant(exog)

        aweights = np.repeat(1, len(endog))
        aweights[::5] = 5
        aweights[::13] = 3
        model = sm.GLM(endog, exog, 
                       family=sm.families.Gamma(link=sm.families.links.log()),
                       var_weights=aweights)
        self.res1 = model.fit(rtol=1e-25, atol=0)
        self.res2 = res_r.results_gamma_aweights_nonrobust

    def test_r_llf(self):
        scale = self.res1.deviance / self.res1._iweights.sum()
        ll = self.res1.model.family.loglike(self.res1.model.endog,
                                            self.res1.mu,
                                            self.res1._iweights,
                                            scale)
        assert_allclose(ll, self.res2.ll, atol=1e-6, rtol=1e-7)


class TestGlmGaussianAwNr(CheckWeight):
    @classmethod
    def setupClass(cls):
        self = cls
        import statsmodels.formula.api as smf

        data = sm.datasets.cpunish.load_pandas()
        endog = data.endog
        data = data.exog
        data['EXECUTIONS'] = endog
        data['INCOME'] /= 1000
        aweights = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2,
                             1])
        model = smf.glm(
                'EXECUTIONS ~ INCOME + SOUTH - 1',
                data=data,
                family=sm.families.Gaussian(link=sm.families.links.log()),
                var_weights=aweights
        )
        self.res1 = model.fit(rtol=1e-25, atol=0)
        self.res2 = res_r.results_gaussian_aweights_nonrobust

    def test_r_llf(self):
        res1 = self.res1
        res2 = self.res2
        model = self.res1.model

        # Need to make a few adjustments...
        # First, calculate scale using nobs as denominator
        scale = res1.scale * model.df_resid / model.wnobs
        # Calculate llf using adj scale and wts = freq_weights
        wts = model.freq_weights
        llf = model.family.loglike(model.endog, res1.mu, wts, scale)
        # SM uses (essentially) stat's loglike formula... first term is
        # (endog - mu) ** 2 / scale
        adj_sm = -1 / 2 * ((model.endog - res1.mu) ** 2).sum() / scale
        # R has these 2 terms that stata/sm don't
        adj_r = -model.wnobs / 2 + np.sum(np.log(model.var_weights)) / 2
        llf_adj = llf - adj_sm + adj_r
        assert_allclose(llf_adj, res2.ll, atol=1e-6, rtol=1e-7)
