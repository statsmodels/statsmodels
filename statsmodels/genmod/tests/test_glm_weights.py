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
        assert_allclose(res1.resid_anscombe, resid_all['resid_anscombe'], atol= 1e-6, rtol=2e-6)
        assert_allclose(res1.resid_working, resid_all['resid_working'], atol= 1e-6, rtol=2e-6)


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
                    family=sm.families.Poisson(), freq_weights=aweights).fit()
        # compare with discrete, start close to save time
        modd = discrete.Poisson(cpunish_data.endog, cpunish_data.exog)
        self.res2 = res_stata.results_poisson_aweight_nonrobust

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
class T_estGlmPoissonAwHC(CheckWeight):
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
                        family=sm.families.Poisson(), freq_weights=aweights
                        ).fit(cov_type='HC0') #, cov_kwds={'use_correction':False})
        # compare with discrete, start close to save time
        #modd = discrete.Poisson(cpunish_data.endog, cpunish_data.exog)
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
