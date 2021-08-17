# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 16:19:55 2021

Author: Josef Perktod
License: BSD-3
"""
import numpy as np
from scipy import stats
# import matplotlib.pyplot as plt
from numpy.testing import assert_allclose
from scipy.stats import rankdata

from statsmodels.base.model import GenericLikelihoodModel
import statsmodels.distributions.copula.api as smc
from statsmodels.distributions.copula.api import (
    ClaytonCopula, GaussianCopula, FrankCopula,
    CopulaDistribution)

import statsmodels.distributions.tools as tools
from statsmodels.distributions.copula import depfunc_ev as dep
from statsmodels.distributions.copula.extreme_value import ExtremeValueCopula


class CopulaModel(GenericLikelihoodModel):

    def __init__(self, copula_distribution, endog, k_params=None):
        self.copula_distribution = copula_distribution
        self.endog = endog
        self.exog = None
        if k_params is None:
            k_params = 1
        self.nparams = k_params
        self.k_copparams = 1
        super().__init__(endog, self.exog)

    def split_params(self, params):
        pass

    def loglike(self, params):
        cd = self.copula_distribution
        # ll = cd.logpdf(self.endog, args=(params[:2], params[2:]))
        cop_args = params[:self.k_copparams]
        if len(params) > self.k_copparams:
            marg_args = np.split(params[self.k_copparams:], 2)
        else:
            marg_args = None

        ll = cd.logpdf(self.endog,
                       cop_args=cop_args, marg_args=marg_args
                       ).sum()
        return ll


def get_data(nobs):
    cop_f = FrankCopula()
    cd_f = CopulaDistribution([stats.norm, stats.norm], cop_f)
    # np.random.seed(98645713)  # at some seeds, parameters atol-differ from true
    # TODO: setting seed doesn't work for copula,
    # copula creates new randomly initialized random state, see #7650
    rng = np.random.RandomState(98645713)
    rvs = cd_f.random(nobs, random_state=rng)
    assert_allclose(rvs.mean(0), [-0.02936002,  0.06658304], atol=1e-7)
    return rvs


data_ev = get_data(500)


class CheckEVfit(object):

    def test(self):
        cop = self.copula
        args = self.cop_args

        cev = CopulaDistribution([stats.norm, stats.norm], cop, copargs=None)
        k_marg = 4
        mod = CopulaModel(cev, data_ev + [0.5, -0.1],
                          k_params=self.k_copparams + k_marg)

        # TODO: patching for now
        mod.k_copparams = self.k_copparams
        mod.df_resid = len(mod.endog) - mod.nparams
        mod.df_model = mod.nparams - 0
        res = mod.fit(start_params=list(args) + [0.5, 1, -0.1, 1],
                      method="bfgs")
        res = mod.fit(method="newton", start_params=res.params)

        assert mod.nparams == self.k_copparams + k_marg
        assert res.nobs == len(mod.endog)
        assert_allclose(res.params[[-4, -2]], [0.5, -0.1], atol=0.2)
        res.summary()
        assert res.mle_retvals["converged"]
        assert not np.isnan(res.bse).any()

    def test2m(self):
        cop = self.copula
        args = self.cop_args

        cev = CopulaDistribution([stats.norm, stats.norm], cop, copargs=None)
        k_marg = 2
        mod = CopulaModel(cev, data_ev + [0.5, -0.1],
                          k_params=self.k_copparams + k_marg)

        # TODO: patching for now
        mod.k_copparams = self.k_copparams
        mod.df_resid = len(mod.endog) - mod.nparams
        mod.df_model = mod.nparams - 0
        res = mod.fit(start_params=list(args) + [0.5, -0.1],
                      method="bfgs")
        # the following fails in TestEVAsymLogistic with nan loglike
        # res = mod.fit(method="newton", start_params=res.params)

        assert mod.nparams == self.k_copparams + k_marg
        assert res.nobs == len(mod.endog)
        assert_allclose(res.params[[-2, -1]], [0.5, -0.1], atol=0.2)
        res.summary()
        assert res.mle_retvals["converged"]
        assert not np.isnan(res.bse).any()


class TestEVHR(CheckEVfit):

    @classmethod
    def setup_class(cls):
        cls.copula = ExtremeValueCopula(transform=dep.HR())
        cls.cop_args = (1,)
        cls.k_copparams = 1


class TestEVAsymLogistic(CheckEVfit):

    @classmethod
    def setup_class(cls):
        cls.copula = ExtremeValueCopula(transform=dep.AsymLogistic())
        cls.cop_args = (0.1, 0.7, 0.7)
        cls.k_copparams = 3


class TestEVAsymMixed(CheckEVfit):

    @classmethod
    def setup_class(cls):
        cls.copula = ExtremeValueCopula(transform=dep.AsymMixed())
        cls.cop_args = (0.5, 0.05)
        cls.k_copparams = 2


class TestFrank(CheckEVfit):

    @classmethod
    def setup_class(cls):
        cls.copula = FrankCopula()
        cls.cop_args = (0.5,)
        cls.k_copparams = 1
