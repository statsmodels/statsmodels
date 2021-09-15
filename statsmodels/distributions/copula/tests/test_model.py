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

from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.distributions.copula.api import (
    ClaytonCopula, GaussianCopula, FrankCopula,
    GumbelCopula, IndependenceCopula, CopulaDistribution)

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
        params = np.atleast_1d(params)
        cd = self.copula_distribution
        # ll = cd.logpdf(self.endog, args=(params[:2], params[2:]))
        cop_args = params[:self.k_copparams]
        if cop_args.size == 0:
            cop_args = ()
        if len(params) > self.k_copparams:
            marg_args = np.split(params[self.k_copparams:], 2)
        else:
            marg_args = None

        ll = cd.logpdf(self.endog,
                       cop_args=cop_args, marg_args=marg_args
                       ).sum()
        return ll


def get_data(nobs):
    cop_f = FrankCopula(theta=2)
    cd_f = CopulaDistribution(cop_f, [stats.norm, stats.norm])
    # np.random.seed(98645713)
    # at some seeds, parameters atol-differ from true
    # TODO: setting seed doesn't work for copula,
    # copula creates new randomly initialized random state, see #7650
    rng = np.random.RandomState(98645713)
    rvs = cd_f.rvs(nobs, random_state=rng)
    assert_allclose(rvs.mean(0), [-0.02936002,  0.06658304], atol=1e-7)
    return rvs


data_ev = get_data(500)


class CheckEVfit1(object):

    def test(self):
        cop = self.copula
        args = self.cop_args

        cev = CopulaDistribution(cop, [stats.norm, stats.norm], cop_args=None)
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

        cev = CopulaDistribution(cop, [stats.norm, stats.norm], cop_args=None)
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


# temporarily split for copulas that only have fixed cop_args
class CheckEVfit0(object):

    def test0(self):
        # test with fixed copula params
        cop = getattr(self, "copula_fixed", None)
        if cop is None:
            # skip test if not yet available
            return
        args = self.cop_args

        cev = CopulaDistribution(cop, [stats.norm, stats.norm], cop_args=args)
        k_marg = 2
        mod = CopulaModel(cev, data_ev + [0.5, -0.1],
                          k_params=0 + k_marg)

        # TODO: patching for now
        mod.k_copparams = 0
        mod.df_resid = len(mod.endog) - mod.nparams
        mod.df_model = mod.nparams - 0
        res = mod.fit(start_params=[0.5, -0.1],
                      method="bfgs")
        # the following fails in TestEVAsymLogistic with nan loglike
        # res = mod.fit(method="newton", start_params=res.params)

        assert mod.nparams == 0 + k_marg
        assert res.nobs == len(mod.endog)
        assert_allclose(res.params, [0.5, -0.1], atol=0.2)
        res.summary()
        assert res.mle_retvals["converged"]
        assert not np.isnan(res.bse).any()


class CheckEVfit(CheckEVfit1, CheckEVfit0):
    # unit test mainly for arg handling, not to verify parameter estimates
    pass


class TestEVHR(CheckEVfit):

    @classmethod
    def setup_class(cls):
        cls.copula = ExtremeValueCopula(transform=dep.HR())
        cls.cop_args = (1,)
        cls.k_copparams = 1
        cls.copula_fixed = ExtremeValueCopula(transform=dep.HR(),
                                              args=cls.cop_args)


class TestEVAsymLogistic(CheckEVfit):

    @classmethod
    def setup_class(cls):
        cls.copula = ExtremeValueCopula(transform=dep.AsymLogistic())
        cls.cop_args = (0.1, 0.7, 0.7)
        cls.k_copparams = 3
        cls.copula_fixed = ExtremeValueCopula(transform=dep.AsymLogistic(),
                                              args=cls.cop_args)


class TestEVAsymMixed(CheckEVfit):

    @classmethod
    def setup_class(cls):
        cls.copula = ExtremeValueCopula(transform=dep.AsymMixed())
        cls.cop_args = (0.5, 0.05)
        cls.k_copparams = 2
        cls.copula_fixed = ExtremeValueCopula(transform=dep.AsymMixed(),
                                              args=cls.cop_args)


class TestFrank(CheckEVfit):

    @classmethod
    def setup_class(cls):
        cls.copula = FrankCopula()
        cls.cop_args = (0.5,)
        cls.k_copparams = 1
        cls.copula_fixed = FrankCopula(*cls.cop_args)


class TestGaussian(CheckEVfit):

    @classmethod
    def setup_class(cls):
        cls.copula = GaussianCopula()
        cls.cop_args = ()
        cls.k_copparams = 0


class TestClayton(CheckEVfit):

    @classmethod
    def setup_class(cls):
        cls.copula = ClaytonCopula()
        cls.cop_args = (1.01,)
        cls.k_copparams = 1
        cls.copula_fixed = ClaytonCopula(*cls.cop_args)


class TestGumbel(CheckEVfit):

    @classmethod
    def setup_class(cls):
        cls.copula = GumbelCopula()
        cls.cop_args = (1.01,)
        cls.k_copparams = 1
        cls.copula_fixed = GumbelCopula(*cls.cop_args)


class TestIndependence(CheckEVfit0):

    @classmethod
    def setup_class(cls):
        cls.copula = IndependenceCopula()
        cls.cop_args = ()
        cls.k_copparams = 0
        cls.copula_fixed = IndependenceCopula(*cls.cop_args)
