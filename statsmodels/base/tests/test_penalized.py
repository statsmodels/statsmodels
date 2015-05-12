# -*- coding: utf-8 -*-
"""
Created on Sun May 10 12:39:33 2015

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.discrete.discrete_model import Poisson, Logit, Probit
from statsmodels.base._penalized import PenalizedMixin
import statsmodels.base._penalties as smpen

class PoissonPenalized(PenalizedMixin, Poisson):
    pass

class LogitPenalized(PenalizedMixin, Logit):
    pass

class ProbitPenalized(PenalizedMixin, Probit):
    pass


class CheckPenalizedPoisson(object):


    @classmethod
    def setup_class(cls):
        # simulate data
        np.random.seed(987865)

        nobs, k_vars = 500, 10
        k_nonzero = 4
        x = (np.random.rand(nobs, k_vars) + 0.5* (np.random.rand(nobs, 1)-0.5)) * 2 - 1
        x *= 1.2
        x[:, 0] = 1
        beta = np.zeros(k_vars)
        beta[:k_nonzero] = 1. / np.arange(1, k_nonzero + 1)
        linpred = x.dot(beta)
        mu = np.exp(linpred)
        y = np.random.poisson(mu)

        cls.k_nonzero = k_nonzero
        cls.x = x
        cls.y = y

        # defaults to be overwritten by subclasses
        cls.rtol = 1e-4
        cls.atol = 1e-6
        cls.exog_index = slice(None, None, None)
        cls.k_params = k_vars
        cls._initialize()


    def test_params_table(self):
        res1 = self.res1
        res2 = self.res2
        assert_equal((res1.params != 0).sum(), self.k_params)
        assert_allclose(res1.params[self.exog_index], res2.params, rtol=self.rtol, atol=self.atol)
        assert_allclose(res1.bse[self.exog_index], res2.bse, rtol=self.rtol, atol=self.atol)
        assert_allclose(res1.pvalues[self.exog_index], res2.pvalues, rtol=self.rtol, atol=self.atol)
        assert_allclose(res1.predict(), res2.predict(), rtol=0.05)

    def test_smoke(self):
        self.res1.summary()



class TestPenalizedPoissonNoPenal(CheckPenalizedPoisson):
    # TODO: check, adjust cov_type

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x

        modp = Poisson(y, x)
        cls.res2 = modp.fit()

        mod = PoissonPenalized(y, x)
        mod.pen_weight = 0
        cls.res1 = mod.fit(method='bfgs', maxiter=100)


class TestPenalizedPoissonOracle(CheckPenalizedPoisson):
    # TODO: check, adjust cov_type

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        modp = Poisson(y, x[:, :cls.k_nonzero])
        cls.res2 = modp.fit()

        mod = PoissonPenalized(y, x)
        mod.pen_weight *= 1.5
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(method='bfgs', maxiter=100)

        cls.exog_index = slice(None, cls.k_nonzero, None)

        cls.atol = 5e-3


class TestPenalizedPoissonOraclePenalized(CheckPenalizedPoisson):
    # TODO: check, adjust cov_type

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        modp = PoissonPenalized(y, x[:, :cls.k_nonzero])
        cls.res2 = modp.fit()

        mod = PoissonPenalized(y, x)
        #mod.pen_weight *= 1.5
        #mod.penal.tau = 0.05
        cls.res1 = mod.fit(method='bfgs', maxiter=100, trim=False)

        cls.exog_index = slice(None, cls.k_nonzero, None)

        cls.atol = 1e-3


class TestPenalizedPoissonOraclePenalized2(CheckPenalizedPoisson):
    # TODO: check, adjust cov_type

    @classmethod
    def _initialize(cls):
        y, x = cls.y, cls.x
        modp = PoissonPenalized(y, x[:, :cls.k_nonzero])
        cls.res2 = modp.fit()

        mod = PoissonPenalized(y, x)
        mod.pen_weight *= 1.5  # meed to penalize more to get oracle selection
        #mod.penal.tau = 0.05
        cls.res1 = mod.fit(method='bfgs', maxiter=100, trim=True)

        cls.exog_index = slice(None, cls.k_nonzero, None)

        cls.atol = 1e-8
        cls.k_params = cls.k_nonzero

    def test_zeros(self):

        # first test for trimmed result
        assert_equal(self.res1.params[self.k_nonzero:], 0)
        # we also set bse to zero, TODO: check fit_regularized
        assert_equal(self.res1.bse[self.k_nonzero:], 0)
