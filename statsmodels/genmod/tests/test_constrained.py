# -*- coding: utf-8 -*-
"""
Unit tests for fit_constrained
Tests for Poisson and Binomial are in discrete


Created on Sun Jan  7 09:21:39 2018

Author: Josef Perktold
"""


import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.genmod.generalized_linear_model import GLM


class ConstrainedCompareMixin(object):

    @classmethod
    def setup_class(cls):
        nobs, k_exog = 100, 5
        np.random.seed(987125)
        x = np.random.randn(nobs, k_exog - 1)
        x = add_constant(x)

        y_true = x.sum(1) / 2
        y = y_true + 2 * np.random.randn(nobs)
        cls.endog = y
        cls.exog = x
        cls.idx_uc = [0, 2, 3, 4]
        cls.idx_p_uc = np.array(cls.idx_uc)
        cls.idx_c = [1]
        cls.exogc = xc = x[:, cls.idx_uc]
        mod_ols_c = OLS(y - 0.5 * x[:, 1], xc)
        mod_ols_c.exog_names[:] = ['const', 'x2', 'x3', 'x4']
        cls.mod2 = mod_ols_c
        cls.init()

    def test_params(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.params[self.idx_p_uc], res2.params, rtol=1e-10)

    def test_se(self):
        res1 = self.res1
        res2 = self.res2

        assert_equal(res1.df_resid, res2.df_resid)
        assert_allclose(res1.scale, res2.scale, rtol=1e-10)
        assert_allclose(res1.bse[self.idx_p_uc], res2.bse, rtol=1e-10)
        assert_allclose(res1.cov_params()[self.idx_p_uc[:, None],
                        self.idx_p_uc], res2.cov_params(), rtol=1e-10)

    def test_resid(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.resid_response, res2.resid, rtol=1e-10)


class TestGLMGaussianOffset(ConstrainedCompareMixin):

    @classmethod
    def init(cls):
        cls.res2 = cls.mod2.fit()
        mod = GLM(cls.endog, cls.exogc,
                  offset=0.5 * cls.exog[:, cls.idx_c].squeeze())
        mod.exog_names[:] = ['const', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit()
        cls.idx_p_uc = np.arange(cls.exogc.shape[1])


class TestGLMGaussianConstrained(ConstrainedCompareMixin):

    @classmethod
    def init(cls):
        cls.res2 = cls.mod2.fit()
        mod = GLM(cls.endog, cls.exog)
        mod.exog_names[:] = ['const', 'x1', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit_constrained('x1=0.5')


class TestGLMGaussianOffsetHC(ConstrainedCompareMixin):

    @classmethod
    def init(cls):
        cov_type = 'HC0'
        cls.res2 = cls.mod2.fit(cov_type=cov_type)
        mod = GLM(cls.endog, cls.exogc,
                  offset=0.5 * cls.exog[:, cls.idx_c].squeeze())
        mod.exog_names[:] = ['const', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit(cov_type=cov_type)
        cls.idx_p_uc = np.arange(cls.exogc.shape[1])


class TestGLMGaussianConstrainedHC(ConstrainedCompareMixin):

    @classmethod
    def init(cls):
        cov_type = 'HC0'
        cls.res2 = cls.mod2.fit(cov_type=cov_type)
        mod = GLM(cls.endog, cls.exog)
        mod.exog_names[:] = ['const', 'x1', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit_constrained('x1=0.5', cov_type=cov_type)


class ConstrainedCompareWtdMixin(ConstrainedCompareMixin):

    @classmethod
    def setup_class(cls):
        nobs, k_exog = 100, 5
        np.random.seed(987125)
        x = np.random.randn(nobs, k_exog - 1)
        x = add_constant(x)
        cls.aweights = np.random.randint(1, 10, nobs)

        y_true = x.sum(1) / 2
        y = y_true + 2 * np.random.randn(nobs)
        cls.endog = y
        cls.exog = x
        cls.idx_uc = [0, 2, 3, 4]
        cls.idx_p_uc = np.array(cls.idx_uc)
        cls.idx_c = [1]
        cls.exogc = xc = x[:, cls.idx_uc]
        mod_ols_c = WLS(y - 0.5 * x[:, 1], xc, weights=cls.aweights)
        mod_ols_c.exog_names[:] = ['const', 'x2', 'x3', 'x4']
        cls.mod2 = mod_ols_c
        cls.init()


class TestGLMWtdGaussianOffset(ConstrainedCompareWtdMixin):

    @classmethod
    def init(cls):
        cls.res2 = cls.mod2.fit()
        mod = GLM(cls.endog, cls.exogc,
                  offset=0.5 * cls.exog[:, cls.idx_c].squeeze(),
                  var_weights=cls.aweights)
        mod.exog_names[:] = ['const', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit()
        cls.idx_p_uc = np.arange(cls.exogc.shape[1])


class TestGLMWtdGaussianConstrained(ConstrainedCompareWtdMixin):

    @classmethod
    def init(cls):
        cls.res2 = cls.mod2.fit()
        mod = GLM(cls.endog, cls.exog, var_weights=cls.aweights)
        mod.exog_names[:] = ['const', 'x1', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit_constrained('x1=0.5')


class TestGLMWtdGaussianOffsetHC(ConstrainedCompareWtdMixin):

    @classmethod
    def init(cls):
        cov_type = 'HC0'
        cls.res2 = cls.mod2.fit(cov_type=cov_type)
        mod = GLM(cls.endog, cls.exogc,
                  offset=0.5 * cls.exog[:, cls.idx_c].squeeze(),
                  var_weights=cls.aweights)
        mod.exog_names[:] = ['const', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit(cov_type=cov_type)
        cls.idx_p_uc = np.arange(cls.exogc.shape[1])


class TestGLMWtdGaussianConstrainedHC(ConstrainedCompareWtdMixin):

    @classmethod
    def init(cls):
        cov_type = 'HC0'
        cls.res2 = cls.mod2.fit(cov_type=cov_type)
        mod = GLM(cls.endog, cls.exog, var_weights=cls.aweights)
        mod.exog_names[:] = ['const', 'x1', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit_constrained('x1=0.5', cov_type=cov_type)
