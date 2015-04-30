# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 06:46:13 2015

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_less, assert_equal
import statsmodels.stats.outliers_influence as smio
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multicollinearity import (vif, vif_selection,
                         MultiCollinearity, MultiCollinearitySequential)


def assert_allclose_large(x, y, rtol=1e-6, atol=0, ltol=1e30):
    """ assert x and y are allclose or x is large if y is inf
    """
    mask_inf = np.isinf(y) & ~np.isinf(x)
    assert_allclose(x[~mask_inf], y[~mask_inf], rtol=rtol, atol=atol)
    assert_array_less(ltol, x[mask_inf])


class CheckMuLtiCollinear(object):


    @classmethod
    def get_data(cls):
        np.random.seed(987536)
        nobs, k_vars = 100, 4
        rho_coeff = np.linspace(0.5, 0.9, k_vars - 1)
        x = np.random.randn(nobs, k_vars - 1) * (1 - rho_coeff)
        x += rho_coeff * np.random.randn(nobs, 1)

        cls.x = x
        cls.xs = (x - x.mean(0)) / x.std(0)
        cls.xf = np.column_stack((np.ones(nobs), x))
        cls.check_pandas = False

    def test_sequential(self):
        xf = self.xf
        ols_results = [OLS(xf[:,k], xf[:, :k]).fit()
                           for k in range(1, xf.shape[1])]
        rsquared0 = np.array([res.rsquared for res in ols_results])
        vif0 = 1. / (1. - rsquared0)

        mcoll = MultiCollinearitySequential(self.x)

        assert_allclose(mcoll.partial_corr, rsquared0, rtol=1e-13, atol=1e-15)
        # infs could be just large values because of floating point imprecision
        #assert_allclose(mcoll.vif, vif0, rtol=1e-13)
        #mask_inf = np.isinf(vif0) & ~np.isinf(mcoll.vif)
        #assert_allclose(mcoll.vif[~mask_inf], vif0[~mask_inf], rtol=1e-13)
        #assert_array_less(1e30, mcoll.vif[mask_inf])
        assert_allclose_large(mcoll.vif, vif0, rtol=1e-13)


    def test_multicoll(self):
        xf = np.asarray(self.xf)
        nobs, k_vars = self.xf.shape
        ols_results = []

        idx = list(range(k_vars))
        for k in range(1, k_vars):
            idx_k = idx.copy()
            del idx_k[k]
            ols_results.append(OLS(xf[:,k], xf[:, idx_k]).fit())

        rsquared0 = np.array([res.rsquared for res in ols_results])
        vif0 = 1. / (1. - rsquared0)

        mcoll = MultiCollinearity(self.x)

        assert_allclose(mcoll.partial_corr, rsquared0, rtol=1e-13)
        assert_allclose_large(mcoll.vif, vif0, rtol=1e-13)

        vif1_ = vif(self.x)
        vif1 = np.asarray(vif1_)   # check values if pandas.Series
        # TODO: why does mcoll.vif have infs but vif1 doesn't?
        assert_allclose_large(vif1, mcoll.vif, rtol=1e-13, ltol=1e-14)
        assert_allclose_large(vif1, vif0, rtol=1e-13, ltol=1e-14)

        if self.check_pandas:
            assert_equal(vif1_.index.values, self.names)


class TestMultiCollinearSingular1(CheckMuLtiCollinear):
    # Example: with singular continuous data
    @classmethod
    def setup_class(cls):
        cls.get_data()
        x = np.column_stack((cls.x[:, -2:].sum(1), cls.x))

        cls.x = x
        cls.xs = (x - x.mean(0)) / x.std(0)
        cls.xf = np.column_stack((np.ones(x.shape[0]), x))


class TestMultiCollinearSingular2(CheckMuLtiCollinear):
    # Example: with singular dummy variables
    @classmethod
    def setup_class(cls):
        cls.get_data()
        nobs = cls.x.shape[0]
        xd = np.tile(np.arange(5), nobs // 5)[:, None] == np.arange(5)
        x = np.column_stack((cls.x, xd))

        cls.x = x
        cls.xs = (x - x.mean(0)) / x.std(0)
        cls.xf = np.column_stack((np.ones(x.shape[0]), x))


class TestMultiCollinearPandas(CheckMuLtiCollinear):
    # Example: with singular continuous data
    @classmethod
    def setup_class(cls):
        cls.get_data()
        import pandas
        cls.names = ['var%d' % i for i in range(cls.x.shape[1])]
        cls.x = pandas.DataFrame(cls.x, columns=cls.names)
        cls.check_pandas = True
