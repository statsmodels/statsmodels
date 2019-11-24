# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:02:53 2019

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose

from statsmodels.regression.linear_model import OLS
from statsmodels.regression._stepwise import (
        StepwiseOLSSweep, get_sweep_matrix_data, all_subset)


def test_sweep_matrix():
    nobs, k_vars = 50, 4
    np.random.seed(85325783)
    x = np.random.randn(nobs, k_vars)
    x[:, 0] = 1.
    y = x[:, :k_vars-1].sum(1) + np.random.randn(nobs)
    xy = np.column_stack((x, y))
    swmat = get_sweep_matrix_data(xy, [0, 1])

    sto = StepwiseOLSSweep(y, x)
    sto.sweep(0)
    sto.sweep(1)

    assert_allclose(sto.rs_current, swmat, rtol=1e-12)


class TestSweep(object):

    def setup(self):
        # fit for each test, because results can be changed by test
        self.stols = StepwiseOLSSweep(self.endog, self.exog)

    @classmethod
    def setup_class(cls):
        nobs, k_vars = 50, 4

        np.random.seed(85325783)
        x = np.random.randn(nobs, k_vars)
        x[:, 0] = 1.
        y = x[:, :k_vars-1].sum(1) + np.random.randn(nobs)
        cls.endog, cls.exog = y, x

        cls.ols_cache = {}

    def cached_ols(self, is_exog):
        key = tuple(is_exog)
        if key in self.ols_cache:
            res = self.ols_cache[key]
        else:
            res = OLS(self.endog, self.exog[:, is_exog]).fit()
            self.ols_cache[key] = res
        return res

    def test_sequence(self):
        # test that params_new and rss_diff correspond to one sweep step
        stols = self.stols
        for k in range(stols.k_vars_x-1):  # last index is endog
            # store anticipated results
            params_new = stols.params_new().copy()
            rss_new = stols.rss - stols.rss_diff()
            # make next sweep
            stols.sweep(k)
            res = self.cached_ols(stols.is_exog[:-1])  # last col is endog
            assert_allclose(params_new[k, k], stols.params[0, k])
            assert_almost_equal(params_new[k, k], res.params[k], decimal=13)
            assert_almost_equal(rss_new[k], stols.rss, decimal=13)
            assert_almost_equal(stols.rss, res.ssr, decimal=13)

    def test_sequence_basic(self):
        # test without anticipated results
        stols = self.stols
        for k in range(stols.k_vars_x-1):  # last index is endog
            stols.sweep(k)
            res = self.cached_ols(stols.is_exog[:-1])  # last col is endog

            assert_almost_equal(stols.params[0][k], res.params[k], decimal=13)
            assert_almost_equal(stols.params[0], res.params, decimal=13)
            assert_almost_equal(stols.rss, res.ssr, decimal=13)


class TestAllSubsetsSweep(object):

    @classmethod
    def setup_class(cls):
        nobs, k_vars = 50, 4
        np.random.seed(85325783)
        x = np.random.randn(nobs, k_vars)
        x[:, 0] = 1.
        beta = 1 / np.arange(1, k_vars + 1)
        y = x[:, :k_vars].dot(beta) + np.random.randn(nobs)
        cls.endog, cls.exog = y, x

    def test_simple(self):
        res_all = all_subset(self.endog, self.exog, keep_exog=1)
        res_aic = np.array([131.84520748, 126.37159826, 126.64133021,
                            133.77530407, 135.29444877, 126.44745325,
                            127.11637884, 133.43356509])
        assert_allclose(res_all.aic, res_aic, rtol=1e-8)
