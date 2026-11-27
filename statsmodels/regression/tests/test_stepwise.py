# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:02:53 2019

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose, assert_equal

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


class CheckSubsetsSweeps(object):

    def test_compare_OLS(self):
        # shortcut alias
        keep_exog = self.keep_exog
        res_all = self.res_all

        # get arrays TODO: this might change, convert to array in Results class
        params = np.asarray(res_all.params_keep_all)
        bse = np.asarray(res_all.bse_keep_all)
        df_resid = np.asarray(res_all.df_resid_all)
        idx_sort_aic = res_all.idx_sort_aic

        k_best = 5
        df_summ = res_all.sorted_frame()
        df_best = df_summ.iloc[:k_best]
        df_resid_best = df_resid[idx_sort_aic[:k_best]]
        if self.keep_exog > 0:
            params_best = params[idx_sort_aic[:k_best]]
            bse_best = bse[idx_sort_aic[:k_best]]
        for i in range(df_best.shape[0]):
            mod = df_best.iloc[i]
            ex_idx = mod["exog_idx"]
            res_all_ssi = OLS(self.endog, self.exog[:, ex_idx]).fit()
            assert_allclose(mod["aic"], res_all_ssi.aic, rtol=1e-13)

            assert_allclose(df_resid_best[i], res_all_ssi.df_resid, rtol=1e-13)
            if self.keep_exog > 0:
                assert_allclose(params_best[i], res_all_ssi.params[:keep_exog],
                                rtol=1e-13)
                assert_allclose(bse_best[i], res_all_ssi.bse[:keep_exog],
                                rtol=1e-13)


class TestAllSubsetsSweep(CheckSubsetsSweeps):

    @classmethod
    def setup_class(cls):
        nobs, k_vars = 50, 4
        np.random.seed(85325783)
        x = np.random.randn(nobs, k_vars)
        x[:, 0] = 1.
        beta = 1 / np.arange(1, k_vars + 1)
        y = x[:, :k_vars].dot(beta) + np.random.randn(nobs)
        cls.endog, cls.exog = y, x
        cls.keep_exog = 1
        cls.res_all = all_subset(cls.endog, cls.exog, keep_exog=cls.keep_exog)

    def test_simple(self):
        res_all = self.res_all
        res_aic = np.array([131.84520748, 126.37159826, 126.64133021,
                            133.77530407, 135.29444877, 126.44745325,
                            127.11637884, 133.43356509])
        assert_allclose(res_all.aic, res_aic, rtol=1e-8)


class TestAllSubsetsSweepkeep(CheckSubsetsSweeps):

    @classmethod
    def setup_class(cls):
        nobs, k_vars = 50, 6
        np.random.seed(85325783)
        x = np.random.randn(nobs, k_vars)
        x[:, 0] = 1.
        beta = 1 / np.arange(1, k_vars + 1)
        y = x.dot(beta) + np.random.randn(nobs)
        cls.endog, cls.exog = y, x
        cls.keep_exog = 2
        cls.res_all = all_subset(cls.endog, cls.exog, keep_exog=cls.keep_exog)


class TestAllSubsetsSweepKMax(CheckSubsetsSweeps):

    @classmethod
    def setup_class(cls):
        nobs, k_vars = 50, 6
        np.random.seed(85325783)
        x = np.random.randn(nobs, k_vars)
        x[:, 0] = 1.
        beta = 1 / np.arange(1, k_vars + 1)
        y = x.dot(beta) + np.random.randn(nobs)
        cls.endog, cls.exog = y, x
        cls.keep_exog = 2
        cls.k_max = 2
        cls.res_all = all_subset(cls.endog, cls.exog, keep_exog=cls.keep_exog,
                                 k_max=cls.k_max)

    def test_kmax(self):
        res_all = self.res_all
        k_largest = res_all.isexog.sum(1).max()
        assert_equal(k_largest, self.k_max)
