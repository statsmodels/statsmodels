# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 23:39:45 2016

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from scipy import sparse
import pandas as pd

from statsmodels.regression.linear_model import OLS
from statsmodels.regression.special_linear_model import (
      OLSAbsorb, _group_demean_iterative, cat2dummy_sparse)
from statsmodels.tools._sparse import PartialingSparse, dummy_sparse

from numpy.testing import assert_allclose

# COMPAT: for numpy <= 1.7 until we increase minimum
from statsmodels.compat.scipy import NumpyVersion
np_smaller_108 = True if (NumpyVersion(np.__version__) < '1.8.0') else False
from numpy.testing import dec
from nose import SkipTest


class CompareModels(object):

    def test_attributes(self):
        res1 = self.res1
        res2 = self.res2
        slice_res1 = self.slice_res1
        slice_res2 = self.slice_res2

        good = ['HC0_se', 'HC1_se', '_data_attr', 'aic', 'bic', 'bse',
                'centered_tss', 'df_model', 'df_resid', 'ess', 'f_pvalue',
                'fvalue', 'k_constant', 'llf', 'mse_model', 'mse_resid',
                'mse_total', 'nobs', 'params', 'pvalues', 'resid_pearson',
                'rsquared', 'rsquared_adj', 'scale', 'ssr', 'tvalues',
                'use_t', 'wresid']

        known_fails = ['HC2_se', 'HC3_se', '_wexog_singular_values',
                       'condition_number', 'eigenvals', 'fittedvalues',
                       'resid', 'uncentered_tss']

        for a in good:
            a2 = getattr(res2, a)
            a1 = getattr(res1, a)
            alen = getattr(a2, '__len__', lambda:0)()
            if alen > 0 and not alen == len(a1):
                a1 = a1[slice_res1]
                a2 = a2[slice_res2]
            err_msg = a
            # resid_pearson fails at tol=1e-13
            assert_allclose(a1, a2, atol=1e-12, rtol=1e-12, err_msg=err_msg)


class TestSparseAbsorb(CompareModels):

    @classmethod
    def setup_class(self):
        #def generate_sample2():
        xcat1 = np.repeat(np.arange(5), 10)
        xcat2 = np.tile(np.arange(5), 10)
        exog_absorb = np.column_stack((xcat1, xcat2))

        df1 = pd.get_dummies(xcat1)  #sparse=True) #sparse requires v 0.16
        df2 = pd.get_dummies(xcat2)
        x = np.column_stack((np.asarray(df1)[:, 1:], np.asarray(df2)))
        exog = sparse.csc_matrix(x) #.astype(np.int8))
        beta = 1. / np.r_[np.arange(1, df1.shape[1]), np.arange(1, df2.shape[1] + 1)]

        np.random.seed(999)
        betad = np.ones(3)
        exogd = np.column_stack((np.ones(exog.shape[0]), np.random.randn(exog.shape[0], len(betad) - 1)))
        y = exogd.dot(betad) + exog.dot(beta) + 0.01 * np.random.randn(exog.shape[0])

        exog_full = np.column_stack((exogd, exog.toarray()))

        res_ols = OLS(y, exog_full[:, :-1]).fit()
        #print(res_ols.params)
        #print(res_ols.bse)

        mod_absorb = OLSAbsorb(y, exogd, exog_absorb)
        res_absorb = mod_absorb.fit()
        #print(res_absorb.params)
        #print(res_absorb.bse)

        # recover and check fixed effects parameters
        res_fe = mod_absorb._get_fixed_effects(res_absorb.resid)
        #print(res_fe.params[1:] - res_ols.params[3:])

        # check constant, corresponds to mean prediction of absorbed factors and constant
        #print(res_ols.predict(np.concatenate(([1, 0, 0], exog.toarray()[:,:-1].mean(0))))
        #      - res_absorb.params[0])
        # same as res_absorb.predict([1, 0, 0])

        self.slice_res1 = slice(1, 3, None)
        self.slice_res2 = slice(1, 3, None)

        self.res1 = res_absorb
        self.res2 = res_ols

        # keep as attributes
        self.endog = y
        self.exog_absorb = exog_absorb
        self.exog_dense = exogd


    def test_temp(self):
        res1 = self.res1
        res2 = self.res2
        slice_res1 = self.slice_res1
        slice_res2 = self.slice_res2

        assert_allclose(res1.params[slice_res1], res2.params[slice_res2], rtol=1e-13)
        assert_allclose(res1.bse[slice_res1], res2.bse[slice_res2], rtol=1e-13)
        assert_allclose(res1.pvalues[slice_res1], res2.pvalues[slice_res2], rtol=1e-10)
        #assert_allclose(res_fe.params[1:], res_ols.params[3:], rtol=1e-13)

    # the following test currently fails (len(y) is 25 instead of 50
    # the example is not a pure two-way effect
    def t_est_demean(self):
        exog_absorb = self.exog_absorb
        exog_dense = self.exog_dense
        endog = self.endog

        xm, xd, it = _group_demean_iterative(exog_dense, exog_absorb,
                                             add_mean=True, max_iter=50)
        ym, yd, it = _group_demean_iterative(endog[:,None], exog_absorb,
                                             add_mean=True, max_iter=50)
        mod_ols2 = OLS(yd, xd)
        k_cat = (exog_absorb.max(0) + 1)
        ddof = k_cat - 2
        mod_ols2.df_resid = mod_ols2.df_resid - ddof
        mod_ols2.df_model = mod_ols2.df_model + ddof
        res1 = mod_ols2.fit()
        res2 = self.res1   # compare with sparse OLSAbsorb

        assert_allclose(res1.params, res2.params, rtol=1e-13)
        assert_allclose(res1.bse, res2.bse, rtol=1e-13)
        assert_allclose(res1.pvalues, res2.pvalues, rtol=1e-10)


class TestAbsorb():

    @classmethod
    def setup_class(cls):
        if np_smaller_108: raise SkipTest("numpy too old for nanmean")
        # use decorator if we have more tests in here that need skipping
        k_cat1, k_cat2 = 50, 10
        k_vars = 3

        np.random.seed(654)
        keep = (np.random.rand(k_cat1 * k_cat2) > 0.1).astype(bool)

        xcat1 = np.repeat(np.arange(k_cat1), k_cat2)[keep]
        xcat2 = np.tile(np.arange(k_cat2), k_cat1)[keep]
        exog_absorb = np.column_stack((xcat1, xcat2))
        nobs = len(xcat1)
        exog_dense = np.random.randn(nobs, k_vars)
        xm, xd1, it = _group_demean_iterative(exog_dense, exog_absorb,
                                             add_mean=True, max_iter=50)

        absorb = cat2dummy_sparse(exog_absorb)
        projector = PartialingSparse(absorb, method='lu')
        exog_dense_mean = exog_dense.mean(0)
        xd2 = projector.partial_sparse(exog_dense)[1] + exog_dense_mean

        cls.res1 = xd1
        cls.res2 = xd2

    #@dec.skipif(np_smaller_108)
    def test_absorb(self):
        # Note: agreement (atol) depends on convergence tolerance
        assert_allclose(self.res1, self.res2, atol=1e-9)
