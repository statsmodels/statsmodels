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
from statsmodels.regression.special_linear_model import OLSAbsorb

from numpy.testing import assert_allclose


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

    def test_temp(self):
        res1 = self.res1
        res2 = self.res2
        slice_res1 = self.slice_res1
        slice_res2 = self.slice_res2

        assert_allclose(res1.params[slice_res1], res2.params[slice_res2], rtol=1e-13)
        assert_allclose(res1.bse[slice_res1], res2.bse[slice_res2], rtol=1e-13)
        assert_allclose(res1.pvalues[slice_res1], res2.pvalues[slice_res2], rtol=1e-10)
        #assert_allclose(res_fe.params[1:], res_ols.params[3:], rtol=1e-13)
