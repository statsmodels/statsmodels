# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:18:12 2018

Author: Josef Perktold
"""

import os.path
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
try:
    import pandas.testing as pdt
except ImportError:
    import pandas.util.testing as pdt

try:
    import matplotlib.pyplot as plt
    have_matplotlib = True
    plt.switch_backend('Agg')
except:
    have_matplotlib = False

from statsmodels.regression.linear_model import OLS
from statsmodels.discrete.discrete_model import Logit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families

import statsmodels.stats.outliers_influence as oi
from statsmodels.stats.outliers_influence import (GLMInfluence,
                                                  MLEInfluence)

from statsmodels.compat.testing import skipif

cur_dir = os.path.abspath(os.path.dirname(__file__))

file_name = 'binary_constrict.csv'
file_path = os.path.join(cur_dir, 'results', file_name)
data_bin = pd.read_csv(file_path, index_col=0)

file_name = 'results_influence_logit.csv'
file_path = os.path.join(cur_dir, 'results', file_name)
results_sas_df = pd.read_csv(file_path, index_col=0)


def test_influence_glm_bernoulli():

    df = data_bin
    results_sas = np.asarray(results_sas_df)

    res = Logit(df['constrict'], df[['const', 'log_rate', 'log_volumne']]).fit()
    print(res.summary())

    res = GLM(df['constrict'], df[['const', 'log_rate', 'log_volumne']],
              family=families.Binomial()).fit(attach_wls=True, atol=1e-10)
    print(res.summary())

    wexog = res.results_wls.model.wexog
    pinv_wexog = res.results_wls.model.pinv_wexog
    hat_diag = (wexog * pinv_wexog.T).sum(1)

    infl = res.get_influence(observed=False)
    # monkey patching to see what's missing
    res._results.mse_resid = res.scale

    mask = np.ones(len(df), np.bool_)
    mask[3] = False
    df_m3 = df[mask]
    res_m3 = GLM(df_m3['constrict'], df_m3[['const', 'log_rate', 'log_volumne']],
                 family=families.Binomial()).fit(attach_wls=True, atol=1e-10)

    mask = np.ones(len(df), np.bool_)
    mask[0] = False
    df_m0 = df[mask]
    res_m0 = GLM(df_m0['constrict'], df_m0[['const', 'log_rate', 'log_volumne']],
                 family=families.Binomial()).fit(attach_wls=True, atol=1e-10)

    hess = res.model.hessian(res.params, observed=False)
    score_obs = res.model.score_obs(res.params)
    cd = (score_obs * np.linalg.solve(-hess, score_obs.T).T).sum(1)
    db = np.linalg.solve(-hess, score_obs.T).T
    hess_oim = res.model.hessian(res.params, observed=True)
    db_oim = np.linalg.solve(-hess_oim, score_obs.T).T



    k_vars = 3
    assert_allclose(infl.dfbetas, results_sas[:, 5:8], atol=1e-4)
    assert_allclose(infl.d_params, results_sas[:, 5:8] * res.bse.values, atol=1e-4)
    assert_allclose(infl.cooks_distance[0] * k_vars, results_sas[:, 8], atol=6e-5)
    assert_allclose(infl.hat_matrix_diag, results_sas[:, 4], atol=6e-5)

    c_bar = infl.cooks_distance[0] * 3 * (1 - infl.hat_matrix_diag)
    assert_allclose(c_bar, results_sas[:, 9], atol=6e-5)


class InfluenceCompareExact(object):


    def test_basics(self):
        infl1 = self.infl1
        infl0 = self.infl0

        assert_allclose(infl0.hat_matrix_diag, infl1.hat_matrix_diag,
                        rtol=1e-12)

        assert_allclose(infl0.resid_studentized,
                        infl1.resid_studentized, rtol=1e-12)
        assert_allclose(infl0.cooks_distance, infl1.cooks_distance, rtol=1e-7)
        assert_allclose(infl0.dfbetas, infl1.dfbetas, rtol=1e-9, atol=1e-14)
        assert_allclose(infl0.d_params, infl1.d_params, rtol=1e-9, atol=1e-14)
        assert_allclose(infl0.d_fittedvalues, infl1.d_fittedvalues, rtol=1e-9)
        assert_allclose(infl0.d_fittedvalues_scaled,
                        infl1.d_fittedvalues_scaled, rtol=1e-9)

    @skipif(not have_matplotlib, reason='matplotlib not available')
    def test_plots(self):
        # SMOKE tests for plots
        infl1 = self.infl1
        infl0 = self.infl0

        fig = infl0.plot_influence()
        plt.close(fig)
        fig = infl1.plot_influence()
        plt.close(fig)

        fig = infl0.plot_index('resid', threshold=0.2, title='')
        plt.close(fig)
        fig = infl1.plot_index('resid', threshold=0.2, title='')
        plt.close(fig)

        fig = infl0.plot_index('dfbeta', idx=1, threshold=0.2, title='')
        plt.close(fig)
        fig = infl1.plot_index('dfbeta', idx=1, threshold=0.2, title='')
        plt.close(fig)

        fig = infl0.plot_index('cook', idx=1, threshold=0.2, title='')
        plt.close(fig)
        fig = infl1.plot_index('cook', idx=1, threshold=0.2, title='')
        plt.close(fig)

        fig = infl0.plot_index('hat', idx=1, threshold=0.2, title='')
        plt.close(fig)
        fig = infl1.plot_index('hat', idx=1, threshold=0.2, title='')
        plt.close(fig)
        plt.close('all')

    def test_summary(self):
        infl1 = self.infl1
        infl0 = self.infl0

        df0 = infl0.summary_frame()
        df1 = infl1.summary_frame()
        assert_allclose(df0.values, df1.values, rtol=1e-7)
        pdt.assert_index_equal(df0.index, df1.index)


def _check_looo(self):
    infl = self.infl1
    # unwrap if needed
    results = getattr(infl.results, '_results', infl.results)

    res_looo = infl._res_looo
    mask_infl = infl.cooks_distance[0] > 2 * infl.cooks_distance[0].std()
    mask_low = ~mask_infl
    diff_params = results.params - res_looo['params']
    assert_allclose(infl.d_params[mask_low], diff_params[mask_low], atol=0.05)
    assert_allclose(infl.params_one[mask_low], res_looo['params'][mask_low], rtol=0.01)


class TestInfluenceLogitGLMMLE(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        df = data_bin
        res = GLM(df['constrict'], df[['const', 'log_rate', 'log_volumne']],
              family=families.Binomial()).fit(attach_wls=True, atol=1e-10)

        cls.infl1 = res.get_influence()
        cls.infl0 = MLEInfluence(res)

    def test_looo(self):
        _check_looo(self)


class TestInfluenceGaussianGLMMLE(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        from .test_diagnostic import get_duncan_data
        endog, exog, labels = get_duncan_data()
        data = pd.DataFrame(np.column_stack((endog, exog)),
                        columns='y const var1 var2'.split(),
                        index=labels)

        res = GLM.from_formula('y ~ const + var1 + var2 - 1', data).fit()
        #res = GLM(endog, exog).fit()

        cls.infl1 = res.get_influence()
        cls.infl0 = MLEInfluence(res)

    def test_looo(self):
        _check_looo(self)


class TestInfluenceGaussianGLMOLS(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        from .test_diagnostic import get_duncan_data
        endog, exog, labels = get_duncan_data()
        data = pd.DataFrame(np.column_stack((endog, exog)),
                        columns='y const var1 var2'.split(),
                        index=labels)

        res0 = GLM.from_formula('y ~ const + var1 + var2 - 1', data).fit()
        res1 = OLS.from_formula('y ~ const + var1 + var2 - 1', data).fit()
        cls.infl1 = res1.get_influence()
        cls.infl0 = res0.get_influence()

    def test_basics(self):
        # needs to override attributes that are not equivalent,
        # i.e. not available or different definition like external vs internal
        infl1 = self.infl1
        infl0 = self.infl0

        assert_allclose(infl0.hat_matrix_diag, infl1.hat_matrix_diag,
                        rtol=1e-12)
        assert_allclose(infl0.resid_studentized,
                        infl1.resid_studentized, rtol=1e-12)
        assert_allclose(infl0.cooks_distance, infl1.cooks_distance, rtol=1e-7)
        assert_allclose(infl0.dfbetas, infl1.dfbetas, rtol=0.1) # changed
        # OLSInfluence only has looo dfbeta/d_params
        assert_allclose(infl0.d_params, infl1.dfbeta, rtol=1e-9, atol=1e-14)
        # d_fittedvalues is not available in OLSInfluence, i.e. only scaled dffits
        # assert_allclose(infl0.d_fittedvalues, infl1.d_fittedvalues, rtol=1e-9)
        assert_allclose(infl0.d_fittedvalues_scaled,
                        infl1.dffits_internal[0], rtol=1e-9)

        # specific to linear link
        assert_allclose(infl0.d_linpred,
                        infl0.d_fittedvalues, rtol=1e-12)
        assert_allclose(infl0.d_linpred_scaled,
                        infl0.d_fittedvalues_scaled, rtol=1e-12)


    def test_summary(self):
        infl1 = self.infl1
        infl0 = self.infl0

        df0 = infl0.summary_frame()
        df1 = infl1.summary_frame()
        # just some basic check on overlap except for dfbetas
        cols = ['cooks_d', 'standard_resid', 'hat_diag', 'dffits_internal']
        assert_allclose(df0[cols].values, df1[cols].values, rtol=1e-5)
        pdt.assert_index_equal(df0.index, df1.index)
