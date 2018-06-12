# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:18:12 2018

Author: Josef Perktold
"""

import os.path
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

from statsmodels.discrete.discrete_model import Logit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families

import statsmodels.stats.outliers_influence as oi
from statsmodels.stats.outliers_influence import Influence

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
    assert_allclose(results_sas[:, 5:8] * res.bse.values, infl.dfbetas, atol=1e-4)
    assert_allclose(infl.cooks_distance[0] * k_vars, results_sas[:, 8], atol=6e-5)
    assert_allclose(infl.hat_matrix_diag, results_sas[:, 4], atol=6e-5)
    
    c_bar = infl.cooks_distance[0] * 3 * (1 - infl.hat_matrix_diag)
    assert_allclose(c_bar, results_sas[:, 9], atol=6e-5)

