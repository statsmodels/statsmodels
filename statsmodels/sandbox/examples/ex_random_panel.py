# -*- coding: utf-8 -*-
"""

Created on Fri May 18 13:05:47 2012

Author: Josef Perktold

moved example from main of random_panel
"""

import numpy as np
from statsmodels.sandbox.panel.panel_short import ShortPanelGLS, ShortPanelGLS2
from statsmodels.sandbox.panel.random_panel import PanelSample
import statsmodels.sandbox.panel.correlation_structures as cs

import statsmodels.stats.sandwich_covariance as sw
#from statsmodels.stats.sandwich_covariance import (
#                   S_hac_groupsum, weights_bartlett, _HCCM2)
from statsmodels.stats.moment_helpers import cov2corr, se_cov
cov_nw_panel2 = sw.cov_nw_groupsum


examples = ['ex1']


if 'ex1' in examples:
    nobs = 100
    nobs_i = 5
    n_groups = nobs // nobs_i
    k_vars = 3

#    dgp = PanelSample(nobs, k_vars, n_groups, corr_structure=cs.corr_equi,
#                      corr_args=(0.6,))
#    dgp = PanelSample(nobs, k_vars, n_groups, corr_structure=cs.corr_ar,
#                      corr_args=([1, -0.95],))
    dgp = PanelSample(nobs, k_vars, n_groups, corr_structure=cs.corr_arma,
                      corr_args=([1], [1., -0.9],), seed=377769)
    print('seed', dgp.seed)
    y = dgp.generate_panel()
    noise = y - dgp.y_true
    print(np.corrcoef(y.reshape(-1,n_groups, order='F')))
    print(np.corrcoef(noise.reshape(-1,n_groups, order='F')))

    mod = ShortPanelGLS2(y, dgp.exog, dgp.groups)
    res = mod.fit()
    print(res.params)
    print(res.bse)
    #Now what?
    #res.resid is of transformed model
    #np.corrcoef(res.resid.reshape(-1,n_groups, order='F'))
    y_pred = np.dot(mod.exog, res.params)
    resid = y - y_pred
    print(np.corrcoef(resid.reshape(-1,n_groups, order='F')))
    print(resid.std())
    err = y_pred - dgp.y_true
    print(err.std())
    #OLS standard errors are too small
    mod.res_pooled.params
    mod.res_pooled.bse
    #heteroscedasticity robust doesn't help
    mod.res_pooled.HC1_se
    #compare with cluster robust se

    print(sw.se_cov(sw.cov_cluster(mod.res_pooled, dgp.groups.astype(int))))
    #not bad, pretty close to panel estimator
    #and with Newey-West Hac
    print(sw.se_cov(sw.cov_nw_panel(mod.res_pooled, 4, mod.group.groupidx)))
    #too small, assuming no bugs,
    #see Peterson assuming it refers to same kind of model
    print(dgp.cov)

    mod2 = ShortPanelGLS(y, dgp.exog, dgp.groups)
    res2 = mod2.fit_iterative(2)
    print(res2.params)
    print(res2.bse)
    #both implementations produce the same results:
    from numpy.testing import assert_almost_equal
    assert_almost_equal(res.params, res2.params, decimal=12)
    assert_almost_equal(res.bse, res2.bse, decimal=13)
    mod5 = ShortPanelGLS(y, dgp.exog, dgp.groups)
    res5 = mod5.fit_iterative(5)
    print(res5.params)
    print(res5.bse)
    #fitting once is the same as OLS
    #note: I need to create new instance, otherwise it continuous fitting
    mod1 = ShortPanelGLS(y, dgp.exog, dgp.groups)
    res1 = mod1.fit_iterative(1)
    res_ols = mod1._fit_ols()
    assert_almost_equal(res1.params, res_ols.params, decimal=12)
    assert_almost_equal(res1.bse, res_ols.bse, decimal=13)

    #cov_hac_panel with uniform_kernel is the same as cov_cluster for balanced
    #panel with full length kernel
    #I fixe default correction to be equal
    mod2._fit_ols()
    cov_clu = sw.cov_cluster(mod2.res_pooled, dgp.groups.astype(int))
    clubse = se_cov(cov_clu)
    cov_uni = sw.cov_nw_panel(mod2.res_pooled, 4, mod2.group.groupidx,
                              weights_func=sw.weights_uniform,
                              use_correction='cluster')
    assert_almost_equal(cov_uni, cov_clu, decimal=7)

    #without correction
    cov_clu2 = sw.cov_cluster(mod2.res_pooled, dgp.groups.astype(int),
                              use_correction=False)
    cov_uni2 = sw.cov_nw_panel(mod2.res_pooled, 4, mod2.group.groupidx,
                              weights_func=sw.weights_uniform,
                              use_correction=False)
    assert_almost_equal(cov_uni2, cov_clu2, decimal=8)

    cov_white = sw.cov_white_simple(mod2.res_pooled)
    cov_pnw0 = sw.cov_nw_panel(mod2.res_pooled, 0, mod2.group.groupidx,
                              use_correction='hac')
    assert_almost_equal(cov_pnw0, cov_white, decimal=13)

    time = np.tile(np.arange(nobs_i), n_groups)
    #time = mod2.group.group_int
    cov_pnw1 = sw.cov_nw_panel(mod2.res_pooled, 4, mod2.group.groupidx)
    cov_pnw2 = cov_nw_panel2(mod2.res_pooled, 4, time)
    #s = sw.group_sums(x, time)

    c2, ct, cg = sw.cov_cluster_2groups(mod2.res_pooled, time, dgp.groups.astype(int), use_correction=False)
    ct_nw0 = cov_nw_panel2(mod2.res_pooled, 0, time, weights_func=sw.weights_uniform, use_correction=False)
    cg_nw0 = cov_nw_panel2(mod2.res_pooled, 0, dgp.groups.astype(int), weights_func=sw.weights_uniform, use_correction=False)
    assert_almost_equal(ct_nw0, ct, decimal=13)
    assert_almost_equal(cg_nw0, cg, decimal=13)   #pnw2 0 lags
    assert_almost_equal(cov_clu2, cg, decimal=13)
    assert_almost_equal(cov_uni2, cg, decimal=8)  #pnw all lags




    import pandas as pa
    #pandas.DataFrame doesn't do inplace append
    se = pa.DataFrame(res_ols.bse[None,:], index=['OLS'])
    se = se.append(pa.DataFrame(res5.bse[None,:], index=['PGLSit5']))
    clbse = sw.se_cov(sw.cov_cluster(mod.res_pooled, dgp.groups.astype(int)))
    se = se.append(pa.DataFrame(clbse[None,:], index=['OLSclu']))
    pnwse = sw.se_cov(sw.cov_nw_panel(mod.res_pooled, 4, mod.group.groupidx))
    se = se.append(pa.DataFrame(pnwse[None,:], index=['OLSpnw']))
    print(se)
    #list(se.index)
    from statsmodels.iolib.table import SimpleTable
    headers = [str(i) for i in se.columns]
    stubs=list(se.index)
#    print SimpleTable(np.round(np.asarray(se), 4),
#                      headers=headers,
#                      stubs=stubs)
    print(SimpleTable(np.asarray(se), headers=headers, stubs=stubs,
                      txt_fmt=dict(data_fmts=['%10.4f']),
                      title='Standard Errors'))
