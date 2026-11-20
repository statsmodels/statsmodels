
from __future__ import division

import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse

from statsmodels.regression.linear_model import OLS
from statsmodels.multivariate.sur_model import SURCompact, cov_func_spherical


def test_sur_ex():
    # just a dump of example
    # currently mainly special case with spherical cov equivalence to OLS

    # simulate data
    nobs = 1000
    ks = [2, 3, 4]
    n_eq = len(ks)
    np.random.seed(987125)

    xs = [np.column_stack((np.ones(nobs), np.random.rand(nobs, ki-1)))
          for ki in ks]
    ys = [0.5 * x.sum(1) + 0.5 * np.random.randn(nobs) for x in xs]


    # print('\nusing class')
    mod = SURCompact(ys, xs)
    res = mod.fit(maxiter=0)
    # print(res.params_block.todense())
    # print()
    # print(res.cov_resid)
    # print()
    # print(res.lmtest_uncorr())
    tt = res.t_test(np.eye(len(res.params)))
    # print(tt.summary(xname=res.model.data.param_names))
    # print()
    tt = res.t_test((np.eye(len(res.params)), 0.5 * np.ones(len(res.params))))
    # print(tt.summary(xname=['bias:%s' % s for s in res.model.data.param_names]))

    # print('\nbse')
    # print(res.bse)
    cov_hc0 = res._cov_hc0()
    bse_hc0 = np.sqrt(np.diag(cov_hc0))
    ## print(bse_hc0)
    # print(res.bse_hc0)
    xu = res._get_score_values()
    c = np.cov(xu, rowvar=0)
    # print(np.max(np.abs(res.normalized_cov_params - np.linalg.inv(c) / nobs)))
    # without demeaning score
    # print(np.max(np.abs(res.normalized_cov_params - np.linalg.inv(xu.T.dot(xu)))))




    mod_sph = SURCompact(ys, xs, cov_func=cov_func_spherical)
    res_sph = mod_sph.fit(maxiter=1)

    mod_ols = OLS(np.concatenate(ys), sparse.block_diag(xs).todense())
    res_ols = mod_ols.fit()
    time = np.tile(np.arange(nobs), len(ks))
    res_ols_clu = mod_ols.fit(cov_type='cluster', cov_kwds={'groups':time})

    # print(res_sph.bse_hc0 / res_ols_clu.bse)
    # some degrees of freedom differences like
    np.sqrt(((3000 - 9) / 3000) / ((1000 - 1) / 1000))

    cov_func_spherical(res_ols.resid[:,None]) / res_ols.scale
    fact = 0.99866533140389435   # regression number, df correction is not clear
    assert_allclose(res_sph.bse_hc0, fact * res_ols_clu.bse, rtol=1e-7)
    assert_allclose(res_sph.params, res_ols.params, rtol=1e-13)
    fact = 0.99899849749536518
    assert_allclose(res_sph.bse, fact * res_ols.bse, rtol=1e-13)

