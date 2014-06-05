# -*- coding: utf-8 -*-
"""
Created on Fri May 30 22:56:57 2014

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from numpy.testing import assert_allclose, assert_raises

from statsmodels.base._constraints import (TransformRestriction,
                   transform_params_constraint, fit_constrained)


if __name__ == '__main__':



    R = np.array([[1, 1, 0, 0, 0], [0, 0, 1, -1, 0]])



    k_constr, k_vars = R.shape

    m = np.eye(k_vars) - R.T.dot(np.linalg.pinv(R).T)
    evals, evecs = np.linalg.eigh(m)

    L = evecs[:, :k_constr]
    T = evecs[:, k_constr:]

    print T.T.dot(np.eye(k_vars))

    tr = np.column_stack((T, R.T))

    q = [2, 0]
    tr0 = TransformRestriction(R, q)

    p_reduced = [1,1,1]
    #round trip test
    assert_allclose(tr0.reduce(tr0.expand(p_reduced)), p_reduced, rtol=1e-14)


    p = tr0.expand(p_reduced)
    assert_allclose(R.dot(p), q, rtol=1e-14)

    # inconsistent restrictions
    #Ri = np.array([[1, 1, 0, 0, 0], [0, 0, 1, -1, 0], [0, 0, 1, -2, 0]])
    R = np.array([[1, 1, 0, 0, 0], [0, 0, 1, -1, 0], [0, 0, 1, 0, -1]])
    q = np.zeros(R.shape[0])
    tr1 = TransformRestriction(R, q)  # bug raises error with q
    p = tr1.expand([1,1])

    # inconsistent restrictions that has a solution with p3=0
    Ri = np.array([[1, 1, 0, 0, 0], [0, 0, 1, -1, 0], [0, 0, 1, -2, 0]])
    tri = TransformRestriction(Ri, [0, 1, 1])
    # Note: the only way this can hold is if variable 3 is zero
    p = tri.expand([1,1])
    print(p[[2,3]])
    #array([  1.00000000e+00,  -1.34692639e-17])

    # inconsistent without any possible solution
    Ri2 = np.array([[0, 0, 0, 1, 0], [0, 0, 1, -1, 0], [0, 0, 1, -2, 0]])
    q = [1, 1]
    #tri2 = TransformRestriction(Ri2, q)
    #p = tri.expand([1,1])
    assert_raises(ValueError, TransformRestriction, Ri2, q)
    # L doesn't have full row rank, calculating constant fails with Singular Matrix

    # transform data xr = T x
    np.random.seed(1)
    x = np.random.randn(10, 5)
    xr = tr1.reduce(x)
    # roundtrip
    x2 = tr1.expand(xr)
    # this doesn't hold ? don't use constant? don't need it anyway ?
    #assert_allclose(x2, x, rtol=1e-14)


    from patsy import DesignInfo

    names = 'a b c d'.split()
    LC = DesignInfo(names).linear_constraint('a + b = 0')
    LC = DesignInfo(names).linear_constraint(['a + b = 0', 'a + 2*c = 1', 'b-a', 'c-a', 'd-a'])
    #LC = DesignInfo(self.model.exog_names).linear_constraint(r_matrix)
    r_matrix, q_matrix = LC.coefs, LC.constants

    np.random.seed(123)
    nobs = 20
    x = 1 + np.random.randn(nobs, 4)
    exog = np.column_stack((np.ones(nobs), x))
    endog = exog.sum(1) + np.random.randn(nobs)

    from statsmodels.regression.linear_model import OLS
    res2 = OLS(endog, exog).fit()
    #transf = TransformRestriction(np.eye(exog.shape[1])[:2], res2.params[:2] / 2)
    transf = TransformRestriction([[0, 0, 0,1,1]], res2.params[-2:].sum())
    exog_st = transf.reduce(exog)
    res1 = OLS(endog, exog_st).fit()
    # need to correct for constant/offset in the optimization
    res1 = OLS(endog - exog.dot(transf.constant.squeeze()), exog_st).fit()
    params = transf.expand(res1.params).squeeze()
    assert_allclose(params, res2.params, rtol=1e-13)
    print(res2.params)
    print(params)
    print(res1.params)

    res3_ols = OLS(endog - exog[:, -1], exog[:, :-2]).fit()
    #transf = TransformRestriction(np.eye(exog.shape[1])[:2], res2.params[:2] / 2)
    transf3 = TransformRestriction([[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]], [0, 1])
    exog3_st = transf3.reduce(exog)
    res3 = OLS(endog, exog3_st).fit()
    # need to correct for constant/offset in the optimization
    res3 = OLS(endog - exog.dot(transf3.constant.squeeze()), exog3_st).fit()
    params = transf3.expand(res3.params).squeeze()
    assert_allclose(params[:-2], res3_ols.params, rtol=1e-13)
    print(res3.params)
    print(params)
    print(res3_ols.params)
    print(res3_ols.bse)
    # the following raises `ValueError: can't test a constant constraint`
    #tt = res3.t_test(transf3.transf_mat, transf3.constant.squeeze())
    #print tt.sd
    cov_params3 = transf3.transf_mat.dot(res3.cov_params()).dot(transf3.transf_mat.T)
    bse3 = np.sqrt(np.diag(cov_params3))
    print(bse3)

    tp = transform_params_constraint(res2.params, res2.normalized_cov_params,
                                     transf3.R, transf3.q)
    tp = transform_params_constraint(res2.params, res2.cov_params(), transf3.R, transf3.q)


    from statsmodels.discrete.discrete_model import Poisson
    import statsmodels.api as sm
    rand_data = sm.datasets.randhie.load()
    rand_exog = rand_data.exog.view(float).reshape(len(rand_data.exog), -1)
    rand_exog = sm.add_constant(rand_exog, prepend=False)


    # Fit Poisson model:
    poisson_mod0 = sm.Poisson(rand_data.endog, rand_exog)
    poisson_res0 = poisson_mod0.fit(method="newton")

    R = np.zeros((2, 10))
    R[0, -2] = 1
    R[1, -1] = 1
    transfp = TransformRestriction(R, [0, 1])
    poisson_mod = sm.Poisson(rand_data.endog, rand_exog[:, :-2])
    # note wrong offset, why did I put offset in fit ? it's ignored
    poisson_res = poisson_mod.fit(method="newton", offset=rand_exog.dot(transfp.constant.squeeze()))

    exogp_st = transfp.reduce(rand_exog)
    poisson_modr = sm.Poisson(rand_data.endog, exogp_st)
    poisson_resr = poisson_modr.fit(method="newton")
    paramsp = transfp.expand(poisson_resr.params).squeeze()
    print('\nPoisson')
    print(paramsp)
    print(poisson_res.params)
    # error because I don't use the unconstrained basic model
#    tp = transform_params_constraint(poisson_res.params, poisson_res.cov_params(), transfp.R, transfp.q)
#    cov_params3 = transf3.transf_mat.dot(res3.cov_params()).dot(transf3.transf_mat.T)
#    bse3 = np.sqrt(np.diag(cov_params3))


    poisson_mod0 = sm.Poisson(rand_data.endog, rand_exog)
    poisson_res0 = poisson_mod0.fit(method="newton")
    tp = transform_params_constraint(poisson_res0.params, poisson_res0.cov_params(), transfp.R, transfp.q)
    cov_params3 = transf3.transf_mat.dot(res3.cov_params()).dot(transf3.transf_mat.T)
    bse3 = np.sqrt(np.diag(cov_params3))

    # try again same example as it was intended

    poisson_mod = sm.Poisson(rand_data.endog, rand_exog[:, :-2], offset=rand_exog[:, -1])
    poisson_res = poisson_mod.fit(method="newton")

    exogp_st = transfp.reduce(rand_exog)
    poisson_modr = sm.Poisson(rand_data.endog, exogp_st, offset=rand_exog.dot(transfp.constant.squeeze()))
    poisson_resr = poisson_modr.fit(method="newton")
    paramsp = transfp.expand(poisson_resr.params).squeeze()
    print('\nPoisson')
    print(paramsp)
    print(poisson_resr.params)
    tp = transform_params_constraint(poisson_res0.params, poisson_res0.cov_params(), transfp.R, transfp.q)
    cov_paramsp = transfp.transf_mat.dot(poisson_resr.cov_params()).dot(transfp.transf_mat.T)
    bsep = np.sqrt(np.diag(cov_paramsp))
    print(bsep)
    p, cov, res_r = fit_constrained(poisson_mod0, transfp.R, transfp.q)
    se = np.sqrt(np.diag(cov))
    print(p)
    print(se)
