# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 23:43:23 2017

Author: Josef Perktold
Licence: BSD-3

"""

from __future__ import division

import numpy as np
from scipy import sparse

from numpy.testing import assert_allclose

from statsmodels.nonparametric.variance import (var_differencing,
                                               VarianceDiffProjector)

def test_var_nonparametric():

    nobs = 30
    t = np.linspace(0, 1, nobs)
    x = np.sin(2 * 5 * t) + 0.2 * np.random.randn(nobs) #* (1 + 1. * t) #heteroscedasticity
    X = np.asarray(x, float)
    if X.ndim > 1:
        X = X.squeeze()
    nobs = len(X)
    I = sparse.eye(nobs,nobs)
    offsets = np.array([0,1,2])
    #data = np.repeat([[1.],[-2.],[1.]], nobs, axis=1)
    data = np.repeat([[0.5], [-1.], [0.5]], nobs, axis=1)
    K = sparse.dia_matrix((data, offsets), shape=(nobs-2,nobs))

    v = x[1:-1].dot(K.toarray().dot(x)) * (2 / (3 * (nobs - 2)))
    #print(v, np.sqrt(v))
    xf = K.toarray().dot(x)
    v_ = xf.dot(xf) * (2 / (3 * (nobs - 2)))
    #print(v_, np.sqrt(v_))

    offsets2 = np.array([0, 1])
    #data = np.repeat([[1.],[-2.],[1.]], nobs, axis=1)
    data2 = np.repeat([[1], [-1.]], nobs, axis=1)
    k2 = sparse.dia_matrix((data2, offsets2), shape=(nobs-1,nobs))

    #v2 = x[1:-1].dot(k2.toarray().dot(x)) / (2 * (nobs - 1))
    #print(v2, np.sqrt(v2))

    xf2 = k2.toarray().dot(x)
    v2_ = xf2.dot(xf2) / (2 * (nobs - 1))
    #print(v2_, np.sqrt(v2_))

    v21 = x.dot(k2.T.dot(k2).dot(x)) / (2 * (nobs - 1))
    #print(v21, np.sqrt(v21))


    s, resid = var_differencing(x, x=None, order=2, method=None)
    k = 10
    #print(resid[:k * (resid.shape[0] // k)].reshape(k, -1, order="F").var(0))


    # building Difference matrix for unequal spaced
    offsets = np.array([0, 1])
    data1 = np.repeat([[-1], [1.]], nobs, axis=1)
    diff1_mat = sparse.dia_matrix((data1, offsets2), shape=(nobs-1,nobs))

    order = 2
    xd = []
    d1 = []
    xt = t # temporary alias for debugging
    for i in range(1, order+1):
        end = len(xt)
        #xd_diff = xt[order:] - xt[order - i:-i]  # backward difference
        #xd_diff = xt[i: end-order + i] - xt[:-i]  # forward difference
        xd_diff = xt[i:] - xt[:-i]
        assert len(xd_diff) == nobs - i
        xd_mat = sparse.dia_matrix((xd_diff, [0]), shape=(nobs-i,nobs-i))
        xd.append(xd_mat)
        d1_mat = sparse.dia_matrix((data1, offsets), shape=(nobs-i, nobs-i+1))
        d1.append(d1_mat)


    D1_ = xd[0].dot(d1[0])
    D2_ = xd[1].dot(d1[1].dot(D1_))
    D = D1_
    norm = np.sqrt(D.power(2).sum(1).A.ravel())
    #norm[norm==0] = 1
    Dn = (sparse.dia_matrix((1. / norm, [0]), shape=[D.shape[0], D.shape[0]]))

    D1 = Dn.dot(D)
    resid1 = D1.dot(x)
    vv1 = resid1.dot(resid1) / resid1.shape[0]

    #############
    D = D2_
    norm = np.sqrt(D.power(2).sum(1).A.ravel())
    #norm[norm==0] = 1
    Dn = (sparse.dia_matrix((1. / norm, [0]), shape=[D.shape[0], D.shape[0]]))

    D2 = Dn.dot(D)
    resid2 = D2.dot(x)
    vv2 = resid2.dot(resid2) / resid2.shape[0]

    #print(vv1, vv2)
    p = VarianceDiffProjector(x, t)
    #print(p.var(order=1), p.var(order=2))

    assert_allclose(vv2, v_, rtol=1e-13)
    assert_allclose(s, v_, rtol=1e-13)
    assert_allclose(p.var(order=2), v_, rtol=1e-13)

    assert_allclose(v21, v2_, rtol=1e-13)
    assert_allclose(vv1, v2_, rtol=1e-13)
    assert_allclose(p.var(order=1), v2_, rtol=1e-13)

    # vectorized in endog
    xx2 = np.column_stack((x, x**2))
    cov_xx = p.var(xx2)
    vxx = np.diag(cov_xx)
    vxx1 = np.array([p.var(x), p.var(x**2)])
    assert_allclose(vxx, vxx1, rtol=1e-13)

    p = VarianceDiffProjector(xx2, t)
    assert_allclose(p.var(), cov_xx, rtol=1e-13)
