# -*- coding: utf-8 -*-
"""
symmetric matrix helper functions for use with covariance and correlation
matrices

follows Neudecker and Wesselman 1990
more details in Magnus and Neudecker 1996
some correlation functions based on Staiger and Hakstian 1982

Warning: These are a reference implementation and the matrices can become
vary large. The objective is to replace these functions with efficient
operators or sparse matrices and use the matrix functions as reference and
test cases and for prototyping.



Created on Thu Nov 23 13:39:47 2017

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_equal, assert_allclose
from statsmodels.stats._cov_tools import (Dup, E, K, L, Md, Md_0,
    Ms, _cov_corr, _cov_cov, _cov_cov_mom4, _cov_cov_vech,
    _gls_cov_vech, _mom4_normal, cov_corr_coef, cov_corr_coef_normal,
    cov_cov_data, cov_cov_fisherz, dg, mom4, ravel_indices, unvec,
    unvech, vec, vech, veclow, vech_cross_product)
import statsmodels.stats._cov_tools as ct

def test_matrix_tools():
    xf = np.arange(12).reshape(3, 4).T
    xsf = np.arange(9).reshape(3, 3).T
    xs = (xsf + xsf.T)

    res = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    assert_equal(vec(xf), res)
    assert_equal(unvec(vec(xf), *xf.shape), xf)

    res = np.array([ 0,  1,  2,  3,  5,  6,  7, 10, 11])
    assert_equal(vech(xf), res)
    res = np.array([ 1,  2,  5])
    assert_equal(veclow(xsf), res)
    # it works for non-square matrices, but that's not the usecase
    res = np.array([ 1,  2,  3,  6,  7, 11])
    assert_equal(veclow(xf), res)

    assert_equal(dg(xsf), np.diag(np.diag(xsf)))

    # check just one case
    assert_equal(E(1, 2, 2, 3), np.array([[0, 0, 0], [0, 0, 1]]))

    n = 3
    n2 = n**2
    nh = n * (n + 1) //2
    assert_equal(K(n).shape, (n2, n2))
    assert_equal(Ms(n).shape, (n2, n2))
    assert_equal(Md(n).shape, (n2, n2))
    assert_equal(Dup(n).shape, (n2, nh))
    assert_equal(L(n).shape, (nh, n2))

    # check K is vec of transpose
    assert_equal(K(n).dot(vec(xsf)), vec(xsf.T))

    # check Ms is vec of symmetrized matrix
    vs = vec(xs) / 2
    assert_equal(Ms(n).dot(vec(xsf)), vs)
    assert_equal(vec((xsf + xsf.T) / 2), vs)

    # check L
    assert_equal(L(n).dot(vec(xs)), vech(xs))

    # check Md is vec of diagonalized matrix (off-diagonals set to zero)
    # Md_0 produces same result as Md, different way of computing it
    md = vec(dg(xs))
    assert_equal(Md_0(n).dot(vec(xs)), md)
    assert_equal(Md(n).dot(vec(xs)), md)

    # Dup
    dd = Dup(n)
    assert_equal(dd.dot(vech(xs)), vec(xs))
    assert_equal(dd.dot(L(n).dot(vec(xs))), vec(xs))
    dinv = np.linalg.pinv(dd)
    assert_allclose(dinv.dot(vec(xs)), vech(xs), rtol=1e-14)
    assert_allclose(L(n).dot(Ms(n)), dinv, rtol=1e-14)


def test_cov_corr():
    # example from Steiger and Hakstian 1982
    # I don't know whether they use rounded intermediate results in calculations

    # standardized values, z-scores rounded as in table
    z0 = np.array([[-0.18, -0.835, -0.702],
                   [-0.465, 0.076, -0.467],
                   [-0.514, -0.781, -0.744],
                   [0.006, 0.009, -0.178],
                   [-0.559, -0.377, -0.539],
                   [-0.476, -0.511, -0.574],
                   [-0.252, 1.257, 0.240],
                   [-0.596, -0.615, -0.67],
                   [-0.534, -0.715, -0.698],
                   [3.360, 1.050, 1.830],
                   [-0.348, -0.579, -0.526],
                   [-0.057, 0.74, 0.278],
                   [-0.287, -0.419, 0.392],
                   [-0.240, 0.657, 0.988],
                   [0.738, 0.697, 0.614],
                   [-0.402, -0.68, -0.456],
                   [-0.572, -0.784, -0.676],
                   [-0.086, -0.457, -0.514],
                   [-0.606, -0.853, -0.678],
                   [2.070, 3.118, 3.081]])

    # original data
    x = np.array([[1.365, 0.182, 0.244],
                 [0.648, 1.520, 0.689],
                 [0.524, 0.261, 0.165],
                 [1.832, 1.421, 1.235],
                 [0.411, 0.854, 0.552],
                 [0.620, 0.657, 0.487],
                 [1.183, 3.254, 2.026],
                 [0.318, 0.505, 0.305],
                 [0.475, 0.357, 0.252],
                 [10.263, 2.950, 5.034],
                 [0.942, 0.558, 0.577],
                 [1.673, 2.494, 2.099],
                 [1.095, 0.793, 2.314],
                 [1.213, 2.373, 3.441],
                 [3.671, 2.432, 2.734],
                 [0.807, 0.409, 0.710],
                 [0.379, 0.257, 0.293],
                 [1.601, 0.736, 0.599],
                 [0.292, 0.155, 0.289],
                 [7.020, 5.987, 7.403]])

    z = (x - x.mean(0)) / x.std(0, ddof=1)  # not rounded as in z0
    ct.mom4(z0)

    def ztest_correlated(diff, var0, var1, covar, nobs):
        std = np.sqrt((var0 + var1 - 2 * covar) / nobs)
        statistic = diff / std
        from scipy import stats
        pvalue = stats.norm.sf(np.abs(statistic)) * 2
        return statistic, pvalue

    corr = np.corrcoef(z, rowvar=0)
    #corr = z0.dot(z0) / len(z0)
    # constrained correlation under corr_r[0, 1] = corr_r[0, 2]
    # estimate: replace by simple average (OLS)
    corr_r = corr.copy()
    corr_r[0, 1] = (corr[0, 1] + corr[0, 2]) / 2
    corr_r[1, 0] = corr_r[0, 2] = corr_r[2,0] = corr_r[0, 1]
    diff = corr[0, 1] - corr[0, 2]

    # two cases under normality and with empirical mom4
    # using functions for individual correlation coefficients

    # without normality assumption for 4th moments
    m4 = ct.mom4(z, ddof=1).reshape(3,3,3,3)

    i, j, k, h = 0, 1, 0, 1
    g1 = cov_corr_coef(i, j, k, h, corr_r, m4)
    assert_allclose(g1, 0.20200121658014503, rtol=1e-8)
    res = 0.2020
    assert_allclose(g1, res, atol=5e-4)

    i, j, k, h = 0, 2, 0, 2;
    g2 = cov_corr_coef(i, j, k, h, corr_r, m4)
    0.10611046338328745
    res = 0.1062
    assert_allclose(g2, res, atol=5e-4)

    i, j, k, h = 0, 1, 0, 2;
    g3 = cov_corr_coef(i, j, k, h, corr_r, m4)
    0.048605932563082987
    res = 0.0486
    assert_allclose(g3, res, atol=5e-4)

    i, j, k, h = 1, 2, 1, 2;
    g4 = cov_corr_coef(i, j, k, h, corr_r, m4)
    0.051162679039107317

    var0, var1, covar, nobs = g1, g2, g3, 20
    zt, pv = ztest_correlated(diff, var0, var1, covar, nobs)
    (-1.2978383834464489, 0.19434287689430729)
    zt1 = -1.25   # difference is most likely rounding error, g3 differs a bit

    # with normality assumption for 4th moments

    i, j, k, h = 0, 1, 0, 1
    g1 = ct.cov_corr_coef_normal(i, j, k, h, corr_r)
    0.17012301158351617
    res = 0.1701

    i, j, k, h = 0, 2, 0, 2
    g2 = ct.cov_corr_coef_normal(i, j, k, h, corr_r)
    0.17012301158351617
    res = 0.1701
    i, j, k, h = 0, 1, 0, 2
    g3 = ct.cov_corr_coef_normal(i, j, k, h, corr_r)
    0.13832670991613805
    res = 0.1383

    # normal theory variance of correlation coefficient is
    cr0 = (1-corr_r**2)**2
    assert_allclose(cr0[1,0], g1, rtol=1e-14)
    assert_allclose(cr0[1,0], g2, rtol=1e-14)

    cr0low = ct.veclow((1-corr**2)**2)
    crn = [ct.cov_corr_coef_normal(0, 1, 0, 1, corr),
           ct.cov_corr_coef_normal(0, 2, 0, 2, corr),
           ct.cov_corr_coef_normal(1, 2, 1, 2, corr)]
    assert_allclose(cr0low, crn, atol=1e-14)

    var0, var1, covar, nobs = g1, g2, g3, 20
    ztest_correlated(diff, var0, var1, covar, nobs)
    (-2.3642128782748038, 0.018068426888646225)
    zt1 = -2.30

    # working with indices and masks, e.g. select lower from vec
    kv = 3
    order = 'F'
    ravel_idx_mom4 = np.mgrid[:kv, :kv, :kv, :kv].reshape((4, -1), order=order).T
    mg2 = np.mgrid[:kv, :kv].reshape((2, -1), order=order).T
    mask_low = (np.diff(mg2, 1) < 0).squeeze()
    idx2_dict = dict((tuple(rii.tolist()), ii) for ii, rii in enumerate(mg2))

    covcorr = ct._cov_corr(corr_r, ct._cov_cov(corr_r, 20)*20, 20) * 20

    # extract and individual covariance of correlation coefficients
    covcorr[idx2_dict[(1, 2)], idx2_dict[(1, 2)]]


    # add index or labels to DataFrame
    import pandas as pd
    mdlabels = list(map(tuple, mg2[mask_low]))  #convert to list of tuples
    pd.DataFrame(covcorr[mask_low][:, mask_low], index=mdlabels, columns=mdlabels)
    """
              (0, 1)    (0, 2)    (1, 2)
    (0, 1)  0.170123  0.138327  0.028239
    (0, 2)  0.138327  0.170123  0.028239
    (1, 2)  0.028239  0.028239  0.024656
    """

    cre = ct._cov_corr_elliptical_vech(corr_r, 1, kurt=0)
    dd = Dup(3)
    di = np.linalg.pinv(dd)
    #(di.dot(cre).dot(di.T))[np.array([[1,2,4]]).T, np.array([[1,2,4]])] / covcorr[mask_low][:, mask_low]
    #print(cre[np.array([[1,2,4]]).T, np.array([[1,2,4]])] / covcorr[mask_low][:, mask_low])
    # Note create index arrays

    # selector masks vor vec, vech and veclow
    mask_vech = (np.diff(mg2, 1) <= 0).squeeze()
    mg2vech = mg2[mask_vech]
    mask_vech2veclow = (np.diff(mg2vech, 1) < 0).squeeze()
    cr1 = cre[np.ix_(mask_vech2veclow, mask_vech2veclow)]
    cr2 = di.dot(ct._cov_corr(corr_r, ct._cov_cov(corr_r, 1), 1)).dot(di.T)[np.ix_(mask_vech2veclow, mask_vech2veclow)]
    vr1 = ((1 -vech(corr_r)**2)**2)[mask_vech2veclow]
    assert_allclose(cr1.diagonal(), vr1, rtol=1e-14)
    # same as
    # Note we need to set nobs=1 in call to _cov_cov  Bug or Definition?
    cr3 = ct._cov_corr(corr_r, ct._cov_cov(corr_r, 1), 1)[np.ix_(mask_low, mask_low)]
    # same as
    cr4 = ct._cov_corr_elliptical_vech(corr_r, 1, kurt=0, drop=True)

    n = 3
    n_low = n * (n - 1) / 2
    assert_equal(cr1.shape, (n_low, n_low))
    assert_allclose(cr2, cr1, rtol=1e-14)
    assert_allclose(cr3, cr1, rtol=1e-14)
    assert_allclose(cr4, cr1, rtol=1e-14)

