# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 09:31:36 2019

Author: Josef Perktold
License: BSD-3

"""

# flake8: noqa: E201

import numpy as np
import statsmodels.tools._linalg_updating as smlu

from numpy.testing import assert_array_equal, assert_allclose, assert_equal

# Testing helper functions for splitting and combining matrices
# -------------------------------------------------------------


def test_split_block():
    m = np.arange(4 * 4).reshape(4, 4)
    k = 3
    blocks = smlu._split4(m, k)
    assert_array_equal([(3, 3), (3, 1), (1, 3), (1, 1)],
                       [i.shape for i in blocks])

    res_blocks = (np.array([[0, 1,  2], # noqa
                            [4, 5,  6],
                            [8, 9, 10]]),
                  np.array([[3], [7], [11]]),
                  np.array([[12, 13, 14]]),
                  np.array([[15]]))

    assert_equal(blocks, res_blocks)
    m_roundtrip = smlu._block4(*blocks)
    assert_array_equal(m_roundtrip, m)

    # with empty blocks
    assert_array_equal([(4, 2), (4, 2), (0, 2), (0, 2)],
                       [i.shape for i in smlu._split4(m, 4, 2)])

    # non-square matrix
    m2 = np.arange(4 * 5).reshape(4, 5)
    assert_array_equal([(1, 3), (1, 2), (3, 3), (3, 2)],
                       [i.shape for i in smlu._split4(m2, 1, 3)])


def test_split_block_scalar():
    # verify scalar handling with filing
    m = np.arange(4 * 4).reshape(4, 4)

    blocks = smlu._split4(m, 3, 2)
    z11, z12, z21, z22 = blocks
    res_z12 = np.array([[ 2,  3],
                        [ 6,  7],
                        [10, 11]])
    assert_array_equal(z12, res_z12)
    assert_array_equal(z21, [[12, 13]])
    mb = smlu._block4(z11, 0, 0, z22)

    res_mat = np.array([[0, 1,  0,  0],
                        [4, 5,  0,  0],
                        [8, 9,  0,  0],
                        [0, 0, 14, 15]])
    assert_array_equal(mb, res_mat)


def test_split_matrix_idx():
    m = np.arange(4 * 4).reshape(4, 4)
    res = smlu.split_matrix_idx(m, 1)
    res2 = (np.array([[ 0,  2,  3],
                      [ 8, 10, 11],
                      [12, 14, 15]]),
            np.array([[4, 5, 6, 7]]),
            np.array([[1], [5], [9], [13]])
            )  # noqa
    assert_equal(res, res2)

    res = smlu.split_matrix_idx(m, [1, -1])
    res2 = (np.array([[0,  2],
                      [8, 10]]),
            np.array([[4, 5, 6, 7],
                      [12, 13, 14, 15]]),
            np.array([[ 1,  3],
                      [ 5,  7],
                      [ 9, 11],
                      [13, 15]]))
    assert_equal(res, res2)


# testing up- and down-dating functions
# -------------------------------------

# generate some data and test cases for matrix updating
np.random.seed(987123)
k_vars = 4
zz = np.random.randn(100, k_vars)
pm = zz.T.dot(zz)
pmi = np.linalg.inv(pm)

# sub matrices
pm_not3 = pm[:-1, :-1]
pmi_not3 = np.linalg.inv(pm_not3)

idx_not2 = np.arange(k_vars)
idx_not2 = idx_not2[idx_not2 != 2]
pm_not2 = pm[idx_not2[:, None], idx_not2]
pmi_not2 = np.linalg.inv(pm_not2)


def test_update_inv_obs():
    # check adding or removing observations

    nobsz = zz.shape[0]
    zz_sub0 = zz[:nobsz - 10]
    zz_sub1 = zz[nobsz - 10:]
    # updating with new observations
    pmis0 = np.linalg.inv(zz_sub0.T.dot(zz_sub0))
    res_pmi = smlu.update_inv(pmis0, zz_sub1.T, zz_sub1)
    assert_allclose(res_pmi, pmi, rtol=1e-13)

    # downdating, make either u or v negative
    res_pmis0 = smlu.update_inv(pmi, -zz_sub1.T, zz_sub1)
    assert_allclose(res_pmis0, pmis0, rtol=1e-13)
    res_pmis0 = smlu.update_inv(pmi, zz_sub1.T, -zz_sub1)
    assert_allclose(res_pmis0, pmis0, rtol=1e-13)

    # using wrapper function
    res_pmi = smlu.update_symm_inv(pmis0, observations=zz_sub1)
    assert_allclose(res_pmi, pmi, rtol=1e-13)


def test_update_inv_var():
    # check adding or removing one variable using update_symm_inv
    # Note, inverse matrix does not change shape, does not shrink or grow

    # remove last variable
    pmi_updated = smlu.update_symm_inv(pmi, variable=pm[-1], var_idx=-1,
                                       add=False)
    assert_allclose(pmi_updated[:-1, :-1], pmi_not3, rtol=1e-13)
    # round trip
    pmi_reversed = smlu.update_symm_inv(pmi_updated, variable=pm[-1],
                                        var_idx=-1, add=True)
    assert_allclose(pmi_reversed, pmi, rtol=1e-13)


def test_update_inv_addvar():
    # check adding and removing variables using _update_inv_addvar
    idx = 3
    pmi_not = pmi_not3
    res1 = smlu.update_inv_addvar(pmi_not, variable=pm[idx], var_idx=idx,
                                  add=True)
    assert_allclose(res1, pmi, rtol=1e-13)
    res2 = smlu.update_inv_addvar(pmi, variable=pm[idx], var_idx=idx,
                                  add=False)
    assert_allclose(res2, pmi_not, rtol=1e-13)

    idx = -1
    pmi_not = pmi_not3
    res1 = smlu.update_inv_addvar(pmi_not, variable=pm[idx], var_idx=idx,
                                  add=True)
    assert_allclose(res1, pmi, rtol=1e-13)
    res2 = smlu.update_inv_addvar(pmi, variable=pm[idx], var_idx=idx,
                                  add=False)
    assert_allclose(res2, pmi_not, rtol=1e-13)

    idx = 2
    pmi_not = pmi_not2
    res1 = smlu.update_inv_addvar(pmi_not, variable=pm[idx], var_idx=idx,
                                  add=True)
    assert_allclose(res1, pmi, rtol=1e-13)
    res2 = smlu.update_inv_addvar(pmi, variable=pm[idx], var_idx=idx,
                                  add=False)
    assert_allclose(res2, pmi_not, rtol=1e-13)


def test_replace_symm_inv():
    zz2 = zz.copy()
    idx = 1
    zz2[:, idx] = zz2[:, idx]**2
    pm2 = zz2.T.dot(zz2)
    pmi2 = np.linalg.inv(pm2)
    row_diff = pm[idx] - pm2[idx]
    res_pmi = smlu.replace_symm_inv(pmi2, row_diff, idx)
    assert_allclose(res_pmi, pmi, rtol=1e-13)


def junk():

    np.random.seed(987123)
    zz = np.random.randn(100, 4)
    pm = zz.T.dot(zz)

    m = np.arange(4 * 4).reshape(4, 4)
    blocks = smlu._split4(m, 1, 2)
    z11, z12, z21, z22 = blocks
    shapes = [i.shape for i in [z11, z12, z21, z22]]

    smlu._block4(z11, z12, z21, z22)
    smlu._block4(z11, z12, 0, z22)
