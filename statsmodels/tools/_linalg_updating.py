# -*- coding: utf-8 -*-
"""Helper functions for updating inverse matrices

This functions are intended mainly as internal helper functions.
API and options might change depending on internal needs.

Created on Sat Nov 16 08:48:05 2019

Author: Josef Perktold
License: BSD-3

"""

import numpy as np


def _split4(mat, k0, k1=None):
    """split or partition 2-dim array into 4 blocks.

    Parameters
    ----------
    mat : array_like that allows double indexing, 2-dim
        2-dimensional array to be partitioned
    k0 : int
        number of rows for upper blocks
    k1 : None or int
        number of columns in first block.
        If k1 is None, then it is set to k0 so the first block is square

    Returns
    -------
    z11, z12, z21, z22 : array_like
        4 blocks of the partitioned matrix

    """
    if k1 is None:
        k1 = k0

    z11 = mat[:k0, :k1]
    z12 = mat[:k0, k1:]
    z21 = mat[k0:, :k1]
    z22 = mat[k0:, k1:]
    return z11, z12, z21, z22


def _block4(z11, z12, z21, z22):
    """convenience function for partitioned matrix with 2 by 2 blocks

        If a value is scalar, then it will be used as a fill value for the
        matrix of matching shape. If the shape cannot be determined, then an
        IndexError is raised.

    Parameters
    ----------
    z11, z12, z21, z22 : array_like, 2-dim or scalar
        4 blocks of the partitioned matrix with matching shape.
        Blocks need to be 2-dimensional or scalar.
        If a value is scalar, then it will be used as a fill value for the
        matrix of matching shape. If the shape cannot be determined, then an
        IndexError is raised.

    Returns
    -------
    mat : ndarray
        Partitioned matrix given the 4 blocks.
        ``dtype`` of matrix is determined by `np.block`

    Raises
    ------
    `IndexError: tuple index out of range` if there are too many scalars so
    that the shape of submatrices cannot be determined

    See Also
    --------
    numpy.block

    """
    z11, z12, z21, z22 = map(np.array, [z11, z12, z21, z22])
    # the following will raise IndexError if blocks to determine the shapes
    #   are not 2-dimensional.
    if z11.shape == ():
        z11_ = z11
        shape_sub = z12.shape[0], z21.shape[1]
        z11 = np.full(shape_sub, z11_)
    if z12.shape == ():
        z12_ = z12
        shape_sub = z11.shape[0], z22.shape[1]
        z12 = np.full(shape_sub, z12_)
    if z21.shape == ():
        z21_ = z21
        shape_sub = z22.shape[0], z11.shape[1]
        z21 = np.full(shape_sub, z21_)
    if z22.shape == ():
        z22_ = z22
        shape_sub = z21.shape[0], z12.shape[1]
        z22 = np.full(shape_sub, z22_)

    assert z11.shape[0] == z12.shape[0]
    assert z21.shape[0] == z22.shape[0]
    assert z11.shape[1] == z21.shape[1]
    assert z12.shape[1] == z22.shape[1]

    mat = np.block([[z11, z12], [z21, z22]])
    return mat


def split_matrix_idx(mat, idx):
    """remove rows and columns from square matrix

    Parameters
    ----------
    mat : array_like
        2 dim array-like object that allows double indexing
        pandas DataFrame does not work
    idx : int or list of ints
        index or indices of rows and columns to remove.
        If idx is a list of integers, then multiple rows and columns
        are removed.

    Returns
    -------
    mat_noti : array_like
        submatrix without the deleted rows and columns
    row_i, col_i : array_like
        rows and columns removed from the matrix

    Notes
    -----
    This is intended as an internal helper function.

    """
    k_vars = mat.shape[1]
    if not hasattr(idx, '__iter__'):
        idx = [idx]

    # handle negative indices
    idx = [k_vars + i if i < 0 else i for i in idx]
    if any(i < 0 for i in idx):
        raise IndexError("index `idx` out of range")

    idx_noti = np.array([i for i in range(k_vars) if i not in idx])
    # numpy style fancy indexing, does not work with pandas DataFrames
    # requires rectangular indexing
    mat_noti = mat[idx_noti[:, None], idx_noti]
    row_i = mat[idx]
    col_i = mat[:, idx]
    return mat_noti, row_i, col_i


def update_inv(matinv_old, u_mat, v_mat):
    """ This updates a matrix inverse for a rank k addition [A + U V]^{-1}

    For subtraction or down dating use `-umat`.


    Notes
    -----
    This implements the updating formula for `Bnew` in Hager (1998) p. 225
    [Bnew] = B-l-B-lU[I+VB-'U]->VB-l.

    References
    ----------
    William W. Hager (1998) Updating the Inverse of a Matrix.
    SIAM Review, Vol. 31, No. 2 (Jun., 1989), pp. 221-239

    """
    bi = matinv_old
    eye = np.eye(u_mat.shape[1])
    vbi = v_mat @ bi
    mat_inv_new = bi - bi @ u_mat @ np.linalg.inv(eye + vbi @ u_mat) @ vbi
    return mat_inv_new


def _get_update_uv(row_diff, idx, zero_diagonal=True):
    """find u and v for updating matrix inverse by adjusting row and column

    The `u3.T.dot(v3)` has pm in the row and the column corresponding to `idx`
    and zero otherwise.
    The value in the diagonal element (idx, idx) depends on the `zero_diagonal`
    options.

    zero_diagonal = False  # used in adding observations
    zero_diagonal = 0    # this fully adjusts for the row_diff diag element
    otherwise it depends it will adjust to final diagonal element by a
    different amount than the full amount.

    I'm not sure about the details, in OLS this corresponds to adjustments
    to the total sum of squares or the variance of the variable.
    If tss is constant, then zero_diagonal=True applies.
    otherwise zero_diagonal specified a correction amount to full adjustment.
    This can be used if the initial diagonal element, was arbitrarily set, e.g.
    to 1. If zero_diagonal is False, then zero_diagonal is set to 1.
    If initial product matrix is initialized at the variance, then
    zero_diagonal should be True.

    The definition and API of zero_diagonal might still change.



    """

    k = idx
    pm0 = np.atleast_2d(row_diff)
    indic = np.zeros((1, pm0.shape[1]))
    indic[0, k] = 1
    # v3 = np.concatenate((pm[k:k + 1].copy(), indic), axis=0)
    v3 = np.concatenate((pm0.copy(), indic), axis=0)

    u3 = v3[::-1, :].copy()
    v3[0, k] = 0
    if zero_diagonal is True:
        u3[1, k] = 0
    else:
        if zero_diagonal is False:
            # we assume that a new line has been added with 1 on diagonal
            base = 1
        else:
            # assume numeric
            base = zero_diagonal
        u3[1, k] -= base

    # u3.T.dot(v3) for checking
    return u3, v3


def update_symm_inv(symm_inv, observations=None, variable=None, var_idx=-1,
                    add=True):
    """update an symmetric inverse matrix

    This updates $A^{-1}$ to $[A + V.T V]^{-1}$

    Parameters
    ----------
    symm_inv : ndarray
        initial matrix to be updated
    observations : None or ndarray
        New observations that are added to the symmetric product matrix.
        This can contain one or several observations.
    variable : None or ndarray
        Variables, i.e. columns and matrices to be added or dropped.
        This needs to provide the values of the original matrix, not the
        inverse matrix. Note, ``symm_inv`` does not change shape.
    add : boolean
        If add is True, then observations or variables are added to the initial
        matrix. Otherwise, they will be dropped

    Returns
    -------
    symm_inv_new : ndarray
        updated symmetric inverse matrix
        TODO: currently same shape as symm_inv,
        maybe change shape when adding or dropping a variable

    Notes
    -----
    Warning: This does not handle division by nobs for covariance matrices.
    If observations are added or dropped, then the user needs to adjust for
    the change in the number of observations when this is used for covariance
    matrices.

    This currently delegates to the generic rank k updating function. Adding or
    dropping columns is based on creating a corresponding `V` matrix which is
    only available for adding or dropping a single variable.

    TODO: check repeated or iterated calls to this function.

    If a variable is dropped, then symm_inv_new does still have the same shape
    as symm_inv but with a block diagonal structure.

    References
    ----------
    William W. Hager (1998) Updating the Inverse of a Matrix.
    SIAM Review, Vol. 31, No. 2 (Jun., 1989), pp. 221-239

    "hardmath" for converting dropping a variable to standard rank k update,
    answer in
    https://math.stackexchange.com/questions/1248220/find-the-inverse-of-a-submatrix-of-a-given-matrix

    """
    if variable is not None:
        k_vars = symm_inv.shape[1]
        if var_idx < 0:
            var_idx = k_vars + var_idx

        pm0 = np.atleast_2d(variable)

        indic = np.zeros((1, k_vars))
        indic[0, var_idx] = 1
        v = np.concatenate((pm0.copy(), indic), axis=0)
        v[0, var_idx] = 0
        u = v[::-1, :]
        # print(u.T.dot(v))
    if observations is not None:
        u = v = np.atleast_2d(observations)

    if add is not True:
        u = -u  # don't change inplace because `u` might not be a copy

    symm_inv_new = update_inv(symm_inv, u.T, v)
    return symm_inv_new


def replace_symm_inv(symm_inv, row_diff, var_idx):
    """update inverse matrix when one row and column changes in original matrix


    """
    u, v = _get_update_uv(row_diff, var_idx, zero_diagonal=0)
    symm_inv_new = update_inv(symm_inv, u.T, v)
    return symm_inv_new


def _symm_inv_addvar(cov_inv, cov_new, diag_new):
    """update symmetric inverse matrix by adding new row and column at end

    The inverse matrix is extended by a new column and a new row at the last
    position.

    This function might be deleted as redundant
    """

    col_n = np.atleast_2d(cov_new).T
    diag_n = diag_new
    alpha = cov_inv @ col_n
    gamma = diag_n - col_n.T @ alpha
    gamma = np.squeeze(gamma)
    inv_new = np.block([[gamma * cov_inv + alpha @ alpha.T, -alpha],
                        [-alpha.T, np.array([[1.]])]]) / gamma
    return inv_new


def _inv_replace_var(symm_inv, cov_new, cov_old, idx):
    """

    doesn't seem to work correctly for diagonal element of replaced column

    found on internet as matlab function, but I don't understand algorithm
    """

    coli = symm_inv[idx:idx+1, :]

    r = np.atleast_2d(cov_new - cov_old).T
    Astar = symm_inv - (symm_inv @  r  @ coli) / (1 + r.T @ coli.T)
    r[idx] = 0  # try this to fix diagnoal element
    A = Astar - ((Astar[:, idx:idx+1] @ r.T @ Astar) /
                 (1 + r.T @ Astar[:, idx:idx+1]))
    return A


def update_inv_addvar(symm_inv, variable=None, var_idx=-1, add=True):
    """add or remove variable from symmetric inverse matrix

    This returns the new inverse matrix after adding or removing a row and
    column, the inverse matrix changes shape in each call.

    Based on Emtiyaz, CS, UBC
    """

    mi0 = symm_inv  # shorthand
    k_old = symm_inv.shape[1]
    if add is True:
        k_new = k_old + 1
        mask_i = np.zeros(k_new, np.bool_)
        mask_i[var_idx] = True
        # new column of product matrix
        u1 = variable[~mask_i, None]  # cross product, covariance
        v2 = variable[mask_i]  # diagonal element, variance
        # label as if 2 by 2 block matrix

        proj = mi0 @ u1  # (x'x)^{-1} @ x' y
        # 1 / residual var of new,  scalar (2-D?)
        mi22 = 1 / (v2 - u1.T @ proj)
        mi11 = mi0 + mi22 * proj @ proj.T  # update (x'x)^{-1}
        mi12 = -mi22 * proj

        # create new inv matrix with inserted row and col
        mi = np.zeros((k_new, k_new), dtype=symm_inv.dtype)
        mi[~mask_i[:, None] * ~mask_i] = mi11.ravel()
        mi[mask_i, ~mask_i] = mi12.squeeze()
        mi[~mask_i, mask_i] = mi12.squeeze()
        mi[mask_i, mask_i] = mi22
    else:
        # delete col and row
        k_new = k_old - 1
        mask_i = np.zeros(k_old, np.bool_)
        mask_i[var_idx] = True
        mi11 = mi0[~mask_i[:, None] * ~mask_i].reshape(k_new, k_new)
        mi22 = mi0[mask_i, mask_i]
        mi12 = -mi0[mask_i, ~mask_i][:, None]
        u2 = mi12 / mi22
        mi = mi11 - mi22 * u2 @ u2.T

    return mi
