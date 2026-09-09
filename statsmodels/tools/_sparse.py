# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 00:15:11 2016

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from scipy import sparse
import scipy.sparse.linalg as sparsela
import pandas as pd


def generate_sample1():
    xcat = np.repeat(np.arange(5), 10)

    df = pd.get_dummies(xcat)  #sparse=True) #sparse requires v 0.16

    exog = sparse.csc_matrix(df.values)
    beta = 1. / np.arange(1, 6)

    np.random.seed(999)
    y = exog.dot(beta) + 0.01 * np.random.randn(exog.shape[0])

    return y, exog


def generate_sample2():
    xcat1 = np.repeat(np.arange(5), 10)
    xcat2 = np.tile(np.arange(5), 10)

    df1 = pd.get_dummies(xcat1)  #sparse=True) #sparse requires v 0.16
    df2 = pd.get_dummies(xcat2)
    x = np.column_stack((np.asarray(df1)[:, 1:], np.asarray(df2)))
    exog = sparse.csc_matrix(x)   #.astype(np.int8))  int breaks in sparse
    beta = 1. / np.r_[np.arange(1, df1.shape[1]), np.arange(1, df2.shape[1] + 1)]

    np.random.seed(999)
    y = exog.dot(beta) + 0.01 * np.random.randn(exog.shape[0])

    return y, exog


from scipy import sparse

def dummy_sparse(groups, dtype=np.float64):
    '''create a sparse indicator from a group array with integer labels

    Parameters
    ----------
    groups: ndarray, int, 1d (nobs,)
        an array of group indicators for each observation. Group levels are assumed
        to be defined as consecutive integers, i.e. range(n_groups) where
        n_groups is the number of group levels.
    dtype : numpy dtype
        dtype of sparse matrix.
        We need float64 for some applications with sparse linear algebra. For other
        uses we can use an integer type which will use less memory.

    Returns
    -------
    indi : ndarray, int8, 2d (nobs, n_groups)
        an indicator array with one row per observation, that has 1 in the
        column of the group level for that observation

    Examples
    --------

    >>> g = np.array([0, 0, 2, 1, 1, 2, 0])
    >>> indi = dummy_sparse(g, dtype=np.int8)
    >>> indi
    <7x3 sparse matrix of type '<type 'numpy.int8'>'
        with 7 stored elements in Compressed Sparse Row format>
    >>> indi.todense()
    matrix([[1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]], dtype=int8)


    current behavior with missing groups
    >>> g = np.array([0, 0, 2, 0, 2, 0])
    >>> indi = dummy_sparse(g, dtype=np.int8)
    >>> indi.todense()
    matrix([[1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0]], dtype=int8)

    '''

    indptr = np.arange(len(groups)+1)
    data = np.ones(len(groups), dtype=dtype)
    indi = sparse.csr_matrix((data, groups, indptr))
    #indi = sparse.csr_matrix((data, groups, indptr)).tocsc()
    #TODO: construct csc directly

    return indi


class PartialingSparse(object):
    """class to condition an partial out a sparse matrix

    Parameters
    ----------
    x_conditioning : sparse matrix
        stored as attribute `xcond`
        TODO: check this version needs full rank matrix
        `lsqr` could gain a Ridge penalization implemented in scipy.
        `lu` could gain Ridge penalization implemented here.
        Using tiny Ridge penalization is close to using pinv.
    method : 'lu' or 'lsqr'
        sparse method to solf least squares problem. 'lu' stores
        the LU factorization and should be faster in repeated calls.
        'lsqr' does not store intermediate results.

    Notes
    -----
    This solves a least squares problem and has methods for the parameters or
    the projection. Those values are the same as params, fittedvalues and
    resid in OLS results.


    """

    def __init__(self, x_conditioning, method='lu'):
        self.xcond = x = x_conditioning
        self.method = method.lower()
        if self.method == 'lu':
            self.xtx_solve = sparsela.factorized(x.T.dot(x))
            # x.T.dot(x) should be csc for efficiency, warning if csr
        else:
            if self.method not in ['lsqr']:
                raise ValueError('method can only be lu or lsqr')


    def partial_params(self, x):
        """find least squares parameters

        This uses the method given in initialization.

        Parameters
        ----------
        x : ndarray, 1D or 2D

        Returns
        -------
        params : ndarray
            least squares parameters


        """
        if self.method == 'lu':
            # this fails in scipy < 0.14
            try:
                p = self.xtx_solve( self.xcond.T.dot(x))
            except SystemError:
                p =  np.column_stack([self.xtx_solve(xc) for
                                      xc in self.xcond.T.dot(x).T])
            return p
        else:
            return sparsela.lsqr(self.xcond, x)


    def partial_sparse(self, x):
        '''calculate projection of x on x_conditioning

        Note: x needs to be dense

        Parameters
        ----------
        x : ndarray, 1D or 2D

        Returns
        -------
        xhat : ndarray
            projection of x on xcond
        resid : ndarray
            orthogonal projection on null space



        TODO: this needs a pandas wrapper so we can create new DataFrames

        '''
        xhat = self.xcond.dot(self.partial_params(x))
        resid = x - xhat
        return xhat, resid


if __name__ == '__main__':
    y, x = generate_sample2()
    #print(x.todense())

    #x = x.todense()
    yhat_dense = np.squeeze(x.dot(np.linalg.pinv(x.todense()).dot(y).T))
    resid_dense = y - yhat_dense

    yh, resid = PartialingSparse(x).partial_sparse(y)

    print(yh - yhat_dense)
    print(resid - resid_dense)




