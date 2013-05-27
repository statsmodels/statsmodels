# -*- coding: utf-8 -*-
"""

Created on Fri Aug 17 13:10:52 2012

Author: Josef Perktold
License: BSD-3
"""

import numpy as np

def clip_evals(x, value=0): #threshold=0, value=0):
    evals, evecs = np.linalg.eigh(x)
    clipped = np.any(evals < 0)
    x_new = np.dot(evecs * np.maximum(evals, value), evecs.T)
    return x_new, clipped


def corr_nearest(corr, threshold=1e-15, n_fact=100):
    '''
    Find the nearest correlation matrix that is positive semi-definite

    Parameters
    ----------
    corr : ndarray, (k,k)
        initial correlation matrix
    threshold : float
        clipping threshold for smallest eigen value, see Notes
    n_fact : int or float
        factor to determine the maximum number of iterations. The maximum
        number of iterations is the integer part of the number of columns in
        the correlation matrix times n_fact.

    Returns
    -------
    corr_new : ndarray, (optional)
        corrected correlation matrix

    Notes
    -----
    The smallest eigenvalue of the corrected correlation matrix is
    approximately equal to the ``threshold`` (smaller).
    If the threshold=0, then the smallest eigenvalue of the correlation matrix
    might be negative, but zero within a numerical error, for example in the
    range of -1e-16.

    Assumes input correlation matrix is symmetric.

    Stops after the first step if correlation matrix is already positive
    semi-definite.

    See Also
    --------
    corr_clipped
    cov_nearest


    '''
    k_vars = corr.shape[0]
    if k_vars != corr.shape[1]:
        raise ValueError("matrix is not square")

    diff = np.zeros(corr.shape)
    x_new = corr.copy()
    diag_idx = np.arange(k_vars)

    for ii in range(int(len(corr) * n_fact)):
        x_adj = x_new - diff
        x_psd, clipped = clip_evals(x_adj, value=threshold)
        if not clipped:
            x_new = x_psd
            break
        diff = x_psd - x_adj
        x_new = x_psd.copy()
        x_new[diag_idx, diag_idx] = 1
    else:
        import warnings
        warnings.warn('maximum iteration reached')

    return x_new

def corr_clipped(corr, threshold=1e-15):
    '''
    Find the nearest correlation matrix that is positive semi-definite

    Parameters
    ----------
    corr : ndarray, (k,k)
        initial correlation matrix
    threshold : float
        clipping threshold for smallest eigen value, see Notes

    Returns
    -------
    corr_new : ndarray, (optional)
        corrected correlation matrix


    Notes
    -----
    The smallest eigenvalue of the corrected correlation matrix is
    approximately equal to the ``threshold`` (smaller). In examples, the
    smallest eigenvalue can be by a factor of 10 smaller then the threshold,
    e.g. threshold 1e-8 can result in smallest eigenvalue in the range
    between 1e-9 and 1e-8.
    If the threshold=0, then the smallest eigenvalue of the correlation matrix
    might be negative, but zero within a numerical error, for example in the
    range of -1e-16.

    Assumes input correlation matrix is symmetric.

    Stops after the first step if correlation matrix is already positive
    semi-definite.

    See Also
    --------
    corr_nearest
    cov_nearest
        40 or more times faster in simple example with a bit larger
        approximation error


    '''
    x_new, clipped = clip_evals(corr, value=threshold)
    if not clipped:
        return corr

    #cov2corr
    x_std = np.sqrt(np.diag(x_new))
    x_new = x_new / x_std / x_std[:,None]
    return x_new

def cov_nearest(cov, method='clipped', threshold=1e-15, n_fact=100,
                return_all=False):

    '''
    Find the nearest covariance matrix that is postive (semi) definite

    This leaves the diagonal, i.e. the variance unchanged

    Parameters
    ----------
    cov : ndarray, (k,k)
        initial covariance matrix
    method : string
        if "clipped", then the faster but less accurate ``clipped_corr`` is used.
        if "nearest", then ``nearest_corr`` is used
    threshold : float
        clipping threshold for smallest eigen value, see Notes
    nfact : int or float
        factor to determine the maximum number of iterations in
        ``nearest_corr``. See its doc string
    return_all : bool
        if False (default), then only the covariance matrix is returned.
        If True, then correlation matrix and standard deviation are
        additionally returned.

    Returns
    -------
    cov_ : ndarray
        corrected covariance matrix
    corr_ : ndarray, (optional)
        corrected correlation matrix
    std_ : ndarray, (optional)
        standard deviation


    Notes
    -----
    This converts the covariance matrix to a correlation matrix. Then, finds
    finds the nearest correlation matrix that is positive semidefinite and
    converts it back to a covariance matrix using the initial standard
    deviation.

    The smallest eigenvalue of the intermediate correlation matrix is
    approximately equal to the ``threshold`` (smaller).
    If the threshold=0, then the smallest eigenvalue of the correlation matrix
    might be negative, but zero within a numerical error, for example in the
    range of -1e-16.

    Assumes input covariance matrix is symmetric.

    Stops after the first step if correlation matrix is already positive
    semi-definite.

    See Also
    --------
    corr_nearest
    corr_clipped

    '''

    from statsmodels.stats.moment_helpers import cov2corr, corr2cov
    cov_, std_ = cov2corr(cov, return_std=True)
    if method == 'clipped':
        corr_ = corr_clipped(cov_, threshold=threshold)
    elif method == 'nearest':
        corr_ = corr_nearest(cov_, threshold=threshold, n_fact=n_fact)

    cov_ = corr2cov(corr_, std_)

    if return_all:
        return cov_, corr_, std_
    else:
        return cov_

if __name__ == '__main__':
    pass
