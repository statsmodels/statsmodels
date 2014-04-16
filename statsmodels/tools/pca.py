# -*- coding: utf-8 -*-
"""Principal Component Analysis


Created on Tue Sep 29 20:11:23 2009
Author: josef-pktd

TODO : add class for better reuse of results
"""

import numpy as np


class PCA(object):
    """
    Principal Components Analysis

    Parameters
    ----------
    X : array-like
        2d array-like variable with variables in columns
    corr : bool
        If True, perform PCA on the correlation matrix. If False, use the
        covariance matrix.
    normalize : bool, optional
        False - Principal components are normed to 1
        True - Principal components are normed to the associated eigenvalues
    demean : bool, optional
        Whether or not to demean X.
    use_svd : bool
        If True, compute via the singular value decomposition ``np.linalg.svd``
        If False, compute via the eigenvalues and eigenvectors directly
        using ``np.linalg.eig``.

    Attributes
    ----------
    """
    def __init__(self, X, corr=True, normalize=False, demean=False,
                 use_svd=True):

        (proj, factors,
         evals, evecs) = pca(X, None, normalize, demean, corr, use_svd)

        self.projections = proj
        self.factors = factors
        self.components = evals
        self.loadings = evecs


def _pca_eig(x, corr):
    if corr:
        Y = np.corrcoef(x, rowvar=0, ddof=1)
    else:
        Y = np.cov(x, rowvar=0, ddof=1)

    # Compute eigenvalues and sort into descending order
    evals, evecs = np.linalg.eig(Y)
    indices = np.argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:, indices]
    evals = evals[indices]

    return evals, evecs


def _pca_svd(x, corr):
    if corr:
        # standardize
        Y = x/np.std(x, 0)
        ddof = 0
    else:
        Y = x
        ddof = 1

    U, s, v = np.linalg.svd(Y)
    evecs = v.T
    evals = s**2/(x.shape[0] - ddof)

    return evals, evecs


def pca(data, keepdim=None, normalize=False, demean=True, corr=True,
        use_svd=True):
    """
    Function for principal components analysis

    Parameters
    ----------
    data : ndarray, 2d
        data with observations by rows and variables in columns
    keepdim : integer
        Number of eigenvectors to keep. If keepdim is None, then all
        eigenvectors are included
    normalize : bool
        If True, then eigenvectors are normalized by sqrt of eigenvalues
    demean : boolean
        If true, then the column mean is subtracted from the data.
    corr : bool
        If True, perform PCA on correlation matrix. If False, perform PCA
        on covariance matrix.
    use_svd : bool
        If True, uses np.linalg.svd. If False, uses np.linalg.eig.

    Returns
    -------
    xreduced : ndarray, 2d, (nobs, nvars)
        projection of the data x on the kept eigenvectors
    factors : ndarray, 2d, (nobs, nfactors)
        factor matrix, given by np.dot(x, evecs)
    evals : ndarray, 2d, (nobs, nfactors)
        eigenvalues
    evecs : ndarray, 2d, (nobs, nfactors)
        eigenvectors, normalized if normalize is true

    """
    x = np.array(data, copy=True)
    #make copy so original doesn't change, maybe not necessary anymore
    if demean:
        m = x.mean(0)
    else:
        m = np.zeros(x.shape[1])
    x -= m

    if use_svd:
        evals, evecs = _pca_svd(x, corr)
    else:
        evals, evecs = _pca_eig(x, corr)

    if keepdim is not None:
        if keepdim == 0:
            raise ValueError("keepdim of 0 not supported")
        elif keepdim > x.shape[1]:
            raise ValueError("keepdim is larger than the number of variables")
        evecs = evecs[:, :keepdim]
        evals = evals[:keepdim]

    if normalize:
        #for i in range(shape(evecs)[1]):
        #    evecs[:,i] / linalg.norm(evecs[:,i]) * sqrt(evals[i])
        evecs = evecs/np.sqrt(evals)  # np.sqrt(np.dot(evecs.T, evecs) * evals)

    # get factor matrix
    factors = np.dot(x, evecs)
    # get original data from reduced number of components
    xreduced = np.dot(factors, evecs.T) + m
    return xreduced, factors, evals, evecs


__all__ = ['pca', 'PCA']
