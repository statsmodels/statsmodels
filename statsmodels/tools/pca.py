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
    norm : bool, optional
        False - Principal components are normed to 1
        True - Principal components are normed to the associated eigenvalues
    use_svd : bool
        If True, compute via the singular value decomposition ``np.linalg.svd``
        If False, compute via the eigenvalues and eigenvectors directly
        using ``np.linalg.eig``.

    Attributes
    ----------
    """
    def __init__(self, X, corr=True, normalize=False, demean=False,
                 use_svd=True):
        if not use_svd:
            proj, fact, evals, evecs = pca(X, 0, normalize, demean, corr)
        else:
            if corr:
                ddof = 0
            else:
                ddof = 1
            demean = True
            proj, fact, evals, evecs = pcasvd(X, 0, normalize, demean, corr)

        self.projections = proj
        self.factors = fact
        self.components = evals
        self.loadings = evecs


def pca(data, keepdim=0, normalize=False, demean=True, corr=True):
    """
    Principal components with eigenvector decomposition

    Parameters
    ----------
    data : ndarray, 2d
        data with observations by rows and variables in columns
    keepdim : integer
        number of eigenvectors to keep
        if keepdim is zero, then all eigenvectors are included
    normalize : bool
        If True, then eigenvectors are normalized by sqrt of eigenvalues
    demean : boolean
        if true, then the column mean is subtracted from the data
    cov : bool
        If True, perform PCA on covariance matrix. If False, perform PCA
        on correlation.

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

    Notes
    -----

    See Also
    --------
    pcasvd : principal component analysis using svd
    """
    x = np.array(data, copy=True)
    #make copy so original doesn't change, maybe not necessary anymore
    if demean:
        m = x.mean(0)
    else:
        m = np.zeros(x.shape[1])
    x -= m

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

    if keepdim > 0 and keepdim < Y.shape[1]:
        evecs = evecs[:, :keepdim]
        evals = evals[:keepdim]

    if normalize:
        #for i in range(shape(evecs)[1]):
        #    evecs[:,i] / linalg.norm(evecs[:,i]) * sqrt(evals[i])
        evecs = evecs/np.sqrt(evals)  # np.sqrt(np.dot(evecs.T, evecs) * evals)

    # get factor matrix
    #x = np.dot(evecs.T, x.T)
    factors = np.dot(x, evecs)
    # get original data from reduced number of components
    #xreduced = np.dot(evecs.T, factors) + m
    #print x.shape, factors.shape, evecs.shape, m.shape
    xreduced = np.dot(factors, evecs.T) + m
    return xreduced, factors, evals, evecs


def pcasvd(data, keepdim=0, normalize=0, demean=True, corr=True):
    """
    Principal components computed by SVD

    Parameters
    ----------
    data : ndarray, 2d
        data with observations by rows and variables in columns
    keepdim : integer
        number of eigenvectors to keep
        if keepdim is zero, then all eigenvectors are included
    demean : boolean
        if true, then the column mean is subtracted from the data
    on : str {'corr', 'cov'}, optonal
        Perform PCA on correlation ('corr') or covariance ('cov'). Note
        that if on == 'corr', the data is standardized and `demean` is
        ignored.
    ddof : int
        The degrees of freedom correction for the correlation or covariance
        calculation. Result is normalized by (N - ddof) where N is the number
        of observations.


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

    See Also
    -------
    pca : principal component analysis using eigenvector decomposition

    Notes
    -----
    This doesn't have yet the normalize option of pca.
    """
    x = np.array(data, copy=True)
    nobs, nvars = x.shape
    #print nobs, nvars, keepdim
    if demean:
        m = x.mean(0)
    else:
        m = np.zeros(x.shape[1])
    x -= m

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

    if keepdim:
        evals = evals[:keepdim]
        evecs = evecs[:, :keepdim]

    factors = np.dot(x, evecs)  # princomps
    xreduced = np.dot(factors, evecs.T) + m

    return xreduced, factors, evals, evecs


__all__ = ['pca', 'pcasvd', 'PCA']
