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
    on : str {'corr', 'cov'}, optonal
        Perform PCA on correlation ('corr') or covariance ('cov')
    norm : bool, optional
        False - Principal components are normed to 1
        True - Principal components are normed to the associated eigenvalues
    method : str {'eig', 'svd'}, optional
        'eig' - compute the eigenvalues and eigenvectors directly
        'svd' - compute via the singular value decomposition
    ddof : int
        The degrees of freedom correction for the correlation or covariance
        calculation. Result is normalized by (N - ddof) where N is the number
        of observations.

    Attributes
    ----------
    """
    def __init__(self, X, on='corr', norm=0, demean=False, method='eig',
                 ddof=1):
        if method == 'eig':
            proj, fact, evals, evecs = pca(X, 0, norm, demean, on, ddof)
        elif method == 'svd':
            if on == 'corr':
                ddof = 0
            else:
                ddof = 1
            demean = True
            proj, fact, evals, evecs = pcasvd(X, 0, norm, demean, on, ddof)

        self.projections = proj
        self.factors = fact
        self.components = evals
        self.loadings = evecs


def pca(data, keepdim=0, normalize=0, demean=True, on='cov', ddof=1):
    '''principal components with eigenvector decomposition
    similar to princomp in matlab

    Parameters
    ----------
    data : ndarray, 2d
        data with observations by rows and variables in columns
    keepdim : integer
        number of eigenvectors to keep
        if keepdim is zero, then all eigenvectors are included
    normalize : boolean
        if true, then eigenvectors are normalized by sqrt of eigenvalues
    demean : boolean
        if true, then the column mean is subtracted from the data
    on : str {'corr', 'cov'}, optonal
        Perform PCA on correlation ('corr') or covariance ('cov')

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

    '''
    x = np.asarray(data)
    #make copy so original doesn't change, maybe not necessary anymore
    if demean:
        m = x.mean(0)
    else:
        m = np.zeros(x.shape[1])
    x -= m

    if on == 'corr':
        Y = np.corrcoef(x, rowvar=0, ddof=ddof)
    elif on == 'cov':
        Y = np.cov(x, rowvar=0, ddof=ddof)

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


def pcasvd(data, keepdim=0, normalize=0, demean=True, on='cov', ddof=1):
    '''principal components with svd

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

    '''
    nobs, nvars = data.shape
    #print nobs, nvars, keepdim
    x = np.array(data)
    #make copy so original doesn't change
    if demean:
        m = x.mean(0)
    else:
        m = np.zeros(x.shape[1])
    x -= m

    if on == 'corr':  # center and standardize
        if not demean:  # do it anyway
            m = x.mean(0)
            Y = x - m
        else:
            Y = x
        Y /= np.std(x, 0)
    elif on == 'cov':
        Y = x

    U, s, v = np.linalg.svd(Y)
    factors = np.dot(U.T, Y).T  # princomps
    if keepdim:
        xreduced = np.dot(factors[:, :keepdim], U[:, :keepdim].T) + m
    else:
        xreduced = data
        keepdim = nvars
        "print reassigning keepdim to max", keepdim

    # s = evals, U = evecs
    # no idea why denominator for s is with minus 1
    evals = s**2/(x.shape[0]-ddof)
    #print keepdim
    return xreduced, factors[:, :keepdim], evals[:keepdim], U[:, :keepdim]
    # , v


__all__ = ['pca', 'pcasvd', 'PCA']
