# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 20:11:23 2009

Author: josef-pktd
"""

import numpy as np


def pca(data, keepdim=0, normalize=0, demean=True):
    '''principal components with eigenvector decomposition
    similar to princomp in matlab
    '''
    x = np.array(data)
    #make copy so original doesn't change, maybe not necessary anymore
    if demean:
        m = x.mean(0)
    else:
        m = np.zeros(x.shape[1])
    x -= m

    # Covariance matrix
    xcov = np.cov(x, rowvar=0)

    # Compute eigenvalues and sort into descending order
    evals, evecs = np.linalg.eig(xcov)
    indices = np.argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:,indices]
    evals = evals[indices]

    if keepdim > 0 and keepdim < x.shape[1]:
        evecs = evecs[:,:keepdim]
        evals = evals[:keepdim]

    if normalize:
        #for i in range(shape(evecs)[1]):
        #    evecs[:,i] / linalg.norm(evecs[:,i]) * sqrt(evals[i])
        evecs = evecs/np.sqrt(evals) #np.sqrt(np.dot(evecs.T, evecs) * evals)

    # get factor matrix
    #x = np.dot(evecs.T, x.T)
    factors = np.dot(x, evecs)
    # get original data from reduced number of components
    #xreduced = np.dot(evecs.T, factors) + m
    print x.shape, factors.shape, evecs.shape, m.shape
    xreduced = np.dot(factors, evecs.T) + m
    return xreduced, factors, evals, evecs



def pcasvd(data, keepdim=0, demean=True):
    '''principal components with svd
    '''
    nobs, nvars = data.shape
    print nobs, nvars, keepdim
    x = np.array(data)
    #make copy so original doesn't change
    if demean:
        m = x.mean(0)
    else:
        m = 0
##    if keepdim == 0:
##        keepdim = nvars
##        "print reassigning keepdim to max", keepdim
    x -= m
    U, s, v = np.linalg.svd(x.T, full_matrices=1)
    factors = np.dot(U.T, x.T).T #princomps
    if keepdim:
        xreduced = np.dot(factors[:,:keepdim], U[:,:keepdim].T) + m
    else:
        xreduced = data
        keepdim = nvars
        "print reassigning keepdim to max", keepdim

    # s = evals, U = evecs
    # no idea why denominator for s is with minus 1
    evals = s**2/(x.shape[0]-1)
    print keepdim
    return xreduced, factors[:,:keepdim], evals[:keepdim], U[:,:keepdim] #, v
