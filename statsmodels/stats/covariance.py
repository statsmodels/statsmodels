"""

Author: Josef Perktold
License: BSD-3
"""
from __future__ import division

import numpy as np
from scipy import integrate, stats

pi2 = np.pi**2
pi2i = 1. / pi2


def _term_integrate(rho):
    # needs other terms for spearman rho var calculation
    # TODO: streamline calculation and save to linear interpolation, maybe
    sin, cos = np.sin, np.cos
    f1 = lambda t, x: np.arcsin(sin(x) / (1 + 2 * cos(2 * x)))
    f2 = lambda t, x: np.arcsin(sin(2 * x) /
                                np.sqrt(1 + 2 * cos(2 * x)))
    f3 = lambda t, x: np.arcsin(sin(2 * x) /
                                (2 * np.sqrt(cos(2 * x))))
    f4 = lambda t, x: np.arcsin(( 3 * sin(x) - sin(3 * x)) /
                                (4 * cos(2 * x)))

    fact = pi2i * (f1(None, rho) +
                   2 * pi2i * f2(None, rho) +
                   f3(None, rho) +
                   0.5 * f4(None, rho))

    return fact


def transform_corr_normal(corr, method, return_var=False, possdef=True):
    """transform correlation matrix to be consistent at normal distribution

    Parameters
    ----------
    corr : array_like
        correlation matrix, either Pearson, Gaussian-rank, Spearman, Kendall
        or quadrant correlation matrix
    method : string
        type of covariance matrix
        supported types are 'pearson', 'gauss_rank', 'kendal', 'spearman' and
        'quadrant'
    return_var : bool
        If true, then the asymptotic variance of the normalized correlation
        is also returned. The variance of the spearman correlation requires
        numerical integration which is calculated with scipy's odeint.
    possdef : not implemented yet
        Check whether resulting correlation matrix for positive semidefinite
        and return a positive semidefinite approximation if not.

    Returns
    -------
        corr : ndarray
            correlation matrix, consistent with correlation for a multivariate
            normal distribution
        var : ndarray (optional)
            asymptotic variance of the correlation if requested by `return_var`.

    Notes
    -----
    Pearson and Gaussian-rank correlation are consistent at the normal
    distribution and will be returned without changes.

    The other correlation matrices are not guaranteed to be positive
    semidefinite in small sample after conversion, even if the underlying
    untransformed correlation matrix is positive (semi)definite. Croux and
    ... mention that nobs / k_vars should be larger than 3 for kendall and
    larger than 2 for spearman.

    """
    method = method.lower()
    rho = np.asarray(corr)

    if method in ['pearson', 'gauss_rank']:
        corr_n = corr
        if return_var:
            var = (1 - rho**2)**2


    elif method.startswith('kendal'):
        corr_n = np.sin(np.pi / 2 * corr)
        if return_var:
            var = (1 - rho**2) * np.pi**2 * (
                  1./9 - 4 / np.pi**2 * np.arcsin(rho / 2)**2)


    elif method == 'quadrant':
        corr_n = np.sin(np.pi / 2 * corr)
        if return_var:
            var = (1 - rho**2) * (np.pi**2 / 4 - np.arcsin(rho)**2)

    elif method.startswith('spearman'):
        corr_n =  2 * np.sin(np.pi / 6 * corr)
        # not clear which rho is in formula, should be normalized rho,
        # but original corr coefficient seems to match results in articles
        #rho = corr_n
        if return_var:
            # odeint only works if grid of rho is large, i.e. many points
            # e.g. rho = np.linspace(0, 1, 101)
            rho = np.atleast_1d(rho)
            idx = np.argsort(rho)
            rhos = rho[idx]
            rhos = np.concatenate(([0], rhos))
            t = np.arcsin(rhos / 2)
            # drop np namespace here
            sin, cos = np.sin, np.cos
            var = (1 - rho**2 / 4) * pi2 / 9 # leading factor
            f1 = lambda t, x: np.arcsin(sin(x) / (1 + 2 * cos(2 * x)))
            f2 = lambda t, x: np.arcsin(sin(2 * x) /
                                        np.sqrt(1 + 2 * cos(2 * x)))
            f3 = lambda t, x: np.arcsin(sin(2 * x) /
                                        (2 * np.sqrt(cos(2 * x))))
            f4 = lambda t, x: np.arcsin(( 3 * sin(x) - sin(3 * x)) /
                                        (4 * cos(2 * x)))
            # todo check dimension, odeint return column (n, 1) array
            hmax = 1e-1
            rf1 = integrate.odeint(f1 , 0, t=t, hmax=hmax).squeeze()
            rf2 = integrate.odeint(f2 , 0, t=t, hmax=hmax).squeeze()
            rf3 = integrate.odeint(f3 , 0, t=t, hmax=hmax).squeeze()
            rf4 = integrate.odeint(f4 , 0, t=t, hmax=hmax).squeeze()
            fact = 1 + 144 * (-9 / 4. * pi2i * np.arcsin(rhos / 2)**2 +
                              pi2i * rf1 +
                              2 * pi2i * rf2 + pi2i * rf3 +
                              0.5 * pi2i * rf4)
            #fact = 1 - 9 / 4 * pi2i * np.arcsin(rhos / 2)**2
            fact2 = np.zeros_like(var) * np.nan
            fact2[idx] = fact[1:]
            var *= fact2
    else:
        raise ValueError('method not recognized')

    if return_var:
        return corr_n, var
    else:
        return corr_n



def corr_normal_scores(data):
    """Gaussian rank (normal scores) correlation

    Status: unverified, subject to change

    Parameters
    ----------
    data : array_like
        2-D data with observations in rows and variables in columns

    Returns
    -------
    corr : ndarray
        correlation matrix

    """
    # TODO: a full version should be same as scipy spearmanr
    # needs verification for the p-value calculation
    x = np.asarray(data)
    nobs, k_vars = x.shape
    axisout = 0
    ar = np.apply_along_axis(stats.rankdata, axisout, x)
    ar = stats.norm.ppf(ar / (nobs + 1))
    corr = np.corrcoef(ar, rowvar=axisout)
    return corr

def corr_quadrant(data, transform=np.sign, normalize=False):
    """Quadrant correlation

    Status: unverified, subject to change

    Parameters
    ----------
    data : array_like
        2-D data with observations in rows and variables in columns

    Returns
    -------
    corr : ndarray
        correlation matrix

    """

    # try also with tanh transform, a starting corr for DetXXX
    # tanh produces a cov not a corr
    x = np.asarray(data)
    nobs, k_vars = x.shape
    med = np.median(x, 0)
    x_dm = transform(x - med)
    corr = x_dm.T.dot(x_dm) / nobs
    if normalize:
        std = np.sqrt(np.diag(corr))
        corr /= std
        corr /= std[:, None]
    return corr
