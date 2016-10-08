"""robust location, scatter and covariance estimators

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from scipy import linalg, stats
import statsmodels.robust as robust

# shorthand functions
mad = robust.mad
mad0 = lambda x: mad(x, center=0)

class Holder(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def _weight_mean(x, c):
    x = np.asarray(x)
    w = (1 - (x / c)**2)**2 * (np.abs(x) <= c)
    return w

def _winsor(x, c):
    return np.minimum(x**2, c**2)


def scale_tau(data, cm=4.5, cs=3, weight_mean=_weight_mean,
              weight_scale=_winsor, normalize=True, ddof=2):
    """tau estimator of univariate scale

    Experimental, API will change

    Parameters
    ----------
    data : array_like, 1-D or 2-D
        If data is 2d, then the location and scale estimates
        are calculated for each column
    cm : float
        constant used in call to weight_mean
    cs : float
        constant used in call to weight_scale
    weight_mean : callable
        function to calculate weights for weighted mean
    weight_scale : callable
        function to calculate scale, "rho" function
    normalize : bool
        rescale the scale estimate so it is consistent when the data is
        normally distributed. The computation assumes winsorized (truncated)
        variance.

    Returns
    -------
    mean : nd_array
        robust mean
    std : nd_array
        robust estimate of scale (standard deviation)

    Notes
    -----
    Uses definition of Maronna and Zamar 2002, with weighted mean and
    trimmed variance.
    The normalization has been added to match R robustbase.
    R robustbase uses by default ddof=0, with option to set it to 2.

    """

    x = np.asarray(data)
    nobs = x.shape[0]

    med_x = np.median(x, 0)
    xdm = x - med_x
    mad_x = np.median(np.abs(xdm), 0)
    wm = weight_mean(xdm / mad_x, cm)
    mean = (wm * x).sum(0) / wm.sum(0)
    var = mad_x**2 * weight_scale((x - mean) / mad_x, cs).sum(0) / (nobs - ddof)

    cf = 1
    if normalize:
        c =  cs * stats.norm.ppf(0.75)
        cf = 2 * ((1 - c**2) * stats.norm.cdf(c) - c * stats.norm.pdf(c)
                 + c**2) - 1
    #return Holder(loc=mean, scale=np.sqrt(var / cf))
    return mean, np.sqrt(var / cf)



def mahalanobis(data, cov=None, cov_inv=None):
    """Mahalanobis distance

    """
    x = np.asarray(data)
    if cov_inv is not None:
        d = (x * cov_inv.dot(x)).sum(1)
    elif cov is not None:
        d = (x * np.linalg.solve(cov, x.T).T).sum(1)
    else:
        raise ValueError('either cov or cov_inv needs to be given')

    return d


def cov_gk1(x, y, scale_func=mad):
    """Gnanadesikan and Kettenring covariance between two variables

    """
    s1 = scale_func((x + y))
    s2 = scale_func((x - y))
    return (s1**2 - s2**2) / 4


def cov_gk(data, scale_func=mad):
    """Gnanadesikan and Kettenring covariance estimator

    uses loop with cov_gk1

    """
    x = np.asarray(data)
    if x.ndim != 2:
        raise ValueError('data needs to be two dimensional')
    nobs, k_vars = x.shape
    cov = np.diag(scale_func(x)**2)
    for i in range(k_vars):
        for j in range(i):
            cij = cov_gk1(x[:, i], x[:, j], scale_func=scale_func)
            cov[i, j] = cov[j, i] = cij
    return cov


def cov_ogk(data, maxiter=2, scale_func=mad, cov_func=cov_gk,
            loc_func=lambda x:np.median(x, axis=0), reweight=0.9, ddof=1):
    """orthogonalized Gnanadesikan and Kettenring covariance estimator


    based on Maronna and Zamar 2002

    Parameters
    ----------
    data : array_like, 2-D

    maxiter : int
        number of iteration steps. According to Maronna and Zamar the
        estimate doesn't improve much after the second iteration and the
        iterations do not converge.

    scale_func : callable

    cov_func : callable

    loc_func : callable

    reweight : float in (0, 1) or None
        API for this will change.
        If reweight is None, then the reweighting step is skipped.
        Otherwise, reweight is the chisquare probability beta for the
        trimming based on estimated robust distances.
        Hard-rejection is currently the only weight function.
    ddof : int
        Degrees of freedom correction for the reweighted sample
        covariance.

    Notes
    -----
    compared to R: In robustbase covOGK the default scale and location are
    given by tau_scale with normalization but ddof=0.
    CovOGK of R package rrcov does not agree with this in the default options.

    """
    beta = reweight  #alias, need more reweighting options
    x = np.asarray(data)
    if x.ndim != 2:
        raise ValueError('data needs to be two dimensional')
    nobs, k_vars = x.shape
    z = x
    transf0 = np.eye(k_vars)
    for i in range(maxiter):
        scale = scale_func(z)
        zs = z / scale
        corr = cov_func(zs, scale_func=scale_func)
        # Maronna, Zamar set diagonal to 1, otherwise small difference to 1
        corr[np.arange(k_vars), np.arange(k_vars)] = 1
        evals, evecs = np.linalg.eigh(corr)
        transf = evecs * scale[:, None]   # A matrix in Maronna, Zamar
        #z = np.linalg.solve(transf, z.T).T
        z = zs.dot(evecs)
        transf0 = transf0.dot(transf)

    scale_z = scale_func(z)
    cov = (transf0 * scale_z**2).dot(transf0.T)

    loc_z = loc_func(z)
    loc = transf0.dot(loc_z)
    # prepare for results
    cov_ogk_raw = cov
    loc_ogk_raw = loc
    mask = None
    d = None
    if reweight is not None:
        d = (((z - loc_z) / scale_z)**2).sum(1)
        #d = mahalanobis(x - loc, cov)
        # only hard thresholding right now
        dmed = np.median(d)
        cutoff = dmed * stats.chi2.isf(1-beta, k_vars) / stats.chi2.ppf(0.5, k_vars)
        mask = d <= cutoff
        sample = x[mask]
        loc = sample.mean(0)
        cov = np.cov(sample.T, ddof=ddof)

    res = Holder(cov=cov, loc=loc, mask=mask, mahalanobis=d,
                 cov_ogk_raw=cov_ogk_raw, loc_ogk_raw=loc_ogk_raw,
                 transf0=transf0)

    return res
