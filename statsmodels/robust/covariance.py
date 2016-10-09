"""robust location, scatter and covariance estimators

Author: Josef Perktold
License: BSD-3


---  for Tyler
copied from "M:\josef\eclipsegworkspace\statsmodels-git\local_scripts\local_scripts\try_cov_tyler.py"

Created on Tue Nov 18 11:53:19 2014

Author: Josef Perktold
License: BSD-3


based on iteration in equ. (14) in Soloveychik and Wiesel

Soloveychik, I., and A. Wiesel. 2014. “Tyler’s Covariance Matrix Estimator in
Elliptical Models With Convex Structure.”
IEEE Transactions on Signal Processing 62 (20): 5251–59.
doi:10.1109/TSP.2014.2348951.

see also related articles by Frahm (which are not easy to read,
too little explanation, too many strange font letters)

shrinkage version is based on article

Chen, Yilun, A. Wiesel, and A.O. Hero. 2011. “Robust Shrinkage Estimation of
High-Dimensional Covariance Matrices.”
IEEE Transactions on Signal Processing 59 (9): 4097–4107.
doi:10.1109/TSP.2011.2138698.


"""

import numpy as np
from scipy import linalg, stats
from statsmodels import robust
import statsmodels.robust as robust

# shorthand functions
mad = robust.mad
mad0 = lambda x: mad(x, center=0)

class Holder(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


### GK and OGK ###

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

### Tyler ###



def cov_tyler(data, start_cov=None, normalize=False, maxiter=100, eps=1e-13):
    """Tyler's M-estimator for normalized covariance (scatter)

    The underlying (population) mean of the data is assumed to be zero.

    Parameters
    ----------
    data : ndarray
        data array with observations in rows and variables in columns
    start_cov : None or ndarray
        starting covariance for iterative solution
    normalize : bool
        If True, then the scatter matrix is normalized to have trace equalt
        to the number of columns in the data.
    maxiter : int
        maximum number of iterations to find the solution
    eps : float
        convergence criterion. The maximum absolute distance needs to be
        smaller than eps for convergence.

    Returns
    -------
    scatter : ndarray
        estimate of the scatter matrix
    iter : int
        number of iterations used in finding a solution. If iter is less than
        maxiter, then the iteration converged.
    """
    x = np.asarray(data)
    nobs, k_vars = x.shape
    kn = k_vars * 1. / nobs
    if start_cov is not None:
        c = start_cov
    else:
        c = np.diag(robust.mad(x, center=0)**2)

    #Tyler's M-estimator of shape (scatter) matrix
    for i in range(maxiter):
        c_inv = np.linalg.pinv(c)
        c_old = c
        # this could be vectorized but could use a lot of memory
        # TODO:  try to work in vectorized batches
        c = kn * sum(np.outer(xi, xi) / np.inner(xi, c_inv.dot(xi)) for xi in x)
        diff = np.max(np.abs(c - c_old))
        if diff < eps:
            break
    if normalize:
        c /= np.trace(c) / k_vars

    return c, i


def cov_tyler_regularized(data, start_cov=None, normalize=False,
                          shrinkage_factor=None,
                          maxiter=100, eps=1e-13):
    """Tyler's M-estimator for normalized covariance (scatter)

    The underlying (population) mean of the data is assumed to be zero.

    Parameters
    ----------
    data : ndarray
        data array with observations in rows and variables in columns.
    start_cov : None or ndarray
        starting covariance for iterative solution
    normalize : bool
        If True, then the scatter matrix is normalized to have trace equalt
        to the number of columns in the data.
    shrinkage_factor : None or float in [0, 1]
        Shrinkage for the scatter estimate. If it is zero, then no shrinkage
        is performed. If it is None, then the shrinkage factor will be
        determined by a plugin estimator
    maxiter : int
        maximum number of iterations to find the solution
    eps : float
        convergence criterion. The maximum absolute distance needs to be
        smaller than eps for convergence.

    Returns
    -------
    scatter : ndarray
        estimate of the scatter matrix
    iter : int
        number of iterations used in finding a solution. If iter is less than
        maxiter, then the iteration converged.
    shrinkage_factor : float
        shrinkage factor that was used in the estimation. This will be the
        same as the function argument if it was not None.

    Notes
    -----
    If the shrinkage factor is None, then a plugin is used as described in
    Chen and Wiesel 2011. The required trace for a pilot scatter estimate is
    obtained by the covariance rescaled by MAD estimate for the variance.

    """
    x = np.asarray(data)
    nobs, k_vars = x.shape
    kn = k_vars * 1. / nobs

    # calculate MAD only once if needed
    if start_cov is None or shrinkage_factor is None:
        scale_mad = robust.mad(x, center=0)

    corr = None
    if shrinkage_factor is None:
        # maybe some things here are redundant
        xd = x / x.std(0) #scale_mad
        corr = xd.T.dot(xd)
        corr * np.outer(scale_mad, scale_mad)
        corr *= k_vars / np.trace(corr)
        tr = np.trace(corr.dot(corr))

        n, k = nobs, k_vars
        # Chen and Wiesel 2011 equation (13)
        sf = k*k + (1 - 2./k) * tr
        sf /= (k*k - n*k - 2*n) + (n + 1 + 2. * (n - 1.) / k) * tr
        shrinkage_factor = sf

    if start_cov is not None:
        c = start_cov
    else:
        c = np.diag(scale_mad**2)

    identity = np.eye(k_vars)

    for i in range(maxiter):
        c_inv = np.linalg.pinv(c)
        c_old = c
        # this could be vectorized but could use a lot of memory
        # TODO:  try to work in vectorized batches
        c0 = kn * sum(np.outer(xi, xi) / np.inner(xi, c_inv.dot(xi)) for xi in x)
        if shrinkage_factor != 0:
            c = (1 - shrinkage_factor) * c0 + shrinkage_factor * identity
        else:
            c = c0

        c *= k_vars / np.trace(c)

        diff = np.max(np.abs(c - c_old))
        if diff < eps:
            break

    return c, i, shrinkage_factor, corr


def cov_tyler_pairs_regularized(data_iterator, start_cov=None, normalize=False,
                          shrinkage_factor=None, nobs=None, k_vars=None,
                          maxiter=100, eps=1e-13):
    """Tyler's M-estimator for normalized covariance (scatter)

    The underlying (population) mean of the data is assumed to be zero.

    experimental, calculation of startcov and shrinkage factor doesn't work
    This is intended for cluster robust and HAC covariance matrices that need
    to iterate over pairs of observations that are correlated.

    Parameters
    ----------
    data_iterator : restartable iterator
        needs to provide three elements xi, xj and w
    start_cov : None or ndarray
        starting covariance for iterative solution
    normalize : bool
        If True, then the scatter matrix is normalized to have trace equalt
        to the number of columns in the data.
    shrinkage_factor : None or float in [0, 1]
        Shrinkage for the scatter estimate. If it is zero, then no shrinkage
        is performed. If it is None, then the shrinkage factor will be
        determined by a plugin estimator
    maxiter : int
        maximum number of iterations to find the solution
    eps : float
        convergence criterion. The maximum absolute distance needs to be
        smaller than eps for convergence.

    Returns
    -------
    scatter : ndarray
        estimate of the scatter matrix
    iter : int
        number of iterations used in finding a solution. If iter is less than
        maxiter, then the iteration converged.
    shrinkage_factor : float
        shrinkage factor that was used in the estimation. This will be the
        same as the function argument if it was not None.

    Notes
    -----
    If the shrinkage factor is None, then a plugin is used as described in
    Chen and Wiesel 2011. The required trace for a pilot scatter estimate is
    obtained by the covariance rescaled by MAD estimate for the variance.

    """
    x = data_iterator
    #x = np.asarray(data)
    #nobs, k_vars = x.shape

    # calculate MAD only once if needed
    if start_cov is None or shrinkage_factor is None:
        scale_mad = sm.robust.mad(x, center=0)

    corr = None
    if shrinkage_factor is None:
        # maybe some things here are redundant
        xd = x / x.std(0) #scale_mad
        corr = xd.T.dot(xd)
        corr * np.outer(scale_mad, scale_mad)
        corr *= k_vars / np.trace(corr)
        tr = np.trace(corr.dot(corr))

        n, k = nobs, k_vars
        # Chen and Wiesel 2011 equation (13)
        sf = k*k + (1 - 2./k) * tr
        sf /= (k*k - n*k - 2*n) + (n + 1 + 2. * (n - 1.) / k) * tr
        shrinkage_factor = sf

    if start_cov is not None:
        c = start_cov
    else:
        c = np.diag(scale_mad**2)

    identity = np.eye(k_vars)
    kn = k_vars * 1. / nobs
    for i in range(maxiter):
        c_inv = np.linalg.pinv(c)
        c_old = c
        # this could be vectorized but could use a lot of memory
        # TODO:  try to work in vectorized batches
        # weights is a problem if iterator should be ndarray
        #c0 = kn * sum(np.outer(xi, xj) / np.inner(xi, c_inv.dot(xj)) for xi, xj in x)
        c0 = kn * sum(np.outer(xij[0], xij[1]) / np.inner(xij[0], c_inv.dot(xij[1])) for xij in x)
        if shrinkage_factor != 0:
            c = (1 - shrinkage_factor) * c0 + shrinkage_factor * identity
        else:
            c = c0

        c *= k_vars / np.trace(c)

        diff = np.max(np.abs(c - c_old))
        if diff < eps:
            break

    return c, i, shrinkage_factor#, corr


### iterative, M-estimators and related

def cov_weighted(data, weights, center=None):
    wsum = weights.sum()
    if center is None:
        wmean = weights.dot(data) / wsum
    xdm = data - wmean
    wcov = (weights * xdm.T).dot(xdm) / wsum
    return wcov, wmean


def weights_mvt(distance, df, k_vars):
    """weight function based on multivariate t distribution

    Parameters
    ----------
    distance : ndarray
        mahalanobis distance
    df : int or float
        degrees of freedom of the t distribution
    k_vars : int
        number of variables in the multivariate sample

    Returns
    -------
    weights : ndarray
        weights calculated for the given distances.

    References
    ----------

    Finegold, Michael A., and Mathias Drton. 2014. “Robust Graphical
    Modeling with T-Distributions.” arXiv:1408.2033 [Cs, Stat], August.
    http://arxiv.org/abs/1408.2033.

    Finegold, Michael, and Mathias Drton. 2011. “ROBUST GRAPHICAL
    MODELING OF GENE NETWORKS USING CLASSICAL AND ALTERNATIVE
    T-DISTRIBUTIONS.” The Annals of Applied Statistics 5 (2A): 1057–80.


    """
    w = (df + k_vars) / (df + distance)
    return w


def weights_quantile(distance, frac=0.5, rescale=True):
    cutoff = np.percentile(distance, frac * 100)
    w = (distance < cutoff).astype(int)
    return w


def _cov_iter(data, weights_func, weights_args=None, cov_init=None,
             rescale=False, maxiter=3, atol=1e-14, rtol=1e-6):
    """iterative robust covariance estimation using weights

    This is in the style of M-estimators for given weight function.

    Note: ??? Whether this is normalized to be consistent with the
    multivariate normal case depends on the weight function.
    maybe it is consistent, it's just a weighted cov.

    Parameters
    ----------
    data : array_like
    weights_func : callable
        function to calculate weights from the distances and weights_args
    weights_args : tuple
        extra arguments for the weights_func
    cov_init : ndarray, square 2-D
        initial covariance matrix
    rescale : bool
        If true then the resulting covariance matrix is normalized so it is
        approximately consistent with the normal distribution. Rescaling is
        based on the median of the distances and of the chisquare distribution.

    Returns
    -------
    this will still change
    cov, mean, w, dist, it, converged

    Notes
    -----
    This iterates over calculating the mahalanobis distance and weighted
    covariance. See Feingold and Drton 2014 for the motivation using weights
    based on the multivariate t distribution. Note that this does not calculate
    their alternative t distribution which requires numerical or Monte Carlo
    integration.

    """
    data = np.asarray(data)
    nobs, k_vars = data.shape

    if cov_init is None:
        cov_init = np.cov(data.T)

    converged = False
    cov = cov_old = cov_init
    for it in range(maxiter):
        dist = mahalanobis(data, cov=cov)
        w = weights_func(dist, *weights_args)
        cov, mean = cov_weighted(data, w)
        if np.allclose(cov, cov_old, atol=atol, rtol=rtol):
            converged = True
            break

    if rescale:
        s = np.median(dist) / stats.chi2.ppf(0.5, k_vars)
        cov *= s

    return cov, mean, w, dist, it, converged
