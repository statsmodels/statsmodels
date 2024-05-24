r"""robust location, scatter and covariance estimators

Author: Josef Perktold
License: BSD-3

Created on Tue Nov 18 11:53:19 2014

cov tyler based on iteration in equ. (14) in Soloveychik and Wiesel

Soloveychik, I., and A. Wiesel. 2014. Tyler's Covariance Matrix Estimator in
Elliptical Models With Convex Structure.
IEEE Transactions on Signal Processing 62 (20): 5251-59.
doi:10.1109/TSP.2014.2348951.

see also related articles by Frahm (which are not easy to read,
too little explanation, too many strange font letters)

shrinkage version is based on article

Chen, Yilun, A. Wiesel, and A.O. Hero. 2011. Robust Shrinkage Estimation of
High-Dimensional Covariance Matrices.
IEEE Transactions on Signal Processing 59 (9): 4097-4107.
doi:10.1109/TSP.2011.2138698.

"""

import numpy as np
from scipy import stats, linalg
from scipy.linalg.lapack import dtrtri
from .scale import mad
import statsmodels.robust.norms as rnorms
import statsmodels.robust.scale as rscale
import statsmodels.robust.tools as rtools
from statsmodels.tools.testing import Holder

from statsmodels.stats.covariance import corr_rank, corr_normal_scores

mad0 = lambda x: mad(x, center=0)  # noqa: E731
median = lambda x: np.median(x, axis=0)  # noqa: E731


# from scikit-learn
# https://github.com/scikit-learn/scikit-learn/blob/6a7eae5a97b6eba270abaaf17bc82ad56db4290e/sklearn/covariance/tests/test_covariance.py#L190
# Note: this is only intended for internal use
def _naive_ledoit_wolf_shrinkage(x, center):
    # A simple implementation of the formulas from Ledoit & Wolf

    # The computation below achieves the following computations of the
    # "O. Ledoit and M. Wolf, A Well-Conditioned Estimator for
    # Large-Dimensional Covariance Matrices"
    # beta and delta are given in the beginning of section 3.2
    n_samples, n_features = x.shape
    xdm = x - center
    emp_cov = xdm.T.dot(xdm) / n_samples
    mu = np.trace(emp_cov) / n_features
    delta_ = emp_cov.copy()
    delta_.flat[::n_features + 1] -= mu
    delta = (delta_ ** 2).sum() / n_features
    x2 = x ** 2
    beta_ = 1. / (n_features * n_samples) \
        * np.sum(np.dot(x2.T, x2) / n_samples - emp_cov ** 2)

    beta = min(beta_, delta)
    shrinkage = beta / delta
    return Holder(cov=shrinkage * emp_cov, method="naive ledoit wolf")


def coef_normalize_cov_truncated(frac, k_vars):
    """factor for consistency of truncated cov at normal distribution

    This is usually denoted by `b`. Here, it is calculated as `1 / b`.
    Trimming threshold is based on chisquare distribution.

    Parameters
    ----------
    frac: float in (0, 1)
        fraction (probability) of observations that are not trimmed
    k_vars : integer
        number of variables, i.e. dimension of multivariate random variable

    Returns
    -------
    fac : float
        factor to multiply the raw trimmed covariance

    Notes
    -----
    TODO: it might be better to use alpha = 1 - frac as argument instead.
    Uses explicit formula from Riani, Cerioli and Torti (2014) equation (3)
    which is also in Rocke and Woodroff (1996) Outliers equation (5).

    References
    ----------

    .. [1] Riani, Marco, Andrea Cerioli, and Francesca Torti. “On Consistency
       Factors and Efficiency of Robust S-Estimators.” TEST 23, no. 2 (February
       4, 2014): 356–87. https://doi.org/10.1007/s11749-014-0357-7.

    .. [2] Rocke, David M., and David L. Woodruff. “Identification of Outliers
       in Multivariate Data.” Journal of the American Statistical
       Association 91, no. 435 (1996): 1047–61.
       https://doi.org/10.2307/2291724.

    """
    # todo: use isf(alpha, k_vars) instead?
    fac = 1 / (stats.chi2.cdf(stats.chi2.ppf(frac, k_vars), k_vars + 2) / frac)
    return fac


def _coef_normalize_cov_truncated_(frac, k_vars):
    # normalize cov_truncated (example ogk)
    # currently not used except for verification
    # I think it generalized to other weight/transform function than trimming
    ct = k_vars / stats.chi2.expect(lambda x: x,
                                    lb=0,
                                    ub=stats.chi2.ppf(frac, k_vars),
                                    args=(k_vars,)
                                    )
    # correction for using cov of truncated sample which uses nobs of subsample
    # not full nobs
    ct *= frac
    return ct


class _NormalizeTruncCov:
    """Normalization factor for truncation with caching
    """
    _cache = {}

    def __call__(self, frac, k_vars):

        return self._cache.setdefault(
            (frac, k_vars),
            _coef_normalize_cov_truncated(frac, k_vars)
            )


_coef_normalize_cov_truncated = _NormalizeTruncCov()


# reweight adapted from OGK reweight step
def _reweight(x, loc, cov, trim_frac=0.975, ddof=1):
    """Reweighting step, trims data and computes Pearson covariance

    Parameters
    ----------
    x : ndarray
        Multivariate data with observation in rows
    loc : ndarray
        Location, mean or center of the data.
    cov : ndarray
        Covariance for computing Mahalanobis distance
    trim_frac : float in (0, 1)
        # todo: change name
        This is the coverage, (1 - trim_frac) is tail probability for chi2
        distribution.
    ddof : int or float
        Delta degrees of freedom used for trimmed Pearson covariance
        computed with `np.cov`.

    Returns
    -------
    cov : ndarray
        Covariance matrix of trimmed data, not rescaled to account for
        trimming.
    loc : ndarray
        Mean of trimmed data.

    See Also
    --------
    coef_normalize_cov_truncated

    Notes
    -----
    This reweighting step is used in OGK and in literature also for MCD.
    Trimming is metric with cutoff computed under the assumption that the
    Mahalanobis distances are chi-square distributed.

    References
    ----------
    ???


    """
    beta = trim_frac
    nobs, k_vars = x.shape  # noqa: F841
    # d = (((z - loc_z) / scale_z)**2).sum(1) # for orthogonal
    d = mahalanobis(x - loc, cov)
    # only hard thresholding right now
    dmed = np.median(d)
    cutoff = (dmed * stats.chi2.isf(1-beta, k_vars) /
              stats.chi2.ppf(0.5, k_vars))
    mask = d <= cutoff
    sample = x[mask]
    loc = sample.mean(0)
    cov = np.cov(sample.T, ddof=ddof)
    return cov, loc


def _rescale(x, loc, cov, prob=0.5):
    """rescale covariance to be consistent with normal distribution

    This matches median of mahalanobis distance with the chi-square
    distribution. This assumes that the data is normally distributed.

    Parameters
    ----------
    x : array-like
       sample data, 2-dim with observation in rows
    loc : ndarray
       mean or center of data
    cov : ndarray
       covariance estimate

    Returns
    -------
    ndarray: rescaled covariance

    Notes
    -----
    This rescaling is used in several functions to compute rescaled
    Mahalanobis distances for trimming.
    """
    if prob != 0.5:
        raise ValueError("currently only median prob=0.5 supported")

    x = np.asarray(x)
    k_vars = x.shape[1]
    d = mahalanobis(x - loc, cov)
    dmed = np.median(d)
    fac = dmed / stats.chi2.ppf(prob, k_vars)
    return cov * fac


def _outlier_gy(d, distr=None, k_endog=1, trim_prob=0.975):
    """determine outlier fraction given reference distribution

    This implements the outlier cutoff of Gervini and Yohai 2002
    for use in efficient reweighting.

    Parameters
    ----------
    d : array_like, 1-D
        array of squared standardized residuals or Mahalanobis distance
    distr : None or distribution instance
        reference distribution of d, needs cdf and ppf methods.
        If None, then chisquare with k_endog degrees of freedom is
        used. Otherwise, it should be a callable that provides the
        cdf function
    k_endog : int or float
        used only if cdf is None. In that case, it provides the degrees
        of freedom for the chisquare distribution.
    trim_prob : float in (0.5, 1)
        threshold for the tail probability at which the search for
        trimming or outlier fraction starts.

    Returns
    -------
    frac : float
        fraction of outliers
    cutoff : float
        cutoff value, values with `d > cutoff` are considered outliers
    ntail : int
        number of outliers
    ntail0 : int
        initial number of outliers based on trim tail probability.
    cutoff0 : float
        initial cutoff value based on trim tail probability.

    Notes
    -----
    This does not fully correct for multiple testing and does not
    maintain a familywise error rate or false discovery rate.
    The error rate goes to zero asymptotically under the null model,
    i.e. if there are no outliers.

    This might not handle threshold points correctly with discrete
    distribution.
    TODO: check weak versus strict inequalities (e.g. in isf)

    This only checks the upper tail of the distribution and of `d`.

    """
    d = np.asarray(d)
    nobs = d.shape[0]
    if distr is None:
        distr = stats.chi2(k_endog)

    threshold = distr.isf(1 - trim_prob)

    # get sorted array, we only need upper tail
    dtail = np.sort(d[d >= threshold])
    ntail0 = len(dtail)
    if ntail0 == 0:
        # no values above threshold
        return 0, threshold, 0, 0, threshold

    # using (n-1) / n as in GY2002
    ranks = np.arange(nobs - ntail0, nobs) / nobs

    frac = np.maximum(0, distr.cdf(dtail) - ranks).max()
    ntail = int(nobs * frac)  # rounding down
    if ntail > 0:
        cutoff = dtail[-ntail - 1]
    else:
        cutoff = dtail[-1] + 1e-15  # not sure, check inequality
    if (dtail > cutoff).sum() < ntail:
        import warnings
        warnings.warn('ties at cutoff, cutoff rule produces fewer'
                      'outliers than `ntail`')
    return frac, cutoff, ntail, ntail0, threshold


# ## GK and OGK ###

def mahalanobis(data, cov=None, cov_inv=None, sqrt=False):
    """Mahalanobis distance squared

    Note: this is without taking the square root.
    assumes data is already centered.

    Parameters
    ----------
    data : array-like
        Multivariate data with observation in rows.
    cov : None or ndarray
        Covariance matrix used in computing distance.
        This is only used if cov_inv is None.
    cov_inv : None or ndarray
        Inverse ovariance matrix used in computing distance.
        One of cov and cov_inv needs to be provided.
    sqrt : bool
        If False, then the squared distance is returned.
        If True, then the square root is returmend.

    Return
    ------
    ndarray : Mahalanobis distances or squared distance.
    """
    # another option would be to allow also cov^{-0.5) as keyword
    x = np.asarray(data)
    if cov_inv is not None:
        # einsum might be a bit faster
        d = (x * cov_inv.dot(x.T).T).sum(1)
    elif cov is not None:
        d = (x * np.linalg.solve(cov, x.T).T).sum(1)
    else:
        raise ValueError('either cov or cov_inv needs to be given')

    if sqrt:
        d = np.sqrt(d)

    return d


def cov_gk1(x, y, scale_func=mad):
    """Gnanadesikan and Kettenring covariance between two variables.

    Parameters
    ----------
    x : ndarray
        Data array.
    y : ndarray
        Data array
    scale_func : callable
        Scale function used in computing covariance.
        Default is median absolute deviation, MAD.

    Returns
    -------
    ndarray : GK covariance between x and y.

    """
    s1 = scale_func(x + y)
    s2 = scale_func(x - y)
    return (s1**2 - s2**2) / 4


def cov_gk(data, scale_func=mad):
    """Gnanadesikan and Kettenring covariance matrix estimator

    Parameters
    ----------
    data : ndarray
        Multivariate data array with observations in rows.
    scale_func : callable
        Scale function used in computing covariance.
        Default is median absolute deviation, MAD.

    Returns
    -------
    ndarray : GK covariance matrix of the data

    Notes
    -----
    This uses a loop over pairs of variables with cov_gk1 to avoid large
    intermediate arrays.

    """
    x = np.asarray(data)
    if x.ndim != 2:
        raise ValueError('data needs to be two dimensional')
    nobs, k_vars = x.shape  # noqa: F841
    cov = np.diag(scale_func(x)**2)
    for i in range(k_vars):
        for j in range(i):
            cij = cov_gk1(x[:, i], x[:, j], scale_func=scale_func)
            cov[i, j] = cov[j, i] = cij
    return cov


def cov_ogk(data, maxiter=2, scale_func=mad, cov_func=cov_gk,
            loc_func=lambda x: np.median(x, axis=0), reweight=0.9,
            rescale=True, rescale_raw=True, ddof=1):
    """Orthogonalized Gnanadesikan and Kettenring covariance estimator.

    Based on Maronna and Zamar 2002

    Parameters
    ----------
    data : array-like
        Multivariate data set with observation in rows and variables in
        columns.
    maxiter : int
        Number of iteration steps. According to Maronna and Zamar the
        estimate doesn't improve much after the second iteration and the
        iterations do not converge.
    scale_func : callable
        Scale function over axis=0 used in computing covariance.
        Default is median absolute deviation, MAD.
    cov_func : callable
        Bivariate covariance function. Default is GK.
    loc_func : callable
        Function to compute mean or center over axis=0.
    reweight : float in (0, 1) or None
        API for this will change.
        If reweight is None, then the reweighting step is skipped.
        Otherwise, reweight is the chisquare probability beta for the
        trimming based on estimated robust distances.
        Hard-rejection is currently the only weight function.
    rescale: bool
        If rescale is true, then reweighted covariance is rescale to be
        consistent at normal distribution.
        This only applies if reweight is not None.
    ddof : int
        Degrees of freedom correction for the reweighted sample
        covariance.

    Returns
    -------
    Holder instance with main attributes

    - cov : covariance, either raw OGK or reweighted OGK.
    - loc (and alias mean) : mean, either from raw OGK or reweighted OGK.
    - cov_ogk_raw : OGK covariance without reweighting optionally rescaled.
    - loc_ogk_raw : mean or center of OGK without reweighting.

    and extra attributes from intermediate results.

    Notes
    -----
    compared to R: In robustbase covOGK the default scale and location are
    given by tau_scale with normalization but ddof=0.
    CovOGK of R package rrcov does not agree with this in the default options.

    References
    ----------
    .. [1] Maronna, Ricardo A, and Ruben H Zamar. “Robust Estimates of Location
       and Dispersion for High-Dimensional Datasets.” Technometrics 44, no. 4
       (November 1, 2002): 307–17. https://doi.org/10.1198/004017002188618509.

    """
    if reweight is False:
        # treat false the same as None
        reweight = None
    if reweight is not None:
        beta = reweight  # alias, need more reweighting options
    else:
        beta = 0.9
    x = np.asarray(data)
    if x.ndim != 2:
        raise ValueError('data needs to be two dimensional')
    nobs, k_vars = x.shape  # noqa: F841
    z = x
    transf0 = np.eye(k_vars)
    for _ in range(maxiter):
        scale = scale_func(z)
        zs = z / scale
        corr = cov_func(zs, scale_func=scale_func)
        # Maronna, Zamar set diagonal to 1, otherwise small difference to 1
        corr[np.arange(k_vars), np.arange(k_vars)] = 1
        evals, evecs = np.linalg.eigh(corr)  # noqa: F841
        transf = evecs * scale[:, None]  # A matrix in Maronna, Zamar
        # z = np.linalg.solve(transf, z.T).T
        z = zs.dot(evecs)
        transf0 = transf0.dot(transf)

    scale_z = scale_func(z)
    cov = (transf0 * scale_z**2).dot(transf0.T)

    loc_z = loc_func(z)
    loc = transf0.dot(loc_z)
    # prepare for results
    cov_raw = cov
    loc_raw = loc

    # reweighting or rescaling
    # extra results are None if reweight is None
    mask = None
    d = None
    scale_factor_raw = 1.
    scale_factor = 1.
    n_trunc = 0
    if (reweight is not None) or rescale_raw:
        # compute scale_factor_raw and cutoff if needed
        d = (((z - loc_z) / scale_z)**2).sum(1)
        # d = mahalanobis(x - loc, cov)
        # only hard thresholding right now
        dmed = np.median(d)
        scale_factor_raw = dmed / stats.chi2.ppf(0.5, k_vars)
        cutoff = scale_factor_raw * stats.chi2.isf(1-beta, k_vars)

    if reweight is not None:
        mask = d <= cutoff
        n_trunc = nobs - sum(mask)
        sample = x[mask]
        loc = sample.mean(0)
        cov = np.cov(sample.T, ddof=ddof)
        # do we use empirical or theoretical frac, inlier/nobs or 1-beta?
        frac = beta  # n_inlier / nobs
        scale_factor = coef_normalize_cov_truncated(frac, k_vars)
        if rescale:
            cov *= scale_factor

    if rescale_raw:
        cov_raw *= scale_factor_raw

    # duplicate name loc mean center, choose consistent naming
    res = Holder(cov=cov, loc=loc, mean=loc, mask=mask, mahalanobis_raw=d,
                 cov_raw=cov_raw, loc_raw=loc_raw,
                 transf0=transf0, scale_factor=scale_factor,
                 scale_factor_raw=scale_factor_raw,
                 n_trunc=n_trunc,
                 method="ogk"
                 )

    return res


# ## Tyler ###

def cov_tyler(data, start_cov=None, normalize=False, maxiter=100, eps=1e-13):
    """Tyler's M-estimator for normalized covariance (scatter)

    The underlying (population) mean of the data is assumed to be zero.

    Parameters
    ----------
    data : array-like
        data array with observations in rows and variables in columns
    start_cov : None or ndarray
        starting covariance for iterative solution
    normalize : False or string
        If normalize is False (default), then the unscaled tyler scatter matrix
        is returned.

        Three types of normalization, i.e. rescaling are available by defining
        string option:

        - "trace" :
          The scatter matrix is normalized to have trace equal to the number
          of columns in the data.
        - "det" :
          The scatter matrix is normalized to have determinant equal to 1.
        - "normal" :
          The scatter matrix is rescaled to be consistent when data is normally
          distributed. Rescaling is based on median of the mahalanobis
          distances and assuming chisquare distribution of the distances.
        - "weights" :
          The scatter matrix is rescaled by the sum of weights.
          see Ollila et al 2023

    maxiter : int
        maximum number of iterations to find the solution.
    eps : float
        convergence criterion. The maximum absolute distance needs to be
        smaller than eps for convergence.

    Returns
    -------
    Holder instance with the following attributes
    cov : ndarray
        estimate of the scatter matrix
    iter : int
        number of iterations used in finding a solution. If iter is less than
        maxiter, then the iteration converged.

    References
    ----------
    .. [1] Tyler, David E. “A Distribution-Free M-Estimator of Multivariate
       Scatter.” The Annals of Statistics 15, no. 1 (March 1, 1987): 234–51.
    .. [2] Soloveychik, I., and A. Wiesel. 2014. Tyler's Covariance Matrix
       Estimator in Elliptical Models With Convex Structure.
       IEEE Transactions on Signal Processing 62 (20): 5251-59.
       doi:10.1109/TSP.2014.2348951.
    .. [3] Ollila, Esa, Daniel P. Palomar, and Frederic Pascal.
       “Affine Equivariant Tyler’s M-Estimator Applied to Tail Parameter
       Learning of Elliptical Distributions.” arXiv, May 7, 2023.
       https://doi.org/10.48550/arXiv.2305.04330.


    """
    x = np.asarray(data)
    nobs, k_vars = x.shape
    # kn = k_vars * 1. / nobs
    if start_cov is not None:
        c = start_cov
    else:
        c = np.diag(mad(x, center=0)**2)

    # Tyler's M-estimator of shape (scatter) matrix
    for i in range(maxiter):
        # this is old code, slower than new version, but more literal
        # c_inv = np.linalg.pinv(c)
        # c_old = c
        # c = kn * sum(np.outer(xi, xi) / np.inner(xi, c_inv.dot(xi))
        #              for xi in x)
        c_old = c
        ichol, _ = dtrtri(linalg.cholesky(c, lower=False), lower=0)
        v = x @ ichol
        dist_mahal_2 = np.einsum('ij,ji->i', v, v.T)
        weights = k_vars / dist_mahal_2[:, None]
        xw = np.sqrt(weights) * x
        c = xw.T @ xw / nobs

        diff = np.max(np.abs(c - c_old))
        if diff < eps:
            break

    if normalize is False or normalize is None:
        pass
    elif normalize == "trace":
        c /= np.trace(c) / k_vars
    elif normalize == "det":
        c /= np.linalg.det(c)**(1. / k_vars)
    elif normalize == "normal":
        _rescale(x, np.zeros(k_vars), c, prob=0.5)
    elif normalize == "weights":
        c /= weights.mean() / (np.trace(c) / k_vars)
    else:
        msg = 'normalize needs to be False, "trace", "det" or "normal"'
        raise ValueError(msg)

    return Holder(cov=c, n_iter=i, method="tyler")


def cov_tyler_regularized(data, start_cov=None, normalize=False,
                          shrinkage_factor=None,
                          maxiter=100, eps=1e-13):
    """Regularized Tyler's M-estimator for normalized covariance (shape).

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
    result instance with the following attributes
    cov : ndarray
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

    References
    ----------
    .. [1] Chen, Yilun, A. Wiesel, and A.O. Hero. “Robust Shrinkage
       Estimation of High-Dimensional Covariance Matrices.” IEEE Transactions
       on Signal Processing 59, no. 9 (September 2011): 4097–4107.
       https://doi.org/10.1109/TSP.2011.2138698.


    """
    x = np.asarray(data)
    nobs, k_vars = x.shape
    kn = k_vars * 1. / nobs

    # calculate MAD only once if needed
    if start_cov is None or shrinkage_factor is None:
        scale_mad = mad(x, center=0)

    corr = None
    if shrinkage_factor is None:
        # maybe some things here are redundant
        xd = x / x.std(0)  # scale_mad
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
        c0 = kn * sum(np.outer(xi, xi) / np.inner(xi, c_inv.dot(xi))
                      for xi in x)
        if shrinkage_factor != 0:
            c = (1 - shrinkage_factor) * c0 + shrinkage_factor * identity
        else:
            c = c0

        c *= k_vars / np.trace(c)

        diff = np.max(np.abs(c - c_old))
        if diff < eps:
            break

    res = Holder(cov=c, n_iter=i, shrinkage_factor=shrinkage_factor,
                 corr=corr)
    return res


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

    References
    ----------
    .. [1] Chen, Yilun, A. Wiesel, and A.O. Hero. “Robust Shrinkage Estimation
       of High-Dimensional Covariance Matrices.” IEEE Transactions on Signal
       Processing 59, no. 9 (September 2011): 4097–4107.
       https://doi.org/10.1109/TSP.2011.2138698.

    """
    x = data_iterator
    # x = np.asarray(data)
    # nobs, k_vars = x.shape

    # calculate MAD only once if needed
    if start_cov is None or shrinkage_factor is None:
        scale_mad = mad(x, center=0)

    corr = None
    if shrinkage_factor is None:
        # maybe some things here are redundant
        xd = x / x.std(0)  # scale_mad
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
        # c0 = kn * sum(np.outer(xi, xj) / np.inner(xi, c_inv.dot(xj))
        #               for xi, xj in x)
        c0 = kn * sum(np.outer(xij[0], xij[1]) /
                      np.inner(xij[0], c_inv.dot(xij[1])) for xij in x)
        if shrinkage_factor != 0:
            c = (1 - shrinkage_factor) * c0 + shrinkage_factor * identity
        else:
            c = c0

        c *= k_vars / np.trace(c)

        diff = np.max(np.abs(c - c_old))
        if diff < eps:
            break

    res = Holder(cov=c, n_iter=i, shrinkage_factor=shrinkage_factor,
                 corr=corr)
    return res


# ## iterative, M-estimators and related

def cov_weighted(data, weights, center=None, weights_cov=None,
                 weights_cov_denom=None, ddof=1):
    """weighted mean and covariance (for M-estimators)

    wmean = sum (weights * data) / sum(weights)
    wcov = sum (weights_cov * data_i data_i') / weights_cov_denom

    The options for weights_cov_denom are described in Parameters.
    By default both mean and cov are averages based on the same
    weights.

    Parameters
    ----------
    data : array_like, 2-D
        observations in rows, variables in columns
        no missing value handling
    weights : ndarray, 1-D
        weights array with length equal to the number of observations
    center : None or ndarray (optional)
        If None, then the weighted mean is subtracted from the data
        If center is provided, then it is used instead of the
        weighted mean.
    weights_cov : None, ndarray or "det" (optional)
        If None, then the same weights as for the mean are used.
    weights_cov_denom : None, float or "det" (optional)
        specified the denominator for the weighted covariance
        If None, then the sum of weights - ddof are used and the covariance is
        an average cross product.
        If "det", then the weighted covariance is normalized such that
        det(wcov) is 1.
        If weights_cov_denom is 1, then the weighted cross product is returned
        without averaging or scaling (sum of squares).
        Otherwise it is used directly as denominator after subtracting
        ddof.
    ddof : int or float
        covariance degrees of freedom correction, only used if
        weights_cov_denom is None or a float.

    Notes
    -----
    The extra options are available to cover the general M-estimator
    for location and scatter with estimating equations (using data x):

    sum (weights * (x - m)) = 0
    sum (weights_cov * (x_i - m) * (x_i - m)') - weights_cov_denom * cov = 0

    where the weights are functions of the mahalonibis distance of the
    residuals, and m is the mean.

    In the default case
    wmean = ave (w_i x_i)
    wcov = ave (w_i (x_i - m) (x_i - m)')

    References
    ----------
    .. [1] Rocke, D. M., and D. L. Woodruff. 1993. Computation of Robust
       Estimates of Multivariate Location and Shape.
       Statistica Neerlandica 47 (1): 27-42.
       doi:10.1111/j.1467-9574.1993.tb01404.x.


    """

    wsum = weights.sum()
    if weights_cov is None:
        weights_cov = weights
        wsum_cov = wsum
    else:
        wsum_cov = None  # calculate below only if needed

    if center is None:
        wmean = weights.dot(data) / wsum
    else:
        wmean = center

    xdm = data - wmean
    wcov = (weights_cov * xdm.T).dot(xdm)
    if weights_cov_denom is None:
        if wsum_cov is None:
            wsum_cov = weights_cov.sum()
        wcov /= (wsum_cov - ddof)   # * np.sum(weights_cov**2) / wsum_cov)
    elif weights_cov_denom == "det":
        wcov /= np.linalg.det(wcov)**(1 / wcov.shape[0])
    elif weights_cov_denom == 1:
        pass
    else:
        wcov /= (weights_cov_denom - ddof)

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
    .. [1] Finegold, Michael A., and Mathias Drton. 2014. Robust Graphical
       Modeling with T-Distributions. arXiv:1408.2033 [Cs, Stat], August.
       http://arxiv.org/abs/1408.2033.

    .. [2] Finegold, Michael, and Mathias Drton. 2011. Robust graphical
       modeling of gene networks using classical and alternative
       t-distributions. The Annals of Applied Statistics 5 (2A): 1057-80.
    """
    w = (df + k_vars) / (df + distance)
    return w


def weights_quantile(distance, frac=0.5, rescale=True):
    """Weight function for cutoff weights.

    The weight function is an indicator function for distances smaller then
    the frac quantile.

    rescale option is not supported.
    """
    cutoff = np.percentile(distance, frac * 100)
    w = (distance < cutoff).astype(int)
    return w


def _cov_iter(data, weights_func, weights_args=None, cov_init=None,
              rescale='med', maxiter=3, atol=1e-14, rtol=1e-6):
    """Iterative robust covariance estimation using weights.

    This is in the style of M-estimators for given weight function.

    Note: ??? Whether this is normalized to be consistent with the
    multivariate normal case depends on the weight function.
    maybe it is consistent, it's just a weighted cov.

    TODO: options for rescale instead of just median

    Parameters
    ----------
    data : array_like
    weights_func : callable
        function to calculate weights from the distances and weights_args
    weights_args : tuple
        extra arguments for the weights_func
    cov_init : ndarray, square 2-D
        initial covariance matrix
    rescale : "med" or "none"
        If "med" then the resulting covariance matrix is normalized so it is
        approximately consistent with the normal distribution. Rescaling is
        based on the median of the distances and of the chisquare distribution.
        Other options are not yet available.
        If rescale is the string "none", then no rescaling is performed.

    Returns
    -------
    Holder instance with attributes: cov, mean, w, dist, it, converged

    Notes
    -----
    This iterates over calculating the mahalanobis distance and weighted
    covariance. See Feingold and Drton 2014 for the motivation using weights
    based on the multivariate t distribution. Note that this does not calculate
    their alternative t distribution which requires numerical or Monte Carlo
    integration.

    References
    ----------
    .. [1] Finegold, Michael, and Mathias Drton. 2011. Robust graphical
       modeling of gene networks using classical and alternative
       t-distributions. Annals of Applied Statistics 5 (2A): 1057-80.

    """
    data = np.asarray(data)
    nobs, k_vars = data.shape  # noqa: F841

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

    # recompute maha distance at final estimate
    dist = mahalanobis(data, cov=cov)

    if rescale == 'none':
        s = 1
    elif rescale == 'med':
        s = np.median(dist) / stats.chi2.ppf(0.5, k_vars)
        cov *= s
    else:
        raise NotImplementedError('only rescale="med" is currently available')

    res = Holder(cov=cov, mean=mean, weights=w, mahalanobis=dist,
                 scale_factor=s, n_iter=it, converged=converged,
                 method="m-estimator",
                 weights_func=weights_func,
                 )
    return res


def _cov_starting(data, standardize=False, quantile=0.5, retransform=False):
    """compute some robust starting covariances

    The returned covariance matrices are intended as starting values
    for further processing. The main purpose is for algorithms with high
    breakdown point.
    The quality as standalone covariance matrices varies and might not
    be very good.

    Preliminary version. This will still be changed. Options and defaults can
    change, additional covarince methods will be added and return extended.

    Parameters
    ----------
    data : array-like
        Multivariate data with observations in rows (axis=0).
    standardize : bool
        If False, then the data is only centered (by median).
        If True, then the data is standardized using median and mad-scale.
        This scaling is only intermediate, the returned covariance compensates
        for the initial scaling.
    quantile : float in [0.5, 1]
        Parameter used for `_cov_iter` estimation.

    Returns
    -------
    list of Holder instances with `cov` attribute.

    """
    x = np.asarray(data)
    nobs, k_vars = x.shape
    if standardize:
        # there should be a helper function/class
        center = np.median(data, axis=0)
        xs = (x - center)
        std = mad0(data)
        xs /= std
    else:
        center = np.median(data, axis=0)
        xs = x - center
        std = 1

    cov_all = []
    d = mahalanobis(xs, cov=None, cov_inv=np.eye(k_vars))
    percentiles = [(k_vars+2) / nobs * 100 * 2, 25, 50, 85]
    cutoffs = np.percentile(d, percentiles)
    for p, cutoff in zip(percentiles, cutoffs):
        xsp = xs[d < cutoff]
        c = np.cov(xsp.T)
        corr_factor = coef_normalize_cov_truncated(p / 100, k_vars)
        c0 = Holder(
            cov=c * corr_factor,
            mean=xsp.mean(0) * std + center,
            method="pearson truncated",
            )
        c01 = _cov_iter(xs, weights_quantile, weights_args=(quantile,),
                        rescale="med", cov_init=c0.cov, maxiter=100)

        c02 = Holder(
            cov=_naive_ledoit_wolf_shrinkage(xsp, 0).cov * corr_factor,
            mean=xsp.mean(0) * std + center,
            method="ledoit_wolf",
            )
        c03 = _cov_iter(xs, weights_quantile, weights_args=(quantile,),
                        rescale="med", cov_init=c02.cov, maxiter=100)

        if not standardize or not retransform:
            cov_all.extend([c0, c01, c02, c03])
        else:
            # compensate for initial rescaling
            # TODO: this does not return list of Holder anymore
            s = np.outer(std, std)
            cov_all.extend([r.cov * s for r in [c0, c01, c02, c03]])

    c2 = cov_ogk(xs)
    cov_all.append(c2)

    c2raw = Holder(
            cov=c2.cov_raw,
            mean=c2.loc_raw * std + center,
            method="ogk_raw",
            )
    cov_all.append(c2raw)

    z_tanh = np.tanh(xs)
    c_th = Holder(
            cov=np.corrcoef(z_tanh.T),  # not consistently scaled for cov
            mean=center,  # TODO: do we add inverted mean z_tanh ?
            method="tanh",
            )
    cov_all.append(c_th)

    x_spatial = xs / np.sqrt(np.sum(xs**2, axis=1))[:, None]
    c_th = Holder(
            cov=np.cov(x_spatial.T),
            mean=center,
            method="spatial",
            )
    cov_all.append(c_th)

    c_th = Holder(
            # not consistently scaled for cov
            # cov=stats.spearmanr(xs)[0], # not correct shape if k=1 or 2
            cov=corr_rank(xs),  # always returns matrix, np.corrcoef result
            mean=center,
            method="spearman",
            )
    cov_all.append(c_th)

    c_ns = Holder(
            cov=corr_normal_scores(xs),  # not consistently scaled for cov
            mean=center,  # TODO: do we add inverted mean z_tanh ?
            method="normal-scores",
            )
    cov_all.append(c_ns)

    # TODO: rescale back to original space using center and std
    return cov_all


# ####### Det, CovDet and helper functions, might be moved to separate module


class _Standardize():
    """Robust standardization of random variable

    """

    def __init__(self, x, func_center=None, func_scale=None):
        # naming mean or center
        # maybe also allow str func_scale for robust.scale, e.g. for not
        #    vectorized Qn
        if func_center is None:
            center = np.median(x, axis=0)
        else:
            center = func_center(x)
        xdm = x - center
        if func_scale is None:
            scale = mad(xdm, center=0)
        else:
            # assumes vectorized
            scale = func_scale(x)

        self.x_stand = xdm / scale

        self.center = center
        self.scale = scale

    def transform(self, x):
        return (x - self.center) / self.scale

    def untransform_mom(self, m, c):
        mean = self.center + m
        cov = c * np.outer(self.scale, self.scale)
        return mean, cov


def _orthogonalize_det(x, corr, loc_func, scale_func):
    """Orthogonalize

    This is a simplified version of the OGK method.
    version from DetMCD works on zscored data
    (does not return mean and cov of original data)
    so we drop the compensation for scaling in zscoring

    z is the data here, zscored with robust estimators,
    e.g. median and Qn in DetMCD
    """
    evals, evecs = np.linalg.eigh(corr)  # noqa: F841
    z = x.dot(evecs)
    transf0 = evecs

    scale_z = scale_func(z)  # scale of principal components
    cov = (transf0 * scale_z**2).dot(transf0.T)
    # extra step in DetMCD, sphering data with new cov to compute center
    # I think this is equivalent to scaling z
    # loc_z = loc_func(z / scale_z) * scale_z  # center of principal components
    # loc = (transf0 * scale_z).dot(loc_z)
    transf1 = (transf0 * scale_z).dot(transf0.T)
    # transf1inv = (transf0 * scale_z**(-1)).dot(transf0.T)

    # loc = loc_func(x @ transf1inv) @ transf1
    loc = loc_func((z / scale_z).dot(transf0.T)) @ transf1

    return loc, cov


def _get_detcov_startidx(z, h, options_start=None, methods_cov="all"):
    """Starting sets for deterministic robust covariance estimators.

    These are intended as starting sets for DetMCD, DetS and DetMM.
    """

    if options_start is None:
        options_start = {}

    loc_func = options_start.get("loc_func", median)
    scale_func = options_start.get("scale_func", mad)
    z = (z - loc_func(z)) / scale_func(z)

    if np.squeeze(z).ndim == 1:
        # only one random variable
        z = np.squeeze(z)
        nobs = z.shape[0]
        idx_sel = np.argpartition(np.abs(z), h)[:h]
        idx_all = [(idx_sel, "abs-resid")]
        # next uses symmetric equal-tail trimming
        idx_sorted = np.argsort(z)
        h_tail = (nobs - h) // 2
        idx_all.append((idx_sorted[h_tail : h_tail + h], "trimmed-tail"))
        return idx_all

    # continue if more than 1 random variable
    cov_all = _cov_starting(z, standardize=False, quantile=0.5)

    # orthogonalization step
    idx_all = []
    for c in cov_all:
        if not hasattr(c, "method"):
            continue
        method = c.method
        mean, cov = _orthogonalize_det(z, c.cov, loc_func, scale_func)
        d = mahalanobis(z, mean, cov)
        idx_sel = np.argpartition(d, h)[:h]
        idx_all.append((idx_sel, method))

    return idx_all


class CovM:
    """M-estimator for multivariate Mean and Scatter.

    Interface incomplete and experimental.

    Parameters
    ----------
    data : array-like
        Multivariate data set with observation in rows and variables in
        columns.
    norm_mean : norm instance
        If None, then TukeyBiweight norm is used.
        (Currently no other norms are supported for calling the initial
        S-estimator)
    norm_scatter : None or norm instance
        If norm_scatter is None, then the norm_mean will be used.
    breakdown_point : float in (0, 0.5]
        Breakdown point for first stage S-estimator.
    scale_bias : None or float
        Must currently be provided if norm_mean is not None.
    method : str
        Currently only S-estimator has automatic selection of scale function.
    """

    def __init__(self, data, norm_mean=None, norm_scatter=None,
                 scale_bias=None, method="S"):
        # todo: method defines how norm_mean and norm_scatter are linked
        #       currently I try for S-estimator

        if method.lower() not in ["s"]:
            msg = f"method {method} option not recognize or implemented"
            raise ValueError(msg)

        self.data = np.asarray(data)
        self.k_vars = k_vars = self.data.shape[1]

        # Todo: check interface for scale bias
        self.scale_bias = scale_bias
        if norm_mean is None:
            norm_mean = rnorms.TukeyBiweight()
            c = rtools.tuning_s_cov(norm_mean, k_vars, breakdown_point=0.5)
            norm_mean._set_tuning_param(c, inplace=True)
            self.scale_bias = rtools.scale_bias_cov_biw(c, k_vars)[0]

        self.norm_mean = norm_mean
        if norm_scatter is None:
            self.norm_scatter = self.norm_mean
        else:
            self.norm_scatter = norm_scatter

        self.weights_mean = self.norm_mean.weights
        self.weights_scatter = self.weights_mean
        # self.weights_scatter = lambda d: self.norm_mean.rho(d) / d**2
        # this is for S-estimator, M-scale
        self.rho = self.norm_scatter.rho

    def _fit_mean_shape(self, mean, shape, scale):
        """Estimate mean and shape in iteration step.

        This does only one step.

        Parameters
        ----------
        mean : ndarray
            Starting value for mean
        shape : ndarray
            Starting value for shape matrix.
        scale : float
            Starting value for scale.

        Returns
        -------
        Holder instance with updated estimates.
        """
        d = mahalanobis(self.data - mean, shape, sqrt=True) / scale
        weights_mean = self.weights_mean(d)
        weights_cov = self.weights_scatter(d)

        res = cov_weighted(
            self.data,
            weights=weights_mean,
            center=None,
            weights_cov=weights_cov,
            weights_cov_denom="det",
            ddof=1,
            )
        return res

    def _fit_scale(self, maha, start_scale=None, maxiter=100, rtol=1e-5,
                   atol=1e-5):
        """Estimate iterated M-scale.

        Parameters
        ----------
        maha : ndarray
        start_scale : None or float
            Starting scale. If it is None, the mad of maha wi
        maxiter : int
            Maximum iterations to compute M-scale
        rtol, atol : float
            Relative and absolute convergence criteria for scale used with
            allclose.

        Returns
        -------
        float : scale estimate
        """
        if start_scale is None:
            # TODO: this does not really make sense
            # better scale to median of maha and chi or chi2
            start_scale = mad(maha)

        scale = rscale._scale_iter(
            maha,
            scale0=start_scale,
            maxiter=maxiter,
            rtol=rtol,
            atol=atol,
            meef_scale=self.rho,
            scale_bias=self.scale_bias,
            )
        return scale

    def fit(self, start_mean=None, start_shape=None, start_scale=None,
            maxiter=100, update_scale=True):
        """Estimate mean, shape and scale parameters with MM-estimator.

        Parameters
        ----------
        start_mean : None or float
            Starting value for mean, center.
            If None, then median is used.
        start_shape : None or 2-dim ndarray
            Starting value of shape matrix, i.e. scatter matrix normalized
            to det(scatter) = 1.
            If None, then scaled covariance matrix of data is used.
        start_scale : None or float.
            Starting value of scale.
        maxiter : int
            Maximum number of iterations.
        update_scale : bool
            If update_scale is False, then

        Returns
        -------
        results instance with mean, shape, scale, cov and other attributes.

        Notes
        -----
        If start_scale is provided and update_scale is False, then this is
        an M-estimator with a predetermined scale as used in the second
        stage of an MM-estimator.

        """

        converged = False

        if start_scale is not None:
            scale_old = start_scale
        else:
            scale_old = 1
            # will be reset if start_shape is also None.
        if start_mean is not None:
            mean_old = start_mean
        else:
            mean_old = np.median(self.data, axis=0)
        if start_shape is not None:
            shape_old = start_shape
        else:
            shape_old = np.cov(self.data.T)
            scale = np.linalg.det(shape_old) ** (1 / self.k_vars)
            shape_old /= scale
            if start_scale is not None:
                scale_old = scale

        if update_scale is False:
            scale = start_scale

        for i in range(maxiter):
            shape, mean = self._fit_mean_shape(mean_old, shape_old, scale_old)
            d = mahalanobis(self.data - mean, shape, sqrt=True)
            if update_scale:
                scale = self._fit_scale(d, start_scale=scale_old, maxiter=10)

            if (np.allclose(scale, scale_old, rtol=1e-5) and
                    np.allclose(mean, mean_old, rtol=1e-5) and
                    np.allclose(shape, shape_old, rtol=1e-5)
                    ):  # noqa E124
                converged = True
                break
            scale_old = scale
            mean_old = mean
            shape_old = shape

        maha = mahalanobis(self.data - mean, shape / scale, sqrt=True)

        res = Holder(
            mean=mean,
            shape=shape,
            scale=scale,
            cov=shape * scale**2,
            converged=converged,
            n_iter=i,
            mahalanobis=maha,
            )
        return res


class CovDetMCD:
    """Minimum covariance determinant estimator with deterministic starts.

    preliminary version

    reproducability:

    This uses deterministic starting sets and there is no randomness in the
    estimator.
    However, this will not be reprodusible across statsmodels versions
    when the methods for starting sets or tuning parameters for the
    optimization change.

    Parameters
    ----------
    data : array-like
        Multivariate data set with observation in rows and variables in
        columns.

    Notes
    -----
    The correction to the scale to take account of trimming in the reweighting
    estimator is based on the chisquare tail probability.
    This differs from CovMcd in R which uses the observed fraction of
    observations above the metric trimming threshold.

    References
    ----------
    ..[1] Hubert, Mia, Peter Rousseeuw, Dina Vanpaemel, and Tim Verdonck. 2015.
       “The DetS and DetMM Estimators for Multivariate Location and Scatter.”
       Computational Statistics & Data Analysis 81 (January): 64–75.
       https://doi.org/10.1016/j.csda.2014.07.013.
    ..[2] Hubert, Mia, Peter J. Rousseeuw, and Tim Verdonck. 2012. “A
       Deterministic Algorithm for Robust Location and Scatter.” Journal of
       Computational and Graphical Statistics 21 (3): 618–37.
       https://doi.org/10.1080/10618600.2012.672100.
    """

    def __init__(self, data):
        # no options yet, methods were written as functions
        self.data = np.asarray(data)

    def _cstep(self, x, mean, cov, h, maxiter=2, tol=1e-8):
        """C-step for mcd iteration

        x is data, perc is percentile h / nobs, don't need perc when we
        use np.argpartition
        requires starting mean and cov
        """

        converged = False

        for _ in range(maxiter):
            d = mahalanobis(x - mean, cov)
            idx_sel = np.argpartition(d, h)[:h]
            x_sel = x[idx_sel]
            mean = x_sel.mean(0)
            cov_new = np.cov(x_sel.T, ddof=1)

            if ((cov - cov_new)**2).mean() < tol:
                cov = cov_new
                converged = True
                break

            cov = cov_new

        return mean, cov, converged

    def _fit_one(self, x, idx, h, maxiter=2, mean=None, cov=None):
        """Compute mcd for one starting set of observations.

        Parameters
        ----------
        x : ndarray
            Data.
        idx : ndarray
            Indices or mask of observation in starting set, used as ``x[idx]``
        h : int
            Number of observations in evaluation set for cov.
        maxiter : int
            Maximum number of c-steps.

        Returns
        -------
        mean : ndarray
            Estimated mean.
        cov : ndarray
            Estimated covariance.
        det : float
            Determinant of estimated covariance matrix.

        Notes
        -----
        This does not do any preprocessing of the data and returns the
        empirical mean and covariance of evaluation set of the data ``x``.
        """
        if idx is not None:
            x_sel = x[idx]
        else:
            x_sel = x
        if mean is None:
            mean = x_sel.mean(0)
        if cov is None:
            cov = np.cov(x_sel.T, ddof=1)

        # updated with c-step
        mean, cov, conv = self._cstep(x, mean, cov, h, maxiter=maxiter)
        det = np.linalg.det(cov)

        return mean, cov, det, conv

    def fit(self, h, *, h_start=None, mean_func=None, scale_func=None,
            maxiter=100, options_start=None, reweight=True,
            trim_frac=0.975, maxiter_step=100):
        """
        Compute minimum covariance determinant estimate of mean and covariance.

        x : array-like
            Data with observation in rows and variables in columns.
        h : int
            Number of observations in evaluation set for minimimizing
            determinant.
        h_start : int
            Number of observations used in starting mean and covariance.
        mean_func, scale_func : callable or None.
            Mean and scale function for initial standardization.
            Current defaults, if they are None, are median and mad, but
            default scale_func will likely change.
        options_start : None or dict
            Options for the starting estimators.
            currently not used
            TODO: which options? e.g. for OGK
        reweight : bool
            If reweight is true, then a reweighted estimator is returned. The
            reweighting is based on a chisquare trimming of Mahalanobis
            distances. The raw results are in the ``results_raw`` attribute.
        trim_frac : float in (0, 1)
            Trim fraction used if reweight is true. Used to compute quantile
            of chisquare distribution with tail probability 1 - trim_frac.
        maxiter_step : int
            Number of iteration in the c-step.
            In the current implementation a small maxiter in the c-step does
            not find the optimal solution.

        Returns
        -------
        Holder instance with results
        """

        x = self.data
        nobs, k_vars = x.shape

        if h is None:
            h = (nobs + k_vars + 1) // 2  # check with literature
        if mean_func is None:
            mean_func = lambda x: np.median(x, axis=0)  # noqa
        if scale_func is None:
            scale_func = mad
        if options_start is None:
            options_start = {}
        if h_start is None:
            nobs, k_vars = x.shape
            h_start = max(nobs // 2 + 1, k_vars + 1)

        m = mean_func(x)
        s = scale_func(x)
        z = (x - m) / s
        # get initial mean, cov of standardized data, we only need ranking
        # of obs
        starts = _get_detcov_startidx(z, h_start, options_start)

        fac_trunc = coef_normalize_cov_truncated(h / nobs, k_vars)

        res = {}
        for ii, ini in enumerate(starts):
            idx_sel, method = ini
            mean, cov, det, _ = self._fit_one(x, idx_sel, h,
                                              maxiter=maxiter_step)
            res[ii] = Holder(
                mean=mean,
                cov=cov * fac_trunc,
                det_subset=det,
                method=method,
                )

        det_all = np.array([i.det_subset for i in res.values()])
        idx_best = np.argmin(det_all)
        best = res[idx_best]
        # mean = best.mean
        # cov = best.cov

        # need to c-step to convergence for best,
        # is with best 2 in original DetMCD
        if maxiter_step < maxiter:
            mean, cov, det, conv = self._fit_one(x, None, h, maxiter=maxiter,
                                                 mean=best.mean, cov=best.cov)
            best = Holder(
                mean=mean,
                cov=cov * fac_trunc,
                det_subset=det,
                method=method,
                converged=conv,
                )

        # include extra info in returned Holder instance
        best.det_all = det_all
        best.idx_best = idx_best

        best.tmean = m
        best.tscale = s

        if reweight:
            cov, mean = _reweight(x, best.mean, best.cov, trim_frac=trim_frac,
                                  ddof=1)
            fac_trunc = coef_normalize_cov_truncated(trim_frac, k_vars)
            best_w = Holder(
                mean=mean,
                cov=cov * fac_trunc,
                # det_subset=det,
                method=method,
                results_raw=best,
                )

            return best_w
        else:
            return best  # is Holder instance already


class CovDetS:
    """S-estimator for mean and covariance with deterministic starts.

    Parameters
    ----------
    data : array-like
        Multivariate data set with observation in rows and variables in
        columns.
    norm : norm instance
        If None, then TukeyBiweight norm is used.
        (Currently no other norms are supported for calling the initial
        S-estimator)
    breakdown_point : float in (0, 0.5]
        Breakdown point for first stage S-estimator.

    Notes
    -----
    Reproducability:

    This uses deterministic starting sets and there is no randomness in the
    estimator.
    However, the estimates may not be reproducable across statsmodels versions
    when the methods for starting sets or default tuning parameters for the
    optimization change. With different starting sets, the estimate can
    converge to a different local optimum.

    References
    ----------
    ..[1] Hubert, Mia, Peter Rousseeuw, Dina Vanpaemel, and Tim Verdonck. 2015.
       “The DetS and DetMM Estimators for Multivariate Location and Scatter.”
       Computational Statistics & Data Analysis 81 (January): 64–75.
       https://doi.org/10.1016/j.csda.2014.07.013.
    ..[2] Hubert, Mia, Peter J. Rousseeuw, and Tim Verdonck. 2012. “A
       Deterministic Algorithm for Robust Location and Scatter.” Journal of
       Computational and Graphical Statistics 21 (3): 618–37.
       https://doi.org/10.1080/10618600.2012.672100.
    """

    def __init__(self, data, norm=None, breakdown_point=0.5):
        # no options yet, methods were written as functions
        self.data = np.asarray(data)
        self.nobs, k_vars = self.data.shape
        self.k_vars = k_vars

        # self.scale_bias = scale_bias
        if norm is None:
            norm = rnorms.TukeyBiweight()
            c = rtools.tuning_s_cov(norm, k_vars, breakdown_point=0.5)
            norm._set_tuning_param(c, inplace=True)
            self.scale_bias = rtools.scale_bias_cov_biw(c, k_vars)[0]
        else:
            raise NotImplementedError("only Biweight norm is supported")

        self.norm = norm

        self.mod = CovM(data, norm_mean=norm, norm_scatter=norm,
                        scale_bias=self.scale_bias, method="S")

    def _get_start_params(self, idx):
        """Starting parameters from a subsample given by index

        Parameters
        ----------
        idx : ndarray
            Index used to select observations from the data. The index is used
            for numpy arrays, so it can be either a boolean mask or integers.

        Returns
        -------
        mean : ndarray
            Mean of subsample
        shape : ndarray
            The shape matrix of the subsample which is the covariance
            normalized so that determinant of shape is one.
        scale : float
            Scale of subsample, computed so that cov = shape * scale.
        """
        x_sel = self.data[idx]
        k = x_sel.shape[1]

        mean = x_sel.mean(0)
        cov = np.cov(x_sel.T)

        scale2 = np.linalg.det(cov) ** (1 / k)
        shape = cov / scale2
        scale = np.sqrt(scale2)
        return mean, shape, scale

    def _fit_one(self, mean=None, shape=None, scale=None, maxiter=100):
        """Compute local M-estimator for one starting set of observations.

        Parameters
        ----------
        x : ndarray
            Data.
        idx : ndarray
            Indices or mask of observation in starting set, used as ``x[idx]``
        h : int
            Number of observations in evaluation set for cov.
        maxiter : int
            Maximum number of c-steps.

        Returns
        -------
        mean : ndarray
            Estimated mean.
        cov : ndarray
            Estimated covariance.
        det : float
            Determinant of estimated covariance matrix.

        Notes
        -----
        This uses CovM to solve for the local optimum for given starting
        values.
        """

        res = self.mod.fit(
            start_mean=mean,
            start_shape=shape,
            start_scale=scale,
            maxiter=maxiter,
            update_scale=True
            )

        return res

    def fit(self, *, h_start=None, mean_func=None, scale_func=None,
            maxiter=100, options_start=None, maxiter_step=5):
        """Compute S-estimator of mean and covariance.

        Parameters
        ----------
        h_start : int
            Number of observations used in starting mean and covariance.
        mean_func, scale_func : callable or None.
            Mean and scale function for initial standardization.
            Current defaults, if they are None, are median and mad, but
            default scale_func will likely change.
        options_start : None or dict
           Options for the starting estimators.
           TODO: which options? e.g. for OGK

        Returns
        -------
        Holder instance with results
        """

        x = self.data
        nobs, k_vars = x.shape

        if mean_func is None:
            mean_func = lambda x: np.median(x, axis=0)  # noqa
        if scale_func is None:
            scale_func = mad
        if options_start is None:
            options_start = {}
        if h_start is None:
            nobs, k_vars = x.shape
            h_start = max(nobs // 2 + 1, k_vars + 1)

        m = mean_func(x)
        s = scale_func(x)
        z = (x - m) / s
        # get initial mean, cov of standardized data, we only need ranking
        # of obs
        starts = _get_detcov_startidx(z, h_start, options_start)

        res = {}
        for ii, ini in enumerate(starts):
            idx_sel, method = ini
            mean0, shape0, scale0 = self._get_start_params(idx_sel)
            res_i = self._fit_one(
                mean=mean0,
                shape=shape0,
                scale=scale0,
                maxiter=maxiter_step,
                )

            res_i.method = method
            res[ii] = res_i

        scale_all = np.array([i.scale for i in res.values()])
        idx_best = np.argmin(scale_all)
        best = res[idx_best]
        # mean = best.mean
        # cov = best.cov

        # need to c-step to convergence for best,
        # is with best 2 in original DetMCD
        if maxiter_step < maxiter:
            best = self._fit_one(
                mean=best.mean,
                shape=best.shape,
                scale=best.scale,
                maxiter=maxiter,
                )

        # include extra info in returned Holder instance
        best.scale_all = scale_all
        best.idx_best = idx_best

        best.tmean = m
        best.tscale = s

        return best  # is Holder instance already


class CovDetMM():
    """MM estimator using DetS as first stage estimator.

    Note: The tuning parameter for second stage M estimator is currently only
    available for a small number of variables and only three values of
    efficiency. For other cases, the user has to provide the norm instance
    with desirec tuning parameter.

    Parameters
    ----------
    data : array-like
        Multivariate data set with observation in rows and variables in
        columns.
    norm : norm instance
        If None, then TukeyBiweight norm is used.
        (Currently no other norms are supported for calling the initial
        S-estimator)
        If ``norm`` is an instance of TukeyBiweight, then it will be used in
        the second stage M-estimation. The ``efficiency`` argument is ignored
        and the tuning parameter of the user provided instance is not changed.
    breakdown_point : float in (0, 0.5]
        Breakdown point for first stage S-estimator.
    efficiency : float
        Asymptotic efficiency of second stage M estimator.

    Notes
    -----
    Current limitation is that only TukeyBiweight is supported.

    The tuning parameter for second stage M estimator uses a table of values
    for number of variables up to 15 and efficiency in
    [0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]. The tuning parameter for other
    cases needs to be computed by numerical integration and rootfinding.
    Alternatively, the user can provide a norm instance with desired tuning
    parameter.

    References
    ----------
    ..[1] Hubert, Mia, Peter Rousseeuw, Dina Vanpaemel, and Tim Verdonck. 2015.
       “The DetS and DetMM Estimators for Multivariate Location and Scatter.”
       Computational Statistics & Data Analysis 81 (January): 64–75.
       https://doi.org/10.1016/j.csda.2014.07.013.
    ..[2] Hubert, Mia, Peter J. Rousseeuw, and Tim Verdonck. 2012. “A
       Deterministic Algorithm for Robust Location and Scatter.” Journal of
       Computational and Graphical Statistics 21 (3): 618–37.
       https://doi.org/10.1080/10618600.2012.672100.
    ..[3] Lopuhaä, Hendrik P. 1989. “On the Relation between S-Estimators and
       M-Estimators of Multivariate Location and Covariance.” The Annals of
       Statistics 17 (4): 1662–83.
    ..[4] Salibián-Barrera, Matías, Stefan Van Aelst, and Gert Willems. 2006.
       “Principal Components Analysis Based on Multivariate MM Estimators with
       Fast and Robust Bootstrap.” Journal of the American Statistical
       Association 101 (475): 1198–1211.
    ..[5] Tatsuoka, Kay S., and David E. Tyler. 2000. “On the Uniqueness of
       S-Functionals and M-Functionals under Nonelliptical Distributions.” The
       Annals of Statistics 28 (4): 1219–43.
    """

    def __init__(self, data, norm=None, breakdown_point=0.5, efficiency=0.95):
        self.data = np.asarray(data)
        self.nobs, k_vars = self.data.shape
        self.k_vars = k_vars
        self.breakdown_point = breakdown_point

        # self.scale_bias = scale_bias
        if norm is None:
            norm = rnorms.TukeyBiweight()
            c = rtools.tukeybiweight_mvmean_eff(k_vars, efficiency)
            norm._set_tuning_param(c, inplace=True)
            # scale_bias is not used for second stage MM norm
            # self.scale_bias = rtools.scale_bias_cov_biw(c, k_vars)[0]
        elif not isinstance(norm, rnorms.TukeyBiweight):
            raise NotImplementedError("only Biweight norm is supported")
        # We allow tukeybiweight norm instance with user provided c

        self.norm = norm
        # model for second stage M-estimator
        self.mod = CovM(data, norm_mean=norm, norm_scatter=norm,
                        scale_bias=None, method="S")

    def fit(self, maxiter=100):
        """Estimate model parameters.

        Parameters
        ----------
        maxiter : int
            Maximum number of iterations in the second stage M-estimation.
        fit args : dict
            currently missing

        Returns
        -------
        Instance of a results or holder class.

        Notes
        -----
        This uses CovDetS for the first stage estimation and CovM with fixed
        scale in the second stage MM-estimation.

        TODO: fit options are missing.

        """
        # first stage estimate
        mod_s = CovDetS(
            self.data,
            norm=None,
            breakdown_point=self.breakdown_point
            )
        res_s = mod_s.fit()

        res = self.mod.fit(
            start_mean=res_s.mean,
            start_shape=res_s.shape,
            start_scale=res_s.scale,
            maxiter=maxiter,
            update_scale=False,
            )

        return res
