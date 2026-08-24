"""Multivariate Distribution

Probability of a multivariate t distribution

Now also mvstnormcdf has tests against R mvtnorm

Still need non-central t, extra options, and convenience function for
location, scale version.

Author: Josef Perktold
License: BSD (3-clause)

Reference:
Genz and Bretz for formula

"""

from statsmodels.compat.pandas import deprecate_kwarg

import numpy as np
from numpy import exp as np_exp, log as np_log
from scipy import integrate, special
from scipy.special import gamma as sps_gamma, gammaln as sps_gammaln
from scipy.stats import chi

from statsmodels.tools.rng_qrng import check_random_state

from .extras import mvstdnormcdf


def chi2_pdf(x, df):
    """pdf of chi-square distribution"""
    # from scipy.stats.distributions
    Px = x ** (df / 2.0 - 1) * np.exp(-x / 2.0)
    Px /= special.gamma(df / 2.0) * 2 ** (df / 2.0)
    return Px


def chi_pdf(x, df):
    tmp = (
        (df - 1.0) * np_log(x)
        + (-x * x * 0.5)
        - (df * 0.5 - 1) * np_log(2.0)
        - sps_gammaln(df * 0.5)
    )
    return np_exp(tmp)
    # return x**(df-1.)*np_exp(-x*x*0.5)/(2.0)**(df*0.5-1)/sps_gamma(df*0.5)


def chi_logpdf(x, df):
    tmp = (
        (df - 1.0) * np_log(x)
        + (-x * x * 0.5)
        - (df * 0.5 - 1) * np_log(2.0)
        - sps_gammaln(df * 0.5)
    )
    return tmp


def funbgh(s, a, b, R, df):
    sqrt_df = np.sqrt(df + 0.5)
    ret = chi_logpdf(s, df)
    ret += np_log(
        mvstdnormcdf(s * a / sqrt_df, s * b / sqrt_df, R, maxpts=1000000, abseps=1e-6)
    )
    ret = np_exp(ret)
    return ret


def funbgh2(s, a, b, R, df):
    n = len(a)
    sqrt_df = np.sqrt(df)
    # np.power(s, df-1) * np_exp(-s*s*0.5)
    return np_exp((df - 1) * np_log(s) - s * s * 0.5) * mvstdnormcdf(
        s * a / sqrt_df,
        s * b / sqrt_df,
        R[np.tril_indices(n, -1)],
        maxpts=1000000,
        abseps=1e-4,
    )


def bghfactor(df):
    return np.power(2.0, 1 - df * 0.5) / sps_gamma(df * 0.5)


def mvstdtprob(a, b, R, df, ieps=1e-5, quadkwds=None):
    """
    Probability of rectangular area of standard t distribution

    assumes mean is zero and R is correlation matrix

    Parameters
    ----------
    a : array_like
        Lower integration limits.
    b : array_like
        Upper integration limits.
    R : array_like
        Correlation matrix.
    df : int or float
        Degrees of freedom.
    ieps : float
        Tail probability used to set the integration limits over the
        chi distribution.
    quadkwds : dict, optional
        Keyword arguments passed to `scipy.integrate.quad`.

    Notes
    -----
    This function does not calculate the estimate of the combined error
    between the underlying multivariate normal probability calculations
    and the integration.
    """
    kwds = dict(args=(a, b, R, df), epsabs=1e-4, epsrel=1e-2, limit=150)
    if quadkwds is not None:
        kwds.update(quadkwds)
    lower, upper = chi.ppf([ieps, 1 - ieps], df)
    res, err = integrate.quad(funbgh2, lower, upper, **kwds)
    prob = res * bghfactor(df)
    return prob


# written by Enzo Michelangeli, style changes by josef-pktd
# Student's T random variable
@deprecate_kwarg("random_state", "rng")
def multivariate_t_rvs(m, S, df=np.inf, n=1, rng=None):
    """generate random variables of multivariate t distribution

    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    rng : int, array_like of int, numpy.random.Generator, numpy.random.RandomState, optional
        If `rng` is None, a new ``Generator`` is created using fresh
        entropy from the operating system. If `rng` is an int or array
        of ints, a new ``Generator`` is created, seeded with `rng`. If
        `rng` is already a ``Generator`` or ``RandomState`` instance,
        that instance is used.
    rng : int, array_like of int, numpy.random.Generator, numpy.random.RandomState, optional
        .. deprecated:: 0.15

           random_state has been deprecated. In-line with SPEC-007, use
           rng for passing a random number generator or seed.

    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable


    """
    rng = check_random_state(rng)
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = np.ones(n)
    else:
        x = rng.chisquare(df, n) / df
    z = rng.multivariate_normal(np.zeros(d), S, (n,))
    return (
        m + z / np.sqrt(x)[:, None]
    )  # same output format as random.multivariate_normal
