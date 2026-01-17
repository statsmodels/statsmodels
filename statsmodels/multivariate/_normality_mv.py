"""
Created on May 5, 2025 5:09:31 p.m.

Author: Josef Perktold
License: BSD-3

original is from 2012:

Multivariate Normality Tests and supporting code

Created on Thu Feb 16 23:40:46 2012

Author: Josef Perktold


based on
Doornik Hansen 1994, An Omnibus Test for Univariate and Multivariate Normality,
    published 2008

corrections/issues
* in Doornik-Hansen it looks like a typo for curtosis (power 4, instead of
  power 2) cross-checked with matlab fileexchange (add thanks to some authors)
* R MardiaTest doesn't multiply p-value by 2 for two-sided test
  I cross-checked with equivalent chi-square instead of normal distribution,
  p-value of chisquare based test and of normal test are identical.

I separated out helper functions like transformations because they might come
in handy for other things.

"""

import numpy as np

from scipy import stats, linalg


def transform_skewness(skew, n):
    """transform in Doornik Hansen to make skew more normally distributed

    transform by D'Agostino (1970)

    """
    if n < 3:
        raise ValueError('n needs to be at least 3 to calculate transformation')
    skew = np.asarray(skew)
    beta = 3. * (n**2. + 27. * n - 70.) * (n + 1.) * (n + 3.)
    beta = beta / ((n - 2.) * (n + 5.) * (n + 7.) * (n + 9.))
    w2 = np.sqrt(2. * (beta - 1.)) - 1.
    delta = 1. / np.sqrt(np.log(np.sqrt(w2)))
    y = skew * np.sqrt(0.5 * (w2 - 1.) * (n + 1.) * (n + 3.) / 6. / (n - 2.))
    z1 = delta * np.log(y + np.sqrt(y*y + 1))
    return z1


def transform_kurtosis(skew, kurt, n):
    """transform in Doornik Hansen to make kurtosis more chi-square distributed
    """
    if n < 3:
        raise ValueError('n needs to be at least 3 to calculate transformation')
    skew = np.asarray(skew)
    kurt = np.asarray(kurt)
    d = (n - 3.) * (n + 1) * (n**2. + 15. * n - 4.)
    a = (n - 2.) * (n + 5.) * (n + 7.) * (n**2. + 27. * n - 70.) / 6. / d
    c = (n - 7.) * (n + 5.) * (n + 7.) * (n*n + 2 * n - 5.) / 6. / d
    k = (n + 5.) * (n + 7.) * (n**3. + 37 * n*n + 11 * n - 313.) / 12. / d
    alpha = a + skew*skew * c
    chi = (kurt - 1 - skew*skew) * 2 * k
    # next part is Wilson Hilferty transform
    z2 = (np.power(chi * 0.5 / alpha, 1./3.) - 1 + 1. / 9. / alpha)
    z2 *= np.sqrt(9. * alpha)
    return z2


def transform_wilson_hilferty(x, alpha):
    # not sure what alpha is supposed to be in standalone version
    z2 = (np.power(x * 0.5 / alpha, 1./3.) - 1 + 1. / 9. / alpha)
    z2 *= np.sqrt(9. * alpha)
    return z2


def normal_dhm(skew, kurt, nobs):
    """Doornik-Hansen test for normality given skew, kurtosis and nobs

    skew and kurtosis are assumed to be from the standardized data, i.e. data
    with mean zero and identity covariance matrix:

    Parameters
    ----------
    skew : ndarray
        skew of standardized data
    kurtosis : ndarray
        kurtosis of standardized data
    nobs : int
        number of observations in the data

    Returns
    -------
    e_p : float
        Doornik-Hansen test statistic, chi-square distributed
    pval : float
        pvalue of the statistic under the Null hypothesis of normality
    p : int
        number of variables in multivariate data, for testing only will be
        removed

    Notes
    -----
    originally written to test the example in the Doornik Hansen paper,
    agrees with their results at printed precision

    """
    # TODO: replace arguments with data
    # valid for both univariate and multivariate ?
    # maybe not vectorized, except univariate is column array (k,1)
    n = nobs
    p = skew.shape[-1]
    z1 = transform_skewness(skew, n)
    z2 = transform_kurtosis(skew, kurt, n)
    e_p = (z1 * z1).sum(-1) + (z2 * z2).sum(-1)
    pval = stats.chi2.sf(e_p, 2*p)
    return e_p, pval, p


def normal_dh(data):
    """Doornik-Hansen test for normality

    Parameters
    ----------
    data : ndarray
        assmumed to have observations in rows and variables in columns

    Returns
    -------
    e_p : float
        Doornik-Hansen test statistic, chi-square distributed
    pval : float
        pvalue of the statistic under the Null hypothesis of normality
    p : int
        number of variables in multivariate data, for testing only will be
        removed

    """
    nobs = data.shape[0]
    ds = standardize(data, use_corr=True)
    sk = stats.skew(ds)
    ks = stats.kurtosis(ds, fisher=False)
    return normal_dhm(sk, ks, nobs)


def standardize(data, cov=None, demean=True, use_corr=False, ddof = 1):
    """transform data to identity correlation

    Parameters
    ----------
    data : ndarray (nobs, k_vars)
        data
    cov : ndarray
        optional covariance to use for standardization. If this is None, then
        the covariance of the data is used.
    demean : bool
        If true and the covariance is None, then the data is demeaned
        doesn't make sense, maybe option demean if cov is given

    Notes
    -----
    only simple case with default options has been checked.
    """
    # the transformed will have fewer columns in the case of perfect collinearity
    # not implemented yet
    nobs = data.shape[0]
    if cov is None:
        if demean:
            data = data - data.mean(0)
            #demean doesn't make sense in this case, because cov demeans anyway

        if use_corr:
            std = np.sqrt((data * data).sum(0) / (nobs - ddof))
            data = data / std

        cov = data.T @ data / (nobs - ddof)
    evals, evecs = linalg.eigh(cov)
    if evals.min() < 1e-13:
        import warnings
        warnings.warn("cov is(almost) singular")
    sigma_inv_half = evecs / np.sqrt(evals) @ evecs.T
    data_stzd = np.dot(data, sigma_inv_half)
    return data_stzd


def distance_mahalonibis_stzd(data, cov=None):
    """mahalonibis distance of standardized data

    this creates (nob,nobs) distance matrix
    """
    ds = standardize(data, cov=cov)
    distance = np.dot(ds, ds.T)
    return distance


def skewkurtosis_mardia(data):
    """Multivariate Skewness and Kurtosis as defined by Mardia

    This calculates a (nobs, nobs) distance matrix.
    """
    nobs, k_vars = data.shape
    ds = distance_mahalonibis_stzd(data)
    b1 = (ds**3).sum() * 1. / nobs**2  # mean
    b2 = (np.diag(ds)**2).sum() / nobs    # mean
    return b1, b2


def normalmv_mardia(data):
    """Mardia's test for multivariate normality

    Notes
    -----
    This calculates a (nobs, nobs) distance matrix. If the dataset is large,
    then I recommend to use only the cumputationally less demanding tests,
    especially normal_dh.


    """
    nobs, k_vars = data.shape
    b1, b2 = skewkurtosis_mardia(data)
    # ds = distance_mahalonibis_stzd(data)
    # b1 = (ds**3).sum() * 1. / nobs**2  #mean
    # b2 = (np.diag(ds)**2).sum() / nobs    #mean
    sk = nobs * b1 / 6.
    sk_demeaned = (b2 - k_vars * (k_vars + 2.))**2
    ku2 = nobs * sk_demeaned / (8. * k_vars * (k_vars + 2))
    m_p = sk + ku2
    ku = np.sqrt(ku2) * np.sign(sk_demeaned)
    ku_pval = stats.norm.sf(np.abs(ku))*2

    df_sk = k_vars * (k_vars + 1.) * (k_vars + 2.) / 6.
    df_ku = 1
    sk_pval = stats.chi2.sf(sk, df_sk)
    m_pval = stats.chi2.sf(m_p, df_sk + df_ku)
    return m_p, m_pval, (b1, sk_pval, b2, ku_pval, df_ku, sk, ku)
