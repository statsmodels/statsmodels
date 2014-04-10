'''helper functions conversion between moments

contains:

* conversion between central and non-central moments, skew, kurtosis and
  cummulants
* cov2corr : convert covariance matrix to correlation matrix


Author: Josef Perktold
License: BSD-3

'''

from statsmodels.compat.python import range
import numpy as np
from scipy.misc import comb


## start moment helpers

def mc2mnc(mc):
    '''convert central to non-central moments, uses recursive formula
    optionally adjusts first moment to return mean

    '''
    n = len(mc)
    mean = mc[0]
    mc = [1] + list(mc)    # add zero moment = 1
    mc[1] = 0  # define central mean as zero for formula
    mnc = [1, mean] # zero and first raw moments
    for nn,m in enumerate(mc[2:]):
        n=nn+2
        mnc.append(0)
        for k in range(n+1):
            mnc[n] += comb(n,k,exact=1) * mc[k] * mean**(n-k)

    return mnc[1:]


def mnc2mc(mnc, wmean = True):
    '''convert non-central to central moments, uses recursive formula
    optionally adjusts first moment to return mean

    '''
    n = len(mnc)
    mean = mnc[0]
    mnc = [1] + list(mnc)    # add zero moment = 1
    mu = [] #np.zeros(n+1)
    for n,m in enumerate(mnc):
        mu.append(0)
        #[comb(n-1,k,exact=1) for k in range(n)]
        for k in range(n+1):
            mu[n] += (-1)**(n-k) * comb(n,k,exact=1) * mnc[k] * mean**(n-k)
    if wmean:
        mu[1] = mean
    return mu[1:]


def cum2mc(kappa):
    '''convert non-central moments to cumulants
    recursive formula produces as many cumulants as moments

    References
    ----------
    Kenneth Lange: Numerical Analysis for Statisticians, page 40
    (http://books.google.ca/books?id=gm7kwttyRT0C&pg=PA40&lpg=PA40&dq=convert+cumulants+to+moments&source=web&ots=qyIaY6oaWH&sig=cShTDWl-YrWAzV7NlcMTRQV6y0A&hl=en&sa=X&oi=book_result&resnum=1&ct=result)


    '''
    mc = [1,0.0] #_kappa[0]]  #insert 0-moment and mean
    kappa0 = kappa[0]
    kappa = [1] + list(kappa)
    for nn,m in enumerate(kappa[2:]):
        n = nn+2
        mc.append(0)
        for k in range(n-1):
            mc[n] += comb(n-1,k,exact=1) * kappa[n-k]*mc[k]

    mc[1] = kappa0 # insert mean as first moments by convention
    return mc[1:]


def mnc2cum(mnc):
    '''convert non-central moments to cumulants
    recursive formula produces as many cumulants as moments

    http://en.wikipedia.org/wiki/Cumulant#Cumulants_and_moments
    '''
    mnc = [1] + list(mnc)
    kappa = [1]
    for nn,m in enumerate(mnc[1:]):
        n = nn+1
        kappa.append(m)
        for k in range(1,n):
            kappa[n] -= comb(n-1,k-1,exact=1) * kappa[k]*mnc[n-k]

    return kappa[1:]


def mc2cum(mc):
    '''just chained because I have still the test case
    '''
    return mnc2cum(mc2mnc(mc))


def mvsk2mc(args):
    '''convert mean, variance, skew, kurtosis to central moments'''
    mu,sig2,sk,kur = args

    cnt = [None]*4
    cnt[0] = mu
    cnt[1] = sig2
    cnt[2] = sk * sig2**1.5
    cnt[3] = (kur+3.0) * sig2**2.0
    return tuple(cnt)


def mvsk2mnc(args):
    '''convert mean, variance, skew, kurtosis to non-central moments'''
    mc, mc2, skew, kurt = args
    mnc = mc
    mnc2 = mc2 + mc*mc
    mc3  = skew*(mc2**1.5) # 3rd central moment
    mnc3 = mc3+3*mc*mc2+mc**3 # 3rd non-central moment
    mc4  = (kurt+3.0)*(mc2**2.0) # 4th central moment
    mnc4 = mc4+4*mc*mc3+6*mc*mc*mc2+mc**4
    return (mnc, mnc2, mnc3, mnc4)


def mc2mvsk(args):
    '''convert central moments to mean, variance, skew, kurtosis
    '''
    mc, mc2, mc3, mc4 = args
    skew = np.divide(mc3, mc2**1.5)
    kurt = np.divide(mc4, mc2**2.0) - 3.0
    return (mc, mc2, skew, kurt)


def mnc2mvsk(args):
    '''convert central moments to mean, variance, skew, kurtosis
    '''
    #convert four non-central moments to central moments
    mnc, mnc2, mnc3, mnc4 = args
    mc = mnc
    mc2 = mnc2 - mnc*mnc
    mc3 = mnc3 - (3*mc*mc2+mc**3) # 3rd central moment
    mc4 = mnc4 - (4*mc*mc3+6*mc*mc*mc2+mc**4)

    return mc2mvsk((mc, mc2, mc3, mc4))


#def mnc2mc(args):
#    '''convert four non-central moments to central moments
#    '''
#    mnc, mnc2, mnc3, mnc4 = args
#    mc = mnc
#    mc2 = mnc2 - mnc*mnc
#    mc3 = mnc3 - (3*mc*mc2+mc**3) # 3rd central moment
#    mc4 = mnc4 - (4*mc*mc3+6*mc*mc*mc2+mc**4)
#    return mc, mc2, mc

    #TODO: no return, did it get lost in cut-paste?


def cov2corr(cov, return_std=False):
    '''convert covariance matrix to correlation matrix

    Parameters
    ----------
    cov : array_like, 2d
        covariance matrix, see Notes

    Returns
    -------
    corr : ndarray (subclass)
        correlation matrix
    return_std : bool
        If this is true then the standard deviation is also returned.
        By default only the correlation matrix is returned.

    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires
    that division is defined elementwise. np.ma.array and np.matrix are allowed.

    '''
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    if return_std:
        return corr, std_
    else:
        return corr

def corr2cov(corr, std):
    '''convert correlation matrix to covariance matrix given standard deviation

    Parameters
    ----------
    corr : array_like, 2d
        correlation matrix, see Notes
    std : array_like, 1d
        standard deviation

    Returns
    -------
    cov : ndarray (subclass)
        covariance matrix

    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires
    that multiplication is defined elementwise. np.ma.array are allowed, but
    not matrices.

    '''
    corr = np.asanyarray(corr)
    std_ = np.asanyarray(std)
    cov = corr * np.outer(std_, std_)
    return cov

def se_cov(cov):
    '''get standard deviation from covariance matrix

    just a shorthand function np.sqrt(np.diag(cov))

    Parameters
    ----------
    cov : array_like, square
        covariance matrix

    Returns
    -------
    std : ndarray
        standard deviation from diagonal of cov

    '''
    return np.sqrt(np.diag(cov))
