'''helper functions conversion between moments

contains:

* conversion between central and non-central moments, skew, kurtosis and
  cummulants



Author: josef-pktd
'''

import numpy as np
#fix these imports
import scipy
#from scipy import stats   #not used here

## start moment helpers

def mc2mnc(_mc):
    '''convert central to non-central moments, uses recursive formula
    optionally adjusts first moment to return mean

    '''
    n = len(_mc)
    mean = _mc[0]
    mc = [1] + list(_mc)    # add zero moment = 1
    mc[1] = 0  # define central mean as zero for formula
    mnc = [1, mean] # zero and first raw moments
    for nn,m in enumerate(mc[2:]):
        n=nn+2
        mnc.append(0)
        for k in range(n+1):
            mnc[n] += scipy.comb(n,k,exact=1) * mc[k] * mean**(n-k)

    return mnc[1:]


def mnc2mc(_mnc, wmean = True):
    '''convert non-central to central moments, uses recursive formula
    optionally adjusts first moment to return mean

    '''
    n = len(_mnc)
    mean = _mnc[0]
    mnc = [1] + list(_mnc)    # add zero moment = 1
    mu = [] #np.zeros(n+1)
    for n,m in enumerate(mnc):
        mu.append(0)
        #[scipy.comb(n-1,k,exact=1) for k in range(n)]
        for k in range(n+1):
            mu[n] += (-1)**(n-k) * scipy.comb(n,k,exact=1) * mnc[k] * mean**(n-k)
    if wmean:
        mu[1] = mean
    return mu[1:]


def cum2mc(_kappa):
    '''convert non-central moments to cumulants
    recursive formula produces as many cumulants as moments

    References
    ----------
    Kenneth Lange: Numerical Analysis for Statisticians, page 40
    (http://books.google.ca/books?id=gm7kwttyRT0C&pg=PA40&lpg=PA40&dq=convert+cumulants+to+moments&source=web&ots=qyIaY6oaWH&sig=cShTDWl-YrWAzV7NlcMTRQV6y0A&hl=en&sa=X&oi=book_result&resnum=1&ct=result)


    '''
    mc = [1,0.0] #_kappa[0]]  #insert 0-moment and mean
    kappa = [1] + list(_kappa)
    for nn,m in enumerate(kappa[2:]):
        n = nn+2
        mc.append(0)
        for k in range(n-1):
            mc[n] += scipy.comb(n-1,k,exact=1) * kappa[n-k]*mc[k]

    mc[1] = _kappa[0] # insert mean as first moments by convention
    return mc[1:]


def mnc2cum(_mnc):
    '''convert non-central moments to cumulants
    recursive formula produces as many cumulants as moments

    http://en.wikipedia.org/wiki/Cumulant#Cumulants_and_moments
    '''
    mnc = [1] + list(_mnc)
    kappa = [1]
    for nn,m in enumerate(mnc[1:]):
        n = nn+1
        kappa.append(m)
        for k in range(1,n):
            kappa[n] -= scipy.comb(n-1,k-1,exact=1) * kappa[k]*mnc[n-k]

    return kappa[1:]


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

def mnc2mc(args):
    '''convert four non-central moments to central moments
    '''
    mnc, mnc2, mnc3, mnc4 = args
    mc = mnc
    mc2 = mnc2 - mnc*mnc
    mc3 = mnc3 - (3*mc*mc2+mc**3) # 3rd central moment
    mc4 = mnc4 - (4*mc*mc3+6*mc*mc*mc2+mc**4)

    #TODO: no return, did it get lost in cut-paste?

def cov2corr(cov):
    '''convert covariance matrix to correlation matrix

    Parameter
    ---------
    cov : array_like, 2d
        covariance matrix, see Notes

    Returns
    -------
    corr : ndarray (subclass)
        correlation matrix

    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires
    that division is defined elementwise. np.ma.array and np.matrix are allowed.

    '''
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    return corr
