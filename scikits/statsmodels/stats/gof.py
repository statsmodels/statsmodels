'''extra statistical function and helper functions

contains:

* goodness-of-fit tests
  - powerdiscrepancy
  - gof_chisquare_discrete
  - gof_binning_discrete



Author: Josef Perktold (josef-pktd)
'''

import numpy as np
#fix these imports
import scipy
from scipy import stats


# copied from regression/stats.utils
def powerdiscrepancy(observed, expected, lambd=0.0, axis=0, ddof=0):
    """Calculates power discrepancy, a class of goodness-of-fit tests
    as a measure of discrepancy between observed and expected data.

    This contains several goodness-of-fit tests as special cases, see the
    describtion of lambd, the exponent of the power discrepancy. The pvalue
    is based on the asymptotic chi-square distribution of the test statistic.

    freeman_tukey:
    D(x|\theta) = \sum_j (\sqrt{x_j} - \sqrt{e_j})^2

    Parameters
    ----------
    o : Iterable
        Observed values
    e : Iterable
        Expected values
    lambd : float or string
        * float : exponent `a` for power discrepancy
        * 'loglikeratio': a = 0
        * 'freeman_tukey': a = -0.5
        * 'pearson': a = 1   (standard chisquare test statistic)
        * 'modified_loglikeratio': a = -1
        * 'cressie_read': a = 2/3
        * 'neyman' : a = -2 (Neyman-modified chisquare, reference from a book?)
    axis : int
        axis for observations of one series
    ddof : int
        degrees of freedom correction,

    Returns
    -------
    D_obs : Discrepancy of observed values
    pvalue : pvalue


    References
    ----------
    Cressie, Noel  and Timothy R. C. Read, Multinomial Goodness-of-Fit Tests,
        Journal of the Royal Statistical Society. Series B (Methodological),
        Vol. 46, No. 3 (1984), pp. 440-464

    Campbell B. Read: Freeman-Tukey chi-squared goodness-of-fit statistics,
        Statistics & Probability Letters 18 (1993) 271-278

    Nobuhiro Taneichi, Yuri Sekiya, Akio Suzukawa, Asymptotic Approximations
        for the Distributions of the Multinomial Goodness-of-Fit Statistics
        under Local Alternatives, Journal of Multivariate Analysis 81, 335?359 (2002)
    Steele, M. 1,2, C. Hurst 3 and J. Chaseling, Simulated Power of Discrete
        Goodness-of-Fit Tests for Likert Type Data

    Examples
    --------

    >>> observed = np.array([ 2.,  4.,  2.,  1.,  1.])
    >>> expected = np.array([ 0.2,  0.2,  0.2,  0.2,  0.2])

    for checking correct dimension with multiple series

    >>> powerdiscrepancy(np.column_stack((observed,observed)).T, 10*expected, lambd='freeman_tukey',axis=1)
    (array([[ 2.745166,  2.745166]]), array([[ 0.6013346,  0.6013346]]))
    >>> powerdiscrepancy(np.column_stack((observed,observed)).T, 10*expected,axis=1)
    (array([[ 2.77258872,  2.77258872]]), array([[ 0.59657359,  0.59657359]]))
    >>> powerdiscrepancy(np.column_stack((observed,observed)).T, 10*expected, lambd=0,axis=1)
    (array([[ 2.77258872,  2.77258872]]), array([[ 0.59657359,  0.59657359]]))
    >>> powerdiscrepancy(np.column_stack((observed,observed)).T, 10*expected, lambd=1,axis=1)
    (array([[ 3.,  3.]]), array([[ 0.5578254,  0.5578254]]))
    >>> powerdiscrepancy(np.column_stack((observed,observed)).T, 10*expected, lambd=2/3.0,axis=1)
    (array([[ 2.89714546,  2.89714546]]), array([[ 0.57518277,  0.57518277]]))
    >>> powerdiscrepancy(np.column_stack((observed,observed)).T, expected, lambd=2/3.0,axis=1)
    (array([[ 2.89714546,  2.89714546]]), array([[ 0.57518277,  0.57518277]]))
    >>> powerdiscrepancy(np.column_stack((observed,observed)), expected, lambd=2/3.0, axis=0)
    (array([[ 2.89714546,  2.89714546]]), array([[ 0.57518277,  0.57518277]]))

    each random variable can have different total count/sum

    >>> powerdiscrepancy(np.column_stack((observed,2*observed)), expected, lambd=2/3.0, axis=0)
    (array([[ 2.89714546,  5.79429093]]), array([[ 0.57518277,  0.21504648]]))
    >>> powerdiscrepancy(np.column_stack((observed,2*observed)), expected, lambd=2/3.0, axis=0)
    (array([[ 2.89714546,  5.79429093]]), array([[ 0.57518277,  0.21504648]]))
    >>> powerdiscrepancy(np.column_stack((2*observed,2*observed)), expected, lambd=2/3.0, axis=0)
    (array([[ 5.79429093,  5.79429093]]), array([[ 0.21504648,  0.21504648]]))
    >>> powerdiscrepancy(np.column_stack((2*observed,2*observed)), 20*expected, lambd=2/3.0, axis=0)
    (array([[ 5.79429093,  5.79429093]]), array([[ 0.21504648,  0.21504648]]))
    >>> powerdiscrepancy(np.column_stack((observed,2*observed)), np.column_stack((10*expected,20*expected)), lambd=2/3.0, axis=0)
    (array([[ 2.89714546,  5.79429093]]), array([[ 0.57518277,  0.21504648]]))
    >>> powerdiscrepancy(np.column_stack((observed,2*observed)), np.column_stack((10*expected,20*expected)), lambd=-1, axis=0)
    (array([[ 2.77258872,  5.54517744]]), array([[ 0.59657359,  0.2357868 ]]))


    """
    o = np.array(observed)
    e = np.array(expected)

    if np.isfinite(lambd) == True:  # check whether lambd is a number
        a = lambd
    else:
        if   lambd == 'loglikeratio': a = 0
        elif lambd == 'freeman_tukey': a = -0.5
        elif lambd == 'pearson': a = 1
        elif lambd == 'modified_loglikeratio': a = -1
        elif lambd == 'cressie_read': a = 2/3.0
        else:
            raise ValueError('lambd has to be a number or one of ' + \
                    'loglikeratio, freeman_tukey, pearson, ' +\
                    'modified_loglikeratio or cressie_read')

    n = np.sum(o, axis=axis)
    nt = n
    if n.size>1:
        n = np.atleast_2d(n)
        if axis == 1:
            nt = n.T     # need both for 2d, n and nt for broadcasting
        if e.ndim == 1:
            e = np.atleast_2d(e)
            if axis == 0:
                e = e.T

    if np.all(np.sum(e, axis=axis) == n):
        p = e/(1.0*nt)
    elif np.all(np.sum(e, axis=axis) == 1):
        p = e
        e = nt * e
    else:
        raise ValueError('observed and expected need to have the same ' +\
                          'number of observations, or e needs to add to 1')
    k = o.shape[axis]
    if e.shape[axis] != k:
        raise ValueError('observed and expected need to have the same ' +\
                          'number of bins')

    # Note: taken from formulas, to simplify cancel n
    if a == 0:   # log likelihood ratio
        D_obs = 2*n * np.sum(o/(1.0*nt) * np.log(o/e), axis=axis)
    elif a == -1:  # modified log likelihood ratio
        D_obs = 2*n * np.sum(e/(1.0*nt) * np.log(e/o), axis=axis)
    else:
        D_obs = 2*n/a/(a+1) * np.sum(o/(1.0*nt) * ((o/e)**a - 1), axis=axis)

    return D_obs, stats.chi2.sf(D_obs,k-1-ddof)



#todo: need also binning for continuous distribution
#      and separated binning function to be used for powerdiscrepancy

def gof_chisquare_discrete(distfn, arg, rvs, alpha, msg):
    '''perform chisquare test for random sample of a discrete distribution

    Parameters
    ----------
    distname : string
        name of distribution function
    arg : sequence
        parameters of distribution
    alpha : float
        significance level, threshold for p-value

    Returns
    -------
    result : bool
        0 if test passes, 1 if test fails

    Notes
    -----
    originally written for scipy.stats test suite,
    still needs to be checked for standalone usage, insufficient input checking
    may not run yet (after copy/paste)

    refactor: maybe a class, check returns, or separate binning from
        test results
    '''

    # define parameters for test
##    n=2000
    n = len(rvs)
    nsupp = 20
    wsupp = 1.0/nsupp

##    distfn = getattr(stats, distname)
##    np.random.seed(9765456)
##    rvs = distfn.rvs(size=n,*arg)

    # construct intervals with minimum mass 1/nsupp
    # intervalls are left-half-open as in a cdf difference
    distsupport = xrange(max(distfn.a, -1000), min(distfn.b, 1000) + 1)
    last = 0
    distsupp = [max(distfn.a, -1000)]
    distmass = []
    for ii in distsupport:
        current = distfn.cdf(ii,*arg)
        if  current - last >= wsupp-1e-14:
            distsupp.append(ii)
            distmass.append(current - last)
            last = current
            if current > (1-wsupp):
                break
    if distsupp[-1]  < distfn.b:
        distsupp.append(distfn.b)
        distmass.append(1-last)
    distsupp = np.array(distsupp)
    distmass = np.array(distmass)

    # convert intervals to right-half-open as required by histogram
    histsupp = distsupp+1e-8
    histsupp[0] = distfn.a

    # find sample frequencies and perform chisquare test
    #TODO: move to compatibility.py
    if np.__version__ < '1.5':
        freq,hsupp = np.histogram(rvs, histsupp, new=True)
    else:
        freq,hsupp = np.histogram(rvs,histsupp)
    cdfs = distfn.cdf(distsupp,*arg)
    (chis,pval) = stats.chisquare(np.array(freq),n*distmass)

    return chis, pval, (pval > alpha), 'chisquare - test for %s' \
           'at arg = %s with pval = %s' % (msg,str(arg),str(pval))

# copy/paste, remove code duplication when it works
def gof_binning_discrete(rvs, distfn, arg, nsupp=20):
    '''get bins for chisquare type gof tests for a discrete distribution

    Parameters
    ----------
    rvs : array
        sample data
    distname : string
        name of distribution function
    arg : sequence
        parameters of distribution
    nsupp : integer
        number of bins. The algorithm tries to find bins with equal weights.
        depending on the distribution, the actual number of bins can be smaller.

    Returns
    -------
    freq : array
        empirical frequencies for sample; not normalized, adds up to sample size
    expfreq : array
        theoretical frequencies according to distribution
    histsupp : array
        bin boundaries for histogram, (added 1e-8 for numerical robustness)

    Notes
    -----
    The results can be used for a chisquare test ::

        (chis,pval) = stats.chisquare(freq, expfreq)

    originally written for scipy.stats test suite,
    still needs to be checked for standalone usage, insufficient input checking
    may not run yet (after copy/paste)

    refactor: maybe a class, check returns, or separate binning from
        test results
    todo :
      optimal number of bins ? (check easyfit),
      recommendation in literature at least 5 expected observations in each bin

    '''

    # define parameters for test
##    n=2000
    n = len(rvs)

    wsupp = 1.0/nsupp

##    distfn = getattr(stats, distname)
##    np.random.seed(9765456)
##    rvs = distfn.rvs(size=n,*arg)

    # construct intervals with minimum mass 1/nsupp
    # intervalls are left-half-open as in a cdf difference
    distsupport = xrange(max(distfn.a, -1000), min(distfn.b, 1000) + 1)
    last = 0
    distsupp = [max(distfn.a, -1000)]
    distmass = []
    for ii in distsupport:
        current = distfn.cdf(ii,*arg)
        if  current - last >= wsupp-1e-14:
            distsupp.append(ii)
            distmass.append(current - last)
            last = current
            if current > (1-wsupp):
                break
    if distsupp[-1]  < distfn.b:
        distsupp.append(distfn.b)
        distmass.append(1-last)
    distsupp = np.array(distsupp)
    distmass = np.array(distmass)

    # convert intervals to right-half-open as required by histogram
    histsupp = distsupp+1e-8
    histsupp[0] = distfn.a

    # find sample frequencies and perform chisquare test
    if np.__version__ < '1.5':
        freq,hsupp = np.histogram(rvs, histsupp, new=True)
    else:
        freq,hsupp = np.histogram(rvs,histsupp)
    #freq,hsupp = np.histogram(rvs,histsupp,new=True)
    cdfs = distfn.cdf(distsupp,*arg)
    return np.array(freq), n*distmass, histsupp
