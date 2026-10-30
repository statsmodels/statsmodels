# -*- coding: utf-8 -*-
"""Statistics confint and hypothesis tests for multinomial proportions


Warning: several functions here are internal and incompletely tested and
might not work correctly in some corner cases.



Created on Sat May  7 00:17:13 2016
initial draft in notebook

Author: Josef Perktold
License: BSD-3
"""

from __future__ import division

import numpy as np
from scipy import stats, special

##############################

def factmom_truncpoisson(n_moms, rates, low, upp):
    """factorial moments of truncated poisson random variables

    TODO: including or excluding bounds? should be including ?

    Parameter
    ---------
    n_moms : int
        number of moments to calculate, starting at first factorial moment
    rates : array_like
        Poisson rate parameters
    low, upp : array_like
        truncation limits for truncated Poisson random variables

    Returns
    -------
    mfact : ndarray
        factorial moments
    prob : float
        probability of truncated set

    Notes
    -----
    broadcasting behavior ???
    no input checking


    """
    rate = np.asarray(rates)
    k = np.arange(1, n_moms + 1)[:, None]

    # This is a bit tricky, I'm using 2-dim and 3-dim arrays for broadcasting
    cdf1 = stats.poisson.cdf(np.array([low - 1, upp])[:,None], rate)
    cdf2 = stats.poisson.cdf(np.array((low - k - 1, upp - k)), rate)
    prob = cdf1[1] - cdf1[0]
    tmp = ((cdf1[1] - cdf2[1]) - (cdf1[0] - cdf2[0])) / prob
    mfact = rate**k * (1 - tmp)
    return mfact, prob


def _check_factmom_truncpoisson(k_mom, rate, low, upp):
    """k'th factorial moments of truncated poisson random variables

    This is mainly for cross-checking the computation in factmom_truncpoisson.
    """

    r = k_mom
    b, a = low, upp  # May Johnson have reverse bound labels

    c = lambda kk: stats.poisson(rate).cdf(kk)

    fmom = rate**r * (1 - (c(a) - c(a - r) - (c(b - 1) - c(b - 1 - r))) /
                      (c(a) - c(b-1)))

    return fmom, (c(a) - c(b-1))



def sterling2(nk):
    """Sterling numbers of second kind

    coded from basic formula, not designed or verified for numerical stability
    with large numbers

    Parameters
    ----------
    nk : int

    Returns
    -------
    snk : list
        Sterling numbers for n=nk, for all k in (0, 1, 2, ..., nk)

    References
    ----------
    Wikipedia

    """
    snk = []
    for kk in range(nk+1):
        j = np.arange(kk + 1)
        sn = 1 / special.factorial(kk) * ( (-1)**(kk - j) * special.comb(kk, j) * j**nk).sum()
        snk.append(sn)
    return snk


def mfc2mnc(mfact):
    """convert factorial moments to non-central moments

    uses loops, but each moment can be array

    TODO: check, currently mf is without zero'th moment

    Parameters
    ----------
    mfact : array_like
        factorial moments starting at 1st moment

    Returns
    -------
    mnc : ndarray   TODO: maybe return list for consistency with moment_helpers
        noncentral moments

    """
    mf = np.asarray(mfact)
    m_all = []
    for km in range(1, len(mf)+1):
        # Sterling number of km'th moment
        snk = sterling2(km)[1:]
        m_all.append((mf[:len(snk)].T * snk).sum(1))
    print(m_all)
    m_all = np.asarray(m_all)
    return m_all


def mfc2mc_four(mfact):
    """convert four factorial moments to central moments

    uses explicit formulas for four moments
    `mfc2mnc` is the more general version that uses Sterling numbers

     Parameters
    ----------
    mfact : iterable
        four factorial moments starting at 1st moment
        (used through tuple unpacking)

    Returns
    -------
    mc : ndarray   TODO: maybe return list for consistency with moment_helpers
        central moments

    Notes
    -----
    mfact are unpacked, but each moment can be array.

    """
    mf1, mf2, mf3, mf4 = mfact

    m1 = mf1
    m2 = mf2 + m1 - m1**2
    m3 = mf3 + mf2 * 3 * (1 - m1)  + (m1 - 3 * m1**2 + 2 * m1**3)
    m4 = (mf4 +
          mf3 * (6 - 4 * m1) +
          mf2 * (7 - 12 * m1 + 6 * m1**2) +
          m1 * (1  - 4 * m1 + 6 * m1**2 - 3 * m1**3))

    return np.column_stack((m1, m2, m3, m4))


class BootstrapResults(object):
    """class to hold results of Monte Carlo or Bootstrap

    """

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def summary_progress_frame(self):
        """create pandas DataFrame to summarize result across batches

        The main purpose of this is to be able to check if the Monte Carlo is large
        enough, and the estimates have converged at sufficient precision.

        Returns
        -------
        summary_frame : DataFrame

        """
        import pandas as pd
        res = self.bresult
        sum_batch = np.diff(np.concatenate((np.zeros((1, res.shape[1])), res)), axis=0)
        mean = res[:, 1] / res[:, 0]
        diff_mean = np.concatenate(([0], np.diff(mean)))
        names = ['n_batch', 'sum_batch', 'n_cumsum', 'cumsum', 'mean', 'diff_mean']
        frame = pd.DataFrame(np.column_stack((sum_batch, res, mean, diff_mean)), columns=names)

        return frame


def prob_multinomial_simulated(nobs, probs, low, upp, n_rep=50000, batch=None):
    """box probability of a multinomial distribution obtained by simulation

    """
    if batch is None:
        batch = n_rep
    nr = 0
    boxprob = 0
    res = []
    while nr < n_rep:
        #print(nobs, prob, batch)
        rvs = np.random.multinomial(nobs, probs, size=(batch,))
        boxprob += ((low <= rvs) & (rvs <= upp)).all(1).sum()
        nr += batch
        res.append([nr, boxprob])
    boxprob /= nr
    bres = BootstrapResults(bresult=np.array(res))

    return boxprob, bres


def prob_multinomial(nobs, probs, low, upp, method="simulated", kwds=None):
    """box probability of a multinomial distribution

    currently computed only by simulation

    """
    if kwds is None:
        kwds = {}

    if method in ['sim', 'simulated']:
        return prob_multinomial_simulated(nobs, probs, low, upp, **kwds)
    else:
        raise(ValueError, method + " is not available")



def prob_multinomial_simulated_func(nobs, probs, func, n_rep=50000, batch=None):
    """probability of a multinomial distribution obtained by simulation

    This is a generalized version of prob_multinomial_simulated that uses
    a call back function instead of
    """
    if batch is None:
        batch = n_rep
    nr = 0
    boxprob = 0
    res = []
    while nr < n_rep:
        rvs = np.random.multinomial(nobs, probs, size=(batch,))
        boxprob += func(rvs).sum()
        nr += batch
        res.append([nr, boxprob])
    boxprob /= nr
    bres = BootstrapResults(bresult=np.array(res))

    return boxprob, bres



def chisquare_multinomial(count, values=None, method='simulated',
                          kwds=None):
    '''Chisquare test with exact p-values based on multinomial distribution

    Parameters
    ----------
    count : array_like, 1-D
        counts of a multinomial random variable, number of observations is
        taken to be the sum of counts.
    values : None or array_like, 1-D
        probabilities or expected frequencies. If they do not add up to one,
        then they are normalized to sum to one.
    method : string
        Only 'simulated' is currently implemented, which uses Monte Carlo
        integration based on multinomial random numbers.

    Returns
    -------
    chi-square statistic : float
        chisquare statistic returned by scipy.stats.chisquare
    p-value : float
        exact (by simulation) p-value of the chisquare test
    chi-square p-value : float
        p-value returned by scipy.stats.chisquare

    '''
    count = np.asarray(count)

    if values is None:
        probs = np.ones(len(count)) / len(count)
    else:
        probs = np.asarray(values)
        if np.abs(np.sum(probs) - 1) > 1e-15:
            probs = probs / probs.sum()

    if count.ndim > 1 or probs.ndim > 1:
        raise ValueError('count and probs cannot have more than 1 dimension')

    nobs = count.sum()

    stat_chi2, pval_chi2 = stats.chisquare(count, probs * nobs)

    if method == 'simulated':

        ch = lambda xc : stats.chisquare(xc.T, probs[:, None] * nobs)
        chind = lambda xx: ch(xx)[0] >= stat_chi2 - 1e-10

        mc_kwds = dict(n_rep=50000, batch=None)
        if kwds is not None:
            mc_kwds.update(kwds)
        pb = prob_multinomial_simulated_func(nobs, probs, chind, **mc_kwds)
        return stat_chi2, pb[0], pval_chi2

    else:
        raise NotImplementedError('currently only simulated')



#############################

_norm_pdf_C = np.sqrt(2*np.pi)
def _norm_pdf(x):
    return np.exp(-x**2/2.0) / _norm_pdf_C

def pdf_edgeworth_mvsk_simple(x, mvsk):
    """pdf of Edgeworth expansion based on mean, variance, skew and kurtosis

    This uses explicit power polynomial formulas.
    The moments in `mvsk` are tuple unpacked and all calculations are
    elementwise, so numpy broadcasting applies.

    """
    m, v, skew, kurt = mvsk

    s = np.sqrt(v)
    xs = (x - m) / s
    del x, v, m

    xs2 = xs * xs
    xs3 = xs2 * xs
    xs4 = xs2 * xs2

    pdf = (1 +
           skew / 6 * (xs3 - 3 * xs) +
           kurt / 24 * (xs4 - 6 * xs2 + 3) +
           skew**2 / 72 * (xs3 * xs2 - 15 * xs4 + 45 * xs2 - 15))
    pdf *= _norm_pdf(xs) / s

    return pdf


def _pdf_edgeworth_mvsk_simple2(x, mvsk):
    """

    rearrange polynomial calculation,

    Essentially no numerical difference in example in double precision range
    (around 1e-16 relative)
    """
    m, v, skew, kurt = mvsk

    s = np.sqrt(v)
    xs = (x - m) / s
    del x, v, m

    xs2 = xs * xs
    xs3 = xs2 * xs
    xs4 = xs2 * xs2

    # temp variables
    a = skew / 6
    b = kurt / 24
    c = skew**2 / 72

    pdf = (1 + b * 3 - c * 15 +
           xs * (- a * 3) +
           xs2 * (- b * 6 + c * 45) +
           xs3 * a +
           xs4 * (b - c * 15) +
           xs3 * xs2 * c)

    pdf *= _norm_pdf(xs) / s

    return pdf


from numpy.polynomial import Polynomial

class EdgeworthMvskSimple(object):
    """Edgeworth expansion based on mean, variance, skew and kurtosis

    This precalculates and combines numpy Polynomial in init.
    The moments cannot be vectorized, i.e. mvsk is required to be 1-D,
    in this version because of the limitation of numpy Polynomial.

    This uses the power polynomial representation and not directly
    Hermit polynomials.



    """

    def __init__(self, mvsk):
        m, v, skew, kurt = mvsk

        self.m = m
        self.s = np.sqrt(v)
        #xs = (x - m) / s
        p = (Polynomial([1]) +
             skew / 6 * Polynomial([0, -3, 0, 1]) +
             kurt / 24 * Polynomial([3, 0, -6, 0, 1]) +
             skew**2 / 72 * Polynomial([-15, 0, 45, 0, -15, 0, 1]))
        self.poly = p

    def pdf(self, x) :
        m, s = self.m, self.s
        p = self.poly
        xs = (x - m) / s
        pdf = _norm_pdf(xs) / s * p(xs)
        return pdf

###############################

def prob_multinomial_edgeworth(lower, upper, nobs, rates):
    """box probability of multinomial distribution using Edgeworth approximation

    """

    mfact, prob_poi = factmom_truncpoisson(4, rates, lower, upper)
    mc = mfc2mc_four(mfact)  # central moments

    # calculate aggregated mvsk
    # TODO: transpose or make return of mfc2mc into list ?
    mc = mc.T
    mc1, mc2, mc3, mc4 = mc.sum(1)
    mc22 = (mc[1]**2).sum()
    #k = mc.shape[1]
    #sqrtk = np.sqrt(k)
    # equations from Sison and Glaz JASA 1995 with np.sqrt(k),
    # now from May and Johnson without sqrt(k), sum not average
    skew = mc3 / mc2**(3 / 2) #/ sqrtk
    kurt = (mc4 - 3 * mc22) / mc2**2 #/ sqrtk
    #mvsk = [mc1, mc2 / sqrtk, skew, kurt]
    mvsk = [mc1, mc2, skew, kurt]
    # edgworth approximation to probability of sum of truncated Poisson
    print('mvsk', mvsk)
    prob_w = pdf_edgeworth_mvsk_simple(nobs, mvsk)
    prob_poi = np.exp(np.log(prob_poi).sum())
    box_prob = prob_poi * prob_w / stats.poisson.pmf(nobs, nobs)

    return box_prob
