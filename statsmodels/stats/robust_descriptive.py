# -*- coding: utf-8 -*-
"""Robust Measures of Skew and Kurtosis

Created on Fri May 31 09:39:23 2013

Author: Josef Perktold

This currently implements parts of Kim and White 2004.

TODO:
 - refactor to calculate several measures at the same time
 - refactor to reuse intermediate calculations
 - add shortcut names for the `kind`
   the full names are too long and difficult to spell
 - relies currently on frozen distribution, generalize (explicit args, kwds ?)

Matteo Bonato 2011 Robust estimation of skewness and kurtosis in distributions with infinite higher moments
 this paper has the sign corrected for `crow-siddiqui` kurtosis


"""

import numpy as np
from scipy import stats

def frozen_expect(self, *args, **kwds):
    kwds.update(self.kwds)
    return self.dist.expect(*args, **kwds)

if not hasattr(stats.distributions.rv_frozen, 'expect'):
    stats.distributions.rv_frozen.expect = frozen_expect

def quantile(x, frac):
    #replace this function
    #find more robust check than __iter__
    if not hasattr(frac, '__iter__'):
        frac = [frac]

    q = [stats.scoreatpercentile(x, fr * 100) for fr in frac]
    return q

def tail_mean(data, upp, low):
    '''

    Note:
    upp and low refer to bounds not tail.
    the left tail includes observations to upp, the right tail includes
    observations starting at low.
    '''
    x = np.sort(data)
    nobs = len(data)
    m_upp = [x[:int(np.round(upp_ * nobs))].mean(0) for upp_ in upp]
    m_low = [x[int(np.round(low_ * nobs)):].mean(0) for low_ in low]
    return m_upp, m_low

def tail_expectation(distr, lb=0, ub=1):
    '''

    lb, ub : integration bounds in probabilities (cdf)
    '''
    from scipy.integrate import quad

    def func(x):
        return distr.ppf(x)

    res = quad(func, lb, ub)
    return res


def skew(data, kind='standard', arg=None):
    '''robust measures of sample skewness

    This implements skewness measures used in Kim and White 2004

    '''
    data = np.asarray(data)
    if kind == 'standard':
        #TODO implement directly
        sk = stats.skew(data)
    elif kind == 'bowley-hinkley':
        if arg is None:
            frac = 0.25
        q1, q2, q3 = quantile(data, [frac, 0.5, 1 - frac])
        sk = (q3 + q1 - 2. * q2) / (q3 - q1)
    elif kind == 'groeneveld-meeden':
        q2 = np.median(data, axis=0)
        sk = (data.mean(0) - q2) / np.abs(data - q2).mean(0)
    elif kind == 'pearson':
        q2 = np.median(data, axis=0)
        sk = (data.mean(0) - q2) / data.std(0)


    return sk


def skew_normal(kind='standard', arg=()):
    return 0

def skew_distr(distr, kind='standard', arg=None):
    if kind == 'standard':
        #TODO implement directly
        sk = distr.stats(moments='s')
    elif kind == 'bowley-hinkley':
        if arg is None:
            frac = 0.25
        else:
            frac = arg
        q1, q2, q3 = distr.ppf([frac, 0.5, 1 - frac])
        sk = (q3 + q1 - 2. * q2) / (q3 - q1)
    elif kind == 'groeneveld-meeden':
        q2 = distr.ppf(0.5)
        mean_ = distr.mean()
        if np.abs(q2 - mean_) < 1e-15:
            #stop early if mean = median
            sk = 0
        else:
            func = lambda x: np.abs(x - q2)
            if hasattr(distr, 'expect'):
                denom = distr.expect(func)
            else:
                denom = distr.dist.expect(func)#, args=distr.args, **distr.kwds)
            sk = (mean_ - q2) / denom
    elif kind == 'pearson':
        q2 = distr.ppf(0.5)
        mean_, var_ = distr.stats()
        if np.abs(q2 - mean_) < 1e-15:
            #stop early if mean = median
            sk = 0
        else:
            sk = (mean_ - q2) / np.sqrt(var_)
    return sk

def kurtosis(data, kind='standard', arg=None, center=True):
    '''robust measures of sample skewness

    This implements skewness measures used in Kim and White 2004

    '''
    centering = {'standard': 3,
                 'moore': 1.23,
                 'hogg': 2.59,
                 'crow-siddiqui': 2.91}

    data = np.asarray(data)
    if kind == 'standard':
        #TODO implement directly
        kurt = stats.kurtosis(data) + 3  # we remove 3 later
    elif kind == 'moore':
        q = quantile(data, np.arange(0, 8.) / 8)
        # I add a leading zero, so I can index as in the paper
        # index 1 is first octile, ...
        kurt = (q[7] - q[5] + q[3] - q[1]) / (q[6] - q[2])

    elif kind == 'hogg':
        if arg is None:
            alpha, beta = 0.05, 0.5
        else:
            alpha, beta = arg

        frac_left = [alpha, beta]
        frac_right = [1. - alpha, 1 - beta]
        (ea_low, eb_low), (ea_upp, eb_upp) = tail_mean(data, frac_left, frac_right)
        print ea_upp, ea_low, eb_upp, eb_low
        kurt = (ea_upp - ea_low) / (eb_upp - eb_low)

    elif kind == 'crow-siddiqui':
        if arg is None:
            alpha, beta = 0.025, 0.25
        else:
            alpha, beta = arg
        frac = [alpha, 1. - alpha, beta, 1 - beta]
        qa1, qa2, qb1, qb2 = quantile(data, frac)
        kurt = (qa2 - qa1) / (qb2 - qb1) # sign correct compared to KW 2004

    if center:
        kurt -= centering[kind]
    return kurt

def kurtosis_distr(distr, kind='standard', arg=None, center=True):
    '''robust measures of sample skewness

    This implements skewness measures used in Kim and White 2004

    '''
    centering = {'standard': 3,
                 'moore': 1.23,
                 'hogg': 2.59,
                 'crow-siddiqui': 2.91 # or 0, see below,
                                #I don't get the result in the paper
                 }

    if kind == 'standard':
        #TODO implement directly
        kurt = distr.stats(moments='k') + 3 # we remove 3 later
    elif kind == 'moore':
        q = distr.ppf(np.arange(0, 8.) / 8)
        # I add a leading zero, so I can index as in the paper
        # index 1 is first octile, ...
        kurt = (q[7] - q[5] + q[3] - q[1]) / (q[6] - q[2])

    elif kind == 'hogg':
        if arg is None:
            alpha, beta = 0.05, 0.5
        else:
            alpha, beta = arg

        func = lambda x: x
        dargs = distr.args
        ea_low = distr.dist.expect(func, args=dargs, ub=distr.ppf(alpha), **distr.kwds) / alpha
        ea_upp = distr.dist.expect(func, args=dargs, lb=distr.isf(alpha), **distr.kwds) / alpha
        eb_low = distr.dist.expect(func, args=dargs, ub=distr.ppf(beta), **distr.kwds) / beta
        eb_upp = distr.dist.expect(func, args=dargs, lb=distr.isf(beta), **distr.kwds) / beta
        print ea_upp, ea_low, eb_upp, eb_low
        kurt = (ea_upp - ea_low) / (eb_upp - eb_low)

    elif kind == 'crow-siddiqui':
        if arg is None:
            alpha, beta = 0.025, 0.25
        else:
            alpha, beta = arg
        frac = [alpha, 1. - alpha, beta, 1 - beta]
        qa1, qa2, qb1, qb2 = distr.ppf(frac)
        #kurt = (qa2 + qa1) / (qb2 - qb1) # equation in KW
        kurt = (qa2 - qa1) / (qb2 - qb1) # this gives numbers in KW

    if center:
        kurt -= centering[kind]
    return kurt

class MixtureUnivariate(stats.distributions.rv_continuous):
    def __init__(self, distr, prob, args, kwds):
        self.distr = distr
        self.prob = [prob, 1 - prob]
        self.args = args
        self.kwds = kwds
        self.dist = self # another Hack for frozen
        super(MixtureUnivariate, self).__init__(name='Mixture', xa=-100, xb=100)

    def rvs(self, size=1):
        nobs1, nobs2 = np.random.multinomial(size, self.prob)
        print nobs1, nobs2
        rvs_ = np.empty(size)
        rvs_.fill(np.nan)
        kwds = {'size' : nobs1}
        kwds.update(self.kwds[0])
        rvs_[:nobs1] = self.distr.rvs(*self.args[0], **kwds)
        kwds = {'size' : nobs2}
        kwds.update(self.kwds[1])
        rvs_[nobs1:] = self.distr.rvs(*self.args[1], **kwds)
        return rvs_


    def _pdf(self, x):
        #for dist in enumerate(self.distr):
        pdf_ = self.prob[0] * self.distr.pdf(x, *self.args[0], **self.kwds[0])
        pdf_ += self.prob[1] * self.distr.pdf(x, *self.args[1], **self.kwds[1])
        return pdf_

    def _cdf(self, x):
        #for dist in enumerate(self.distr):
        cdf_ = self.prob[0] * self.distr.cdf(x, *self.args[0], **self.kwds[0])
        cdf_ += self.prob[1] * self.distr.cdf(x, *self.args[1], **self.kwds[1])
        return cdf_

    def expect(self, func, lb=None, ub=None, **kwds):
        # Hack to work around usage of frozen distribution
        return super(MixtureUnivariate, self).expect(func, lb=lb, ub=ub)

example = 't'

# example lognormal
distr = stats.lognorm(0.4, scale=np.exp(1))
loc = distr.mean()
distr = stats.lognorm(0.4, loc=loc, scale=np.exp(1))
distr = stats.norm()
distr = stats.t(5)

if example == 'mixture':
    # for examples from Kim and White 2004
    #distr = MixtureUnivariate(distr, prob, args, kwds)
    distr_m = stats.t
    kwds1 = dict(loc=0, scale=1)
    kwds2 = dict(loc=-7, scale=10)
    df = 5
    distr = MixtureUnivariate(distr_m, 0.9988, [(df,), (df,)], kwds=[kwds1, kwds2])

    distr_m = stats.norm
    distr = MixtureUnivariate(distr_m, 0.9988, [(), ()], kwds=[kwds1, kwds2])

x_rvs = distr.rvs(size=500000)

sk_kinds = ['standard', 'bowley-hinkley', 'groeneveld-meeden', 'pearson']
kurt_kinds = ['standard', 'moore', 'hogg', 'crow-siddiqui']

print '\nsample lognormal'
for kind in sk_kinds:
    print skew(x_rvs, kind=kind, arg=None)

print '\ndistr lognormal'
for kind in sk_kinds:
    print skew_distr(distr, kind=kind, arg=None)

print '\nsample lognormal'
for kind in kurt_kinds:
    print kurtosis(x_rvs, kind=kind, arg=None, center=True)

print '\ndistr lognormal'
for kind in kurt_kinds:
    print kurtosis_distr(distr, kind=kind, arg=None, center=True)

te_l = tail_expectation(distr, lb=1.-0.05, ub=1)
print te_l, te_l[0] / 0.05
te_u = tail_expectation(distr, lb=0, ub=0.05)
print te_u, te_u[0] / 0.05
