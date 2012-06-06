# -*- coding: utf-8 -*-
"""Goodness of fit tests with estimated parameters

Created on Wed Jun 06 11:53:07 2012

Author: Josef Perktold
"""

import numpy as np
from scipy import stats
from statsmodels.stats.lilliefors import ksstat

def gof_mc(data, distr, test_func, fit_func=None, n_repl=100, side="upper"):
    '''gof test with estimated parameter and Monte Carlo p-value

    Parameters
    ----------
    data : ndarray, 1d
       data sample
    distr : instance of distribution
       distribution in the pattern of scipy stats with rvs method, and
       fit method (if fit_func is None), and cdf method for ksstat
    test_func : function
       function that calculates the test statistic
       currently arguments are specific for ksstat
    fit_func : None or function
       if None, then the fit method of the distribution instance is called,
       otherwise this function is used for fitting
    n_repl : int
       number of replications to use in the Monte Carlo for the p-value.
    side : "upper"
       which side of the test statistic to use,
       currently only upper

    Returns
    -------
    pvalue : float
       Monte Carlo p-value
    stat : float
       test statistic of the data
    res : ndarray
       test statistic for all Monte Carlo results
       will be dropped or made optional

    Notes
    -----
    first version: ksstat specific parts
    Short Monte Carlo shows that results is correctly sized.


    '''
    nobs = len(data)
    if fit_func is None:
        fit_func = distr.fit

    def gof_fit(x):
        est = fit_func(x)
        #cdf_func = lambda x: distr.cdf
        #the arguments here are specific to ksstat
        stat = test_func(x, distr.cdf, args=est)
        return stat, est

    stat, est = gof_fit(data)

    res = np.empty(n_repl)
    res.fill(np.nan)
    for ii in xrange(n_repl):
        x_rvs = distr.rvs(*est, **dict(size=nobs))
        #could just count >stat instead
        res[ii] = gof_fit(x_rvs)[0]

    if side == 'upper':
        return (res>stat).mean(), stat, res
    else:
        raise NotImplementedError


#Example: exponential distribution with estimated shape parameter

rvs = stats.expon.rvs(loc=0, scale=5, size=200)

fit_func = lambda x: stats.expon.fit(x, floc=0)

p, st, res = gof_mc(rvs, stats.expon, ksstat, fit_func=fit_func)

import time
t0 = time.time()
mcres = []
for _ in xrange(1000):
    rvs = stats.expon.rvs(loc=0, scale=5, size=200)
    p, st, res = gof_mc(rvs, stats.expon, ksstat, fit_func=fit_func)
    mcres.append([p, st])

t1 = time.time()
print "time for mc", t1 - t0

mcres = np.asarray(mcres)
print (mcres[:, :1] < 0.05).mean()

print np.linspace(0,1,11)
print (mcres[:, :1] < np.linspace(0,1,11)).mean(0)

import matplotlib.pyplot as plt
#distribution of pvalues
_ = plt.hist(mcres[:,0], bins=100, normed=True)
#distribution of test statistic
#_ = plt.hist(mcres[:,1], bins=100, normed=True)
#plt.show()


''' slow but size looks ok
>>> execfile('try_gof_est.py')
time for mc 228.508000135
0.045
[ 0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1. ]
[ 0.  0.098  0.203  0.328  0.426  0.537  0.634  0.728  0.8    0.886
  0.99 ]

second run
time for mc 234.578000069
0.051
[ 0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1. ]
[ 0.  0.098  0.2    0.32   0.395  0.486  0.599  0.707  0.795  0.895
  0.988]

'''
