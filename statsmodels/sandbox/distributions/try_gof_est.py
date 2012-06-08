# -*- coding: utf-8 -*-
"""Goodness of fit tests with estimated parameters

Created on Wed Jun 06 11:53:07 2012

Author: Josef Perktold
"""

from numpy.testing import assert_equal, assert_almost_equal

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
for _ in xrange(1):  #1000
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

third run
time for mc 257.38499999
0.048
[ 0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1. ]
[ 0.     0.097  0.191  0.291  0.378  0.494  0.595  0.691  0.779  0.878
  0.995]

'''


print pvalue_st(t, cutoff, coef)

alpha = [0.15, 0.10, 0.05, 0.025, 0.01]
a2_exp = [0.916, 1.062, 1.321, 1.591, 1.959]

for t, aa in zip(a2_exp, alpha):
    print pvalue_st(t, cutoff, coef), aa

for t, aa in zip(crit_normal_a2, alpha):
    print pvalue_st(t, acut2, acoef2), aa

#compare with linear interpolation
from scipy import interpolate
nu2 = interpolate.interp1d(crit_normal_u2, alpha)
#check
assert_equal(nu2(crit_normal_u2), alpha)

for b in np.linspace(0, 1, 11):
    z = crit_normal_u2[:-1] + b * np.diff(crit_normal_u2)
    print (nu2(z) - [pvalue_st(zz, ucut2, ucoef2)[1] for zz in z])
    assert_almost_equal(nu2(z),
                        [pvalue_st(zz, ucut2, ucoef2)[1] for zz in z],
                        decimal=2)

for te in ['w2', 'u2', 'a2']:
    assert_almost_equal([pvalue_expon(z, te) for z in crit_expon[te]],
                         alpha, decimal=2)
    assert_almost_equal([pvalue_normal(z, te) for z in crit_normal[te]],
                         alpha, decimal=2)

from statsmodels.sandbox.distributions import gof_new as gofn

class GOFUniform(gofn.GOF):

    def __init__(self, rvs):
        if np.min(rvs) < 0 or np.max(rvs) > 1:
            raise ValueError('some values are out of bounds')

        vals = np.sort(rvs)
        cdfvals = vals
        self.nobs = len(vals)
        self.vals_sorted = vals
        self.cdfvals = cdfvals

class GOFNormal(GOFUniform):

    def __init__(self, rvs, ddof=1):
        rvs = np.asarray(rvs)
        vals = stats.norm.cdf((rvs - rvs.mean()) / rvs.std(ddof=ddof))
        super(GOFNormal, self).__init__(vals)

    def get_test(self, testid='a2', pvals='davisstephens89upp'):
        '''get p-value for a test

        uses Stephens approximation formula for 'w2', 'u2', 'a2' and
        interpolated table for 'd', 'v'

        '''

        stat = getattr(self, testid.replace('2', 'squ'))
        stat_modified = modify_normal[testid](stat, self.nobs)
        if (testid in ['w2', 'u2', 'a2']) and pvals == 'davisstephens89upp':
            pval = pvalue_normal(stat_modified, testid)
        elif (testid in ['d', 'v']) or pvals == 'interpolated':
            pval = pvalue_interp(stat_modified, test=testid, dist='normal')
        else:
            raise NotImplementedError
        return stat, pval, stat_modified

class GOFExpon(GOFUniform):
    '''Goodness-of-fit tests for exponential distribution with estimated scale


    available tests

    "d" Kolmogorov-Smirnov
    "v" Kuiper
    "w2" Cramer-Von Mises
    "u2" Watson U^2 statistic, a modified W^2 test statistic
    "a2" Anderson-Darling A^2

    In genral "a2" is recommended as the most powerful test of the above.


    '''

    def __init__(self, rvs):
        rvs = np.asarray(rvs)
        vals = 1 - np.exp(-rvs / rvs.mean())
        super(GOFExpon, self).__init__(vals)

    def get_test(self, testid='a2', pvals='davisstephens89upp'):
        '''get p-value for a test

        '''
        #mostly copy paste from normal, not DRY
        stat = getattr(self, testid.replace('2', 'squ'))
        stat_modified = modify_expon[testid](stat, self.nobs)
        if (testid in ['w2', 'u2', 'a2']) and pvals == 'davisstephens89upp':
            pval = pvalue_expon(stat_modified, testid)
        elif (testid in ['d', 'v']) or pvals == 'interpolated':
            pval = pvalue_interp(stat_modified, test=testid, dist='expon')
        else:
            raise NotImplementedError
        return stat, pval, stat_modified

ge = GOFExpon(rvs)
print ge.get_test()  #default is 'a2'
print ge.get_test('w2')
print ge.get_test('u2')
rvsn = np.random.randn(50)
print GOFExpon(rvsn**2).get_test()

#rvsm = (np.random.randn(50, 2) * [1, 2]).sum(1)
rvsm = (np.random.randn(50, 2) * [1, 2]).ravel()
gn = GOFNormal(rvsm)
print gn.get_test()
from statsmodels.stats import diagnostic as dia
print dia.normal_ad(rvsm)
print gn.get_test('d')
print dia.lillifors(rvsm)

#created for copying to R with
#np.random.seed(9768)
#xx = np.round(1000 * stats.expon.rvs(size=20), 3).astype(int)
xx = np.array([1580,  179, 1471,  328,  492, 1008, 1412, 4820, 2840,  559,
               223, 871,  791,  837, 1722, 1247,  985, 4378,  620,  530])
x = xx / 1000.

ge = GOFExpon(x)

b_list = [0, 0.3, 0.35, 0.5, 2]  #chosen to get a good spread of pvalues

#the following doesn't work well because for some case the tests differ too
#much
for b in b_list:
    ge2 = GOFExpon(x + b * x**2)
    ad = ge2.get_test('a2')
    for ti in ['d', 'v', 'w2', 'u2']:
        oth = ge2.get_test(ti)
        #check pvalues
        if oth[1] == 0.15:  #upper boundary for pval of d and v
            if not (ti == 'v' and b in [0.5]):  #skip one test for Kuiper
                assert_array_less(0.11, ad[1])
        elif oth[1] == 0.01:  #upper boundary for pval of d and v
            #if not ti == 'v':  #skip for Kuiper
            assert_array_less(ad[1], 0.01)
        else:
            #assert_almost_equal(ad[1], oth[1], 1)
            #assert_array_less(np.abs(ad[1] / oth[1] - 1), 0.6) #25)
            #assert_array_less(np.abs(oth[1] - ad[1]) / ad[1]**2, 1)
            assert_array_less(np.abs(ad[1] - oth[1]), 0.01 + 0.6 * oth[1]) #25)

#b in rows, ti in columns
res_r = np.array([0.1564240118194638, 0.09436760924796966, 0.6314329797982694,
                  0.1793511287764876, 0.1506515020772339, 0.887774219046726,
                  0.184369741403067, 0.1697126047249459, 0.978323412566048,
                  0.2055857239850029, 0.225018414134194, 1.247031661161905,
                  0.285678730877463, 0.510830687544145, 2.808327043589259]
                  ) #.reshape(-1,3)


res_gof = [ GOFExpon(x + b * x**2).get_test(ti)[0] for b in b_list for ti in ['d', 'w2', 'a2']]

assert_almost_equal(res_gof, res_r, 7)
