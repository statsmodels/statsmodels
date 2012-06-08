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

#The following implements parts of
#Algorithm AS 248: Empirical Distribution Function Goodness-of-Fit Tests
#Author(s): Charles S. Davis and Michael A. Stephens
#Source: Journal of the Royal Statistical Society. Series C (Applied Statistics), Vol. 38, No. 3(1989), pp. 535-543
#Published by: Blackwell Publishing for the Royal Statistical Society
#Stable URL: http://www.jstor.org/stable/2347751 .

#F(x) completely specified
def modify_fs_d(stat, nobs):
    return stat * (np.sqrt(nobs) + 0.12 + 0.11 / np.sqrt(nobs))
crit_fs_d = [1.138, 1.224, 1.358, 1.480, 1.628]
def modify_fs_v(stat, nobs):
    return stat * (np.sqrt(nobs) + 0.155 + 0.24 / np.sqrt(nobs))
crit_fs_v = [1.537, 1.620, 1.747, 1.862, 2.001]
def modify_fs_w2(stat, nobs):
    return (stat -0.4 / nobs + 0.6 / nobs**2) * (1.0 + 1.0 / nobs)
crit_fs_w2 = [0.284, 0.347, 0.461, 0.581, 0.743]
def modify_fs_u2(stat, nobs):
    return (stat -0.1 / nobs + 0.1 / nobs**2) * (1.0 + 0.8 / nobs)
crit_fs_u2 = [0.131, 0.152, 0.187, 0.222, 0.268]
def modify_fs_a2(stat, nobs):
    return stat  #unmodified (for all nobs > 5)
crit_fs_a2 = [1.610, 1.933, 2.492, 3.070, 3.857]

#F(x) is the normal distribution, mu and sigma^2 unspecified
def modify_normal_d(stat, nobs):
    return stat * (np.sqrt(nobs) - 0.01 + 0.85 / np.sqrt(nobs))
crit_normal_d = [0.775, 0.819, 0.895, 0.955, 1.035]
def modify_normal_v(stat, nobs):
    return stat * (np.sqrt(nobs) + 0.05 + 0.82 / np.sqrt(nobs))
crit_normal_v = [1.320, 1.386, 1.489, 1.585, 1.693]
def modify_normal_w2(stat, nobs):
    return stat * (1.0 + 0.5 / nobs)
crit_normal_w2 = [0.091, 0.104, 0.126, 0.148, 0.179]
def modify_normal_u2(stat, nobs):
    return stat * (1.0 + 0.5 / nobs)
crit_normal_u2 = [0.085, 0.096, 0.117, 0.136, 0.164]
def modify_normal_a2(stat, nobs):
    return stat * (1.0 + 0.75 / nobs + 2.25 / nobs**2)
crit_normal_a2 = [0.561, 0.631, 0.752, 0.873, 1.035]

#F(x) is the exponential distribution, 0 unspecified
def modify_expon_d(stat, nobs):
    return (stat - 0.2 / nobs) * (np.sqrt(nobs) + 0.26 + 0.5 / np.sqrt(nobs))
crit_expon_d = [0.926, 0.995, 1.094, 1.184, 1.298]
def modify_expon_v(stat, nobs):
    return (stat - 0.2 / nobs) * (np.sqrt(nobs) + 0.24 + 0.35 / np.sqrt(nobs))
crit_expon_v = [1.445, 1.527, 1.655, 1.774, 1.910]
def modify_expon_w2(stat, nobs):
    return stat * (1.0 + 0.16 / nobs)
crit_expon_w2 = [0.148, 0.175, 0.222, 0.271, 0.338]
def modify_expon_u2(stat, nobs):
    return stat * (1.0 + 0.16 / nobs)
crit_expon_u2 = [0.112, 0.129, 0.159, 0.189, 0.230]
def modify_expon_a2(stat, nobs):
    return stat * (1.0 + 0.6 / nobs)
crit_expon_a2 = [0.916, 1.062, 1.321, 1.591, 1.959]


#I didn't look what this is used for
#test statistics are inherited from gof_new, i.e. Stephens' 1970 paper
tmp = (0.01, 0.05, 0.1, 0.11, 0.12, 0.155, 0.16, 0.2, 0.24)
pt01, pt05, pt1, pt11, pt12, pt155, pt16, pt2, pt24 = tmp
tmp = (0.26, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 0.8, 0.82, 0.85)
pt26, pt3, pt35, pt4, pt5, pt6, pt75, pt8, pt82, pt85 = tmp

#A2
#pvalue_st(A2, acut3, acoef3)
#wcut2 = wcut3 = ucut2 = ucut3 = acut2 = acut3
cut = np.array([0.0275, 0.051, 0.092, 0.035, 0.074, 0.160, 0.0262, 0.048, 0.094,
         0.029, 0.062, 0.120, 0.200, 0.340, 0.600, 0.260, 0.510, 0.950]).reshape(-1,3)
wcut2, wcut3, ucut2, ucut3, acut2, acut3 = cut
cutoff_normal_w2, cutoff_normal_u2, cutoff_normal_a2 = cut[::2]
cutoff_expon_w2, cutoff_expon_u2, cutoff_expon_a2 = cut[1::2]

acut3 = cut[-1]

coef_normal_w2 = wcoef2 = np.array([-13.953, 775.5, -12542.61, -5.903, 179.546, -1515.29,
                   0.886, -31.62, 10.897, 1.111, -34.242, 12.832]
                   ).reshape(4, 3, order='c')
coef_expon_w2 = wcoef3 = np.array([-11.334, 459.098, -5652.1, -5.779, 132.89, -866.58, 0.586,
                   -17.87, 7.417, 0.447, -16.592, 4.849]
                   ).reshape(4, 3, order='c')
coef_normal_u2 = ucoef2 = np.array([-13.642, 766.31, -12432.74, -6.3328, 214.57, -2022.28,
                   0.8510, -32.006, -3.45, 1.325, -38.918, 16.45]
                   ).reshape(4, 3, order='c')
coef_expon_u2 = ucoef3 = np.array([-11.703, 542.5, -7574.59, -6.3288, 178.1, -1399.49, 0.8071,
                   -25.166, 8.44, 0.7663, -24.359, 4.539]
                   ).reshape(4, 3, order='c')
coef_normal_a2 = acoef2 = np.array([-13.436, 101.14, -223.73, -8.318, 42.796, -59.938, 0.9177,
                   -4.279, -1.38, 1.2937, -5.709, 0.0186]
                   ).reshape(4, 3, order='c')
coef_expon_a2 = acoef3 = np.array([-12.2204, 67.459, -110.3, -6.1327, 20.218, -18.663, 0.9209, -3.353,
          0.300, 0.731, -3.009, 0.15]).reshape(4, 3, order='c')


modify_expon = {}
crit_expon = {}
coef_expon = {}
cutoff_expon = {}
distr = 'expon'
for test_ in 'd v w2 u2 a2'.split():
    modify_expon.setdefault(test_, locals()["modify_" + distr + "_" + test_])
    crit_expon.setdefault(test_, locals()["crit_" + distr + "_" + test_])
for test_ in 'w2 u2 a2'.split():
    cutoff_expon.setdefault(test_, locals()["cutoff_" + distr + "_" + test_])
    coef_expon.setdefault(test_, locals()["coef_" + distr + "_" + test_])

modify_normal = {}
crit_normal = {}
coef_normal = {}
cutoff_normal = {}
distr = 'normal'
for test_ in 'd v w2 u2 a2'.split():
    modify_normal.setdefault(test_, locals()["modify_" + distr + "_" + test_])
    crit_normal.setdefault(test_, locals()["crit_" + distr + "_" + test_])

for test_ in 'w2 u2 a2'.split():
    cutoff_normal.setdefault(test_, locals()["cutoff_" + distr + "_" + test_])
    coef_normal.setdefault(test_, locals()["coef_" + distr + "_" + test_])

modify_fs = {}
crit_fs = {}
distr = 'fs'
for test_ in 'd v w2 u2 a2'.split():
    modify_fs.setdefault(test_, locals()["modify_" + distr + "_" + test_])
    crit_fs.setdefault(test_, locals()["crit_" + distr + "_" + test_])




cutoff = acut3
coef = acoef3

t = 0.01#1.32 #0.724 #1.959

def pvalue_st(t, cutoff, coef):
    #vectorized
    i = 3 * np.ones(np.shape(t), int)
    i = np.searchsorted(cutoff, t)
    pval = np.exp(coef[i,0] + t * (coef[i,1] + t * coef[i,2]))
    if pval.shape == ():
        if i < 2:
            pval = 1 - pval
    else:
        pval[i<2] = 1 - pval[i<2]
    return i, pval

def pvalue_expon(t, test='a2'):
    return pvalue_st(t, cutoff_expon[test], coef_expon[test])[1]

def pvalue_normal(t, test='a2'):
    return pvalue_st(t, cutoff_normal[test], coef_normal[test])[1]

def pvalue_interp(t, test='a2', dist='normal'):
    #vectorized
    if np.shape(t) == ():
        scalar = True
    t = np.atleast_1d(t)
    if dist == 'normal':
        crit = crit_normal[test]
    elif dist == 'expon':
        crit = crit_expon[test]
    elif dist == 'fs':
        crit = crit_fs[test]
    else:
        raise NotImplementedError('currently only normal, exponential and' + \
                                  'fully specified are supported')

    interp = interpolate.interp1d(crit, alpha)
    if not np.all(np.diff(crit) > 0): #check for increasing
        raise NotImplementedError('please call us and tell np.diff(crit)<=0')

    tlow = t < crit[0]
    tupp = t > crit[-1]
    mask = (~tlow) & (~tupp)
    pval = np.empty(t.shape, float)
    pval.fill(np.nan)
    pval[tlow] = alpha[0]
    pval[tupp] = alpha[-1]
    pval[mask] = interp(t[mask])
    if scalar:
        pval = np.squeeze(pval)[()]  #return scalar
    return pval

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
            if not ti == 'v':  #skip for Kuiper
                assert_array_less(0.11, ad[1])
        elif oth[1] == 0.01:  #upper boundary for pval of d and v
            #if not ti == 'v':  #skip for Kuiper
            assert_array_less(ad[1], 0.01)
        else:
            #assert_almost_equal(ad[1], oth[1], 1)
            #assert_array_less(np.abs(ad[1] / oth[1] - 1), 0.6) #25)
            #assert_array_less(np.abs(oth[1] - ad[1]) / ad[1]**2, 1)
            assert_array_less(np.abs(ad[1] - oth[1]), 0.01 + 0.6 * oth[1]) #25)

