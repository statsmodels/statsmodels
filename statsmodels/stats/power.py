# -*- coding: utf-8 -*-
"""Statistical power, solving for nobs, ... - trial version

Created on Sat Jan 12 21:48:06 2013

Author: Josef Perktold

Example
roundtrip - root with respect to all variables

       calculated, desired
nobs   33.367204205 33.367204205
effect 0.5 0.5
alpha  0.05 0.05
beta   0.8 0.8

"""

import numpy as np
from scipy import stats

#power
d = 0.68  #effect size
nobs = 20  #observations
alpha = 0.05
alpha_two = alpha / 2.
crit = stats.t.isf(alpha_two, nobs-1) #critical value at df=nobs
pow_t = stats.nct(nobs-1, d*np.sqrt(nobs)).sf(stats.t.isf(alpha_two, nobs-1))
pow_f = stats.ncf(1, nobs-1, d**2*nobs).sf(stats.f.isf(alpha, 1, nobs-1))


def ttest_power(effect_size, nobs, alpha, df=None, alternative='two-sided'):
    '''Calculate power of a ttest
    '''
    d = effect_size
    if df is None:
        df = nobs - 1

    if alternative in ['two-sided', '2s']:
        alpha_ = alpha / 2.  #no inplace changes, doesn't work
    elif alternative in ['one-sided', '1s']:
        alpha_ = alpha
    else:
        raise ValueError("alternative has to be 'two-sided' or 'one-sided'")

    pow_ = stats.nct(df, d*np.sqrt(nobs)).sf(stats.t.isf(alpha_, df))
    return pow_


from scipy import optimize

def solve_power_nobs(effect_size, alpha, beta):
    '''solve for required sample size for t-test to obtain power beta
    '''

    pow_n = lambda nobs: ttest_power(effect_size, nobs, alpha) - beta
    return optimize.fsolve(pow_n, 10).item()

effect_size, alpha, beta = 0.5, 0.05, 0.8
nobs_p = solve_power_nobs(effect_size, alpha, beta)
print nobs_p
print ttest_power(effect_size, nobs_p, alpha), beta


#generic implementation
#----------------------

#to be reused for other tests

def ttest_power_id(effect_size=None, nobs=None, alpha=None, beta=None,
                   df=None, alternative='two-sided'):
    '''identity for power calculation, should be zero

    '''
    return ttest_power(effect_size=effect_size, nobs=nobs, alpha=alpha,
                       df=df, alternative=alternative) - beta

#import functools as ft   #too limited

start_ttp = dict(effect_size=0.01, nobs=10., alpha=0.15, beta=0.6,
                 nobs1=10, ratio=1)
#TODO: nobs1 and rati are for ttest_ind,
#      need start_ttp for each test/class separately, added default start_value
#possible rootfinding problem for effect_size, starting small seems to work

def solve_ttest_power(**kwds):
    '''solve for any one of the parameters of a t-test

    for t-test the keywords are:
        effect_size, nobs, alpha, beta

    exactly one needs to be `None`, all others need numeric values

    '''
    #TODO: maybe use explicit kwds,
    #      nicer but requires inspect? and not generic across tests
    #TODO: use explicit calculation for beta=None
    key = [k for k,v in kwds.iteritems() if v is None]
    #print kwds, key;
    if len(key) != 1:
        raise ValueError('need exactly one keyword that is None')
    key = key[0]

    def func(x):
        kwds[key] = x
        return ttest_power_id(**kwds)

    #start_value = start_ttp.get(key, 0.9)
    #TODO: I'm using the following so I get a warning when start_ttp is not defined
    try:
        start_value = start_ttp[key]
    except KeyError:
        start_value = 0.9
        print 'Warning: using default start_value'

    return optimize.fsolve(func, start_value).item() #scalar


print '\nroundtrip - root with respect to all variables'
print '\n       calculated, desired'

print 'nobs  ', solve_ttest_power(effect_size=effect_size, nobs=None, alpha=alpha, beta=beta), nobs_p
print 'effect', solve_ttest_power(effect_size=None, nobs=nobs_p, alpha=alpha, beta=beta), effect_size

print 'alpha ', solve_ttest_power(effect_size=effect_size, nobs=nobs_p, alpha=None, beta=beta), alpha
print 'beta  ', solve_ttest_power(effect_size=effect_size, nobs=nobs_p, alpha=alpha, beta=None), beta


#class based implementation
#--------------------------

class Power(object):
    '''Statistical Power calculations

    so far this could all be class methods
    '''

    def power(self, *args, **kwds):
        raise NotImplementedError

    def power_identity(self, *args, **kwds):
        beta_ = kwds.pop('beta')
        return self.power(*args, **kwds) - beta_

    def solve_power(self, **kwds):
        '''solve for any one of the parameters of a t-test

        for t-test the keywords are:
            effect_size, nobs, alpha, beta

        exactly one needs to be `None`, all others need numeric values

        '''
        #TODO: maybe use explicit kwds,
        #      nicer but requires inspect? and not generic across tests
        #TODO: use explicit calculation for beta=None
        key = [k for k,v in kwds.iteritems() if v is None]
        #print kwds, key;
        if len(key) != 1:
            raise ValueError('need exactly one keyword that is None')
        key = key[0]

        def func(x):
            kwds[key] = x
            return self.power_identity(**kwds)

        #TODO: I'm using the following so I get a warning when start_ttp is not defined
        try:
            start_value = start_ttp[key]
        except KeyError:
            start_value = 0.9
            print 'Warning: using default start_value'

        return optimize.fsolve(func, start_value).item() #scalar

class TTestPower(Power):
    '''Statistical Power calculations for one sample or paired sample t-test

    '''

    def power(self, effect_size, nobs, alpha, df=None,
              alternative='two-sided'):
        return ttest_power(effect_size, nobs, alpha, df=None,
                           alternative=alternative)

class TTestIndPower(Power):
    '''Statistical Power calculations for t-test for two independent sample

    check that nobs in non-centrality parameter is correct

    currently only uses pooled variance

    '''


    def power(self, effect_size, nobs1, alpha, ratio=1, df=None,
              alternative='two-sided'):
        #pooled variance
        nobs2 = nobs1*ratio
        df = (nobs1 - 1 + nobs2 - 1)
        nobs = 1./ (1. / nobs1 + 1. / nobs2)
        return ttest_power(effect_size, nobs, alpha, df=df,
                           alternative=alternative)

tt_solve_power = TTestPower().solve_power

ttest_pow = TTestPower()
print '\nroundtrip - root with respect to all variables'
print '\n       calculated, desired'

print 'nobs  ', ttest_pow.solve_power(effect_size=effect_size, nobs=None, alpha=alpha, beta=beta), nobs_p
print 'effect', ttest_pow.solve_power(effect_size=None, nobs=nobs_p, alpha=alpha, beta=beta), effect_size

print 'alpha ', ttest_pow.solve_power(effect_size=effect_size, nobs=nobs_p, alpha=None, beta=beta), alpha
print 'beta  ', ttest_pow.solve_power(effect_size=effect_size, nobs=nobs_p, alpha=alpha, beta=None), beta

print '\nroundtrip - root with respect to all variables'
print '\n       calculated, desired'

print 'nobs  ', tt_solve_power(effect_size=effect_size, nobs=None, alpha=alpha, beta=beta), nobs_p
print 'effect', tt_solve_power(effect_size=None, nobs=nobs_p, alpha=alpha, beta=beta), effect_size

print 'alpha ', tt_solve_power(effect_size=effect_size, nobs=nobs_p, alpha=None, beta=beta), alpha
print 'beta  ', tt_solve_power(effect_size=effect_size, nobs=nobs_p, alpha=alpha, beta=None), beta

print '\none sided'
nobs_p1 = tt_solve_power(effect_size=effect_size, nobs=None, alpha=alpha, beta=beta, alternative='1s')
print 'nobs  ', nobs_p1
print 'effect', tt_solve_power(effect_size=None, nobs=nobs_p1, alpha=alpha, beta=beta, alternative='1s'), effect_size
print 'alpha ', tt_solve_power(effect_size=effect_size, nobs=nobs_p1, alpha=None, beta=beta, alternative='1s'), alpha
print 'beta  ', tt_solve_power(effect_size=effect_size, nobs=nobs_p1, alpha=alpha, beta=None, alternative='1s'), beta

#start_ttp = dict(effect_size=0.01, nobs1=10., alpha=0.15, beta=0.6)

ttind_solve_power = TTestIndPower().solve_power

print '\nroundtrip - root with respect to all variables'
print '\n       calculated, desired'

nobs_p2 = ttind_solve_power(effect_size=effect_size, nobs1=None, alpha=alpha, beta=beta)
print 'nobs  ', nobs_p2
print 'effect', ttind_solve_power(effect_size=None, nobs1=nobs_p2, alpha=alpha, beta=beta), effect_size

print 'alpha ', ttind_solve_power(effect_size=effect_size, nobs1=nobs_p2, alpha=None, beta=beta), alpha
print 'beta  ', ttind_solve_power(effect_size=effect_size, nobs1=nobs_p2, alpha=alpha, beta=None), beta
print 'ratio  ', ttind_solve_power(effect_size=effect_size, nobs1=nobs_p2, alpha=alpha, beta=beta, ratio=None), 1

print '\ncheck ratio'
print 'smaller beta', ttind_solve_power(effect_size=effect_size, nobs1=nobs_p2, alpha=alpha, beta=0.7, ratio=None), '< 1'
print 'larger beta ', ttind_solve_power(effect_size=effect_size, nobs1=nobs_p2, alpha=alpha, beta=0.9, ratio=None), '> 1'
