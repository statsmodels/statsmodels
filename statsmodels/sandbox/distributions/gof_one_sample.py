'''More Goodness of fit tests

contains

GOF : 1 sample gof tests based on Stephens 1970, plus AD A^2
bootstrap : vectorized bootstrap p-values for gof test with fitted parameters


Created : 2011-05-21
Author : Josef Perktold

parts based on ks_2samp and kstest from scipy.stats.stats
(license: Scipy BSD, but were completely rewritten by Josef Perktold)


References
----------

'''

import numpy as np

from scipy import stats
#from scipy.stats import distributions

from statsmodels.tools.decorators import cache_readonly

from _gof_stephens2 import (pvalue_normal, pvalue_expon, pvalue_interp,
                            modify_normal, modify_expon)
from _gof_stephens70 import gof_pvals




class GOF(object):
    '''One Sample Goodness of Fit tests

    includes Kolmogorov-Smirnov D, D+, D-, Kuiper V, Cramer-von Mises W^2, U^2 and
    Anderson-Darling A, A^2. The p-values for all tests except for A^2 are based on
    the approximatiom given in Stephens 1970. A^2 has currently no p-values. For
    the Kolmogorov-Smirnov test the tests as given in scipy.stats are also available
    as options.




    design: I might want to retest with different distributions, to calculate
    data summary statistics only once, or add separate class that holds
    summary statistics and data (sounds good).




    '''




    def __init__(self, rvs, cdf, args=(), N=20):
        if isinstance(rvs, basestring):
            #cdf = getattr(stats, rvs).cdf
            if (not cdf) or (cdf == rvs):
                cdf = getattr(distributions, rvs).cdf
                rvs = getattr(distributions, rvs).rvs
            else:
                raise AttributeError('if rvs is string, cdf has to be the same distribution')


        if isinstance(cdf, basestring):
            cdf = getattr(distributions, cdf).cdf
        if callable(rvs):
            kwds = {'size':N}
            vals = np.sort(rvs(*args,**kwds))
        else:
            vals = np.sort(rvs)
            N = len(vals)
        cdfvals = cdf(vals, *args)

        self.nobs = N
        self.vals_sorted = vals
        self.cdfvals = cdfvals



    @cache_readonly
    def d_plus(self):
        nobs = self.nobs
        cdfvals = self.cdfvals
        return (np.arange(1.0, nobs+1)/nobs - cdfvals).max()

    @cache_readonly
    def d_minus(self):
        nobs = self.nobs
        cdfvals = self.cdfvals
        return (cdfvals - np.arange(0.0, nobs)/nobs).max()

    @cache_readonly
    def d(self):
        return np.max([self.d_plus, self.d_minus])

    @cache_readonly
    def v(self):
        '''Kuiper'''
        return self.d_plus + self.d_minus

    @cache_readonly
    def wsqu(self):
        '''Cramer von Mises'''
        nobs = self.nobs
        cdfvals = self.cdfvals
        #use literal formula, TODO: simplify with arange(,,2)
        wsqu = ((cdfvals - (2. * np.arange(1., nobs+1) - 1)/nobs/2.)**2).sum() \
               + 1./nobs/12.
        return wsqu

    @cache_readonly
    def usqu(self):
        nobs = self.nobs
        cdfvals = self.cdfvals
        #use literal formula, TODO: simplify with arange(,,2)
        usqu = self.wsqu - nobs * (cdfvals.mean() - 0.5)**2
        return usqu

    @cache_readonly
    def a(self):
        nobs = self.nobs
        cdfvals = self.cdfvals

        #one loop instead of large array
        msum = 0
        for j in xrange(1,nobs):
            mj = cdfvals[j] - cdfvals[:j]
            mask = (mj > 0.5)
            mj[mask] = 1 - mj[mask]
            msum += mj.sum()

        a = nobs / 4. - 2. / nobs * msum
        return a

    @cache_readonly
    def asqu(self):
        '''Stephens 1974, doesn't have p-value formula for A^2'''
        nobs = self.nobs
        cdfvals = self.cdfvals

        asqu = -((2. * np.arange(1., nobs+1) - 1) *
                (np.log(cdfvals) + np.log(1-cdfvals[::-1]) )).sum()/nobs - nobs

        return asqu


    def get_test(self, testid='d', pvals='stephens70upp'):
        '''

        '''
        #print gof_pvals[pvals][testid]
        stat = getattr(self, testid)
        if pvals == 'stephens70upp':
            return gof_pvals[pvals][testid](stat, self.nobs), stat
        else:
            return gof_pvals[pvals][testid](stat, self.nobs)



class GOFUniform(GOF):

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






def gof_mc(randfn, distr, nobs=100):
    #print '\nIs it correctly sized?'
    from collections import defaultdict
    all_gofs = ['d', 'd_plus', 'd_minus', 'v', 'wsqu', 'usqu', 'a']

    results = defaultdict(list)
    for i in xrange(1000):
        rvs = randfn(nobs)
        goft = GOF(rvs, distr)
        for ti in all_gofs:
            results[ti].append(goft.get_test(ti, 'stephens70upp')[0][1])

    resarr = np.array([results[ti] for ti in all_gofs])
    print '         ', '      '.join(all_gofs)
    print 'at 0.01:', (resarr < 0.01).mean(1)
    print 'at 0.05:', (resarr < 0.05).mean(1)
    print 'at 0.10:', (resarr < 0.1).mean(1)

def asquare(cdfvals, axis=0):
    '''vectorized Anderson Darling A^2, Stephens 1974'''
    ndim = len(cdfvals.shape)
    nobs = cdfvals.shape[axis]
    slice_reverse = [slice(None)] * ndim  #might make copy if not specific axis???
    islice = [None] * ndim
    islice[axis] = slice(None)
    slice_reverse[axis] = slice(None, None, -1)
    asqu = -((2. * np.arange(1., nobs+1)[islice] - 1) *
            (np.log(cdfvals) + np.log(1-cdfvals[slice_reverse]))/nobs).sum(axis) \
            - nobs

    return asqu


#class OneSGOFFittedVec(object):
#    '''for vectorized fitting'''
    # currently I use the bootstrap as function instead of full class

    #note: kwds loc and scale are a pain
    # I would need to overwrite rvs, fit and cdf depending on fixed parameters

    #def bootstrap(self, distr, args=(), kwds={}, nobs=200, nrep=1000,
def bootstrap(distr, args=(), nobs=200, nrep=100, value=None, batch_size=None):
    '''Monte Carlo (or parametric bootstrap) p-values for gof

    currently hardcoded for A^2 only

    assumes vectorized fit_vec method,
    builds and analyses (nobs, nrep) sample in one step

    rename function to less generic

    this works also with nrep=1

    '''
    #signature similar to kstest ?
    #delegate to fn ?

    #rvs_kwds = {'size':(nobs, nrep)}
    #rvs_kwds.update(kwds)


    #it will be better to build a separate batch function that calls bootstrap
    #keep batch if value is true, but batch iterate from outside if stat is returned
    if (not batch_size is None):
        if value is None:
            raise ValueError('using batching requires a value')
        n_batch = int(np.ceil(nrep/float(batch_size)))
        count = 0
        for irep in xrange(n_batch):
            rvs = distr.rvs(args, **{'size':(batch_size, nobs)})
            params = distr.fit_vec(rvs, axis=1)
            params = map(lambda x: np.expand_dims(x, 1), params)
            cdfvals = np.sort(distr.cdf(rvs, params), axis=1)
            stat = asquare(cdfvals, axis=1)
            count += (stat >= value).sum()
        return count / float(n_batch * batch_size)
    else:
        #rvs = distr.rvs(args, **kwds)  #extension to distribution kwds ?
        rvs = distr.rvs(args, **{'size':(nrep, nobs)})
        params = distr.fit_vec(rvs, axis=1)
        params = map(lambda x: np.expand_dims(x, 1), params)
        cdfvals = np.sort(distr.cdf(rvs, params), axis=1)
        stat = asquare(cdfvals, axis=1)
        if value is None:           #return all bootstrap results
            stat_sorted = np.sort(stat)
            return stat_sorted
        else:                       #calculate and return specific p-value
            return (stat >= value).mean()



def bootstrap2(value, distr, args=(), nobs=200, nrep=100):
    '''Monte Carlo (or parametric bootstrap) p-values for gof

    currently hardcoded for A^2 only

    non vectorized, loops over all parametric bootstrap replications and calculates
    and returns specific p-value,

    rename function to less generic

    '''
    #signature similar to kstest ?
    #delegate to fn ?

    #rvs_kwds = {'size':(nobs, nrep)}
    #rvs_kwds.update(kwds)


    count = 0
    for irep in xrange(nrep):
        #rvs = distr.rvs(args, **kwds)  #extension to distribution kwds ?
        rvs = distr.rvs(args, **{'size':nobs})
        params = distr.fit_vec(rvs)
        cdfvals = np.sort(distr.cdf(rvs, params))
        stat = asquare(cdfvals, axis=0)
        count += (stat >= value)
    return count * 1. / nrep


class NewNorm(object):
    '''just a holder for modified distributions
    '''

    def fit_vec(self, x, axis=0):
        return x.mean(axis), x.std(axis)

    def cdf(self, x, args):
        return distributions.norm.cdf(x, loc=args[0], scale=args[1])

    def rvs(self, args, size):
        loc=args[0]
        scale=args[1]
        return loc + scale * distributions.norm.rvs(size=size)


