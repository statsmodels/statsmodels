'''More Goodness of fit tests

contains

GOF : 1 sample gof tests based on Stephens 1970, plus AD A^2
bootstrap : vectorized bootstrap p-values for gof test with fitted parameters


Created : 2011-05-21
Author : Josef Perktold

parts based on ks_2samp and kstest from scipy.stats
(license: Scipy BSD, but were completely rewritten by Josef Perktold)


References
----------

'''
from __future__ import print_function
from statsmodels.compat.python import range, lmap, string_types, callable
import numpy as np

from scipy.stats import distributions, ks_2samp, kstest

from statsmodels.tools.decorators import cache_readonly

from scipy.special import kolmogorov as ksprob


#TODO: split into modification and pvalue functions separately ?
#      for separate testing and combining different pieces

def dplus_st70_upp(stat, nobs):
    mod_factor = np.sqrt(nobs) + 0.12 + 0.11 / np.sqrt(nobs)
    stat_modified = stat * mod_factor
    pval = np.exp(-2 * stat_modified**2)
    digits = np.sum(stat > np.array([0.82, 0.82, 1.00]))
    #repeat low to get {0,2,3}
    return stat_modified, pval, digits

dminus_st70_upp = dplus_st70_upp


def d_st70_upp(stat, nobs):
    mod_factor = np.sqrt(nobs) + 0.12 + 0.11 / np.sqrt(nobs)
    stat_modified = stat * mod_factor
    pval = 2 * np.exp(-2 * stat_modified**2)
    digits = np.sum(stat > np.array([0.91, 0.91, 1.08]))
    #repeat low to get {0,2,3}
    return stat_modified, pval, digits

def v_st70_upp(stat, nobs):
    mod_factor = np.sqrt(nobs) + 0.155 + 0.24 / np.sqrt(nobs)
    #repeat low to get {0,2,3}
    stat_modified = stat * mod_factor
    zsqu = stat_modified**2
    pval = (8 * zsqu - 2) * np.exp(-2 * zsqu)
    digits = np.sum(stat > np.array([1.06, 1.06, 1.26]))
    return stat_modified, pval, digits

def wsqu_st70_upp(stat, nobs):
    nobsinv = 1. / nobs
    stat_modified = (stat - 0.4 * nobsinv + 0.6 * nobsinv**2) * (1 + nobsinv)
    pval = 0.05 * np.exp(2.79 - 6 * stat_modified)
    digits = np.nan  # some explanation in txt
    #repeat low to get {0,2,3}
    return stat_modified, pval, digits

def usqu_st70_upp(stat, nobs):
    nobsinv = 1. / nobs
    stat_modified = (stat - 0.1 * nobsinv + 0.1 * nobsinv**2)
    stat_modified *= (1 + 0.8 * nobsinv)
    pval = 2 * np.exp(- 2 * stat_modified * np.pi**2)
    digits = np.sum(stat > np.array([0.29, 0.29, 0.34]))
    #repeat low to get {0,2,3}
    return stat_modified, pval, digits

def a_st70_upp(stat, nobs):
    nobsinv = 1. / nobs
    stat_modified = (stat - 0.7 * nobsinv + 0.9 * nobsinv**2)
    stat_modified *= (1 + 1.23 * nobsinv)
    pval = 1.273 * np.exp(- 2 * stat_modified / 2. * np.pi**2)
    digits = np.sum(stat > np.array([0.11, 0.11, 0.452]))
    #repeat low to get {0,2,3}
    return stat_modified, pval, digits



gof_pvals = {}

gof_pvals['stephens70upp'] = {
    'd_plus' : dplus_st70_upp,
    'd_minus' : dplus_st70_upp,
    'd' : d_st70_upp,
    'v' : v_st70_upp,
    'wsqu' : wsqu_st70_upp,
    'usqu' : usqu_st70_upp,
    'a' : a_st70_upp }

def pval_kstest_approx(D, N):
    pval_two = distributions.kstwobign.sf(D*np.sqrt(N))
    if N > 2666 or pval_two > 0.80 - N*0.3/1000.0 :
        return D, distributions.kstwobign.sf(D*np.sqrt(N)), np.nan
    else:
        return D, distributions.ksone.sf(D,N)*2, np.nan

gof_pvals['scipy'] = {
    'd_plus' : lambda Dplus, N: (Dplus, distributions.ksone.sf(Dplus, N), np.nan),
    'd_minus' : lambda Dmin, N: (Dmin, distributions.ksone.sf(Dmin,N), np.nan),
    'd' : lambda D, N: (D, distributions.kstwobign.sf(D*np.sqrt(N)), np.nan)
    }

gof_pvals['scipy_approx'] = {
    'd' : pval_kstest_approx }

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
        if isinstance(rvs, string_types):
            #cdf = getattr(stats, rvs).cdf
            if (not cdf) or (cdf == rvs):
                cdf = getattr(distributions, rvs).cdf
                rvs = getattr(distributions, rvs).rvs
            else:
                raise AttributeError('if rvs is string, cdf has to be the same distribution')


        if isinstance(cdf, string_types):
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
        for j in range(1,nobs):
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








def gof_mc(randfn, distr, nobs=100):
    #print '\nIs it correctly sized?'
    from collections import defaultdict

    results = defaultdict(list)
    for i in range(1000):
        rvs = randfn(nobs)
        goft = GOF(rvs, distr)
        for ti in all_gofs:
            results[ti].append(goft.get_test(ti, 'stephens70upp')[0][1])

    resarr = np.array([results[ti] for ti in all_gofs])
    print('         ', '      '.join(all_gofs))
    print('at 0.01:', (resarr < 0.01).mean(1))
    print('at 0.05:', (resarr < 0.05).mean(1))
    print('at 0.10:', (resarr < 0.1).mean(1))

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
        for irep in range(n_batch):
            rvs = distr.rvs(args, **{'size':(batch_size, nobs)})
            params = distr.fit_vec(rvs, axis=1)
            params = lmap(lambda x: np.expand_dims(x, 1), params)
            cdfvals = np.sort(distr.cdf(rvs, params), axis=1)
            stat = asquare(cdfvals, axis=1)
            count += (stat >= value).sum()
        return count / float(n_batch * batch_size)
    else:
        #rvs = distr.rvs(args, **kwds)  #extension to distribution kwds ?
        rvs = distr.rvs(args, **{'size':(nrep, nobs)})
        params = distr.fit_vec(rvs, axis=1)
        params = lmap(lambda x: np.expand_dims(x, 1), params)
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
    for irep in range(nrep):
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




if __name__ == '__main__':
    from scipy import stats
    #rvs = np.random.randn(1000)
    rvs = stats.t.rvs(3, size=200)
    print('scipy kstest')
    print(kstest(rvs, 'norm'))
    goft = GOF(rvs, 'norm')
    print(goft.get_test())

    all_gofs = ['d', 'd_plus', 'd_minus', 'v', 'wsqu', 'usqu', 'a']
    for ti in all_gofs:
        print(ti, goft.get_test(ti, 'stephens70upp'))

    print('\nIs it correctly sized?')
    from collections import defaultdict

    results = defaultdict(list)
    nobs = 200
    for i in range(100):
        rvs = np.random.randn(nobs)
        goft = GOF(rvs, 'norm')
        for ti in all_gofs:
            results[ti].append(goft.get_test(ti, 'stephens70upp')[0][1])

    resarr = np.array([results[ti] for ti in all_gofs])
    print('         ', '      '.join(all_gofs))
    print('at 0.01:', (resarr < 0.01).mean(1))
    print('at 0.05:', (resarr < 0.05).mean(1))
    print('at 0.10:', (resarr < 0.1).mean(1))

    gof_mc(lambda nobs: stats.t.rvs(3, size=nobs), 'norm', nobs=200)

    nobs = 200
    nrep = 100
    bt = bootstrap(NewNorm(), args=(0,1), nobs=nobs, nrep=nrep, value=None)
    quantindex = np.floor(nrep * np.array([0.99, 0.95, 0.9])).astype(int)
    print(bt[quantindex])

    #the bootstrap results match Stephens pretty well for nobs=100, but not so well for
    #large (1000) or small (20) nobs
    '''
    >>> np.array([15.0, 10.0, 5.0, 2.5, 1.0])/100.  #Stephens
    array([ 0.15 ,  0.1  ,  0.05 ,  0.025,  0.01 ])
    >>> nobs = 100
    >>> [bootstrap(NewNorm(), args=(0,1), nobs=nobs, nrep=10000, value=c/ (1 + 4./nobs - 25./nobs**2)) for c in [0.576, 0.656, 0.787, 0.918, 1.092]]
    [0.1545, 0.10009999999999999, 0.049000000000000002, 0.023, 0.0104]
    >>>
    '''

    #test equality of loop, vectorized, batch-vectorized
    np.random.seed(8765679)
    resu1 = bootstrap(NewNorm(), args=(0,1), nobs=nobs, nrep=100,
                      value=0.576/(1 + 4./nobs - 25./nobs**2))
    np.random.seed(8765679)
    tmp = [bootstrap(NewNorm(), args=(0,1), nobs=nobs, nrep=1) for _ in range(100)]
    resu2 = (np.array(tmp) > 0.576/(1 + 4./nobs - 25./nobs**2)).mean()
    np.random.seed(8765679)
    tmp = [bootstrap(NewNorm(), args=(0,1), nobs=nobs, nrep=1,
                     value=0.576/ (1 + 4./nobs - 25./nobs**2),
                     batch_size=10) for _ in range(10)]
    resu3 = np.array(resu).mean()
    from numpy.testing import assert_almost_equal, assert_array_almost_equal
    assert_array_almost_equal(resu1, resu2, 15)
    assert_array_almost_equal(resu2, resu3, 15)
