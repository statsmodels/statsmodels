'''patching scipy to fit distributions with some fixed/frozen parameters

with Bootstrap and Monte Carlo function, not general or verified yet
'''


import numpy as np
#import matplotlib.pyplot as plt
from scipy import stats, optimize


########## patching scipy

stats.distributions.vonmises.a = -np.pi
stats.distributions.vonmises.b = np.pi



def _fitstart(self, x):
    '''method of moment estimator as starting values

    Parameters
    ----------
    x : array
        data for which the parameters are estimated

    Returns
    -------
    est : tuple
        preliminary estimates used as starting value for fitting, not
        necessarily a consistent estimator

    Notes
    -----
    This needs to be written and attached to each individual distribution

    This example was written for the gamma distribution, but not verified
    with literature
    '''
    loc = np.min([x.min(),0])
    a = 4/stats.skew(x)**2
    scale = np.std(x) / np.sqrt(a)
    return (a, loc, scale)

def nnlf_fr(self, thetash, x, frmask):
    # new frozen version
    # - sum (log pdf(x, theta),axis=0)
    #   where theta are the parameters (including loc and scale)
    #
    try:
        if frmask != None:
            theta = frmask.copy()
            theta[np.isnan(frmask)] = thetash
        else:
            theta = thetash
        loc = theta[-2]
        scale = theta[-1]
        args = tuple(theta[:-2])
    except IndexError:
        raise ValueError, "Not enough input arguments."
    if not self._argcheck(*args) or scale <= 0:
        return np.inf
    x = np.array((x-loc) / scale)
    cond0 = (x <= self.a) | (x >= self.b)
    if (np.any(cond0)):
        return np.inf
    else:
        N = len(x)
        #raise ValueError
        return self._nnlf(x, *args) + N*np.log(scale)

def fit_fr(self, data, *args, **kwds):
    '''estimate distribution parameters by MLE taking some parameters as fixed

    Parameters
    ----------
    data : array, 1d
        data for which the distribution parameters are estimated,
    args : list ? check
        starting values for optimization
    kwds :

      - 'frozen' : array_like
           values for frozen distribution parameters and, for elements with
           np.nan, the corresponding parameter will be estimated

    Returns
    -------
    argest : array
        estimated parameters

    Notes
    -----
    self is an instance of a distribution class. This can be attached to
    scipy.stats.distributions.rv_continuous

    *Todo*

    * check if docstring is correct
    * more input checking, args is list ? might also apply to current fit method

    '''
    loc0, scale0 = map(kwds.get, ['loc', 'scale'],[0.0, 1.0])
    Narg = len(args)

    if Narg == 0 and hasattr(self, '_fitstart'):
        x0 = self._fitstart(data)
    elif Narg > self.numargs:
        raise ValueError, "Too many input arguments."
    else:
        args += (1.0,)*(self.numargs-Narg)
        # location and scale are at the end
        x0 = args + (loc0, scale0)

    if 'frozen' in kwds:
        frmask = np.array(kwds['frozen'])
        if len(frmask) != self.numargs+2:
            raise ValueError, "Incorrect number of frozen arguments."
        else:
            # keep starting values for not frozen parameters
            x0  = np.array(x0)[np.isnan(frmask)]
    else:
        frmask = None

    #print x0
    #print frmask
    return optimize.fmin(self.nnlf_fr, x0,
                args=(np.ravel(data), frmask), disp=0)

stats.distributions.rv_continuous.fit_fr = fit_fr
stats.distributions.rv_continuous.nnlf_fr = nnlf_fr

########## end patching scipy


def distfitbootstrap(sample, distr, nrepl=100):
    '''run bootstrap for estimation of distribution parameters

    hard coded: only one shape parameter is allowed and estimated,
        loc=0 and scale=1 are fixed in the estimation

    Parameters
    ----------
    sample : array
        original sample data for bootstrap
    distr : distribution instance with fit_fr method
    nrepl : integer
        number of bootstrap replications

    Returns
    -------
    res : array (nrepl,)
        parameter estimates for all bootstrap replications

    '''
    nobs = len(sample)
    res = np.zeros(nrepl)
    for ii in xrange(nrepl):
        rvsind = np.random.randint(nobs, size=nobs)
        x = sample[rvsind]
        res[ii] = distr.fit_fr(x, frozen=[np.nan, 0.0, 1.0])
    return res

def distfitmc(sample, distr, nrepl=100, distkwds={}):
    '''run Monte Carlo for estimation of distribution parameters

    hard coded: only one shape parameter is allowed and estimated,
        loc=0 and scale=1 are fixed in the estimation

    Parameters
    ----------
    sample : array
        original sample data, in Monte Carlo only used to get nobs,
    distr : distribution instance with fit_fr method
    nrepl : integer
        number of Monte Carlo replications

    Returns
    -------
    res : array (nrepl,)
        parameter estimates for all Monte Carlo replications

    '''
    arg = distkwds.pop('arg')
    nobs = len(sample)
    res = np.zeros(nrepl)
    for ii in xrange(nrepl):
        x = distr.rvs(arg, size=nobs, **distkwds)
        res[ii] = distr.fit_fr(x, frozen=[np.nan, 0.0, 1.0])
    return res

def printresults(sample, arg, bres, kind='bootstrap'):
    '''calculate and print Bootstrap or Monte Carlo result

    Parameters
    ----------
    sample : array
        original sample data
    arg : float   (for general case will be array)
    bres : array
        parameter estimates from Bootstrap or Monte Carlo run
    kind : {'bootstrap', 'montecarlo'}
        output is printed for Mootstrap (default) or Monte Carlo

    Returns
    -------
    None, currently only printing

    Notes
    -----
    still a bit a mess because it is used for both Bootstrap and Monte Carlo

    made correction:
        reference point for bootstrap is estimated parameter

    not clear:
        I'm not doing any ddof adjustment in estimation of variance, do we
        need ddof>0 ?

    todo: return results and string instead of printing

    '''
    print 'true parameter value'
    print arg
    print 'MLE estimate of parameters using sample (nobs=%d)'% (nobs)
    argest = distr.fit_fr(sample, frozen=[np.nan, 0.0, 1.0])
    print argest
    if kind == 'bootstrap':
        #bootstrap compares to estimate from sample
        argorig = arg
        arg = argest

    print '%s distribution of parameter estimate (nrepl=%d)'% (kind, nrepl)
    print 'mean = %f, bias=%f' % (bres.mean(0), bres.mean(0)-arg)
    print 'median', np.median(bres, axis=0)
    print 'var and std', bres.var(0), np.sqrt(bres.var(0))
    bmse = ((bres - arg)**2).mean(0)
    print 'mse, rmse', bmse, np.sqrt(bmse)
    bressorted = np.sort(bres)
    print '%s confidence interval (90%% coverage)' % kind
    print bressorted[np.floor(nrepl*0.05)], bressorted[np.floor(nrepl*0.95)]
    print '%s confidence interval (90%% coverage) normal approximation' % kind
    print stats.norm.ppf(0.05, loc=bres.mean(), scale=bres.std()),
    print stats.norm.isf(0.05, loc=bres.mean(), scale=bres.std())
    print 'Kolmogorov-Smirnov test for normality of %s distribution' % kind
    print ' - estimated parameters, p-values not really correct'
    print stats.kstest(bres, 'norm', (bres.mean(), bres.std()))


if __name__ == '__main__':

    examplecases = ['largenumber', 'bootstrap', 'montecarlo']

    if 'largenumber' in examplecases:

        for nobs in [200]:#[20000, 1000, 100]:
            x = stats.vonmises.rvs(1.23, loc=0, scale=1, size=nobs)
            print '\nnobs:', nobs
            print 'true parameter'
            print '1.23, loc=0, scale=1'
            print 'unconstraint'
            print stats.vonmises.fit(x)
            print stats.vonmises.fit_fr(x, frozen=[np.nan, np.nan, np.nan])
            print 'with fixed loc and scale'
            print stats.vonmises.fit_fr(x, frozen=[np.nan, 0.0, 1.0])


        distr = stats.gamma
        arg, loc, scale = 2.5, 0., 20.

        for nobs in [200]:#[20000, 1000, 100]:
            x = distr.rvs(arg, loc=loc, scale=scale, size=nobs)
            print '\nnobs:', nobs
            print 'true parameter'
            print '%f, loc=%f, scale=%f' % (arg, loc, scale)
            print 'unconstraint'
            print distr.fit(x)
            print distr.fit_fr(x, frozen=[np.nan, np.nan, np.nan])
            print 'with fixed loc and scale'
            print distr.fit_fr(x, frozen=[np.nan, 0.0, 1.0])
            print 'with fixed loc'
            print distr.fit_fr(x, frozen=[np.nan, 0.0, np.nan])




        ex = ['gamma', 'vonmises'][0]

        if ex == 'gamma':
            distr = stats.gamma
            arg, loc, scale = 2.5, 0., 1
        elif ex == 'vonmises':
            distr = stats.vonmises
            arg, loc, scale = 1.5, 0., 1
        else:
            raise ValueError('wrong example')

        nobs = 100
        nrepl = 1000

        sample = distr.rvs(arg, loc=loc, scale=scale, size=nobs)

    if 'bootstrap' in examplecases:
        print 'Bootstrap'
        bres = distfitbootstrap(sample, distr, nrepl=nrepl )
        printresults(sample, arg, bres)

    if 'montecarlo' in examplecases:
        print '\nMonteCarlo'
        mcres = distfitmc(sample, distr, nrepl=nrepl,
                          distkwds=dict(arg=arg, loc=loc, scale=scale))
        printresults(sample, arg, mcres, kind='montecarlo')



