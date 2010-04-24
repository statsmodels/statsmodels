'''estimate distribution parameters by method of moments or matching quantiles,
and Maximum Likelihood estimation based on binned data

initially loosely based on a paper and blog for quantile matching
  by John D. Cook
  formula for gamma quantile (ppf) matching by him (from paper)
  http://www.codeproject.com/KB/recipes/ParameterPercentile.aspx
  http://www.johndcook.com/blog/2010/01/31/parameters-from-percentiles/
  this is what I actually used (in parts):
  http://www.bepress.com/mdandersonbiostat/paper55/

quantile based estimator
^^^^^^^^^^^^^^^^^^^^^^^^
only special cases for number or parameters so far
Is there a literature for GMM estimation of distribution parameters? check


binned estimator
^^^^^^^^^^^^^^^^
* I added this also
* use it for chisquare tests with estimation distribution parameters
* move this to distribution_extras (next to gof tests powerdiscrepancy and
  continuous) or add to distribution_patch


example: t-distribution
* works with quantiles if they contain tail quantiles
* results with momentcondquant don't look as good as mle estimate

TODOs
* rearange and make sure I don't use module globals (as I did initially)
  make two version exactly identified method of moments with fsolve
  and GMM (?) version with fmin
  and maybe the special cases of JD Cook
* add semifrozen version of moment and quantile based estimators,
  e.g. for beta (both loc and scale fixed), or gamma (loc fixed)
* add beta example to the semifrozen MLE, fitfr, code
* start a list of how well different estimators, especially current mle work
  for the different distributions
* need general GMM code (with optimal weights ?), looks like a good example for it
* get example for binned data estimation, mailing list a while ago
* any idea when these are better than mle ?
* check language: I use quantile to mean the value of the random variable, not
  quantile between 0 and 1.

Author : josef-pktd
License : BSD
created : 2010-04-20
'''


import numpy as np
from scipy import stats, optimize, special

alpha = 2
xq = [0.5, 4]
pq = [0.1, 0.9]
print stats.gamma.ppf(pq, alpha)
xq = stats.gamma.ppf(pq, alpha)
print np.diff((stats.gamma.ppf(pq, np.linspace(0.01,4,10)[:,None])*xq[::-1]))
#optimize.bisect(lambda alpha: np.diff((stats.gamma.ppf(pq, alpha)*xq[::-1])))
print optimize.fsolve(lambda alpha: np.diff((stats.gamma.ppf(pq, alpha)*xq[::-1])), 3.)

distfn = stats.gamma
# the next two use distfn from module scope
def gammamomentcond(params, mom2, quantile=None):
    '''estimate distribution parameters based on globals, loc=0 is fixed

    Returns
    -------
    cond : function

    '''
    def cond(params):
        alpha, scale = params
        mom2s = distfn.stats(alpha, 0.,scale)
        #quantil
        return np.array(mom2)-mom2s
    return cond

def gammamomentcond2(params, mom2, quantile=None):
    '''estimate distribution parameters based on globals, loc=0 is fixed

    Returns
    -------
    difference : array
        difference between theoretical and empirical moments

    '''
    alpha, scale = params
    mom2s = distfn.stats(alpha, 0.,scale)
    #quantil
    return np.array(mom2)-mom2s

mcond = gammamomentcond([5.,10], mom2=stats.gamma.stats(alpha, 0.,1.), quantile=None)
print optimize.fsolve(mcond, [1.,2.])
mom2 = stats.gamma.stats(alpha, 0.,1.)
print optimize.fsolve(lambda params:gammamomentcond2(params, mom2), [1.,2.])

grvs = stats.gamma.rvs(alpha, 0.,2., size=1000)
mom2 = np.array([grvs.mean(), grvs.var()])
alphaestq = optimize.fsolve(lambda params:gammamomentcond2(params, mom2), [1.,3.])
print alphaestq
print 'scale = ', xq/stats.gamma.ppf(pq, alphaestq)

######### fsolve doesn't move in small samples, fmin not very accurate
def momentcondunbound(distfn, params, mom2, quantile=None):
    '''estimate distribution parameters based on globals, uses 2 moments and
    one quantile

    Returns
    -------
    difference : array
        difference between theoretical and empirical moments and quantiles

    '''
    shape, loc, scale = params
    mom2diff = np.array(distfn.stats(shape, loc,scale)) - mom2
    if not quantile is None:
        pq, xq = quantile
        #ppfdiff = distfn.ppf(pq, alpha)
        cdfdiff = distfn.cdf(xq, shape, loc, scale) - pq
        return np.concatenate([mom2diff, cdfdiff[:1]])
    return mom2diff

nobs = 1000
distfn = stats.t
pq = np.array([0.1,0.9])
paramsdgp = (5, 0, 1)
trvs = distfn.rvs(5, 0, 1, size=nobs)
xqs = [stats.scoreatpercentile(trvs, p) for p in pq*100]
mom2th = distfn.stats(*paramsdgp)
mom2s = np.array([trvs.mean(), trvs.var()])
tparest_gmm3quantilefsolve = optimize.fsolve(lambda params:momentcondunbound(distfn,params, mom2s,(pq,xqs)), [10,1.,2.])
print 'tparest_gmm3quantilefsolve', tparest_gmm3quantilefsolve
tparest_gmm3quantile = optimize.fmin(lambda params:np.sum(momentcondunbound(distfn,params, mom2s,(pq,xqs))**2), [10,1.,2.])
print 'tparest_gmm3quantile', tparest_gmm3quantile
print distfn.fit(trvs)

###### loc scale only
def momentcondunboundls(distfn, params, mom2, quantile=None, shape=None):
    '''estimate loc and scale using either quantiles or moments (not both)

    Returns
    -------
    difference : array
        difference between theoretical and empirical moments or quantiles

    '''
    loc, scale = params
    mom2diff = np.array(distfn.stats(shape, loc, scale)) - mom2
    if not quantile is None:
        pq, xq = quantile
        #ppfdiff = distfn.ppf(pq, alpha)
        cdfdiff = distfn.cdf(xq, shape, loc, scale) - pq
        #return np.concatenate([mom2diff, cdfdiff[:1]])
        return cdfdiff
    return mom2diff

##distfn = stats.t
##pq = np.array([0.1,0.9])
##paramsdgp = (5, 0, 1)
##trvs = distfn.rvs(5, 0, 1, size=nobs)
##xqs = [stats.scoreatpercentile(trvs, p) for p in pq*100]
##mom2th = distfn.stats(*paramsdgp)
##mom2s = np.array([trvs.mean(), trvs.var()])
print optimize.fsolve(lambda params:momentcondunboundls(distfn, params, mom2s,shape=5), [1.,2.])
print optimize.fmin(lambda params:np.sum(momentcondunboundls(distfn, params, mom2s,shape=5)**2), [1.,2.])
print distfn.fit(trvs)
#loc, scale, based on quantiles
print optimize.fsolve(lambda params:momentcondunboundls(distfn, params, mom2s,(pq,xqs),shape=5), [1.,2.])


######### try quantile GMM with identity weight matrix
#(just a guess that's what it is

def momentcondquant(distfn, params, mom2, quantile=None, shape=None):
    '''moment conditions for estimating distribution parameters by matching
    quantiles

    Returns
    -------
    difference : array
        difference between theoretical and empirical quantiles

    '''
    if len(params) == 2:
        loc, scale = params
    elif len(params) == 3:
        shape, loc, scale = params
    else:
        #raise NotImplementedError
        pass #see whether this might work, seems to work for beta with 2 shape args

    #mom2diff = np.array(distfn.stats(*params)) - mom2
    #if not quantile is None:
    pq, xq = quantile
    #ppfdiff = distfn.ppf(pq, alpha)
    cdfdiff = distfn.cdf(xq, *params) - pq
    #return np.concatenate([mom2diff, cdfdiff[:1]])
    return cdfdiff
    #return mom2diff

pq = np.array([0.01, 0.05,0.1,0.4,0.6,0.9,0.95,0.99])
#paramsdgp = (5, 0, 1)
xqs = [stats.scoreatpercentile(trvs, p) for p in pq*100]
tparest_gmmquantile = optimize.fmin(lambda params:np.sum(momentcondquant(distfn, params, mom2s,(pq,xqs), shape=None)**2), [10, 1.,2.])
print 'tparest_gmmquantile', tparest_gmmquantile
rvsb = stats.beta.rvs(5,15,size=200)
print stats.beta.fit(rvsb)
xqsb = [stats.scoreatpercentile(trvs, p) for p in pq*100]
betaparest_gmmquantile = optimize.fmin(lambda params:np.sum(momentcondquant(stats.beta, params, mom2s,(pq,xqsb), shape=None)**2), [10,10, 0., 1.])
print 'betaparest_gmmquantile',  betaparest_gmmquantile
#result sensitive to initial condition


def fitbinned(distfn, freq, binedges, start, fixed=None):
    '''
    estimate parameters of distribution function for binned data

    Parameters
    ----------
    distfn : distribution instance
        needs to have cdf method, as in scipy.stats
    freq : array, 1d
        frequency count, e.g. obtained by histogram
    binedges : array, 1d
        binedges including lower and upper bound
    start : tuple or array_like ?
        starting values, needs to have correct length

    Returns
    -------
    paramest : array
        estimated parameters

    Notes
    -----
    todo: add fixed parameter option

    added factorial

    '''
    if not fixed is None:
        raise NotImplementedError
    nobs = np.sum(freq)
    lnnobsfact = special.gammaln(nobs+1)
    def nloglike(params):
        '''negative loglikelihood function of binned data

        corresponds to multinomial
        '''
        prob = np.diff(distfn.cdf(binedges, *params))
        return -(lnnobsfact + np.sum(freq*np.log(prob)- special.gammaln(freq+1)))
    return optimize.fmin(nloglike, start)

cache = {}
def fitbinnedgmm(distfn, freq, binedges, start, fixed=None, weightsoptimal=True):
    '''
    estimate parameters of distribution function for binned data

    Parameters
    ----------
    distfn : distribution instance
        needs to have cdf method, as in scipy.stats
    freq : array, 1d
        frequency count, e.g. obtained by histogram
    binedges : array, 1d
        binedges including lower and upper bound
    start : tuple or array_like ?
        starting values, needs to have correct length

    Returns
    -------
    paramest : array
        estimated parameters

    Notes
    -----
    todo: add fixed parameter option

    added factorial

    '''
    if not fixed is None:
        raise NotImplementedError
    nobs = np.sum(freq)
    if weightsoptimal:
        weights = freq/float(nobs)
    else:
        weights = np.ones(len(freq))
    freqnormed = freq/float(nobs)
    # skip turning weights into matrix diag(freq/float(nobs))
    def gmmobjective(params):
        '''negative loglikelihood function of binned data

        corresponds to multinomial
        '''
        prob = np.diff(distfn.cdf(binedges, *params))
        momcond = freqnormed - prob
        return np.dot(momcond*weights, momcond)
    return optimize.fmin(gmmobjective, start)

#use trvs from before
bt = stats.t.ppf(np.linspace(0,1,21),5)
ft,bt = np.histogram(trvs,bins=bt)
print 'fitbinned t-distribution'
tparest_mlebinew = fitbinned(stats.t, ft, bt, [10, 0, 1])
tparest_gmmbinewidentity = fitbinnedgmm(stats.t, ft, bt, [10, 0, 1])
tparest_gmmbinewoptimal = fitbinnedgmm(stats.t, ft, bt, [10, 0, 1], weightsoptimal=False)
print paramsdgp

#Note: this can be used for chisquare test and then has correct asymptotic
#   distribution for a distribution with estimated parameters, find ref again
#TODO combine into test with binning included, check rule for number of bins

#bt2 = stats.t.ppf(np.linspace(trvs.,1,21),5)
ft2,bt2 = np.histogram(trvs,bins=50)
'fitbinned t-distribution'
tparest_mlebinel = fitbinned(stats.t, ft2, bt2, [10, 0, 1])
tparest_gmmbinelidentity = fitbinnedgmm(stats.t, ft2, bt2, [10, 0, 1])
tparest_gmmbineloptimal = fitbinnedgmm(stats.t, ft2, bt2, [10, 0, 1], weightsoptimal=False)
tparest_mle = stats.t.fit(trvs)

np.set_printoptions(precision=6)
print 'sample size', nobs
print 'true (df, loc, scale)      ', paramsdgp
print 'parest_mle                 ', tparest_mle
print
print 'tparest_mlebinel           ', tparest_mlebinel
print 'tparest_gmmbinelidentity   ', tparest_gmmbinelidentity
print 'tparest_gmmbineloptimal    ', tparest_gmmbineloptimal
print
print 'tparest_mlebinew           ', tparest_mlebinew
print 'tparest_gmmbinewidentity   ', tparest_gmmbinewidentity
print 'tparest_gmmbinewoptimal    ', tparest_gmmbinewoptimal
print
print 'tparest_gmmquantileidentity', tparest_gmmquantile
print 'tparest_gmm3quantilefsolve ', tparest_gmm3quantilefsolve
print 'tparest_gmm3quantile       ', tparest_gmm3quantile

''' example results:
standard error for df estimate looks large
note: iI don't impose that df is an integer, (b/c not necessary)
need Monte Carlo to check variance of estimators


sample size 1000
true (df, loc, scale)       (5, 0, 1)
parest_mle                  [ 4.571405 -0.021493  1.028584]

tparest_mlebinel            [ 4.534069 -0.022605  1.02962 ]
tparest_gmmbinelidentity    [ 2.653056  0.012807  0.896958]
tparest_gmmbineloptimal     [ 2.437261 -0.020491  0.923308]

tparest_mlebinew            [ 2.999124 -0.0199    0.948811]
tparest_gmmbinewidentity    [ 2.900939 -0.020159  0.93481 ]
tparest_gmmbinewoptimal     [ 2.977764 -0.024925  0.946487]

tparest_gmmquantileidentity [ 3.940797 -0.046469  1.002001]
tparest_gmm3quantilefsolve  [ 10.   1.   2.]
tparest_gmm3quantile        [ 6.376101 -0.029322  1.112403]
'''
