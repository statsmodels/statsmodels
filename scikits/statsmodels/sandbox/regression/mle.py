'''general non-linear MLE for time series analysis

idea for general version

subclass defines geterrors(parameters) besides loglike,...
and covariance matrix of parameter estimates (e.g. from hessian
or outerproduct of jacobian)

superclass or result class calculates result statistic based
on errors, loglike, jacobian and cov/hessian
  -> aic, bic, ...
  -> test statistics, tvalue, fvalue, ...
  -> new to add: distribution (mean, cov) of non-linear transformation
  -> parameter restrictions or transformation with corrected covparams (?)
  -> sse, rss, rsquared  ??? are they defined from this in general
  -> robust parameter cov ???
  -> additional residual bast tests, NW, ... likelihood ratio, lagrange
     multiplier tests ???

how much can be reused from linear model result classes where
   `errorsest = y - X*beta` ?

for tsa: what's the division of labor between model, result instance
    and process

examples:
 * arma: ls and mle look good
 * arimax: add exog, especially mean, trend, prefilter, e.g. (1-L)
 * arma_t: arma with t distributed errors (just a change in loglike)
 * garch: need loglike and (recursive) errorest
 * regime switching model without unobserved state, e.g. threshold



Created on Feb 6, 2010
@author: "josef pktd"
'''

import numpy as np
from numpy.testing import assert_almost_equal

#from scipy.stats import t, norm
from scipy import optimize, signal, derivative

import numdifftools as ndt

from scikits.statsmodels.model import Model, LikelihoodModelResults
from scikits.statsmodels.sandbox import tsa

def normloglike(x, mu=0, sigma2=1, returnlls=False, axis=0):

    x = np.asarray(x)
    x = np.atleast_1d(x)
    if axis is None:
        x = x.ravel()
    #T,K = x.shape
    if x.ndim > 1:
        nobs = x.shape[axis]
    else:
        nobs = len(x)

    x = x - mu  # assume can be broadcasted
    if returnlls:
    #Compute the individual log likelihoods if needed
        lls = -0.5*(np.log(2*np.pi) + np.log(sigma2) + x**2/sigma2)
        # Use these to comput the LL
        LL = np.sum(lls,axis)
        return LL, lls
    else:
        #Compute the log likelihood
        #print np.sum(np.log(sigma2),axis)
        LL  =  -0.5 * (np.sum(np.log(sigma2),axis) + np.sum((x**2)/sigma2, axis)  +  nobs*np.log(2*np.pi))
        return LL

# copied from model.py
class LikelihoodModel(Model):
    """
    Likelihood model is a subclass of Model.
    """

    def __init__(self, endog, exog=None):
        super(LikelihoodModel, self).__init__(endog, exog)
        self.initialize()

    def _initialize(self):
        """
        Initialize (possibly re-initialize) a Model instance. For
        instance, the design matrix of a linear model may change
        and some things must be recomputed.
        """
        pass
#TODO: if the intent is to re-initialize the model with new data then
# this method needs to take inputs...

    def loglike(self, params):
        """
        Log-likelihood of model.
        """
        raise NotImplementedError

    def score(self, params):
        """
        Score vector of model.

        The gradient of logL with respect to each parameter.
        """
        raise NotImplementedError

    def information(self, params):
        """
        Fisher information matrix of model

        Returns -Hessian of loglike evaluated at params.
        """
        raise NotImplementedError

    def hessian(self, params):
        """
        The Hessian matrix of the model
        """
        raise NotImplementedError

    def fit(self, start_params=None, method='newton', maxiter=35, tol=1e-08):
        """
        Fit method for likelihood based models

        Parameters
        ----------
        start_params : array-like, optional
            An optional

        method : str
            Method can be 'newton', 'bfgs', 'powell', 'cg', or 'ncg'.
            The default is newton.  See scipy.optimze for more information.
        """
        methods = ['newton', 'bfgs', 'powell', 'cg', 'ncg', 'fmin']
        if start_params is None:
            start_params = [0]*self.exog.shape[1] # will fail for shape (K,)
        if not method in methods:
            raise ValueError, "Unknown fit method %s" % method
        f = lambda params: -self.loglike(params)
        score = lambda params: -self.score(params)
#        hess = lambda params: -self.hessian(params)
        hess = None
#TODO: can we have a unified framework so that we can just do func = method
# and write one call for each solver?

        if method.lower() == 'newton':
            iteration = 0
            start = np.array(start_params)
            history = [np.inf, start]
            while (iteration < maxiter and np.all(np.abs(history[-1] - \
                    history[-2])>tol)):
                H = self.hessian(history[-1])
                newparams = history[-1] - np.dot(np.linalg.inv(H),
                        self.score(history[-1]))
                history.append(newparams)
                iteration += 1
            mlefit = LikelihoodModelResults(self, newparams)
            mlefit.iteration = iteration
        elif method == 'bfgs':
            score=None
            xopt, fopt, gopt, Hopt, func_calls, grad_calls, warnflag = \
                optimize.fmin_bfgs(f, start_params, score, full_output=1,
                        maxiter=maxiter, gtol=tol)
            converge = not warnflag
            mlefit = LikelihoodModelResults(self, xopt)
            optres = 'xopt, fopt, gopt, Hopt, func_calls, grad_calls, warnflag'
            self.optimresults = dict(zip(optres.split(', '),[
                xopt, fopt, gopt, Hopt, func_calls, grad_calls, warnflag]))
        elif method == 'ncg':
            xopt, fopt, fcalls, gcalls, hcalls, warnflag = \
                optimize.fmin_ncg(f, start_params, score, fhess=hess,
                        full_output=1, maxiter=maxiter, avextol=tol)
            mlefit = LikelihoodModelResults(self, xopt)
            converge = not warnflag
        elif method == 'fmin':
            #fmin(func, x0, args=(), xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None, full_output=0, disp=1, retall=0, callback=None)
            xopt, fopt, niter, funcalls, warnflag = \
                optimize.fmin(f, start_params,
                        full_output=1, maxiter=maxiter, xtol=tol)
            mlefit = LikelihoodModelResults(self, xopt)
            converge = not warnflag
        self._results = mlefit
        return mlefit


class Arma(LikelihoodModel):
    """
    univariate Autoregressive Moving Average model

    Note: This is not working yet
    """

    def __init__(self, endog, exog=None):
        #need to override p,q (nar,nma) correctly
        super(LikelihoodModel, self).__init__(endog, exog)
        #set default arma(1,1)
        self.nar = 1
        self.nma = 1
        #self.initialize()

    def geterrors(self, params):
        #copied from sandbox.tsa.arima.ARIMA
        p, q = self.nar, self.nma
        rhoy = np.concatenate(([1], params[:p]))
        rhoe = np.concatenate(([1], params[p:p+q]))
        errorsest = signal.lfilter(rhoy, rhoe, self.endog)
        return errorsest

    def loglike(self, params):
        """
        Loglikelihood for arma model

        Notes
        -----
        The ancillary parameter is assumed to be the last element of
        the params vector
        """

#        #copied from sandbox.tsa.arima.ARIMA
#        p = self.nar
#        rhoy = np.concatenate(([1], params[:p]))
#        rhoe = np.concatenate(([1], params[p:-1]))
#        errorsest = signal.lfilter(rhoy, rhoe, self.endog)
        errorsest = self.geterrors(params)
        sigma2 = np.maximum(params[-1]**2, 1e-6)
        axis = 0
        nobs = len(errorsest)
        #this doesn't help for exploding paths
        #errorsest[np.isnan(errorsest)] = 100
#        llike  =  -0.5 * (np.sum(np.log(sigma2),axis)
#                          + np.sum((errorsest**2)/sigma2, axis)
#                          +  nobs*np.log(2*np.pi))
        llike  =  -0.5 * (nobs*np.log(sigma2)
                          + np.sum((errorsest**2)/sigma2, axis)
                          +  nobs*np.log(2*np.pi))
        return llike

    def score(self, params):
        """
        Score vector for Arma model
        """
        #return None
        #print params
        jac = ndt.Jacobian(self.loglike, stepMax=1e-4)
        return jac(params)[-1]



    def hessian(self, params):
        """
        Hessian of arma model.  Currently uses numdifftools
        """
        #return None
        Hfun = ndt.Jacobian(self.score, stepMax=1e-4)
        return Hfun(params)[-1]


    def fit(self, start_params=None, maxiter=5000, method='fmin', tol=1e-08):
        start_params = np.concatenate((0.05*np.ones(self.nar + self.nma), [1]))
        mlefit = super(Arma, self).fit(start_params=start_params,
                maxiter=maxiter, method=method, tol=tol)
        return mlefit

def generate_kindofgarch(nobs, ar, ma, mu=1.):
    '''simulate garch like process but not squared errors in arma
    used for initial trial but produces nice graph
    '''
    #garm1, gmam1 = [0.4], [0.2]
    #pqmax = 1
#    res = np.zeros(nobs+pqmax)
#    rvs = np.random.randn(nobs+pqmax,2)
#    for t in range(pqmax,nobs+pqmax):
#        res[i] =
    #ar = [1.0, -0.99]
    #ma = [1.0,  0.5]
    #this has the wrong distribution, should be eps**2
    #TODO: use new version tsa.arima.??? instead, has distr option
    arest = tsa.arima.ARIMA()
    h = arest.generate_sample(ar,ma,nobs,0.1)
    #h = np.abs(h)
    h = (mu+h)**2
    h = np.exp(h)
    err = np.sqrt(h)*np.random.randn(nobs)
    return err, h

def generate_garch(nobs, ar, ma, mu=1., scale=0.1):
    '''simulate standard garch

    scale : float
       scale/standard deviation of innovation process in GARCH process
    '''

    eta = scale*np.random.randn(nobs)
    # copied from armageneratesample
    h = signal.lfilter(ma, ar, eta**2)

    #
    #h = (mu+h)**2
    #h = np.abs(h)
    #h = np.exp(h)
    #err = np.sqrt(h)*np.random.randn(nobs)
    err = np.sqrt(h)*np.random.standard_t(8, size=nobs)
    return err, h



def generate_gjrgarch(nobs, ar, ma, mu=1., scale=0.1, varinnovation=None):
    '''simulate gjr garch process

    Parameter
    ---------
    ar : array_like, 1d
        autoregressive term for variance
    ma : array_like, 2d
        moving average term for variance, with coefficients for negative
        shocks in second column
    mu : float
        constant in variance law of motion
    scale : float
       scale/standard deviation of innovation process in GARCH process

    Returns
    -------
    err : array 1d, (nobs+?,)
        simulated gjr-garch process,
    h : array 1d, (nobs+?,)
        simulated variance
    etax : array 1d, (nobs+?,)
        data matrix for constant and ma terms in variance equation

    Notes
    -----

    References
    ----------



    '''

    if varinnovation is None:    # rename ?
        eta = scale*np.random.randn(nobs)
    else:
        eta = varinnovation
    # copied from armageneratesample
    etax = np.empty((nobs,3))
    etax[:,0] = mu
    etax[:,1:] = (eta**2)[:,None]
    etax[eta>0,2] = 0
    h = miso_lfilter(ar, ma, etax)[0]

    #
    #h = (mu+h)**2
    #h = np.abs(h)
    #h = np.exp(h)
    #err = np.sqrt(h)*np.random.randn(nobs)
    print 'h.shape', h.shape
    err = np.sqrt(h)*np.random.standard_t(8, size=len(h))
    return err, h, etax

def miso_lfilter(ar, ma, x):
    '''
    use nd convolution to merge inputs,
    then use lfilter to produce output

    arguments for column variables
    return currently 1d

    Parameters
    ----------
    ar : array_like, 1d, float
        autoregressive lag polynomial including lag zero, ar(L)y_t
    ma : array_like, same ndim as x, currently 2d
        moving average lag polynomial ma(L)x_t
    x : array_like, 2d
        input data series, time in rows, variables in columns

    Returns
    -------
    y : array, 1d
        filtered output series
    inp : array, 1d
        combined input series

    Notes
    -----
    currently for 2d inputs only, no choice of axis
    Use of signal.lfilter requires that ar lag polynomial contains
    floating point numbers
    does not cut off invalid starting and final values

    miso_lfilter find array y such that::

            ar(L)y_t = ma(L)x_t

    with shapes y (nobs,), x (nobs,nvars), ar (narlags,), ma (narlags,nvars)

    '''
    ma = np.asarray(ma)
    #inp = signal.convolve(x, ma, mode='valid')
    #inp = signal.convolve(x, ma)[:, (x.shape[1]+1)//2]
    #Note: convolve mixes up the variable left-right flip
    #I only want the flip in time direction
    #this might also be a mistake or problem in other code where I
    #switched from correlate to convolve
    # correct convolve version, for use with fftconvolve in other cases
    inp2 = signal.convolve(x, ma[:,::-1])[:, (x.shape[1]+1)//2]
    inp = signal.correlate(x, ma[::-1,:])[:, (x.shape[1]+1)//2]
    assert_almost_equal(inp2, inp)
    return signal.lfilter([1], ar, inp), inp


def test_misofilter():
    x = np.arange(20).reshape(10,2)
    y, inp = miso_lfilter([1., -1],[[1,1],[0,0]], x)
    assert_almost_equal(y[:-1], x.sum(1).cumsum(), decimal=15)
    inp2 = signal.convolve(np.arange(20),np.ones(2))[1::2]
    assert_almost_equal(inp[:-1], inp2, decimal=15)

    inp2 = signal.convolve(np.arange(20),np.ones(4))[1::2]
    y, inp = miso_lfilter([1., -1],[[1,1],[1,1]], x)
    assert_almost_equal(y, inp2.cumsum(), decimal=15)
    assert_almost_equal(inp, inp2, decimal=15)
    y, inp = miso_lfilter([1., 0],[[1,1],[1,1]], x)
    assert_almost_equal(y, inp2, decimal=15)
    assert_almost_equal(inp, inp2, decimal=15)

    x3 = np.column_stack((np.ones((x.shape[0],1)),x))
    y, inp = miso_lfilter([1., 0],np.array([[-2.0,3,1],[0.0,0.0,0]]),x3)
    y3 = (x3*np.array([-2,3,1])).sum(1)
    assert_almost_equal(y[:-1], y3, decimal=15)
    assert_almost_equal(y, inp, decimal=15)
    y4 = y3.copy()
    y4[1:] += x3[:-1,1]
    y, inp = miso_lfilter([1., 0],np.array([[-2.0,3,1],[0.0,1.0,0]]),x3)
    assert_almost_equal(y[:-1], y4, decimal=15)
    assert_almost_equal(y, inp, decimal=15)
    y4 = y3.copy()
    y4[1:] += x3[:-1,0]
    y, inp = miso_lfilter([1., 0],np.array([[-2.0,3,1],[1.0,0.0,0]]),x3)
    assert_almost_equal(y[:-1], y4, decimal=15)
    assert_almost_equal(y, inp, decimal=15)
    y, inp = miso_lfilter([1., -1],np.array([[-2.0,3,1],[1.0,0.0,0]]),x3)
    assert_almost_equal(y[:-1], y4.cumsum(), decimal=15)
    y4 = y3.copy()
    y4[1:] += x3[:-1,2]
    y, inp = miso_lfilter([1., 0],np.array([[-2.0,3,1],[0.0,0.0,1.0]]),x3)
    assert_almost_equal(y[:-1], y4, decimal=15)
    assert_almost_equal(y, inp, decimal=15)
    y, inp = miso_lfilter([1., -1],np.array([[-2.0,3,1],[0.0,0.0,1.0]]),x3)
    assert_almost_equal(y[:-1], y4.cumsum(), decimal=15)

    y, inp = miso_lfilter([1., 0],[[1,0],[1,0],[1,0]], x)
    yt = np.convolve(x[:,0], [1,1,1])
    assert_almost_equal(y, yt, decimal=15)
    assert_almost_equal(inp, yt, decimal=15)
    y, inp = miso_lfilter([1., 0],[[0,1],[0,1],[0,1]], x)
    yt = np.convolve(x[:,1], [1,1,1])
    assert_almost_equal(y, yt, decimal=15)
    assert_almost_equal(inp, yt, decimal=15)

    y, inp = miso_lfilter([1., 0],[[0,1],[0,1],[1,1]], x)
    yt = np.convolve(x[:,1], [1,1,1])
    yt[2:] += x[:,0]
    assert_almost_equal(y, yt, decimal=15)
    assert_almost_equal(inp, yt, decimal=15)


'''
>>> print signal.correlate(x3, np.array([[-2.0,3,1],[0.0,0.0,0]])[::-1,:],mode='full')[:-1, (x3.shape[1]+1)//2]
[ -1.   7.  15.  23.  31.  39.  47.  55.  63.  71.]
>>> (x3*np.array([-2,3,1])).sum(1)
array([ -1.,   7.,  15.,  23.,  31.,  39.,  47.,  55.,  63.,  71.])
'''

def garchplot(err, h, title='Garch simulation'):
    plt.figure()
    plt.subplot(311)
    plt.plot(err)
    plt.title(title)
    plt.ylabel('y')
    plt.subplot(312)
    plt.plot(err**2)
    plt.ylabel('$y^2$')
    plt.subplot(313)
    plt.plot(h)
    plt.ylabel('conditional variance')

if __name__ == '__main__':

    examples = []
    if 'arma' in examples:
        arest = tsa.arima.ARIMA()
        print "\nExample 1"
        ar = [1.0, -0.8]
        ma = [1.0,  0.5]
        y1 = arest.generate_sample(ar,ma,1000,0.1)
        y1 -= y1.mean() #no mean correction/constant in estimation so far

        arma1 = Arma(y1)
        arma1.nar = 1
        arma1.nma = 1
        arma1res = arma1.fit(method='fmin')
        print arma1res.params

        #Warning need new instance otherwise results carry over
        arma2 = Arma(y1)
        res2 = arma2.fit(method='bfgs')
        print res2.params
        print res2.model.hessian(res2.params)
        print ndt.Hessian(arma1.loglike, stepMax=1e-2)(res2.params)
        resls = arest.fit(y1,1,1)
        print resls[0]
        print resls[1]



        print '\nparameter estimate'
        print 'parameter of DGP ar(1), ma(1), sigma_error'
        print [-0.8, 0.5, 0.1]
        print 'mle with fmin'
        print arma1res.params
        print 'mle with bfgs'
        print res2.params
        print 'cond. least squares uses optim.leastsq ?'
        errls = arest.error_estimate
        print resls[0], np.sqrt(np.dot(errls,errls)/errls.shape[0])

        err = arma1.geterrors(res2.params)
        print 'cond least squares parameter cov'
        #print np.dot(err,err)/err.shape[0] * resls[1]
        #errls = arest.error_estimate
        print np.dot(errls,errls)/errls.shape[0] * resls[1]
    #    print 'fmin hessian'
    #    print arma1res.model.optimresults['Hopt'][:2,:2]
        print 'bfgs hessian'
        print res2.model.optimresults['Hopt'][:2,:2]
        print 'numdifftools inverse hessian'
        print -np.linalg.inv(ndt.Hessian(arma1.loglike, stepMax=1e-2)(res2.params))[:2,:2]

        arma3 = Arma(y1**2)
        res3 = arma3.fit(method='bfgs')
        print res3.params

    nobs = 1000

    err,h = generate_kindofgarch(nobs, [1.0, -0.95], [1.0,  0.1], mu=0.5)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(211)
    plt.plot(err)
    plt.subplot(212)
    plt.plot(h)
    #plt.show()

    seed = 91234  #8837708
    seed = np.random.randint(9999999)
    print 'seed', seed
    np.random.seed(seed)
    ar1 = -0.9
    err,h = generate_garch(nobs, [1.0, ar1], [1.0,  0.50], mu=0.0,scale=0.1)
#    plt.figure()
#    plt.subplot(211)
#    plt.plot(err)
#    plt.subplot(212)
#    plt.plot(h)
#    plt.figure()
#    plt.subplot(211)
#    plt.plot(err[-400:])
#    plt.subplot(212)
#    plt.plot(h[-400:])
    #plt.show()
    garchplot(err, h)
    garchplot(err[-400:], h[-400:])


    np.random.seed(seed)
    errgjr,hgjr, etax = generate_gjrgarch(nobs, [1.0, ar1],
                                [[1,0],[0.5,0]], mu=0.0,scale=0.1)
    garchplot(errgjr[:nobs], hgjr[:nobs], 'GJR-GARCH(1,1) Simulation - symmetric')
    garchplot(errgjr[-400:nobs], hgjr[-400:nobs], 'GJR-GARCH(1,1) Simulation - symmetric')

    np.random.seed(seed)
    errgjr2,hgjr2, etax = generate_gjrgarch(nobs, [1.0, ar1],
                                [[1,0],[0.1,0.9]], mu=0.0,scale=0.1)
    garchplot(errgjr2[:nobs], hgjr2[:nobs], 'GJR-GARCH(1,1) Simulation')
    garchplot(errgjr2[-400:nobs], hgjr2[-400:nobs], 'GJR-GARCH(1,1) Simulation')

    np.random.seed(seed)
    errgjr3,hgjr3, etax3 = generate_gjrgarch(nobs, [1.0, ar1],
                        [[1,0],[0.1,0.9],[0.1,0.9],[0.1,0.9]], mu=0.0,scale=0.1)
    garchplot(errgjr3[:nobs], hgjr3[:nobs], 'GJR-GARCH(1,3) Simulation')
    garchplot(errgjr3[-400:nobs], hgjr3[-400:nobs], 'GJR-GARCH(1,3) Simulation')

    np.random.seed(seed)
    errgjr4,hgjr4, etax4 = generate_gjrgarch(nobs, [1.0, ar1],
                        [[1., 1,0],[0, 0.1,0.9],[0, 0.1,0.9],[0, 0.1,0.9]],
                        mu=0.0,scale=0.1)
    garchplot(errgjr4[:nobs], hgjr4[:nobs], 'GJR-GARCH(1,3) Simulation')
    garchplot(errgjr4[-400:nobs], hgjr4[-400:nobs], 'GJR-GARCH(1,3) Simulation')

    varinno = np.zeros(100)
    varinno[5] = 1.
    errgjr5,hgjr5, etax5 = generate_gjrgarch(100, [1.0, ar1],
                        [[1., 1,0],[0, 0.1,0.9],[0, 0.1,0.9],[0, 0.1,0.9]],
                        mu=0.0,scale=0.1, varinnovation=varinno)
    garchplot(errgjr5[:nobs], hgjr5[:nobs], 'GJR-GARCH(1,3) Simulation')
    #garchplot(errgjr4[-400:nobs], hgjr4[-400:nobs], 'GJR-GARCH(1,3) Simulation')


    plt.show()

    x = np.arange(20).reshape(10,2)
    x3=np.column_stack((np.ones((x.shape[0],1)),x))
    y, inp = miso_lfilter([1., 0],np.array([[-2.0,3,1],[0.0,0.0,0]]),x3)
    test_misofilter()
