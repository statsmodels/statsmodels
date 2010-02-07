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


if __name__ == '__main__':

    arest = tsa.arima.ARIMA()
    print "\nExample 1"
    ar = [1.0, -0.8]
    ma = [1.0,  0.5]
    y1 = arest.generate_sample(ar,ma,1000,0.1)
    y1 -= y1.mean() #no mean correction/constant in estimation so far

    arma1 = Arma(y1)
    arma1.nar = 1
    arma1.nma = 1
    arma1res = arma1.fit()
    print arma1res.params
    res2 = arma1.fit(method='bfgs')
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
    print 'cond. least squares'
    errls = arest.error_estimate
    print resls[0], np.sqrt(np.dot(errls,errls)/errls.shape[0])

    err = arma1.geterrors(res2.params)
    print 'cond least squares parameter cov'
    #print np.dot(err,err)/err.shape[0] * resls[1]
    #errls = arest.error_estimate
    print np.dot(errls,errls)/errls.shape[0] * resls[1]
    print 'bfgs hessian'
    print res2.model.optimresults['Hopt'][:2,:2]
    print 'numdifftools inverse hessian'
    print -np.linalg.inv(ndt.Hessian(arma1.loglike, stepMax=1e-2)(res2.params))[:2,:2]
