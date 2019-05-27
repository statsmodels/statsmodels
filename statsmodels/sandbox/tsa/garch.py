'''general non-linear MLE for time series analysis

idea for general version
------------------------

subclass defines geterrors(parameters) besides loglike,...
and covariance matrix of parameter estimates (e.g. from hessian
or outerproduct of jacobian)
update: I don't really need geterrors directly, but get_h the conditional
    variance process

new version Garch0 looks ok, time to clean up and test
no constraints yet
in some cases: "Warning: Maximum number of function evaluations has been exceeded."

Notes
-----

idea: cache intermediate design matrix for geterrors so it doesn't need
    to be build at each function call

superclass or result class calculates result statistic based
on errors, loglike, jacobian and cov/hessian
  -> aic, bic, ...
  -> test statistics, tvalue, fvalue, ...
  -> new to add: distribution (mean, cov) of non-linear transformation
  -> parameter restrictions or transformation with corrected covparams (?)
  -> sse, rss, rsquared  ??? are they defined from this in general
  -> robust parameter cov ???
  -> additional residual based tests, NW, ... likelihood ratio, lagrange
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


roadmap for garch:
 * simple case
 * starting values: garch11 explicit formulas
 * arma-garch, assumed separable, blockdiagonal Hessian
 * empirical example: DJI, S&P500, MSFT, ???
 * other standard garch: egarch, pgarch,
 * non-normal distributions
 * other methods: forecast, news impact curves (impulse response)
 * analytical gradient, Hessian for basic garch
 * cleaner simulation of garch
 * result statistics, AIC, ...
 * parameter constraints
 * try penalization for higher lags
 * other garch: regime-switching

for pgarch (power garch) need transformation of etax given
   the parameters, but then misofilter should work
   general class aparch (see garch glossary)

References
----------

see notes_references.txt


Created on Feb 6, 2010
@author: "josef pktd"
'''
from __future__ import print_function
from statsmodels.compat.python import zip
import numpy as np
from numpy.testing import assert_almost_equal

from scipy import optimize, signal

import matplotlib.pyplot as plt

import numdifftools as ndt

from statsmodels.base.model import Model, LikelihoodModelResults
from statsmodels.tsa.filters.filtertools import miso_lfilter
from statsmodels.sandbox import tsa


def sumofsq(x, axis=0):
    """Helper function to calculate sum of squares along first axis"""
    return np.sum(x**2, axis=axis)


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
        #print(np.sum(np.log(sigma2),axis))
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

    def initialize(self):
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
        if method not in methods:
            raise ValueError("Unknown fit method %s" % method)
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


#TODO: I take it this is only a stub and should be included in another
# model class?
class TSMLEModel(LikelihoodModel):
    """
    univariate time series model for estimation with maximum likelihood

    Note: This is not working yet
    """

    def __init__(self, endog, exog=None):
        #need to override p,q (nar,nma) correctly
        super(TSMLEModel, self).__init__(endog, exog)
        #set default arma(1,1)
        self.nar = 1
        self.nma = 1
        #self.initialize()

    def geterrors(self, params):
        raise NotImplementedError

    def loglike(self, params):
        """
        Loglikelihood for timeseries model

        Notes
        -----
        needs to be overwritten by subclass
        """
        raise NotImplementedError


    def score(self, params):
        """
        Score vector for Arma model
        """
        #return None
        #print(params
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
        '''estimate model by minimizing negative loglikelihood

        does this need to be overwritten ?
        '''
        if start_params is None and hasattr(self, '_start_params'):
            start_params = self._start_params
        #start_params = np.concatenate((0.05*np.ones(self.nar + self.nma), [1]))
        mlefit = super(TSMLEModel, self).fit(start_params=start_params,
                maxiter=maxiter, method=method, tol=tol)
        return mlefit

class Garch0(TSMLEModel):
    '''Garch model,

    still experimentation stage:
    simplified structure, plain garch, no constraints
    still looking for the design of the base class

    serious bug:
    ar estimate looks ok, ma estimate awful
    -> check parameterization of lagpolys and constant
    looks ok after adding missing constant
        but still difference to garch11 function
    corrected initial condition
    -> only small differences left between the 3 versions
    ar estimate is close to true/DGP model
    note constant has different parameterization
    but design looks better

    '''
    def __init__(self, endog, exog=None):
        #need to override p,q (nar,nma) correctly
        super(Garch0, self).__init__(endog, exog)
        #set default arma(1,1)
        self.nar = 1
        self.nma = 1
        #self.initialize()
        # put this in fit (?) or in initialize instead
        self._etax = endog**2
        self._icetax = np.atleast_1d(self._etax.mean())

    def initialize(self):
        pass

    def geth(self, params):
        '''

        Parameters
        ----------
        params : tuple, (ar, ma)
            try to keep the params conversion in loglike

        copied from generate_gjrgarch
        needs to be extracted to separate function
        '''
        #mu, ar, ma = params
        ar, ma, mu = params

        #etax = self.endog  #this would be enough for basic garch version
        etax = self._etax + mu
        icetax = self._icetax  #read ic-eta-x, initial condition

        #TODO: where does my go with lfilter ?????????????
        #      shouldn't matter except for interpretation

        nobs = etax.shape[0]

        #check arguments of lfilter
        zi = signal.lfiltic(ma,ar, icetax)
        #h = signal.lfilter(ar, ma, etax, zi=zi) #np.atleast_1d(etax[:,1].mean()))
        #just guessing: b/c ValueError: BUG: filter coefficient a[0] == 0 not supported yet
        h = signal.lfilter(ma, ar, etax, zi=zi)[0]
        return h


    def loglike(self, params):
        """
        Loglikelihood for timeseries model

        Notes
        -----
        needs to be overwritten by subclass

        make more generic with using function _convertparams
        which could also include parameter transformation
        _convertparams_in, _convertparams_out

        allow for different distributions t, ged,...
        """
        p, q = self.nar, self.nma
        ar = np.concatenate(([1], params[:p]))

        # check where constant goes

        #ma = np.zeros((q+1,3))
        #ma[0,0] = params[-1]
        #lag coefficients for ma innovation
        ma = np.concatenate(([0], params[p:p+q]))

        mu = params[-1]
        params = (ar, ma, mu) #(ar, ma)

        h = self.geth(params)

        #temporary safe for debugging:
        self.params_converted = params
        self.h = h  #for testing

        sigma2 = np.maximum(h, 1e-6)
        axis = 0
        nobs = len(h)
        #this doesn't help for exploding paths
        #errorsest[np.isnan(errorsest)] = 100
        axis=0 #no choice of axis

        # same as with y = self.endog, ht = sigma2
        # np.log(stats.norm.pdf(y,scale=np.sqrt(ht))).sum()
        llike  =  -0.5 * (np.sum(np.log(sigma2),axis)
                          + np.sum(((self.endog)**2)/sigma2, axis)
                          +  nobs*np.log(2*np.pi))
        return llike

class GarchX(TSMLEModel):
    '''Garch model,

    still experimentation stage:
    another version, this time with exog and miso_filter
    still looking for the design of the base class

    not done yet, just a design idea
    * use misofilter as in garch (gjr)
    * but take etax = exog
      this can include constant, asymetric effect (gjr) and
      other explanatory variables (e.g. high-low spread)

    todo: renames
    eta -> varprocess
    etax -> varprocessx
    icetax -> varprocessic (is actually ic of eta/sigma^2)
    '''
    def __init__(self, endog, exog=None):
        #need to override p,q (nar,nma) correctly
        super(Garch0, self).__init__(endog, exog)
        #set default arma(1,1)
        self.nar = 1
        self.nma = 1
        #self.initialize()
        # put this in fit (?) or in initialize instead
        #nobs defined in super - verify
        #self.nobs = nobs = endog.shape[0]
        #add nexog to super
        #self.nexog = nexog = exog.shape[1]
        self._etax = np.column_stack(np.ones((nobs,1)), endog**2, exog)
        self._icetax = np.atleast_1d(self._etax.mean())

    def initialize(self):
        pass

    def convert_mod2params(ar, ma, mu):
        pass

    def geth(self, params):
        '''

        Parameters
        ----------
        params : tuple, (ar, ma)
            try to keep the params conversion in loglike

        copied from generate_gjrgarch
        needs to be extracted to separate function
        '''
        #mu, ar, ma = params
        ar, ma, mu = params

        #etax = self.endog  #this would be enough for basic garch version
        etax = self._etax + mu
        icetax = self._icetax  #read ic-eta-x, initial condition

        #TODO: where does my go with lfilter ?????????????
        #      shouldn't matter except for interpretation

        nobs = self.nobs

##        #check arguments of lfilter
##        zi = signal.lfiltic(ma,ar, icetax)
##        #h = signal.lfilter(ar, ma, etax, zi=zi) #np.atleast_1d(etax[:,1].mean()))
##        #just guessing: b/c ValueError: BUG: filter coefficient a[0] == 0 not supported yet
##        h = signal.lfilter(ma, ar, etax, zi=zi)[0]
##
        h = miso_lfilter(ar, ma, etax, useic=self._icetax)[0]
        #print('h.shape', h.shape
        hneg = h<0
        if hneg.any():
            #h[hneg] = 1e-6
            h = np.abs(h)
            #todo: raise warning, maybe not during optimization calls

        return h


    def loglike(self, params):
        """
        Loglikelihood for timeseries model

        Notes
        -----
        needs to be overwritten by subclass

        make more generic with using function _convertparams
        which could also include parameter transformation
        _convertparams_in, _convertparams_out

        allow for different distributions t, ged,...
        """
        p, q = self.nar, self.nma
        ar = np.concatenate(([1], params[:p]))

        # check where constant goes

        #ma = np.zeros((q+1,3))
        #ma[0,0] = params[-1]
        #lag coefficients for ma innovation
        ma = np.concatenate(([0], params[p:p+q]))

        mu = params[-1]
        params = (ar, ma, mu) #(ar, ma)

        h = self.geth(params)

        #temporary safe for debugging:
        self.params_converted = params
        self.h = h  #for testing

        sigma2 = np.maximum(h, 1e-6)
        axis = 0
        nobs = len(h)
        #this doesn't help for exploding paths
        #errorsest[np.isnan(errorsest)] = 100
        axis=0 #no choice of axis

        # same as with y = self.endog, ht = sigma2
        # np.log(stats.norm.pdf(y,scale=np.sqrt(ht))).sum()
        llike  =  -0.5 * (np.sum(np.log(sigma2),axis)
                          + np.sum(((self.endog)**2)/sigma2, axis)
                          +  nobs*np.log(2*np.pi))
        return llike


class Garch(TSMLEModel):
    '''Garch model gjrgarch (t-garch)

    still experimentation stage, try with

    '''
    def __init__(self, endog, exog=None):
        #need to override p,q (nar,nma) correctly
        super(Garch, self).__init__(endog, exog)
        #set default arma(1,1)
        self.nar = 1
        self.nma = 1
        #self.initialize()

    def initialize(self):
        pass

    def geterrors(self, params):
        '''

        Parameters
        ----------
        params : tuple, (mu, ar, ma)
            try to keep the params conversion in loglike

        copied from generate_gjrgarch
        needs to be extracted to separate function
        '''
        #mu, ar, ma = params
        ar, ma = params
        eta = self.endog
        nobs = eta.shape[0]

        etax = np.empty((nobs,3))
        etax[:,0] = 1
        etax[:,1:] = (eta**2)[:,None]
        etax[eta>0,2] = 0
        #print('etax.shape', etax.shape
        h = miso_lfilter(ar, ma, etax, useic=np.atleast_1d(etax[:,1].mean()))[0]
        #print('h.shape', h.shape
        hneg = h<0
        if hneg.any():
            #h[hneg] = 1e-6
            h = np.abs(h)

            #print('Warning negative variance found'

        #check timing, starting time for h and eta, do they match
        #err = np.sqrt(h[:len(eta)])*eta #np.random.standard_t(8, size=len(h))
        # let it break if there is a len/shape mismatch
        err = np.sqrt(h)*eta
        return err, h, etax

    def loglike(self, params):
        """
        Loglikelihood for timeseries model

        Notes
        -----
        needs to be overwritten by subclass
        """
        p, q = self.nar, self.nma
        ar = np.concatenate(([1], params[:p]))
        #ar = np.concatenate(([1], -np.abs(params[:p]))) #???
        #better safe than fast and sorry
        #
        ma = np.zeros((q+1,3))
        ma[0,0] = params[-1]
        #lag coefficients for ma innovation
        ma[:,1] = np.concatenate(([0], params[p:p+q]))
        #delta lag coefficients for negative ma innovation
        ma[:,2] = np.concatenate(([0], params[p+q:p+2*q]))

        mu = params[-1]
        params = (ar, ma) #(mu, ar, ma)

        errorsest, h, etax = self.geterrors(params)
        #temporary safe for debugging
        self.params_converted = params
        self.errorsest, self.h, self.etax = errorsest, h, etax
        #h = h[:-1] #correct this in geterrors
        #print('shapes errorsest, h, etax', errorsest.shape, h.shape, etax.shape
        sigma2 = np.maximum(h, 1e-6)
        axis = 0
        nobs = len(errorsest)
        #this doesn't help for exploding paths
        #errorsest[np.isnan(errorsest)] = 100
        axis=0 #not used
#        muy = errorsest.mean()
#        # llike is verified, see below
#        # same as with y = errorsest, ht = sigma2
#        # np.log(stats.norm.pdf(y,scale=np.sqrt(ht))).sum()
#        llike  =  -0.5 * (np.sum(np.log(sigma2),axis)
#                          + np.sum(((errorsest)**2)/sigma2, axis)
#                          +  nobs*np.log(2*np.pi))
#        return llike
        muy = errorsest.mean()
        # llike is verified, see below
        # same as with y = errorsest, ht = sigma2
        # np.log(stats.norm.pdf(y,scale=np.sqrt(ht))).sum()
        llike  =  -0.5 * (np.sum(np.log(sigma2),axis)
                          + np.sum(((self.endog)**2)/sigma2, axis)
                          +  nobs*np.log(2*np.pi))
        return llike


def gjrconvertparams(self, params, nar, nma):
    """
    flat to matrix

    Notes
    -----
    needs to be overwritten by subclass
    """
    p, q = nar, nma
    ar = np.concatenate(([1], params[:p]))
    #ar = np.concatenate(([1], -np.abs(params[:p]))) #???
    #better safe than fast and sorry
    #
    ma = np.zeros((q+1,3))
    ma[0,0] = params[-1]
    #lag coefficients for ma innovation
    ma[:,1] = np.concatenate(([0], params[p:p+q]))
    #delta lag coefficients for negative ma innovation
    ma[:,2] = np.concatenate(([0], params[p+q:p+2*q]))

    mu = params[-1]
    params2 = (ar, ma) #(mu, ar, ma)
    return paramsclass  # noqa:F821  # See GH#5756


#TODO: this should be generalized to ARMA?
#can possibly also leverage TSME above
# also note that this is NOT yet general
# it was written for my homework, assumes constant is zero
# and that process is AR(1)
# examples at the end of run as main below
class AR(LikelihoodModel):
    """
    Notes
    -----
    This is not general, only written for the AR(1) case.

    Fit methods that use super and broyden do not yet work.
    """
    def __init__(self, endog, exog=None, nlags=1):
        if exog is None:    # extend to handle ADL(p,q) model? or subclass?
            exog = endog[:-nlags]
        endog = endog[nlags:]
        super(AR, self).__init__(endog, exog)
        self.nobs += nlags # add lags back to nobs for real T

#TODO: need to fix underscore in Model class.
#Done?
    def initialize(self):
        pass

    def loglike(self, params):
        """
        The unconditional loglikelihood of an AR(p) process

        Notes
        -----
        Contains constant term.
        """
        nobs = self.nobs
        y = self.endog
        ylag = self.exog
        penalty = self.penalty
        if isinstance(params,tuple):
            # broyden (all optimize.nonlin return a tuple until rewrite commit)
            params = np.asarray(params)
        usepenalty=False
        if not np.all(np.abs(params)<1) and penalty:
            oldparams = params
            params = np.array([.9999]) # make it the edge
            usepenalty=True
        diffsumsq = sumofsq(y-np.dot(ylag,params))
        # concentrating the likelihood means that sigma2 is given by
        sigma2 = 1/nobs*(diffsumsq-ylag[0]**2*(1-params**2))
        loglike = -nobs/2 * np.log(2*np.pi) - nobs/2*np.log(sigma2) + \
                .5 * np.log(1-params**2) - .5*diffsumsq/sigma2 -\
                ylag[0]**2 * (1-params**2)/(2*sigma2)
        if usepenalty:
        # subtract a quadratic penalty since we min the negative of loglike
            loglike -= 1000 *(oldparams-.9999)**2
        return loglike

    def score(self, params):
        """
        Notes
        -----
        Need to generalize for AR(p) and for a constant.
        Not correct yet.  Returns numerical gradient.  Depends on package
        numdifftools.
        """
        y = self.endog
        ylag = self.exog
        nobs = self.nobs
        diffsumsq = sumofsq(y-np.dot(ylag,params))
        dsdr = 1/nobs * -2 *np.sum(ylag*(y-np.dot(ylag,params))[:,None])+\
                2*params*ylag[0]**2
        sigma2 = 1/nobs*(diffsumsq-ylag[0]**2*(1-params**2))
        gradient = -nobs/(2*sigma2)*dsdr + params/(1-params**2) + \
                1/sigma2*np.sum(ylag*(y-np.dot(ylag, params))[:,None])+\
                .5*sigma2**-2*diffsumsq*dsdr+\
                ylag[0]**2*params/sigma2 +\
                ylag[0]**2*(1-params**2)/(2*sigma2**2)*dsdr
        if self.penalty:
            pass
        j = ndt.Jacobian(self.loglike)
        return j(params)
#        return gradient


    def information(self, params):
        """
        Not Implemented Yet
        """
        return

    def hessian(self, params):
        """
        Returns numerical hessian for now.  Depends on numdifftools.
        """

        h = ndt.Hessian(self.loglike)
        return h(params)

    def fit(self, start_params=None, method='bfgs', maxiter=35, tol=1e-08,
            penalty=False):
        """
        Fit the unconditional maximum likelihood of an AR(p) process.

        Parameters
        ----------
        start_params : array-like, optional
            A first guess on the parameters.  Defaults is a vector of zeros.
        method : str, optional
            Unconstrained solvers:
                Default is 'bfgs', 'newton' (newton-raphson), 'ncg'
                (Note that previous 3 are not recommended at the moment.)
                and 'powell'
            Constrained solvers:
                'bfgs-b', 'tnc'
            See notes.
        maxiter : int, optional
            The maximum number of function evaluations. Default is 35.
        tol = float
            The convergence tolerance.  Default is 1e-08.
        penalty : bool
            Whether or not to use a penalty function.  Default is False,
            though this is ignored at the moment and the penalty is always
            used if appropriate.  See notes.

        Notes
        -----
        The unconstrained solvers use a quadratic penalty (regardless if
        penalty kwd is True or False) in order to ensure that the solution
        stays within (-1,1).  The constrained solvers default to using a bound
        of (-.999,.999).
        """
        self.penalty = penalty
        method = method.lower()
#TODO: allow user-specified penalty function
#        if penalty and method not in ['bfgs_b','tnc','cobyla','slsqp']:
#            minfunc = lambda params : -self.loglike(params) - \
#                    self.penfunc(params)
#        else:
        minfunc = lambda params: -self.loglike(params)
        if method in ['newton', 'bfgs', 'ncg']:
            super(AR, self).fit(start_params=start_params, method=method,
                    maxiter=maxiter, tol=tol)
        else:
            bounds = [(-.999,.999)]   # assume stationarity
            if start_params is None:
                start_params = np.array([0]) #TODO: assumes AR(1)
            if method == 'bfgs-b':
                retval = optimize.fmin_l_bfgs_b(minfunc, start_params,
                        approx_grad=True, bounds=bounds)
                self.params, self.llf = retval[0:2]
            if method == 'tnc':
                retval = optimize.fmin_tnc(minfunc, start_params,
                        approx_grad=True, bounds = bounds)
                self.params = retval[0]
            if method == 'powell':
                retval = optimize.fmin_powell(minfunc,start_params)
                self.params = retval[None]
#TODO: write regression tests for Pauli's branch so that
# new line_search and optimize.nonlin can get put in.
#http://projects.scipy.org/scipy/ticket/791
#            if method == 'broyden':
#                retval = optimize.broyden2(minfunc, [.5], verbose=True)
#                self.results = retval


class Arma(LikelihoodModel):
    """
    univariate Autoregressive Moving Average model

    Note: This is not working yet, or does it
    this can subclass TSMLEModel
    """

    def __init__(self, endog, exog=None):
        #need to override p,q (nar,nma) correctly
        super(Arma, self).__init__(endog, exog)
        #set default arma(1,1)
        self.nar = 1
        self.nma = 1
        #self.initialize()

    def initialize(self):
        pass

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
        #print(params
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
        if start_params is None:
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
    #arest = tsa.arima.ARIMA()
    #arest = tsa.arima.ARIMA  #try class method, ARIMA needs data in constructor
    from statsmodels.tsa.arima_process import arma_generate_sample
    h = arma_generate_sample(ar,ma,nobs,0.1)
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
    err = np.sqrt(h)*eta #np.random.standard_t(8, size=nobs)
    return err, h



def generate_gjrgarch(nobs, ar, ma, mu=1., scale=0.1, varinnovation=None):
    '''simulate gjr garch process

    Parameters
    ----------
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
    #print('h.shape', h.shape)
    err = np.sqrt(h[:len(eta)])*eta #np.random.standard_t(8, size=len(h))
    return err, h, etax

def loglike_GARCH11(params, y):
    # Computes the likelihood vector of a GARCH11
    # assumes y is centered

    w     =  params[0] # constant (1);
    alpha =  params[1] # coefficient of lagged squared error
    beta  =  params[2] # coefficient of lagged variance

    y2   = y**2
    nobs = y2.shape[0]
    ht    = np.zeros(nobs)
    ht[0] = y2.mean()  #sum(y2)/T;

    for i in range(1,nobs):
        ht[i] = w + alpha*y2[i-1] + beta * ht[i-1]

    sqrtht  = np.sqrt(ht)
    x       = y/sqrtht

    llvalues = -0.5*np.log(2*np.pi) - np.log(sqrtht) - 0.5*(x**2)
    return llvalues.sum(), llvalues, ht


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

def test_gjrgarch():
    # test impulse response of gjr simulator
    varinno = np.zeros(100)
    varinno[0] = 1.
    errgjr5,hgjr5, etax5 = generate_gjrgarch(100, [1.0, 0],
                        [[1., 1,0],[0, 0.1,0.8],[0, 0.05,0.7],[0, 0.01,0.6]],
                        mu=0.0,scale=0.1, varinnovation=varinno)
    ht = np.array([ 1., 0.1, 0.05,  0.01, 0., 0.  ])
    assert_almost_equal(hgjr5[:6], ht, decimal=15)

    errgjr5,hgjr5, etax5 = generate_gjrgarch(100, [1.0, -1.0],
                        [[1., 1,0],[0, 0.1,0.8],[0, 0.05,0.7],[0, 0.01,0.6]],
                        mu=0.0,scale=0.1, varinnovation=varinno)
    assert_almost_equal(hgjr5[:6], ht.cumsum(), decimal=15)

    errgjr5,hgjr5, etax5 = generate_gjrgarch(100, [1.0, 1.0],
                        [[1., 1,0],[0, 0.1,0.8],[0, 0.05,0.7],[0, 0.01,0.6]],
                        mu=0.0,scale=0.1, varinnovation=varinno)
    ht1 = [0]
    for h in ht: ht1.append(h-ht1[-1])
    assert_almost_equal(hgjr5[:6], ht1[1:], decimal=15)

    # negative shock
    varinno = np.zeros(100)
    varinno[0] = -1.
    errgjr5,hgjr5, etax5 = generate_gjrgarch(100, [1.0, 0],
                        [[1., 1,0],[0, 0.1,0.8],[0, 0.05,0.7],[0, 0.01,0.6]],
                        mu=0.0,scale=0.1, varinnovation=varinno)
    ht = np.array([ 1.  ,  0.9 ,  0.75,  0.61,  0.  ,  0.  ])
    assert_almost_equal(hgjr5[:6], ht, decimal=15)

    errgjr5,hgjr5, etax5 = generate_gjrgarch(100, [1.0, -1.0],
                        [[1., 1,0],[0, 0.1,0.8],[0, 0.05,0.7],[0, 0.01,0.6]],
                        mu=0.0,scale=0.1, varinnovation=varinno)
    assert_almost_equal(hgjr5[:6], ht.cumsum(), decimal=15)

    errgjr5,hgjr5, etax5 = generate_gjrgarch(100, [1.0, 1.0],
                        [[1., 1,0],[0, 0.1,0.8],[0, 0.05,0.7],[0, 0.01,0.6]],
                        mu=0.0,scale=0.1, varinnovation=varinno)
    ht1 = [0]
    for h in ht: ht1.append(h-ht1[-1])
    assert_almost_equal(hgjr5[:6], ht1[1:], decimal=15)


'''
>>> print(signal.correlate(x3, np.array([[-2.0,3,1],[0.0,0.0,0]])[::-1,:],mode='full')[:-1, (x3.shape[1]+1)//2]
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

    #test_misofilter()
    #test_gjrgarch()

    examples = ['garch']
    if 'arma' in examples:
        arest = tsa.arima.ARIMA()
        print("\nExample 1")
        ar = [1.0, -0.8]
        ma = [1.0,  0.5]
        y1 = arest.generate_sample(ar,ma,1000,0.1)
        y1 -= y1.mean() #no mean correction/constant in estimation so far

        arma1 = Arma(y1)
        arma1.nar = 1
        arma1.nma = 1
        arma1res = arma1.fit(method='fmin')
        print(arma1res.params)

        #Warning need new instance otherwise results carry over
        arma2 = Arma(y1)
        res2 = arma2.fit(method='bfgs')
        print(res2.params)
        print(res2.model.hessian(res2.params))
        print(ndt.Hessian(arma1.loglike, stepMax=1e-2)(res2.params))
        resls = arest.fit(y1,1,1)
        print(resls[0])
        print(resls[1])



        print('\nparameter estimate')
        print('parameter of DGP ar(1), ma(1), sigma_error')
        print([-0.8, 0.5, 0.1])
        print('mle with fmin')
        print(arma1res.params)
        print('mle with bfgs')
        print(res2.params)
        print('cond. least squares uses optim.leastsq ?')
        errls = arest.error_estimate
        print(resls[0], np.sqrt(np.dot(errls,errls)/errls.shape[0]))

        err = arma1.geterrors(res2.params)
        print('cond least squares parameter cov')
        #print(np.dot(err,err)/err.shape[0] * resls[1])
        #errls = arest.error_estimate
        print(np.dot(errls,errls)/errls.shape[0] * resls[1])
    #    print('fmin hessian')
    #    print(arma1res.model.optimresults['Hopt'][:2,:2])
        print('bfgs hessian')
        print(res2.model.optimresults['Hopt'][:2,:2])
        print('numdifftools inverse hessian')
        print(-np.linalg.inv(ndt.Hessian(arma1.loglike, stepMax=1e-2)(res2.params))[:2,:2])

        arma3 = Arma(y1**2)
        res3 = arma3.fit(method='bfgs')
        print(res3.params)

    nobs = 1000

    if 'garch' in examples:
        err,h = generate_kindofgarch(nobs, [1.0, -0.95], [1.0,  0.1], mu=0.5)
        plt.figure()
        plt.subplot(211)
        plt.plot(err)
        plt.subplot(212)
        plt.plot(h)
        #plt.show()

        seed = 3842774 #91234  #8837708
        seed = np.random.randint(9999999)
        print('seed', seed)
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
        varinno[0] = 1.
        errgjr5,hgjr5, etax5 = generate_gjrgarch(100, [1.0, -0.],
                            [[1., 1,0],[0, 0.1,0.8],[0, 0.05,0.7],[0, 0.01,0.6]],
                            mu=0.0,scale=0.1, varinnovation=varinno)
        garchplot(errgjr5[:20], hgjr5[:20], 'GJR-GARCH(1,3) Simulation')
        #garchplot(errgjr4[-400:nobs], hgjr4[-400:nobs], 'GJR-GARCH(1,3) Simulation')


    #plt.show()
    seed = np.random.randint(9999999)  # 9188410
    print('seed', seed)

    x = np.arange(20).reshape(10,2)
    x3 = np.column_stack((np.ones((x.shape[0],1)),x))
    y, inp = miso_lfilter([1., 0],np.array([[-2.0,3,1],[0.0,0.0,0]]),x3)

    nobs = 1000
    warmup = 1000
    np.random.seed(seed)
    ar = [1.0, -0.7]#7, -0.16, -0.1]
    #ma = [[1., 1, 0],[0, 0.6,0.1],[0, 0.1,0.1],[0, 0.1,0.1]]
    ma = [[1., 0, 0],[0, 0.4,0.0]] #,[0, 0.9,0.0]]
#    errgjr4,hgjr4, etax4 = generate_gjrgarch(warmup+nobs, [1.0, -0.99],
#                        [[1., 1, 0],[0, 0.6,0.1],[0, 0.1,0.1],[0, 0.1,0.1]],
#                        mu=0.2, scale=0.25)

    errgjr4,hgjr4, etax4 = generate_gjrgarch(warmup+nobs, ar, ma,
                         mu=0.4, scale=1.01)
    errgjr4,hgjr4, etax4 = errgjr4[warmup:], hgjr4[warmup:], etax4[warmup:]
    garchplot(errgjr4[:nobs], hgjr4[:nobs], 'GJR-GARCH(1,3) Simulation')
    ggmod = Garch(errgjr4-errgjr4.mean())#hgjr4[:nobs])#-hgjr4.mean()) #errgjr4)
    ggmod.nar = 1
    ggmod.nma = 1
    ggmod._start_params = np.array([-0.6, 0.1, 0.2, 0.0])
    ggres = ggmod.fit(start_params=np.array([-0.6, 0.1, 0.2, 0.0]), maxiter=1000)
    print('ggres.params', ggres.params)
    garchplot(ggmod.errorsest, ggmod.h)
    #plt.show()

    print('Garch11')
    print(optimize.fmin(lambda params: -loglike_GARCH11(params, errgjr4-errgjr4.mean())[0], [0.93, 0.9, 0.2]))

    ggmod0 = Garch0(errgjr4-errgjr4.mean())#hgjr4[:nobs])#-hgjr4.mean()) #errgjr4)
    ggmod0.nar = 1
    ggmod.nma = 1
    start_params = np.array([-0.6, 0.2, 0.1])
    ggmod0._start_params = start_params #np.array([-0.6, 0.1, 0.2, 0.0])
    ggres0 = ggmod0.fit(start_params=start_params, maxiter=2000)
    print('ggres0.params', ggres0.params)

    ggmod0 = Garch0(errgjr4-errgjr4.mean())#hgjr4[:nobs])#-hgjr4.mean()) #errgjr4)
    ggmod0.nar = 1
    ggmod.nma = 1
    start_params = np.array([-0.6, 0.2, 0.1])
    ggmod0._start_params = start_params #np.array([-0.6, 0.1, 0.2, 0.0])
    ggres0 = ggmod0.fit(start_params=start_params, method='bfgs', maxiter=2000)
    print('ggres0.params', ggres0.params)


    if 'rpy' in examples:
        from rpy import r
        f = r.formula('~garch(1, 1)')
        #fit = r.garchFit(f, data = errgjr4)
        x = r.garchSim( n = 500)
        print('R acf', tsa.acf(np.power(x,2))[:15])
        arma3 = Arma(np.power(x,2))
        arma3res = arma3.fit(start_params=[-0.2,0.1,0.5],maxiter=5000)
        print(arma3res.params)
        arma3b = Arma(np.power(x,2))
        arma3bres = arma3b.fit(start_params=[-0.2,0.1,0.5],maxiter=5000, method='bfgs')
        print(arma3bres.params)

    llf = loglike_GARCH11([0.93, 0.9, 0.2], errgjr4)
    print(llf[0])

    erro,ho, etaxo = generate_gjrgarch(20, ar, ma, mu=0.04, scale=0.01,
                      varinnovation = np.ones(20))


    ''' this looks relatively good

    >>> Arma.initialize = lambda x: x
    >>> arma3 = Arma(errgjr4**2)
    >>> arma3res = arma3.fit()
    Warning: Maximum number of function evaluations has been exceeded.
    >>> arma3res.params
    array([-0.775, -0.583, -0.001])
    >>> arma2.nar
    1
    >>> arma2.nma
    1

    unit root ?
    >>> arma3 = Arma(hgjr4)
    >>> arma3res = arma3.fit()
    Optimization terminated successfully.
             Current function value: -3641.529780
             Iterations: 250
             Function evaluations: 458
    >>> arma3res.params
    array([ -1.000e+00,  -3.096e-04,   6.343e-03])

    or maybe not great
    >>> arma3res = arma3.fit(start_params=[-0.8,0.1,0.5],maxiter=5000)
    Warning: Maximum number of function evaluations has been exceeded.
    >>> arma3res.params
    array([-0.086,  0.186, -0.001])
    >>> arma3res = arma3.fit(start_params=[-0.8,0.1,0.5],maxiter=5000,method='bfgs')
    Divide-by-zero encountered: rhok assumed large
    Optimization terminated successfully.
             Current function value: -5988.332952
             Iterations: 16
             Function evaluations: 245
             Gradient evaluations: 49
    >>> arma3res.params
    array([ -9.995e-01,  -9.715e-01,   6.501e-04])
    '''

    '''
    current problems
    persistence in errgjr looks too low, small tsa.acf(errgjr4**2)[:15]
    as a consequence the ML estimate has also very little persistence,
    estimated ar term is much too small
    -> need to compare with R or matlab

    help.search("garch") :  ccgarch, garchSim(fGarch), garch(tseries)
    HestonNandiGarchFit(fOptions)

    > library('fGarch')
    > spec = garchSpec()
    > x = garchSim(model = spec@model, n = 500)
    > acf(x**2)    # has low correlation
    but fit has high parameters:
    > fit = garchFit(~garch(1, 1), data = x)

    with rpy:

    from rpy import r
    r.library('fGarch')
    f = r.formula('~garch(1, 1)')
    fit = r.garchFit(f, data = errgjr4)
    Final Estimate:
    LLH:  -3198.2    norm LLH:  -3.1982
          mu        omega       alpha1        beta1
    1.870485e-04 9.437557e-05 3.457349e-02 1.000000e-08

    second run with ar = [1.0, -0.8]  ma = [[1., 0, 0],[0, 1.0,0.0]]
    Final Estimate:
    LLH:  -3979.555    norm LLH:  -3.979555
          mu        omega       alpha1        beta1
    1.465050e-05 1.641482e-05 1.092600e-01 9.654438e-02
    mine:
    >>> ggres.params
    array([ -2.000e-06,   3.283e-03,   3.769e-01,  -1.000e-06])

    another rain, same ar, ma
    Final Estimate:
    LLH:  -3956.197    norm LLH:  -3.956197
          mu        omega       alpha1        beta1
    7.487278e-05 1.171238e-06 1.511080e-03 9.440843e-01

    every step needs to be compared and tested

    something looks wrong with likelihood function, either a silly
    mistake or still some conceptional problems

    * found the silly mistake, I was normalizing the errors before
      plugging into espression for likelihood function

    * now gjr garch estimation works and produces results that are very
      close to the explicit garch11 estimation

    initial conditions for miso_filter need to be cleaned up

    lots of clean up to to after the bug hunting

    '''
    y = np.random.randn(20)
    params = [0.93, 0.9, 0.2]
    lls, llt, ht = loglike_GARCH11(params, y)
    sigma2 = ht
    axis=0
    nobs = len(ht)
    llike  =  -0.5 * (np.sum(np.log(sigma2),axis)
                          + np.sum((y**2)/sigma2, axis)
                          +  nobs*np.log(2*np.pi))
    print(lls, llike)
    #print(np.log(stats.norm.pdf(y,scale=np.sqrt(ht))).sum())



    '''
    >>> optimize.fmin(lambda params: -loglike_GARCH11(params, errgjr4)[0], [0.93, 0.9, 0.2])
    Optimization terminated successfully.
             Current function value: 7312.393886
             Iterations: 95
             Function evaluations: 175
    array([ 3.691,  0.072,  0.932])
    >>> ar
    [1.0, -0.93000000000000005]
    >>> ma
    [[1.0, 0, 0], [0, 0.90000000000000002, 0.0]]
    '''


    np.random.seed(1)
    tseries = np.zeros(200) # set first observation
    for i in range(1,200): # get 99 more observations based on the given process
        error = np.random.randn()
        tseries[i] = .9 * tseries[i-1] + .01 * error

    tseries = tseries[100:]

    armodel = AR(tseries)
    #armodel.fit(method='bfgs-b')
    #armodel.fit(method='tnc')
    #powell should be the most robust, see Hamilton 5.7
    armodel.fit(method='powell', penalty=True)
    # The below don't work yet
    #armodel.fit(method='newton', penalty=True)
    #armodel.fit(method='broyden', penalty=True)
    print("Unconditional MLE for AR(1) y_t = .9*y_t-1 +.01 * err")
    print(armodel.params)
