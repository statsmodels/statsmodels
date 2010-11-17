"""
Created on Sun Oct 10 14:57:50 2010

Author: josef-pktd, Skipper Seabold
License: BSD

TODO: check everywhere initialization of signal.lfilter

"""

import numpy as np
from scipy import signal, optimize
from scikits.statsmodels.model import LikelihoodModel, GenericLikelihoodModel


#copied from sandbox/regression/mle.py
#rename until merge of classes is complete
class Arma(GenericLikelihoodModel):  #switch to generic mle
    """
    univariate Autoregressive Moving Average model, conditional on initial values

    The ARMA model is estimated either with conditional Least Squares or with
    conditional Maximum Likelihood. The implementation is using scipy.filter.lfilter
    which makes it faster than the Kalman Filter Implementation. The Kalman Filter
    Implementation however uses the exact Maximum Likelihood and will be more
    accurate, statistically more efficent in small samples.

    In large samples conditional LS, conditional MLE and exact MLE should be very
    close to each other, they are equivalent asymptotically.

    Note:
    this can subclass TSMLEModel

    TODO:
    - CondLS return raw estimation results
    - needs checking that there is no wrong state retained, when running fit several times
        with different options
    - still needs consistent order options.
    - Currently assumes that the mean is zero, no mean or effect of exogenous variables
        are included in the estimation.


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
        #try lfilter_zi, it requires same length for ar and ma
        maxlag = 1+max(p,q)
        armax = np.zeros(maxlag)
        armax[:p+1] = rhoy
        mamax = np.zeros(maxlag)
        mamax[:q+1] = rhoe
        zi = signal.lfilter_zi(armax, mamax)
        errorsest = signal.lfilter(rhoy, rhoe, self.endog, zi=zi)[0] #zi is also returned
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

    #add for Jacobian calculation  bsejac in GenericMLE, copied from loglike
    def nloglikeobs(self, params):
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
        llike  =  0.5 * (np.log(sigma2)
                          + (errorsest**2)/sigma2
                          +  np.log(2*np.pi))
        return llike

#use generic instead
#    def score(self, params):
#        """
#        Score vector for Arma model
#        """
#        #return None
#        #print params
#        jac = ndt.Jacobian(self.loglike, stepMax=1e-4)
#        return jac(params)[-1]


#use generic instead
#    def hessian(self, params):
#        """
#        Hessian of arma model.  Currently uses numdifftools
#        """
#        #return None
#        Hfun = ndt.Jacobian(self.score, stepMax=1e-4)
#        return Hfun(params)[-1]

    #copied from arima.ARIMA, needs splitting out of method specific code
    def fit(self, order=(0,0,0), method="ls", rhoy0=None, rhoe0=None):
        '''
        Estimate lag coefficients of an ARIMA process.

        Parameters
        ----------
            order : sequence
                p,d,q where p is the number of AR lags, d is the number of
                differences to induce stationarity, and q is the number of
                MA lags to estimate.
            method : str {"ls", "ssm"}
                Method of estimation.  LS is conditional least squares.
                SSM is state-space model and the Kalman filter is used to
                maximize the exact likelihood.
            rhoy0, rhoe0 : array_like (optional)
                starting values for estimation

        Returns
        -------
            rh, cov_x, infodict, mesg, ier : output of scipy.optimize.leastsq
            rh :
                estimate of lag parameters, concatenated [rhoy, rhoe]
            cov_x :
                unscaled (!) covariance matrix of coefficient estimates
        '''
        if not hasattr(order, '__iter__'):
            raise ValueError("order must be an iterable sequence.  Got type \
%s instead" % type(order))

        p,d,q = order
        self.nar = p  # needed for geterrors, needs cleanup
        self.nma = q

        if d > 0:
            raise ValueError("Differencing not implemented yet")
            # assume no constant, ie mu = 0
            # unless overwritten then use w_bar for mu
            Y = np.diff(endog, d, axis=0) #TODO: handle lags?

        x = self.endog.squeeze() # remove the squeeze might be needed later
#        def errfn( rho):
#            #rhoy, rhoe = rho
#            rhoy = np.concatenate(([1], rho[:p]))
#            rhoe = np.concatenate(([1], rho[p:]))
#            etahatr = signal.lfilter(rhoy, rhoe, x)
#            #print rho,np.sum(etahatr*etahatr)
#            return etahatr

        #replace with start_params
        if rhoy0 is None:
            rhoy0 = 0.5 * np.ones(p)
        if rhoe0 is None:
            rhoe0 = 0.5 * np.ones(q)

        method = method.lower()

        if method == "ls":
            #changes: use self.geterrors  (nobs,):
#            rh, cov_x, infodict, mesg, ier = \
#               optimize.leastsq(errfn, np.r_[rhoy0, rhoe0],ftol=1e-10,full_output=True)
            rh, cov_x, infodict, mesg, ier = \
               optimize.leastsq(self.geterrors, np.r_[rhoy0, rhoe0],ftol=1e-10,full_output=True)
            #TODO: need missing parameter estimates for LS, scale, residual-sdt
            #TODO: integrate this into the MLE.fit framework?
        elif method == "ssm":
            pass
        else:  #this is also conditional least squares
            # fmin_bfgs is slow or doesn't work yet
            errfnsum = lambda rho : np.sum(self.geterrors(rho)**2)
            #xopt, {fopt, gopt, Hopt, func_calls, grad_calls
            rh, fopt, gopt, cov_x, _,_, ier = \
                optimize.fmin_bfgs(errfnsum, np.r_[rhoy0, rhoe0], maxiter=2, full_output=True)
            infodict, mesg = None, None
        self.params = rh
        self.ar_est = np.concatenate(([1], rh[:p]))
        self.ma_est = np.concatenate(([1], rh[p:p+q]))
        #rh[-q:])) doesnt work for q=0, added p+q as endpoint for safety if var is included
        self.error_estimate = self.geterrors(rh)
        return rh, cov_x, infodict, mesg, ier


    #renamed and needs check with other fit
    def fit_mle(self, order=(0,0), start_params=None, method='nm', maxiter=5000, tol=1e-08):
        '''Estimate an ARMA model with given order using Conditional Maximum Likelihood

        Parameters
        ----------
        order : tuple, 2 elements
            specifies the number of lags(nar, nma) to include, not including lag 0
        start_params : array_like, 1d, (nar+nma+1,)
            start parameters for the optimization, the length needs to be equal to the
            number of ar plus ma coefficients plus 1 for the residual variance
        method : str
            optimization method, as described in LikelihoodModel
        maxiter : int
            maximum number of iteration in the optimization
        tol : float
            tolerance (?) for the optimization

        Returns
        -------
        mlefit : instance of (GenericLikelihood ?)Result class
            contains estimation results and additional statistics

        '''
        nar, nma = order
        self.nar, self.nma = nar, nma
        if start_params is None:
            start_params = np.concatenate((0.05*np.ones(nar + nma), [1]))
        mlefit = super(Arma, self).fit(start_params=start_params,
                maxiter=maxiter, method=method, tol=tol)
        #bug fix: running ls and then mle didn't overwrite this
        rh = mlefit.params
        self.params = rh
        self.ar_est = np.concatenate(([1], rh[:p]))
        self.ma_est = np.concatenate(([1], rh[p:p+q]))
        return mlefit

    #copied from arima.ARIMA
    def predicted(self, ar=None, ma=None):
        '''past predicted values of time series
        just added, not checked yet
        '''

#        #ar, ma not used, not useful as arguments for predicted pattern
#        #need it for prediction for other time series, endog
#        if ar is None:
#            ar = self.ar_est
#        if ma is None:
#            ma = self.ma_est
        return self.x + self.error_estimate

    #copied from arima.ARIMA
    def forecast(self, ar=None, ma=None, nperiod=10):
        '''nperiod ahead forecast at the end of the data period
        '''
        eta = np.r_[self.error_estimate, np.zeros(nperiod)]
        if ar is None:
            ar = self.ar_est
        if ma is None:
            ma = self.ma_est
        return signal.lfilter(ma, ar, eta)

    def forecast2(self, step_ahead=1, endog=None):
        '''rolling h-period ahead forecast without reestimation

        Notes
        -----
        just the idea:
        To improve performance with expanding arrays, specify total period by endog
        and the conditional forecast period by step_ahead

        This should be used by/with results which should contain predicted error or
        noise. Could be either a recursive loop or lfilter with a h-step ahead
        forecast filter, but then I need to calculate that one. ???

        further extension: allow reestimation option

        question: return h-step ahead or range(h)-step ahead ?
        '''
        pass

    #copied from arima.ARIMA
    #TODO: is this needed as a method at all?
    #JP: not needed in this form, but can be replace with using the parameters
    @classmethod
    def generate_sample(cls, ar, ma, nsample, std=1):
        eta = std * np.random.randn(nsample)
        return signal.lfilter(ma, ar, eta)

