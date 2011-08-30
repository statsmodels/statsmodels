"""
This is the VAR class refactored from pymaclab.
"""
from __future__ import division
import numpy as np
from numpy import (dot, identity, atleast_2d, atleast_1d, zeros)
from numpy.linalg import inv
from scipy import optimize
from scipy.stats import t, norm, ss as sumofsq
from scikits.statsmodels.regression.linear_model import OLS
from scikits.statsmodels.tsa.tsatools import (lagmat, add_trend,
                _ar_transparams, _ar_invtransparams)
import scikits.statsmodels.tsa.base.tsa_model as tsbase
import scikits.statsmodels.base.model as base
from scikits.statsmodels.tools.decorators import (resettable_cache,
        cache_readonly, cache_writable)
from scikits.statsmodels.tools.compatibility import np_slogdet
from scikits.statsmodels.sandbox.regression.numdiff import approx_fprime
from scikits.statsmodels.sandbox.regression.numdiff import (approx_hess,
        approx_hess_cs)
from scikits.statsmodels.tsa.kalmanf.kalmanfilter import KalmanFilter
import scikits.statsmodels.base.wrapper as wrap


__all__ = ['AR']


class AR(tsbase.TimeSeriesModel):
    """
    Autoregressive AR(p) Model

    Parameters
    ----------
    endog : array-like
        Endogenous response variable.
    exog : array-like
        Exogenous variables. Note that exogenous variables are not yet
        supported for AR.
    """
    def __init__(self, endog, exog=None, dates=None):
        super(AR, self).__init__(endog, exog, dates)
        endog = self.endog # original might not have been an ndarray
        if endog.ndim == 1:
            endog = endog[:,None]
            self.endog = endog  # to get shapes right
        elif endog.ndim > 1 and endog.shape[1] != 1:
            raise ValueError("Only the univariate case is implemented")
        if exog is not None:
            raise ValueError("Exogenous variables are not supported for AR.")

    def initialize(self):
        pass

    def _transparams(self, params):
        """
        Transforms params to induce stationarity/invertability.

        Reference
        ---------
        Jones(1980)
        """
        p,k = self.k_ar, self.k_trend # need to include exog here
        newparams = params.copy()
        newparams[k:k+p] = _ar_transparams(params[k:k+p].copy())
        return newparams

    def _invtransparams(self, start_params):
        """
        Inverse of the Jones reparameterization
        """
        p,k = self.k_ar, self.k_trend
        newparams = start_params.copy()
        newparams[k:k+p] = _ar_invtransparams(start_params[k:k+p].copy())
        return newparams

    def _presample_fit(self, params, start, p, y, predictedvalues):
        """
        Return the pre-sample predicted values using the Kalman Filter

        Notes
        -----
        See predict method for how to use start and p.
        """
        k = self.k_trend

        # build system matrices
        T_mat = KalmanFilter.T(params, p, k, p)
        R_mat = KalmanFilter.R(params, p, k, 0, p)

        # Initial State mean and variance
        alpha = np.zeros((p,1))
        Q_0 = dot(inv(identity(p**2)-np.kron(T_mat,T_mat)),dot(R_mat,
                R_mat.T).ravel('F'))

        Q_0 = Q_0.reshape(p,p, order='F') #TODO: order might need to be p+k
        P = Q_0
        Z_mat = atleast_2d([1] + [0] * (p-k))  # TODO: change for exog
        for i in xrange(start,p): #iterate p-1 times to fit presample
            v_mat = y[i] - dot(Z_mat,alpha)
            F_mat = dot(dot(Z_mat, P), Z_mat.T)
            Finv = 1./F_mat # inv. always scalar
            K = dot(dot(dot(T_mat,P),Z_mat.T),Finv)
            # update state
            alpha = dot(T_mat, alpha) + dot(K,v_mat)
            L = T_mat - dot(K,Z_mat)
            P = dot(dot(T_mat, P), L.T) + dot(R_mat, R_mat.T)
#            P[0,0] += 1 # for MA part, R_mat.R_mat.T above
            predictedvalues[i+1-start] = dot(Z_mat,alpha)
        return predictedvalues

    def predict(self, params, n=-1, start=0, method='dynamic', resid=False,
            confint=False):
        """
        Returns in-sample prediction or forecasts.

        Parameters
        ----------
        params : array
            The fitted model parameters.
        n : int
            Number of periods after start to forecast.  If n==-1, returns in-
            sample forecast starting at `start`.
        start : int
            Zero-indexed observation number at which to start forecasting, ie.,
            the first forecast is start.  If start==-1, forecasting starts from
            the end of the sample.  If the model is fit using 'cmle' or 'yw',
            `start` cannot be less than `k_ar`.  If `start` < `k_ar` for
            'cmle' and 'yw', then `start` is set equal to `k_ar`.
        method : string {'dynamic', 'static'}
            If method is 'dynamic', then fitted values are used in place of
            observed 'endog' to make forecasts.  If 'static', observed 'endog'
            are used.
        resid : bool
            Whether or not to return the residuals.
        confint : bool, float
            Whether to return confidence intervals.  If `confint` == True,
            95 % confidence intervals are returned.  Else if `confint` is a
            float, then it is assumed to be the alpha value of the confidence
            interval.  That is confint == .05 returns a 95% confidence
            interval, and .10 would return a 90% confidence interval.

        Returns
        -------
        predicted values : array
        residuals : array, optional
        confidence intervals : array, optional

        Notes
        -----
        The linear Gaussian Kalman filter is used to return pre-sample fitted
        values. The exact initial Kalman Filter is used. See Durbin and Koopman
        in the references for more information.
        """
        if n == 0 or (n==-1 and start==-1):
            return np.array([])

        y = self.endog.copy()
        nobs = int(self.endog.shape[0])

        if start < 0:
            start = nobs + start # convert negative indexing

        p = self.k_ar
        k = self.k_trend
        method = self.method
        if method != 'mle':
            if start == 0:
                start = p # can't do presample fit for != 'mle'

        if n == -1:
            if start != -1 and start < nobs:
                predictedvalues = np.zeros((nobs-start))
                n = nobs-start
            else:
                return np.array([])
        else:
            predictedvalues = zeros((n))

        mu = 0 # overwritten for 'mle' with constant
        if method == 'mle': # use Kalman Filter to get initial values
            if k>=1:    # if constant, demean, #TODO: handle higher trendorders
                mu = params[0]/(1-np.sum(params[k:]))   # only for constant-only
                                        # and exog
                y -= mu
            predictedvalues = self._presample_fit(params, start, p, y,
                    predictedvalues)

        if start < p and (n > p - start or n == -1):
            if n == -1:
                predictedvalues[p-start:] = dot(self.X, params)
            elif n-(p-start) <= nobs-p:
                predictedvalues[p-start:] = dot(self.X,
                        params)[:nobs-(p-start)] #start:nobs-p?
            else:
                predictedvalues[p-start:nobs-(p-start)] = dot(self.X,
                        params) # maybe p-start) - 1?
            predictedvalues[start:p] += mu # does nothing if no constant
        elif start <= nobs:
            if n <= nobs-start:
                predictedvalues[:] = dot(self.X,
#                        params)[start:n+start]
                        params)[start-p:n+start-p]
            else: # right now this handles when start == p only?
                predictedvalues[:nobs-start] = dot(self.X,
                        params)[start-p:]
        else:
#            predictedvalues[:nobs-start] - dot(self.X,params)[p:]
            pass

#NOTE: it only makes sense to forecast beyond nobs+1 if exog is None
        if start + n > nobs:
            endog = self.endog
            if start < nobs:
                if n-(nobs-start) < p:
                    endrange = n
                else:
                    endrange = nobs-start+p
                for i in range(nobs-start,endrange):
                # mixture of static/dynamic
                    predictedvalues[i] = np.sum(np.r_[[1]*k,
                        predictedvalues[nobs-start:i][::-1],
                        atleast_1d(endog[-p+i-nobs+start:][::-1].squeeze())] *\
                                params)
                # dynamic forecasts
                for i in range(nobs-start+p,n):
                    predictedvalues[i] = np.sum(np.r_[[1]*k,
                        predictedvalues[i-p:i][::-1]] * params)
            else: # start > nobs
# if start < nobs + p?
                tmp = np.zeros((start-nobs)) # still calc interim values
# this is only the range for
                if start-nobs < p:
                    endrange = start-nobs
                else:
                    endrange = p
                for i in range(endrange):
                    # mixed static/dynamic
                    tmp[i] = np.sum(np.r_[[1]*k, tmp[:i][::-1],
                            atleast_1d(endog[-p+i:][::-1].squeeze())] * params)
                for i in range(p,start-nobs):
                    tmp[i] = np.sum(np.r_[[1]*k, tmp[i-p:i][::-1]] * params)
                if start - nobs > p:
                    for i in range(p):
                        # mixed tmp/actual
                        predictedvalues[i] = np.sum(np.r_[[1]*k,
                            predictedvalues[:i][::-1],
                            atleast_1d(tmp[-p+i:][::-1].squeeze())] * params)
                else:
                    endtmp = len(tmp)
                    if n < p:
                        endrange = n
                    else:
                        endrange = p-endtmp
                    for i in range(endrange):
                        # mixed endog/tmp/actual
                        predictedvalues[i] = np.sum(np.r_[[1]*k,
                            predictedvalues[:i][::-1],
                            atleast_1d(tmp[-p+i:][::-1].squeeze()),
                            atleast_1d(endog[-\
                            (p-i-endtmp):][::-1].squeeze())] * params)
                    if n > endrange:
                        for i in range(endrange,p):
                            # mixed tmp/actual
                            predictedvalues[i] = np.sum(np.r_[[1]*k,
                                predictedvalues[:i][::-1],
                                atleast_1d(tmp[-p+i:][::-1].squeeze())] * \
                                    params)
                for i in range(p,n):
                    predictedvalues[i] = np.sum(np.r_[[1]*k,
                        predictedvalues[i-p:i][::-1]] * params)
        return predictedvalues

    def _presample_varcov(self, params):
        """
        Returns the inverse of the presample variance-covariance.

        Notes
        -----
        See Hamilton p. 125
        """
        k = self.k_trend # amend for exog
        p = self.k_ar
        p1 = p+1

        # get inv(Vp) Hamilton 5.3.7
        params0 = np.r_[-1, params[k:]]

        Vpinv = np.zeros((p,p), dtype=params.dtype)
        for i in range(k,p1):
            Vpinv[i-1,i-1:] = np.correlate(params0, params0[:i])[:-1]
            Vpinv[i-1,i-1:] -= np.correlate(params0[-i:], params0)[:-1]

        Vpinv = Vpinv + Vpinv.T - np.diag(Vpinv.diagonal())
        return Vpinv


    def loglike(self, params):
        """
        The loglikelihood of an AR(p) process

        Parameters
        ----------
        params : array
            The fitted parameters of the AR model

        Returns
        -------
        llf : float
            The loglikelihood evaluated at `params`

        Notes
        -----
        Contains constant term.  If the model is fit by OLS then this returns
        the conditonal maximum likelihood.

        .. math:: \\frac{\\left(n-p\\right)}{2}\\left(\\log\\left(2\\pi\\right)+\\log\\left(\\sigma^{2}\\right)\\right)-\\frac{1}{\\sigma^{2}}\\sum_{i}\\epsilon_{i}^{2}

        If it is fit by MLE then the (exact) unconditional maximum likelihood
        is returned.

        .. math:: -\\frac{n}{2}log\\left(2\\pi\\right)-\\frac{n}{2}\\log\\left(\\sigma^{2}\\right)+\\frac{1}{2}\\left|V_{p}^{-1}\\right|-\\frac{1}{2\\sigma^{2}}\\left(y_{p}-\\mu_{p}\\right)^{\\prime}V_{p}^{-1}\\left(y_{p}-\\mu_{p}\\right)-\\frac{1}{2\\sigma^{2}}\\sum_{t=p+1}^{n}\\epsilon_{i}^{2}

        where

        :math:`\\mu_{p}` is a (`p` x 1) vector with each element equal to the
        mean of the AR process and :math:`\\sigma^{2}V_{p}` is the (`p` x `p`)
        variance-covariance matrix of the first `p` observations.
        """
        #TODO: Math is on Hamilton ~pp 124-5
        #will need to be amended for inclusion of exogenous variables
        nobs = self.nobs
        Y = self.Y
        X = self.X
        if self.method == "cmle":
            ssr = sumofsq(Y.squeeze()-np.dot(X,params))
            sigma2 = ssr/nobs
            return -nobs/2 * (np.log(2*np.pi) + np.log(sigma2)) -\
                    ssr/(2*sigma2)
        endog = self.endog
        k_ar = self.k_ar

        if isinstance(params,tuple):
            # broyden (all optimize.nonlin return a tuple until rewrite commit)
            params = np.asarray(params)

# reparameterize according to Jones (1980) like in ARMA/Kalman Filter
        if self.transparams:
            params = self._transparams(params)

        # get mean and variance for pre-sample lags
        yp = endog[:k_ar]
        lagstart = self.k_trend
        exog = self.exog
        if exog is not None:
            lagstart += exog.shape[1]
#            xp = exog[:k_ar]
        if self.k_trend == 1 and lagstart == 1:
            c = [params[0]] * k_ar # constant-only no exogenous variables
        else:   #TODO: this isn't right
                #NOTE: when handling exog just demean and proceed as usual.
            c = np.dot(X[:k_ar, :lagstart], params[:lagstart])
        mup = np.asarray(c/(1-np.sum(params[lagstart:])))
        diffp = yp-mup[:,None]

        # get inv(Vp) Hamilton 5.3.7
        Vpinv = self._presample_varcov(params)

        diffpVpinv = np.dot(np.dot(diffp.T,Vpinv),diffp).item()
        ssr = sumofsq(Y.squeeze() -np.dot(X,params))

        # concentrating the likelihood means that sigma2 is given by
        sigma2 = 1./nobs * (diffpVpinv + ssr)
        logdet = np_slogdet(Vpinv)[1] #TODO: add check for singularity
        loglike = -1/2.*(nobs*(np.log(2*np.pi) + np.log(sigma2)) - \
                logdet + diffpVpinv/sigma2 + ssr/sigma2)
        return loglike

    def score(self, params):
        """
        Return the gradient of the loglikelihood at params.

        Parameters
        ----------
        params : array-like
            The parameter values at which to evaluate the score function.

        Notes
        -----
        Returns numerical gradient.
        """
        loglike = self.loglike
#NOTE: always calculate at out of bounds params for estimation
#TODO: allow for user-specified epsilon?
        return approx_fprime(params, loglike, epsilon=1e-8)


    def information(self, params):
        """
        Not Implemented Yet
        """
        return

    def hessian(self, params):
        """
        Returns numerical hessian for now.
        """
        loglike = self.loglike
        return approx_hess(params, loglike)[0]

    def _stackX(self, k_ar, trend):
        """
        Private method to build the RHS matrix for estimation.

        Columns are trend terms, then exogenous, then lags.
        """
        endog = self.endog
        exog = self.exog
        X = lagmat(endog, maxlag=k_ar, trim='both')
        if exog is not None:
            X = np.column_stack((exog[k_ar:,:], X))
        # Handle trend terms
        if trend == 'c':
            k_trend = 1
        elif trend == 'nc':
            k_trend = 0
        elif trend == 'ct':
            k_trend = 2
        elif trend == 'ctt':
            k_trend = 3
        if trend != 'nc':
            X = add_trend(X,prepend=True, trend=trend)
        self.k_trend = k_trend
        return X

    def fit(self, maxlag=None, method='cmle', ic=None, trend='c',
            transparams=True, start_params=None, solver=None, maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs):
        """
        Fit the unconditional maximum likelihood of an AR(p) process.

        Parameters
        ----------
        maxlag : int
            If `ic` is None, then maxlag is the lag length used in fit.  If
            `ic` is specified then maxlag is the highest lag order used to
            select the correct lag order.  If maxlag is None, the default is
            round(12*(nobs/100.)**(1/4.))
        method : str {'cmle', 'mle'}, optional
            cmle - Conditional maximum likelihood using OLS
            mle - Unconditional (exact) maximum likelihood.  See `solver`
            and the Notes.
        ic : str {'aic','bic','hic','t-stat'}
            Criterion used for selecting the optimal lag length.
            aic - Akaike Information Criterion
            bic - Bayes Information Criterion
            t-stat - Based on last lag
            hq - Hannan-Quinn Information Criterion
            If any of the information criteria are selected, the lag length
            which results in the lowest value is selected.  If t-stat, the
            model starts with maxlag and drops a lag until the highest lag
            has a t-stat that is significant at the 95 % level.
        trend : str {'c','nc'}
            Whether to include a constant or not. 'c' - include constant.
            'nc' - no constant.

        The below can be specified if method is 'mle'

        transparams : bool, optional
            Whether or not to transform the parameters to ensure stationarity.
            Uses the transformation suggested in Jones (1980).
        start_params : array-like, optional
            A first guess on the parameters.  Default is cmle estimates.
        solver : str or None, optional
            Solver to be used.  The default is 'l_bfgs' (limited memory Broyden-
            Fletcher-Goldfarb-Shanno).  Other choices are 'bfgs', 'newton'
            (Newton-Raphson), 'nm' (Nelder-Mead), 'cg' - (conjugate gradient),
            'ncg' (non-conjugate gradient), and 'powell'.
            The limited memory BFGS uses m=30 to approximate the Hessian,
            projected gradient tolerance of 1e-7 and factr = 1e3.  These
            cannot currently be changed for l_bfgs.  See notes for more
            information.
        maxiter : int, optional
            The maximum number of function evaluations. Default is 35.
        tol : float
            The convergence tolerance.  Default is 1e-08.
        full_output : bool, optional
            If True, all output from solver will be available in
            the Results object's mle_retvals attribute.  Output is dependent
            on the solver.  See Notes for more information.
        disp : bool, optional
            If True, convergence information is output.
        callback : function, optional
            Called after each iteration as callback(xk) where xk is the current
            parameter vector.
        kwargs
            See Notes for keyword arguments that can be passed to fit.

        References
        ----------
        Jones, R.H. 1980 "Maximum likelihood fitting of ARMA models to time
            series with missing observations."  `Technometrics`.  22.3.
            389-95.

        See also
        --------
        scikits.statsmodels.model.LikelihoodModel.fit for more information
        on using the solvers.

        Notes
        ------
        The below is the docstring from
        scikits.statsmodels.LikelihoodModel.fit
        """
        self.transparams = transparams
        method = method.lower()
        if method not in ['cmle','yw','mle']:
            raise ValueError("Method %s not recognized" % method)
        self.method = method
        nobs = len(self.endog) # overwritten if method is 'cmle'
        if maxlag is None:
            maxlag = int(round(12*(nobs/100.)**(1/4.)))

        endog = self.endog
        exog = self.exog
        k_ar = maxlag # stays this if ic is None

        # select lag length
        if ic is not None:
            ic = ic.lower()
            if ic not in ['aic','bic','hqic','t-stat']:
                raise ValueError("ic option %s not understood" % ic)
            # make Y and X with same nobs to compare ICs
            Y = endog[maxlag:]
            self.Y = Y  # attach to get correct fit stats
            X = self._stackX(maxlag, trend) # sets k_trend
            self.X = X
            startlag = self.k_trend # k_trend set in _stackX
            if exog is not None:
                startlag += exog.shape[1] # add dim happens in super?
            startlag = max(1,startlag) # handle if startlag is 0
            results = {}
            if ic != 't-stat':
                for lag in range(startlag,maxlag+1):
                    # have to reinstantiate the model to keep comparable models
                    endog_tmp = endog[maxlag-lag:]
                    fit = AR(endog_tmp).fit(maxlag=lag, method=method,
                            full_output=full_output, trend=trend,
                            maxiter=maxiter, disp=disp)
                    results[lag] = eval('fit.'+ic)
                bestic, bestlag = min((res, k) for k,res in results.iteritems())
            else: # choose by last t-stat.
                stop = 1.6448536269514722 # for t-stat, norm.ppf(.95)
                for lag in range(maxlag,startlag-1,-1):
                    # have to reinstantiate the model to keep comparable models
                    endog_tmp = endog[maxlag-lag:]
                    fit = AR(endog_tmp).fit(maxlag=lag, method=method,
                            full_output=full_output, trend=trend,
                            maxiter=maxiter, disp=disp)
                    if np.abs(fit.tvalues[-1]) >= stop:
                        bestlag = lag
                        break
            k_ar = bestlag

        # change to what was chosen by fit method
        self.k_ar = k_ar

        # redo estimation for best lag
        # make LHS
        Y = endog[k_ar:,:]
        # make lagged RHS
        X = self._stackX(k_ar, trend) # sets self.k_trend
        k_trend = self.k_trend
        self.Y = Y
        self.X = X

        if solver:
            solver = solver.lower()
        if method == "cmle":     # do OLS
            arfit = OLS(Y,X).fit()
            params = arfit.params
            self.nobs = nobs - k_ar
        if method == "mle":
            self.nobs = nobs
            if not start_params:
                start_params = OLS(Y,X).fit().params
                start_params = self._invtransparams(start_params)
            loglike = lambda params : -self.loglike(params)
            if solver == None:  # use limited memory bfgs
                bounds = [(None,)*2]*(k_ar+k_trend)
                mlefit = optimize.fmin_l_bfgs_b(loglike, start_params,
                    approx_grad=True, m=30, pgtol = 1e-7, factr=1e3,
                    bounds=bounds, iprint=1)
                self.mlefit = mlefit
                params = mlefit[0]
            else:
                mlefit = super(AR, self).fit(start_params=start_params,
                            method=solver, maxiter=maxiter,
                            full_output=full_output, disp=disp,
                            callback = callback, **kwargs)
                self.mlefit = mlefit
                params = mlefit.params
            if self.transparams:
                params = self._transparams(params)
                self.transparams = False # turn off now for other results

# don't use yw, because we can't estimate the constant
#        elif method == "yw":
#            params, omega = yule_walker(endog, order=maxlag,
#                    method="mle", demean=False)
            # how to handle inference after Yule-Walker?
#            self.params = params #TODO: don't attach here
#            self.omega = omega

        pinv_exog = np.linalg.pinv(X)
        normalized_cov_params = np.dot(pinv_exog, pinv_exog.T)
        arfit = ARResults(self, params, normalized_cov_params)
        return ARResultsWrapper(arfit)

    fit.__doc__ += base.LikelihoodModel.fit.__doc__

class ARResults(tsbase.TimeSeriesModelResults):
    """
    Class to hold results from fitting an AR model.

    Parameters
    ----------
    model : AR Model instance
        Reference to the model that is fit.
    params : array
        The fitted parameters from the AR Model.
    normalized_cov_params : array
        inv(dot(X.T,X)) where X is the exogenous variables including lagged
        values.
    scale : float, optional
        An estimate of the scale of the model.

    Returns
    -------
    **Attributes**

    aic : float
        Akaike Information Criterion using Lutkephol's definition.
        :math:`log(sigma) + 2*(1+k_ar)/nobs`
    bic : float
        Bayes Information Criterion
        :math:`\\log(\\sigma) + (1+k_ar)*\\log(nobs)/nobs`
    bse : array
        The standard errors of the estimated parameters. If `method` is 'cmle',
        then the standard errors that are returned are the OLS standard errors
        of the coefficients. If the `method` is 'mle' then they are computed
        using the numerical Hessian.
    fittedvalues : array
        The in-sample predicted values of the fitted AR model. The `k_ar`
        initial values are computed via the Kalman Filter if the model is
        fit by `mle`.
    fpe : float
        Final prediction error using Lutkepohl's definition
        ((n_totobs+k_trend)/(n_totobs-k_ar-k_trend))*sigma
    hqic : float
        Hannan-Quinn Information Criterion.
    k_ar : float
        Lag length. Sometimes used as `p` in the docs.
    k_trend : float
        The number of trend terms included. 'nc'=0, 'c'=1.
    llf : float
        The loglikelihood of the model evaluated at `params`. See `AR.loglike`
    model : AR model instance
        A reference to the fitted AR model.
    nobs : float
        The number of available observations `nobs` - `k_ar`
    n_totobs : float
        The number of total observations in `endog`. Sometimes `n` in the docs.
    params : array
        The fitted parameters of the model.
    pvalues : array
        The p values associated with the standard errors.
    resid : array
        The residuals of the model. If the model is fit by 'mle' then the pre-sample
        residuals are calculated using fittedvalues from the Kalman Filter.
    roots : array
        The roots of the AR process.
    scale : float
        Same as sigma2
    sigma2 : float
        The variance of the innovations (residuals).
    trendorder : int
        The polynomial order of the trend. 'nc' = None, 'c' or 't' = 0, 'ct' = 1,
        etc.
    tvalues : array
        The t-values associated with `params`.
    """

    _cache = {} # for scale setter

    def __init__(self, model, params, normalized_cov_params=None, scale=1.):
        super(ARResults, self).__init__(model, params, normalized_cov_params,
                scale)
        self._cache = resettable_cache()
        self.nobs = model.nobs
        n_totobs = len(model.endog)
        self.n_totobs = n_totobs
        self.X = model.X # copy?
        self.Y = model.Y
        k_ar = model.k_ar
        self.k_ar = k_ar
        k_trend = model.k_trend
        self.k_trend = k_trend
        trendorder = None
        if k_trend > 0:
            trendorder = k_trend - 1
        self.trendorder = 1
        #TODO: cmle vs mle?
        self.df_resid = self.model.df_resid = n_totobs - k_ar - k_trend

    @cache_writable()
    def sigma2(self):
        #TODO: allow for DOF correction if exog is included
        model = self.model
        if model.method == "cmle": # do DOF correction
            return 1./self.nobs * sumofsq(self.resid)
        else: # we need to calculate the ssr for the pre-sample
              # see loglike for details
            lagstart = self.k_trend #TODO: handle exog
            p = self.k_ar
            params = self.params
            meany = params[0]/(1-params[lagstart:].sum())
            pre_resid = model.endog[:p] - meany
            # get presample var-cov
            Vpinv = model._presample_varcov(params)
            diffpVpinv = np.dot(np.dot(pre_resid.T,Vpinv),pre_resid).item()
            ssr = sumofsq(self.resid[p:]) # in-sample ssr

            return 1/self.nobs * (diffpVpinv+ssr)

    @cache_writable()   # for compatability with RegressionResults
    def scale(self):
        return self.sigma2

    @cache_readonly
    def bse(self): # allow user to specify?
        if self.model.method == "cmle": # uses different scale/sigma definition
            resid = self.resid
            ssr = np.dot(resid,resid)
            ols_scale = ssr/(self.nobs - self.k_ar - self.k_trend)
            return np.sqrt(np.diag(self.cov_params(scale=ols_scale)))
        else:
            hess = approx_hess(self.params, self.model.loglike)
            return np.sqrt(np.diag(-np.linalg.inv(hess[0])))

    @cache_readonly
    def pvalues(self):
        return norm.sf(np.abs(self.tvalues))*2

    @cache_readonly
    def aic(self):
        #JP: this is based on loglike with dropped constant terms ?
# Lutkepohl
#        return np.log(self.sigma2) + 1./self.model.nobs * self.k_ar
# Include constant as estimated free parameter and double the loss
        return np.log(self.sigma2) + 2 * (1 + self.k_ar)/self.nobs
# Stata defintion
#        nobs = self.nobs
#        return -2 * self.llf/nobs + 2 * (self.k_ar+self.k_trend)/nobs

    @cache_readonly
    def hqic(self):
        nobs = self.nobs
# Lutkepohl
#        return np.log(self.sigma2)+ 2 * np.log(np.log(nobs))/nobs * self.k_ar
# R uses all estimated parameters rather than just lags
        return np.log(self.sigma2) + 2 * np.log(np.log(nobs))/nobs * \
                (1 + self.k_ar)
# Stata
#        nobs = self.nobs
#        return -2 * self.llf/nobs + 2 * np.log(np.log(nobs))/nobs * \
#                (self.k_ar + self.k_trend)

    @cache_readonly
    def fpe(self):
        nobs = self.nobs
        k_ar = self.k_ar
        k_trend = self.k_trend
#Lutkepohl
        return ((nobs+k_ar+k_trend)/(nobs-k_ar-k_trend))*self.sigma2

    @cache_readonly
    def llf(self):
        return self.model.loglike(self.params)

    @cache_readonly
    def bic(self):
        nobs = self.nobs
# Lutkepohl
#        return np.log(self.sigma2) + np.log(nobs)/nobs * self.k_ar
# Include constant as est. free parameter
        return np.log(self.sigma2) + (1 + self.k_ar) * np.log(nobs)/nobs
# Stata
#        return -2 * self.llf/nobs + np.log(nobs)/nobs * (self.k_ar + \
#                self.k_trend)

    @cache_readonly
    def resid(self):
        #NOTE: uses fittedvalues because it calculate presample values for mle
        model = self.model
        endog = model.endog.squeeze()
        if model.method == "cmle": # elimate pre-sample
            return endog[self.k_ar:] - self.fittedvalues
        else:
            return model.endog.squeeze() - self.fittedvalues

#    def ssr(self):
#        resid = self.resid
#        return np.dot(resid, resid)

    @cache_readonly
    def roots(self):
        return np.roots(np.r_[1, -self.params[1:]])

    @cache_readonly
    def fittedvalues(self):
        return self.model.predict(self.params)

class ARResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
            'roots' : 'columns'
            }
    _wrap_attrs = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._attrs,
                                    _attrs)
    #_methods = {'conf_int' : 'columns'} #TODO: can't handle something like this yet
    _methods = {'conf_int' : 'columns'}
    _wrap_methods = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._wrap_methods,
                                     _methods)
wrap.populate_wrapper(ARResultsWrapper, ARResults)


if __name__ == "__main__":
    import scikits.statsmodels.api as sm
    sunspots = sm.datasets.sunspots.load()
# Why does R demean the data by defaut?
    ar_ols = AR(sunspots.endog)
    res_ols = ar_ols.fit(maxlag=9)
    ar_mle = AR(sunspots.endog)
    res_mle_bfgs = ar_mle.fit(maxlag=9, method="mle", solver="bfgs",
                    maxiter=500, gtol=1e-10)
#    res_mle2 = ar_mle.fit(maxlag=1, method="mle", maxiter=500, penalty=True,
#            tol=1e-13)

#    ar_yw = AR(sunspots.endog)
#    res_yw = ar_yw.fit(maxlag=4, method="yw")

#    # Timings versus talkbox
#    from timeit import default_timer as timer
#    print "Time AR fit vs. talkbox"
#    # generate a long series of AR(2) data
#
#    nobs = 1000000
#    y = np.empty(nobs)
#    y[0:2] = 0
#    for i in range(2,nobs):
#        y[i] = .25 * y[i-1] - .75 * y[i-2] + np.random.rand()
#
#    mod_sm = AR(y)
#    t = timer()
#    res_sm = mod_sm.fit(method="yw", trend="nc", demean=False, maxlag=2)
#    t_end = timer()
#    print str(t_end - t) + " seconds for sm.AR with yule-walker, 2 lags"
#    try:
#        import scikits.talkbox as tb
#    except:
#        raise ImportError("You need scikits.talkbox installed for timings")
#    t = timer()
#    mod_tb = tb.lpc(y, 2)
#    t_end = timer()
#    print str(t_end - t) + " seconds for talkbox.lpc"
#    print """For higher lag lengths ours quickly fills up memory and starts
#thrashing the swap.  Should we include talkbox C code or Cythonize the
#Levinson recursion algorithm?"""

    ## Try with a pandas series
    import pandas
    dates = np.arange(1700,1700+len(sunspots.endog))
    sunspots = pandas.TimeSeries(sunspots.endog, index=dates)
    mod = AR(sunspots)


# some data for an example in Box Jenkins
    IBM = np.asarray([460,457,452,459,462,459,463,479,493,490.])
    w = np.diff(IBM)
    theta = .5
