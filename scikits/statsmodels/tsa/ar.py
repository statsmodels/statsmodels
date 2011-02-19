"""
This is the VAR class refactored from pymaclab.
"""
from __future__ import division
import numpy as np
from numpy import (dot, identity, atleast_2d, atleast_1d)
from numpy.linalg import inv
from scipy import optimize
from scipy.stats import ss as sumofsq
from scikits.statsmodels.regression.linear_model import OLS
from tsatools import lagmat, add_trend
from scikits.statsmodels.base.model import (LikelihoodModelResults,
        LikelihoodModel)
from scikits.statsmodels.tools.decorators import (resettable_cache,
        cache_readonly, cache_writable)
from scikits.statsmodels.tools.compatibility import np_slogdet
from scikits.statsmodels.sandbox.regression.numdiff import approx_fprime
from scikits.statsmodels.sandbox.regression.numdiff import approx_hess


__all__ = ['AR']


class AR(LikelihoodModel):
    def __init__(self, endog, exog=None):
        """
        Autoregressive AR(p) Model
        """
        super(AR, self).__init__(endog, exog)
        if endog.ndim == 1:
            endog = endog[:,None]
        elif endog.ndim > 1 and endog.shape[1] != 1:
            raise ValueError("Only the univariate case is implemented")
        self.endog = endog  # overwrite endog
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
        p,k = self.laglen, self.trendorder # need to include exog here?
        newparams = params.copy() # no copy below now?
        newparams[k:k+p] = ((1-np.exp(-params[k:k+p]))/
                            (1+np.exp(-params[k:k+p]))).copy()
        tmp = ((1-np.exp(-params[k:k+p]))/
               (1+np.exp(-params[k:k+p]))).copy()

        # levinson-durbin to get pacf
        for j in range(1,p):
            a = newparams[k+j]
            for kiter in range(j):
                tmp[kiter] -= a * newparams[k+j-kiter-1]
            newparams[k:k+j] = tmp[:j]
        return newparams

    def _invtransparams(self, start_params):
        """
        Inverse of the Jones reparameterization
        """
        p,k = self.laglen, self.trendorder
        newparams = start_params.copy()
        arcoefs = newparams[k:k+p].copy()
        # AR coeffs
        tmp = arcoefs.copy()
        for j in range(p-1,0,-1):
            a = arcoefs[j]
            for kiter in range(j):
                tmp[kiter] = (arcoefs[kiter] + a * arcoefs[j-kiter-1])/\
                        (1-a**2)
            arcoefs[:j] = tmp[:j]
        invarcoefs = -np.log((1-arcoefs)/(1+arcoefs))
        newparams[k:k+p] = invarcoefs
        return newparams

    def predict(self, n=-1, start=0, method='dynamic', resid=False,
            confint=False):
        """
        Returns in-sample prediction or forecasts.

        n : int
            Number of periods after start to forecast.  If n==-1, returns in-
            sample forecast starting at `start`.
        start : int
            Zero-indexed observation number at which to start forecasting, ie.,
            the first forecast is start.  If start==-1, forecasting starts from
            the end of the sample.  If the model is fit using 'cmle' or 'yw',
            `start` cannot be less than `laglen`.  If `start` < `laglen` for
            'cmle' and 'yw', then `start` is set equal to `laglen`.
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
        values.  The initial state is assumed to be a zero vector with the
        variance given by ...
        """
        if self._results is None:
            raise ValueError("You must fit the model first")

        if n == 0 or (n==-1 and start==-1):
            return np.array([])

        y = self.endog.copy()
        nobs = int(self.nobs)

        if start < 0:
            start = nobs + start # convert negative indexing

        params = self._results.params
        p = self.laglen
        k = self.trendorder
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
            predictedvalues = np.zeros((n))

        mu = 0 # overwritten for 'mle' with constant
        if method == 'mle':
            # build system matrices
            T_mat = np.zeros((p,p))
            T_mat[:,0] = params[k:]
            T_mat[:-1,1:] = identity(p-1)

            R_mat = np.zeros((p,1))
            R_mat[0] = 1

            # Initial State mean and variance
            alpha = np.zeros((p,1))
            if k>=1:    # if constant, demean, #TODO: handle higher trendorders
                mu = params[0]/(1-np.sum(params[k:]))   # only for constant-only
                                           # and exog
                y -= mu

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
#                P[0,0] += 1 # for MA part, R_mat.R_mat.T above
                predictedvalues[i+1-start] = dot(Z_mat,alpha)
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

    def loglike(self, params):
        """
        The loglikelihood of an AR(p) process

        Notes
        -----
        Contains constant term.  If the model is fit by OLS then this returns
        the conditonal maximum likelihood.  If it is fit by MLE then the
        (exact) unconditional maximum likelihood is returned.
        """
        #TODO: Math is on Hamilton ~pp 124-5
        #will need to be amended for inclusion of exogenous variables
        nobs = self.nobs
        avobs = self.avobs
        Y = self.Y
        X = self.X
        if self.method == "cmle":
            ssr = sumofsq(Y.squeeze()-np.dot(X,params))
            sigma2 = ssr/avobs
            return -avobs/2 * (np.log(2*np.pi) + np.log(sigma2)) -\
                    ssr/(2*sigma2)
        endog = self.endog
        laglen = self.laglen

        if isinstance(params,tuple):
            # broyden (all optimize.nonlin return a tuple until rewrite commit)
            params = np.asarray(params)

# reparameterize according to Jones (1980) like in ARMA/Kalman Filter
        if self.transparams:
            params = self._transparams(params) # will this overwrite?

        # get mean and variance for pre-sample lags
        yp = endog[:laglen]
        lagstart = self.trendorder
        exog = self.exog
        if exog is not None:
            lagstart += exog.shape[1]
#            xp = exog[:laglen]
        if self.trendorder == 1 and lagstart == 1:
            c = [params[0]] * laglen # constant-only no exogenous variables
        else:   #TODO: this probably isn't right
            c = np.dot(X[:laglen, :lagstart], params[:lagstart])
        mup = np.asarray(c/(1-np.sum(params[lagstart:])))
        diffp = yp-mup[:,None]

        # get inv(Vp) Hamilton 5.3.7
        params0 = np.r_[-1, params[lagstart:]]

        p = len(params) - lagstart
        p1 = p+1
        Vpinv = np.zeros((p,p))
        for i in range(lagstart,p1):
            for j in range(lagstart,p1):
                if i <= j and j <= p:
                    part1 = np.sum(params0[:i] * params0[j-i:j])
                    part2 = np.sum(params0[p1-j:p1+i-j]*params0[p1-i:])
                    Vpinv[i-1,j-1] = part1 - part2
        Vpinv = Vpinv + Vpinv.T - np.diag(Vpinv.diagonal())
        # this is correct to here

        diffpVpinv = np.dot(np.dot(diffp.T,Vpinv),diffp).item()
        ssr = sumofsq(Y.squeeze() -np.dot(X,params))

        # concentrating the likelihood means that sigma2 is given by
        sigma2 = 1./avobs * (diffpVpinv + ssr)
        logdet = np_slogdet(Vpinv)[1] #TODO: add check for singularity
        loglike = -1/2.*(nobs*(np.log(2*np.pi) + np.log(sigma2)) - \
                logdet + diffpVpinv/sigma2 + ssr/sigma2)
        return loglike

    def _R(self, params):
        """
        Private method for obtaining fitted presample values via Kalman filter.
        """
        pass

    def _T(self, params):
        """
        Private method for obtaining fitted presample values via Kalman filter.

        See also
        --------
        scikits.statsmodels.tsa.kalmanf.ARMA
        """
        pass

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

    def _stackX(self, laglen, trend):
        """
        Private method to build the RHS matrix for estimation.

        Columns are trend terms, then exogenous, then lags.
        """
        endog = self.endog
        exog = self.exog
        X = lagmat(endog, maxlag=laglen, trim='both')
        if exog is not None:
            X = np.column_stack((exog[laglen:,:], X))
        # Handle trend terms
        if trend == 'c':
            trendorder = 1
        elif trend == 'nc':
            trendorder = 0
        elif trend == 'ct':
            trendorder = 2
        elif trend == 'ctt':
            trendorder = 3
        if trend != 'nc':
            X = add_trend(X,prepend=True, trend=trend)
        self.trendorder = trendorder
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
        nobs = self.nobs
        if maxlag is None:
            maxlag = int(round(12*(nobs/100.)**(1/4.)))

        endog = self.endog
        exog = self.exog
        laglen = maxlag # stays this if ic is None

        # select lag length
        if ic is not None:
            ic = ic.lower()
            if ic not in ['aic','bic','hqic','t-stat']:
                raise ValueError("ic option %s not understood" % ic)
            # make Y and X with same nobs to compare ICs
            Y = endog[maxlag:]
            self.Y = Y  # attach to get correct fit stats
            X = self._stackX(maxlag, trend)
            self.X = X
            startlag = self.trendorder # trendorder set in _stackX
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
                    if np.abs(fit.t(-1)) >= stop:
                        bestlag = lag
                        break
            laglen = bestlag

        # change to what was chosen by fit method
        self.laglen = laglen
        avobs = nobs - laglen
        self.avobs = avobs

        # redo estimation for best lag
        # make LHS
        Y = endog[laglen:,:]
        # make lagged RHS
        X = self._stackX(laglen, trend) # sets self.trendorder
        trendorder = self.trendorder
        self.Y = Y
        self.X = X
        self.df_resid = avobs - laglen - trendorder # for compatiblity
                                                # with Model code
        if solver:
            solver = solver.lower()
        if method == "cmle":     # do OLS
            arfit = OLS(Y,X).fit()
            params = arfit.params
        if method == "mle":
            if not start_params:
                start_params = OLS(Y,X).fit().params
                start_params = self._invtransparams(start_params)
            loglike = lambda params : -self.loglike(params)
            if solver == None:  # use limited memory bfgs
                bounds = [(None,)*2]*(laglen+trendorder)
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
        self._results = arfit
        return arfit

    fit.__doc__ += LikelihoodModel.fit.__doc__


class ARResults(LikelihoodModelResults):
    """
    Class to hold results from fitting an AR model.

    Notes
    -----
    If `method` is 'cmle', then the standard errors that are returned
    are the OLS standard errors of the coefficients.  That is, they correct
    for the degrees of freedom in the estimate of the scale/sigma.  To
    reproduce t-stats using the AR definition of sigma (no dof correction), one
    can do np.sqrt(np.diag(results.cov_params())).
    """

    _cache = {} # for scale setter

    def __init__(self, model, params, normalized_cov_params=None, scale=1.):
        super(ARResults, self).__init__(model, params, normalized_cov_params,
                scale)
        self._cache = resettable_cache()
        self.nobs = model.nobs
        self.avobs = model.avobs
        self.X = model.X # copy?
        self.Y = model.Y
        self.laglen = model.laglen
        self.trendorder = model.trendorder

    @cache_writable()
    def sigma(self):
        #TODO: allow for DOF correction if exog is included
        return 1./self.avobs * self.ssr

    @cache_writable()   # for compatability with RegressionResults
    def scale(self):
        return self.sigma

    @cache_readonly
    def bse(self): # allow user to specify?
        if self.model.method == "cmle": # uses different scale/sigma definition
            ols_scale = self.ssr/(self.avobs - self.laglen - self.trendorder)
            return np.sqrt(np.diag(self.cov_params(scale=ols_scale)))
        else:
            return np.sqrt(np.diag(self.cov_params()))

    @cache_readonly
    def t(self):    # overwrite t()
        return self.params/self.bse

    @cache_readonly
    def pvalues(self):
        return stats.t.pp

    @cache_readonly
    def aic(self):
        #JP: this is based on loglike with dropped constant terms ?
# Lutkepohl
#        return np.log(self.sigma) + 1./self.model.avobs * self.laglen
# Include constant as estimated free parameter and double the loss
        return np.log(self.sigma) + 2 * (1 + self.laglen)/self.avobs
# Stata defintion
#        avobs = self.avobs
#        return -2 * self.llf/avobs + 2 * (self.laglen+self.trendorder)/avobs

    @cache_readonly
    def hqic(self):
        avobs = self.avobs
# Lutkepohl
#        return np.log(self.sigma)+ 2 * np.log(np.log(avobs))/avobs * self.laglen
# R uses all estimated parameters rather than just lags
        return np.log(self.sigma) + 2 * np.log(np.log(avobs))/avobs * \
                (1 + self.laglen)
# Stata
#        avobs = self.avobs
#        return -2 * self.llf/avobs + 2 * np.log(np.log(avobs))/avobs * \
#                (self.laglen + self.trendorder)

    @cache_readonly
    def fpe(self):
        avobs = self.avobs
        laglen = self.laglen
        trendorder = self.trendorder
#Lutkepohl
        return ((avobs+laglen+trendorder)/(avobs-laglen-trendorder))*self.sigma

    @cache_readonly
    def llf(self):
        return self.model.loglike(self.params)

    @cache_readonly
    def bic(self):
        avobs = self.avobs
# Lutkepohl
#        return np.log(self.sigma) + np.log(avobs)/avobs * self.laglen
# Include constant as est. free parameter
        return np.log(self.sigma) + (1 + self.laglen) * np.log(avobs)/avobs
# Stata
#        return -2 * self.llf/avobs + np.log(avobs)/avobs * (self.laglen + \
#                self.trendorder)

    @cache_readonly
    def resid(self):
        model = self.model
        return self.Y.squeeze() - np.dot(self.X, self.params)

    @cache_readonly
    def ssr(self):
        resid = self.resid
        return np.dot(resid, resid)

    @cache_readonly
    def roots(self):
        return np.roots(np.r_[1, -self.params[1:]])

    @cache_readonly
    def fittedvalues(self):
        return self.model.predict()

class ARIMA(LikelihoodModel):
    def __init__(self, endog, exog=None):
        """
        ARIMA Model
        """
        super(ARIMA, self).__init__(endog, exog)
        if endog.ndim == 1:
            endog = endog[:,None]
        elif endog.ndim > 1 and endog.shape[1] != 1:
            raise ValueError("Only the univariate case is implemented")
        self.endog = endog # overwrite endog
        if exog is not None:
            raise ValueError("Exogenous variables are not yet supported.")

    def fit(self, order=(0,0,0), method="ssm"):
        """
        Notes
        -----
        Current method being developed is the state-space representation.

        Box and Jenkins outline many more procedures.
        """
        if not hasattr(order, '__iter__'):
            raise ValueError("order must be an iterable sequence.  Got type \
%s instead" % type(order))
        p,d,q = order
        if d > 0:
            raise ValueError("Differencing not implemented yet")
            # assume no constant, ie mu = 0
            # unless overwritten then use w_bar for mu
            Y = np.diff(endog, d, axis=0) #TODO: handle lags?


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

# some data for an example in Box Jenkins
    IBM = np.asarray([460,457,452,459,462,459,463,479,493,490.])
    w = np.diff(IBM)
    theta = .5
