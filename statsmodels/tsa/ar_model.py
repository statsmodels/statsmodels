# -*- coding: utf-8 -*-
from __future__ import division
from statsmodels.compat.python import iteritems, range, lmap

import numpy as np
from numpy import dot, identity
from numpy.linalg import inv, slogdet
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.tsatools import (lagmat, add_trend,
                                      _ar_transparams, _ar_invtransparams)
import statsmodels.tsa.base.tsa_model as tsbase
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from statsmodels.tsa.kalmanf.kalmanfilter import KalmanFilter
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.vector_ar import util


__all__ = ['AR']


def sumofsq(x, axis=0):
    """Helper function to calculate sum of squares along first axis"""
    return np.sum(x**2, axis=axis)


def _check_ar_start(start, k_ar, method, dynamic):
    if (method == 'cmle' or dynamic) and start < k_ar:
        raise ValueError("Start must be >= k_ar for conditional MLE "
                         "or dynamic forecast. Got %d" % start)


def _ar_predict_out_of_sample(y, params, k_ar, k_trend, steps, start=0):
    mu = params[:k_trend] if k_trend else 0  # only have to worry constant
    arparams = params[k_trend:][::-1]  # reverse for dot

    # dynamic endogenous variable
    endog = np.zeros(k_ar + steps)  # this is one too big but doesn't matter
    if start:
        endog[:k_ar] = y[start-k_ar:start]
    else:
        endog[:k_ar] = y[-k_ar:]

    forecast = np.zeros(steps)
    for i in range(steps):
        fcast = mu + np.dot(arparams, endog[i:i+k_ar])
        forecast[i] = fcast
        endog[i + k_ar] = fcast

    return forecast


class AR(tsbase.TimeSeriesModel):
    __doc__ = tsbase._tsa_doc % {"model": "Autoregressive AR(p) model",
                                 "params": """endog : array-like
        1-d endogenous response variable. The independent variable.""",
                                 "extra_params": base._missing_param_doc,
                                 "extra_sections": ""}

    def __init__(self, endog, dates=None, freq=None, missing='none'):
        super(AR, self).__init__(endog, None, dates, freq, missing=missing)
        endog = self.endog  # original might not have been an ndarray
        if endog.ndim == 1:
            endog = endog[:, None]
            self.endog = endog  # to get shapes right
        elif endog.ndim > 1 and endog.shape[1] != 1:
            raise ValueError("Only the univariate case is implemented")

    def initialize(self):
        pass

    def _transparams(self, params):
        """
        Transforms params to induce stationarity/invertability.

        Reference
        ---------
        Jones(1980)
        """
        p = self.k_ar
        k = self.k_trend
        newparams = params.copy()
        newparams[k:k+p] = _ar_transparams(params[k:k+p].copy())
        return newparams

    def _invtransparams(self, start_params):
        """
        Inverse of the Jones reparameterization
        """
        p = self.k_ar
        k = self.k_trend
        newparams = start_params.copy()
        newparams[k:k+p] = _ar_invtransparams(start_params[k:k+p].copy())
        return newparams

    def _presample_fit(self, params, start, p, end, y, predictedvalues):
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
        alpha = np.zeros((p, 1))
        Q_0 = dot(inv(identity(p**2)-np.kron(T_mat, T_mat)),
                  dot(R_mat, R_mat.T).ravel('F'))

        Q_0 = Q_0.reshape(p, p, order='F')  # TODO: order might need to be p+k
        P = Q_0
        Z_mat = KalmanFilter.Z(p)
        for i in range(end):  # iterate p-1 times to fit presample
            v_mat = y[i] - dot(Z_mat, alpha)
            F_mat = dot(dot(Z_mat, P), Z_mat.T)
            Finv = 1./F_mat  # inv. always scalar
            K = dot(dot(dot(T_mat, P), Z_mat.T), Finv)
            # update state
            alpha = dot(T_mat, alpha) + dot(K, v_mat)
            L = T_mat - dot(K, Z_mat)
            P = dot(dot(T_mat, P), L.T) + dot(R_mat, R_mat.T)
            #P[0,0] += 1 # for MA part, R_mat.R_mat.T above
            if i >= start - 1:  # only record if we ask for it
                predictedvalues[i + 1 - start] = dot(Z_mat, alpha)

    def _get_prediction_index(self, start, end, dynamic, index=None):
        method = getattr(self, 'method', 'mle')
        k_ar = getattr(self, 'k_ar', 0)
        if start is None:
            if method == 'mle' and not dynamic:
                start = 0
            else:  # can't do presample fit for cmle or dynamic
                start = k_ar
            start = self._index[start]
        if end is None:
            end = self._index[-1]

        start, end, out_of_sample, prediction_index = (
            super(AR, self)._get_prediction_index(start, end, index))

        # This replaces the _validate() call
        if 'mle' not in method and start < k_ar:
            raise ValueError("Start must be >= k_ar for conditional MLE or "
                             "dynamic forecast. Got %s" % start)
        # Other validation
        _check_ar_start(start, k_ar, method, dynamic)

        return start, end, out_of_sample, prediction_index

    def predict(self, params, start=None, end=None, dynamic=False):
        """
        Returns in-sample and out-of-sample prediction.

        Parameters
        ----------
        params : array
            The fitted model parameters.
        start : int, str, or datetime
            Zero-indexed observation number at which to start forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type.
        end : int, str, or datetime
            Zero-indexed observation number at which to end forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type.
        dynamic : bool
            The `dynamic` keyword affects in-sample prediction. If dynamic
            is False, then the in-sample lagged values are used for
            prediction. If `dynamic` is True, then in-sample forecasts are
            used in place of lagged dependent variables. The first forecasted
            value is `start`.

        Returns
        -------
        predicted values : array

        Notes
        -----
        The linear Gaussian Kalman filter is used to return pre-sample fitted
        values. The exact initial Kalman Filter is used. See Durbin and Koopman
        in the references for more information.
        """
        # will return an index of a date
        start, end, out_of_sample, _ = (
            self._get_prediction_index(start, end, dynamic))

        if start - end > 1:
            raise ValueError("end is before start")

        k_ar = self.k_ar
        k_trend = self.k_trend
        method = self.method
        endog = self.endog.squeeze()

        if dynamic:
            out_of_sample += end - start + 1
            return _ar_predict_out_of_sample(endog, params, k_ar,
                                             k_trend, out_of_sample, start)

        predictedvalues = np.zeros(end + 1 - start)

        # fit pre-sample
        if method == 'mle':  # use Kalman Filter to get initial values
            if k_trend:
                mu = params[0]/(1-np.sum(params[k_trend:]))
            else:
                mu = 0

            # modifies predictedvalues in place
            if start < k_ar:
                self._presample_fit(params, start, k_ar, min(k_ar-1, end),
                                    endog[:k_ar] - mu, predictedvalues)
                predictedvalues[:k_ar-start] += mu

        if end < k_ar:
            return predictedvalues

        # just do the whole thing and truncate
        fittedvalues = np.dot(self.X, params)

        pv_start = max(k_ar - start, 0)
        fv_start = max(start - k_ar, 0)
        fv_end = min(len(fittedvalues), end-k_ar+1)
        predictedvalues[pv_start:] = fittedvalues[fv_start:fv_end]

        if out_of_sample:
            forecastvalues = _ar_predict_out_of_sample(endog, params,
                                                       k_ar, k_trend,
                                                       out_of_sample)
            predictedvalues = np.r_[predictedvalues, forecastvalues]

        return predictedvalues

    def _presample_varcov(self, params):
        """
        Returns the inverse of the presample variance-covariance.

        Notes
        -----
        See Hamilton p. 125
        """
        k = self.k_trend
        p = self.k_ar
        p1 = p+1

        # get inv(Vp) Hamilton 5.3.7
        params0 = np.r_[-1, params[k:]]

        Vpinv = np.zeros((p, p), dtype=params.dtype)
        for i in range(1, p1):
            Vpinv[i-1, i-1:] = np.correlate(params0, params0[:i],)[:-1]
            Vpinv[i-1, i-1:] -= np.correlate(params0[-i:], params0,)[:-1]

        Vpinv = Vpinv + Vpinv.T - np.diag(Vpinv.diagonal())
        return Vpinv

    def _loglike_css(self, params):
        """
        Loglikelihood of AR(p) process using conditional sum of squares
        """
        nobs = self.nobs
        Y = self.Y
        X = self.X
        ssr = sumofsq(Y.squeeze() - np.dot(X, params))
        sigma2 = ssr/nobs
        return (-nobs/2 * (np.log(2 * np.pi) + np.log(sigma2)) -
                ssr/(2 * sigma2))

    def _loglike_mle(self, params):
        """
        Loglikelihood of AR(p) process using exact maximum likelihood
        """
        nobs = self.nobs
        X = self.X
        endog = self.endog
        k_ar = self.k_ar
        k_trend = self.k_trend

        # reparameterize according to Jones (1980) like in ARMA/Kalman Filter
        if self.transparams:
            params = self._transparams(params)

        # get mean and variance for pre-sample lags
        yp = endog[:k_ar].copy()
        if k_trend:
            c = [params[0]] * k_ar
        else:
            c = [0]
        mup = np.asarray(c / (1 - np.sum(params[k_trend:])))
        diffp = yp - mup[:, None]

        # get inv(Vp) Hamilton 5.3.7
        Vpinv = self._presample_varcov(params)

        diffpVpinv = np.dot(np.dot(diffp.T, Vpinv), diffp).item()
        ssr = sumofsq(endog[k_ar:].squeeze() - np.dot(X, params))

        # concentrating the likelihood means that sigma2 is given by
        sigma2 = 1./nobs * (diffpVpinv + ssr)
        self.sigma2 = sigma2
        logdet = slogdet(Vpinv)[1]  # TODO: add check for singularity
        loglike = -1/2. * (nobs * (np.log(2 * np.pi) + np.log(sigma2)) -
                           logdet + diffpVpinv / sigma2 + ssr / sigma2)
        return loglike

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
        # TODO: Math is on Hamilton ~pp 124-5
        if self.method == "cmle":
            return self._loglike_css(params)

        else:
            return self._loglike_mle(params)

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
        return approx_hess(params, loglike)

    def _stackX(self, k_ar, trend):
        """
        Private method to build the RHS matrix for estimation.

        Columns are trend terms then lags.
        """
        endog = self.endog
        X = lagmat(endog, maxlag=k_ar, trim='both')
        k_trend = util.get_trendorder(trend)
        if k_trend:
            X = add_trend(X, prepend=True, trend=trend, has_constant="raise")
        self.k_trend = k_trend
        return X

    def select_order(self, maxlag, ic, trend='c', method='mle'):
        """
        Select the lag order according to the information criterion.

        Parameters
        ----------
        maxlag : int
            The highest lag length tried. See `AR.fit`.
        ic : str {'aic','bic','hqic','t-stat'}
            Criterion used for selecting the optimal lag length.
            See `AR.fit`.
        trend : str {'c','nc'}
            Whether to include a constant or not. 'c' - include constant.
            'nc' - no constant.

        Returns
        -------
        bestlag : int
            Best lag according to IC.
        """
        endog = self.endog

        # make Y and X with same nobs to compare ICs
        Y = endog[maxlag:]
        self.Y = Y  # attach to get correct fit stats
        X = self._stackX(maxlag, trend)  # sets k_trend
        self.X = X
        k = self.k_trend  # k_trend set in _stackX
        k = max(1, k)  # handle if startlag is 0
        results = {}

        if ic != 't-stat':
            for lag in range(k, maxlag+1):
                # have to reinstantiate the model to keep comparable models
                endog_tmp = endog[maxlag-lag:]
                fit = AR(endog_tmp).fit(maxlag=lag, method=method,
                                        full_output=0, trend=trend,
                                        maxiter=100, disp=0)
                results[lag] = getattr(fit, ic)
            bestic, bestlag = min((res, k) for k, res in iteritems(results))

        else:  # choose by last t-stat.
            stop = 1.6448536269514722  # for t-stat, norm.ppf(.95)
            for lag in range(maxlag, k - 1, -1):
                # have to reinstantiate the model to keep comparable models
                endog_tmp = endog[maxlag - lag:]
                fit = AR(endog_tmp).fit(maxlag=lag, method=method,
                                        full_output=0, trend=trend,
                                        maxiter=35, disp=-1)

                bestlag = 0
                if np.abs(fit.tvalues[-1]) >= stop:
                    bestlag = lag
                    break
        return bestlag

    def fit(self, maxlag=None, method='cmle', ic=None, trend='c',
            transparams=True, start_params=None, solver='lbfgs', maxiter=35,
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
            hqic - Hannan-Quinn Information Criterion
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
            Solver to be used if method is 'mle'.  The default is 'lbfgs'
            (limited memory Broyden-Fletcher-Goldfarb-Shanno).  Other choices
            are 'bfgs', 'newton' (Newton-Raphson), 'nm' (Nelder-Mead),
            'cg' - (conjugate gradient), 'ncg' (non-conjugate gradient),
            and 'powell'.
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

        See Also
        --------
        statsmodels.base.model.LikelihoodModel.fit
        """
        method = method.lower()
        if method not in ['cmle', 'yw', 'mle']:
            raise ValueError("Method %s not recognized" % method)
        self.method = method
        self.trend = trend
        self.transparams = transparams
        nobs = len(self.endog)  # overwritten if method is 'cmle'
        endog = self.endog

        if maxlag is None:
            maxlag = int(round(12*(nobs/100.)**(1/4.)))
        k_ar = maxlag  # stays this if ic is None

        # select lag length
        if ic is not None:
            ic = ic.lower()
            if ic not in ['aic', 'bic', 'hqic', 't-stat']:
                raise ValueError("ic option %s not understood" % ic)
            k_ar = self.select_order(k_ar, ic, trend, method)

        self.k_ar = k_ar  # change to what was chosen by ic

        # redo estimation for best lag
        # make LHS
        Y = endog[k_ar:, :]
        # make lagged RHS
        X = self._stackX(k_ar, trend)  # sets self.k_trend
        k_trend = self.k_trend
        self.exog_names = util.make_lag_names(self.endog_names, k_ar, k_trend)
        self.Y = Y
        self.X = X

        if method == "cmle":     # do OLS
            arfit = OLS(Y, X).fit()
            params = arfit.params
            self.nobs = nobs - k_ar
            self.sigma2 = arfit.ssr/arfit.nobs  # needed for predict fcasterr

        elif method == "mle":
            solver = solver.lower()
            self.nobs = nobs
            if start_params is None:
                start_params = OLS(Y, X).fit().params
            else:
                if len(start_params) != k_trend + k_ar:
                    raise ValueError("Length of start params is %d. There"
                                     " are %d parameters." %
                                     (len(start_params), k_trend + k_ar))
            start_params = self._invtransparams(start_params)
            if solver == 'lbfgs':
                kwargs.setdefault('pgtol', 1e-8)
                kwargs.setdefault('factr', 1e2)
                kwargs.setdefault('m', 12)
                kwargs.setdefault('approx_grad', True)
            mlefit = super(AR, self).fit(start_params=start_params,
                                         method=solver, maxiter=maxiter,
                                         full_output=full_output, disp=disp,
                                         callback=callback, **kwargs)

            params = mlefit.params
            if self.transparams:
                params = self._transparams(params)
                self.transparams = False  # turn off now for other results

        # don't use yw, because we can't estimate the constant
        #elif method == "yw":
        #    params, omega = yule_walker(endog, order=maxlag,
        #            method="mle", demean=False)
        #    # how to handle inference after Yule-Walker?
        #    self.params = params #TODO: don't attach here
        #    self.omega = omega

        pinv_exog = np.linalg.pinv(X)
        normalized_cov_params = np.dot(pinv_exog, pinv_exog.T)
        arfit = ARResults(self, params, normalized_cov_params)
        if method == 'mle' and full_output:
            arfit.mle_retvals = mlefit.mle_retvals
            arfit.mle_settings = mlefit.mle_settings
        return ARResultsWrapper(arfit)


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
        inv(dot(X.T,X)) where X is the lagged values.
    scale : float, optional
        An estimate of the scale of the model.

    Attributes
    ----------

    aic : float
        Akaike Information Criterion using Lutkephol's definition.
        :math:`log(sigma) + 2*(1 + k_ar + k_trend)/nobs`
    bic : float
        Bayes Information Criterion
        :math:`\\log(\\sigma) + (1 + k_ar + k_trend)*\\log(nobs)/nobs`
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
        Final prediction error using Lütkepohl's definition
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
        The residuals of the model. If the model is fit by 'mle' then the
        pre-sample residuals are calculated using fittedvalues from the Kalman
        Filter.
    roots : array
        The roots of the AR process are the solution to
        (1 - arparams[0]*z - arparams[1]*z**2 -...- arparams[p-1]*z**k_ar) = 0
        Stability requires that the roots in modulus lie outside the unit
        circle.
    scale : float
        Same as sigma2
    sigma2 : float
        The variance of the innovations (residuals).
    trendorder : int
        The polynomial order of the trend. 'nc' = None, 'c' or 't' = 0,
        'ct' = 1, etc.
    tvalues : array
        The t-values associated with `params`.
    """

    _cache = {}  # for scale setter

    def __init__(self, model, params, normalized_cov_params=None, scale=1.):
        super(ARResults, self).__init__(model, params, normalized_cov_params,
                                        scale)
        self._cache = {}
        self.nobs = model.nobs
        n_totobs = len(model.endog)
        self.n_totobs = n_totobs
        self.X = model.X  # copy?
        self.Y = model.Y
        k_ar = model.k_ar
        self.k_ar = k_ar
        k_trend = model.k_trend
        self.k_trend = k_trend
        trendorder = None
        if k_trend > 0:
            trendorder = k_trend - 1
        self.trendorder = trendorder
        #TODO: cmle vs mle?
        self.df_model = k_ar + k_trend
        self.df_resid = self.model.df_resid = n_totobs - self.df_model

    @cache_writable()
    def sigma2(self):
        model = self.model
        if model.method == "cmle":  # do DOF correction
            return 1. / self.nobs * sumofsq(self.resid)
        else:
            return self.model.sigma2

    @cache_writable()   # for compatability with RegressionResults
    def scale(self):
        return self.sigma2

    @cache_readonly
    def bse(self):  # allow user to specify?
        if self.model.method == "cmle":  # uses different scale/sigma def.
            resid = self.resid
            ssr = np.dot(resid, resid)
            ols_scale = ssr / (self.nobs - self.k_ar - self.k_trend)
            return np.sqrt(np.diag(self.cov_params(scale=ols_scale)))
        else:
            hess = approx_hess(self.params, self.model.loglike)
            return np.sqrt(np.diag(-np.linalg.inv(hess)))

    @cache_readonly
    def pvalues(self):
        return norm.sf(np.abs(self.tvalues))*2

    @cache_readonly
    def aic(self):
        #JP: this is based on loglike with dropped constant terms ?
        # Lutkepohl
        #return np.log(self.sigma2) + 1./self.model.nobs * self.k_ar
        # Include constant as estimated free parameter and double the loss
        return np.log(self.sigma2) + 2 * (1 + self.df_model)/self.nobs
        # Stata defintion
        #nobs = self.nobs
        #return -2 * self.llf/nobs + 2 * (self.k_ar+self.k_trend)/nobs

    @cache_readonly
    def hqic(self):
        nobs = self.nobs
        # Lutkepohl
        # return np.log(self.sigma2)+ 2 * np.log(np.log(nobs))/nobs * self.k_ar
        # R uses all estimated parameters rather than just lags
        return (np.log(self.sigma2) + 2 * np.log(np.log(nobs))/nobs *
                (1 + self.df_model))
        # Stata
        #nobs = self.nobs
        #return -2 * self.llf/nobs + 2 * np.log(np.log(nobs))/nobs * \
        #        (self.k_ar + self.k_trend)

    @cache_readonly
    def fpe(self):
        nobs = self.nobs
        df_model = self.df_model
        # Lutkepohl
        return ((nobs+df_model)/(nobs-df_model))*self.sigma2

    @cache_readonly
    def bic(self):
        nobs = self.nobs
        # Lutkepohl
        #return np.log(self.sigma2) + np.log(nobs)/nobs * self.k_ar
        # Include constant as est. free parameter
        return np.log(self.sigma2) + (1 + self.df_model) * np.log(nobs)/nobs
        # Stata
        # return -2 * self.llf/nobs + np.log(nobs)/nobs * (self.k_ar + \
        #       self.k_trend)

    @cache_readonly
    def resid(self):
        # NOTE: uses fittedvalues because it calculate presample values for mle
        model = self.model
        endog = model.endog.squeeze()
        if model.method == "cmle":  # elimate pre-sample
            return endog[self.k_ar:] - self.fittedvalues
        else:
            return model.endog.squeeze() - self.fittedvalues

    #def ssr(self):
    #    resid = self.resid
    #    return np.dot(resid, resid)

    @cache_readonly
    def roots(self):
        k = self.k_trend
        return np.roots(np.r_[1, -self.params[k:]]) ** -1

    @cache_readonly
    def fittedvalues(self):
        return self.model.predict(self.params)

    def predict(self, start=None, end=None, dynamic=False):
        params = self.params
        predictedvalues = self.model.predict(params, start, end, dynamic)
        return predictedvalues
        # TODO: consider returning forecast errors and confidence intervals?

    # Same docstring as AR.predict, but with "params" parameter removed
    preddoc = AR.predict.__doc__.split('\n')
    predict.__doc__ = '\n'.join(preddoc[:5] + preddoc[7:])


class ARResultsWrapper(wrap.ResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(
        tsbase.TimeSeriesResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(ARResultsWrapper, ARResults)  # noqa:E305


if __name__ == "__main__":
    import statsmodels.api as sm
    sunspots = sm.datasets.sunspots.load(as_pandas=False)
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
    import scikits.timeseries as ts
    d1 = ts.Date(year=1700, freq='A')
    #NOTE: have to have yearBegin offset for annual data until parser rewrite
    #should this be up to the user, or should it be done in TSM init?
    #NOTE: not anymore, it's end of year now
    ts_dr = ts.date_array(start_date=d1, length=len(sunspots.endog))
    pandas_dr = pandas.DatetimeIndex(start=d1.datetime,
                                 periods=len(sunspots.endog),
                                     freq='A-DEC')
    #pandas_dr = pandas_dr.shift(-1, pandas.datetools.yearBegin)

    dates = np.arange(1700, 1700 + len(sunspots.endog))
    dates = ts.date_array(dates, freq='A')
    #sunspots = pandas.Series(sunspots.endog, index=dates)

    #NOTE: pandas only does business days for dates it looks like
    import datetime
    dt_dates = np.asarray(lmap(datetime.datetime.fromordinal,
                              ts_dr.toordinal().astype(int)))
    sunspots = pandas.Series(sunspots.endog, index=dt_dates)

    #NOTE: pandas can't handle pre-1900 dates
    mod = AR(sunspots, freq='A')
    res = mod.fit(method='mle', maxlag=9)

# some data for an example in Box Jenkins
    IBM = np.asarray([460, 457, 452, 459, 462, 459, 463, 479, 493, 490.])
    w = np.diff(IBM)
    theta = .5
