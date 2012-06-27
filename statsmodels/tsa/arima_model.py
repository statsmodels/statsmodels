from datetime import datetime

import numpy as np
from scipy import optimize
from scipy.stats import t, norm
from scipy.signal import lfilter
from numpy import (dot, identity, kron, log, zeros, pi, exp, eye, abs, empty,
                   zeros_like)
from numpy.linalg import inv, pinv

from statsmodels.tools.decorators import (cache_readonly,
        cache_writable, resettable_cache)
import statsmodels.base.model as base
import statsmodels.tsa.base.tsa_model as tsbase
import statsmodels.base.wrapper as wrap
from statsmodels.regression.linear_model import yule_walker, GLS
from statsmodels.tsa.tsatools import (lagmat, add_trend,
        _ar_transparams, _ar_invtransparams, _ma_transparams,
        _ma_invtransparams)
from statsmodels.tsa.vector_ar import util
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.sandbox.regression.numdiff import (approx_fprime,
        approx_fprime_cs, approx_hess, approx_hess_cs)
from statsmodels.tsa.base.datetools import _index_date
from statsmodels.tsa.kalmanf import KalmanFilter
try:
    from kalmanf import kalman_loglike
    fast_kalman = 1
except:
    fast_kalman = 0

_arma_params = """endog : array-like
    The endogenous variable.
exog : array-like, optional
    An optional arry of exogenous variables. This should *not* include a
    constant or trend. You can specify this in the `fit` method."""

_arma_model = "Autoregressive Moving Average ARMA(p,q) Model"

_arima_model = "Autoregressive Integrated Moving Average ARIMA(p,d,q) Model"

_arima_params = """endog : array-like
    The endogenous variable.
order : iterable
    The (p,d,q) order of the model for the number of AR parameters,
    differences, and MA parameters to use.
exog : array-like, optional
    An optional arry of exogenous variables. This should *not* include a
    constant or trend. You can specify this in the `fit` method."""

def _check_arima_start(start, k_ar, k_diff, method, dynamic):
    if start < 0:
        raise ValueError("The start index %d of the original series "
                             "has been differenced away" % start)
    elif (dynamic or 'mle' not in method) and start < k_ar:
        raise ValueError("Start must be >= k_ar for conditional MLE "
                "or dynamic forecast. Got %d" % start)

def _get_predict_out_of_sample(endog, p, q, k_trend, k_exog, start, errors,
                               trendparam, exparams, arparams, maparams, steps,
                               method):
    """
    Returns endog, resid, mu of appropriate length for out of sample
    prediction.
    """
    if q:
        resid = np.zeros(q)
        if start and 'mle' in method or start == p:
            resid[:q] = errors[start-q:start]
        elif start:
            resid[:q] = errors[start-q-p:start-p]
        else:
            resid[:q] = errors[-q:]
    else:
        resid = None

    y = endog
    if k_trend == 1:
        # use expectation not constant
        mu = trendparam * (1 - arparams.sum())
        mu = np.array([mu]*steps)
    else:
        mu = np.zeros(steps)

    if k_exog > 0:
        mu += np.dot(exparams, self.exog)

    endog = np.zeros(p + steps - 1)

    if p and start:
        endog[:p] = y[start-p:start]
    elif p:
        endog[:p] = y[-p:]

    return endog, resid, mu

def _arma_predict_out_of_sample(params, steps, errors, p, q, k_trend, k_exog,
                                endog, exog=None, start=0, method='mle'):
    (trendparam, exparams,
     arparams, maparams) = _unpack_params(params, (p,q), k_trend,
                                k_exog, reverse=True)
    endog, resid, mu = _get_predict_out_of_sample(endog, p, q, k_trend, k_exog,
                                                   start, errors, trendparam,
                                                   exparams, arparams,
                                                   maparams, steps, method)

    forecast = np.zeros(steps)
    if steps == 1:
        if q:
            return mu[0] + np.dot(arparams, endog[:p]) + np.dot(maparams,
                                                                resid[:q])
        else:
            return mu[0] + np.dot(arparams, endog[:p])

    if q:
        i = 0 # if q == 1
    else:
        i = -1

    for i in range(min(q,steps-1)):
        fcast = mu[i] + np.dot(arparams,endog[i:i+p]) + \
                      np.dot(maparams,resid[i:i+q])
        forecast[i] = fcast
        endog[i+p] = fcast

    for i in range(i+1,steps-1):
        fcast = mu[i] + np.dot(arparams,endog[i:i+p])
        forecast[i] = fcast
        endog[i+p] = fcast

    #need to do one more without updating endog
    forecast[-1] = mu[-1] + np.dot(arparams,endog[steps-1:])
    return forecast

def _arma_predict_in_sample(start, end, endog, resid, k_ar,
                            method):
    """
    Pre- and in-sample fitting for ARMA.
    """
    if 'mle' in method:
        fittedvalues = endog - resid #get them all then trim
    elif k_ar > 0:
        fittedvalues = endog[k_ar:] - resid

    fv_start = start
    if 'mle' not in method:
        fv_start -= k_ar # start is in terms of endog index
    predictedvalues = np.zeros(end + 1 - fv_start)
    fv_end = min(len(fittedvalues), end + 1)
    return fittedvalues[fv_start:fv_end]

def _validate(start, k_ar, k_diff, dates, method):
    if isinstance(start, (basestring, datetime)):
        start_date = start
        start = _index_date(start, dates)
        start -= k_diff
    if 'mle' not in method and start < k_ar - k_diff:
        raise ValueError("Start must be >= k_ar for conditional "
                         "MLE or dynamic forecast. Got %s" % start)

    return start

def _unpack_params(params, order, k_trend, k_exog, reverse=False):
    p, q = order
    k = k_trend + k_exog
    maparams = params[k+p:]
    arparams = params[k:k+p]
    trend = params[:k_trend]
    exparams = params[k_trend:k]
    if reverse:
        return trend, exparams, arparams[::-1], maparams[::-1]
    return trend, exparams, arparams, maparams

def _unpack_order(order):
    k_ar, k_ma, k = order
    k_lags = max(k_ar, k_ma+1)
    return k_ar, k_ma, order, k_lags

def _make_arma_names(data, k_trend, order):
    k_ar, k_ma = order
    exog = data.exog
    if exog is not None:
        exog_names = data._get_names(data._orig_exog) or []
    else:
        exog_names = []
    ar_lag_names = util.make_lag_names([data.ynames], k_ar, 0)
    ar_lag_names = [''.join(('ar.', i))
                              for i in ar_lag_names]
    ma_lag_names = util.make_lag_names([data.ynames], k_ma, 0)
    ma_lag_names = [''.join(('ma.', i)) for i in ma_lag_names]
    trend_name = util.make_lag_names('', 0, k_trend)
    exog_names = trend_name + exog_names + ar_lag_names + ma_lag_names
    return exog_names

def _make_arma_exog(endog, exog, trend):
    k_trend = 1 # overwritten if no constant
    if exog is None and trend == 'c':   # constant only
        exog = np.ones((len(endog),1))
    elif exog is not None and trend == 'c': # constant plus exogenous
        exog = add_trend(exog, trend='c', prepend=True)
    elif exog is not None and trend == 'nc':
        # make sure it's not holding constant from last run
        if exog.var() == 0:
            exog = None
        k_trend = 0
    if trend == 'nc':
        k_trend = 0
    return k_trend, exog

class ARMA(tsbase.TimeSeriesModel):

    __doc__ = tsbase._tsa_doc % {"model" : _arma_model,
                    "params" : _arma_params, "extra" : ""}

    def __init__(self, endog, exog=None, dates=None, freq=None):
        super(ARMA, self).__init__(endog, exog, dates, freq)
        if exog is not None:
            k_exog = exog.shape[1]  # number of exog. variables excl. const
        else:
            k_exog = 0
        self.k_exog = k_exog

    def _fit_start_params_hr(self, order):
        """
        Get starting parameters for fit.

        Parameters
        ----------
        order : iterable
            (p,q,k) - AR lags, MA lags, and number of exogenous variables
            including the constant.

        Returns
        -------
        start_params : array
            A first guess at the starting parameters.

        Notes
        -----
        If necessary, fits an AR process with the laglength selected according
        to best BIC.  Obtain the residuals.  Then fit an ARMA(p,q) model via
        OLS using these residuals for a first approximation.  Uses a separate
        OLS regression to find the coefficients of exogenous variables.

        References
        ----------
        Hannan, E.J. and Rissanen, J.  1982.  "Recursive estimation of mixed
            autoregressive-moving average order."  `Biometrika`.  69.1.
        """
        p,q,k = order
        start_params = zeros((p+q+k))
        endog = self.endog.copy() # copy because overwritten
        exog = self.exog
        if k != 0:
            ols_params = GLS(endog, exog).fit().params
            start_params[:k] = ols_params
            endog -= np.dot(exog, ols_params).squeeze()
        if q != 0:
            if p != 0:
                armod = AR(endog).fit(ic='bic', trend='nc')
                arcoefs_tmp = armod.params
                p_tmp = armod.k_ar
                resid = endog[p_tmp:] - np.dot(lagmat(endog, p_tmp,
                                trim='both'), arcoefs_tmp)
                if p < p_tmp + q:
                    endog_start = p_tmp + q - p
                    resid_start = 0
                else:
                    endog_start = 0
                    resid_start = p - p_tmp - q
                lag_endog = lagmat(endog, p, 'both')[endog_start:]
                lag_resid = lagmat(resid, q, 'both')[resid_start:]
                # stack ar lags and resids
                X = np.column_stack((lag_endog, lag_resid))
                coefs = GLS(endog[max(p_tmp+q,p):], X).fit().params
                start_params[k:k+p+q] = coefs
            else:
                start_params[k+p:k+p+q] = yule_walker(endog, order=q)[0]
        if q==0 and p != 0:
            arcoefs = yule_walker(endog, order=p)[0]
            start_params[k:k+p] = arcoefs
        return start_params

    def _fit_start_params(self, order, method):
        if method != 'css-mle': # use Hannan-Rissanen to get start params
            start_params = self._fit_start_params_hr(order)
        else: # use CSS to get start params
            func = lambda params: -self.loglike_css(params)
            #start_params = [.1]*(k_ar+k_ma+k_exog) # different one for k?
            start_params = self._fit_start_params_hr(order)
            if self.transparams:
                start_params = self._invtransparams(start_params)
            bounds = [(None,)*2]*sum(order)
            mlefit = optimize.fmin_l_bfgs_b(func, start_params,
                        approx_grad=True, m=12, pgtol=1e-7, factr=1e3,
                        bounds = bounds, iprint=-1)
            start_params = self._transparams(mlefit[0])
        return start_params


    def score(self, params):
        """
        Compute the score function at params.

        Notes
        -----
        This is a numerical approximation.
        """
        loglike = self.loglike
        #if self.transparams:
        #    params = self._invtransparams(params)
        #return approx_fprime(params, loglike, epsilon=1e-5)
        return approx_fprime_cs(params, loglike)

    def hessian(self, params):
        """
        Compute the Hessian at params,

        Notes
        -----
        This is a numerical approximation.
        """
        loglike = self.loglike
        #if self.transparams:
        #    params = self._invtransparams(params)
        if not fast_kalman or self.method == "css":
            return approx_hess_cs(params, loglike, epsilon=1e-5)
        else:
            return approx_hess(params, self.loglike, epsilon=1e-3)[0]


    def _transparams(self, params):
        """
        Transforms params to induce stationarity/invertability.

        Reference
        ---------
        Jones(1980)
        """
        k_ar, k_ma = self.k_ar, self.k_ma
        k = self.k_exog + self.k_trend
        newparams = np.zeros_like(params)

        # just copy exogenous parameters
        if k != 0:
            newparams[:k] = params[:k]

        # AR Coeffs
        if k_ar != 0:
            newparams[k:k+k_ar] = _ar_transparams(params[k:k+k_ar].copy())

        # MA Coeffs
        if k_ma != 0:
            newparams[k+k_ar:] = _ma_transparams(params[k+k_ar:].copy())
        return newparams

    def _invtransparams(self, start_params):
        """
        Inverse of the Jones reparameterization
        """
        k_ar, k_ma = self.k_ar, self.k_ma
        k = self.k_exog + self.k_trend
        newparams = start_params.copy()
        arcoefs = newparams[k:k+k_ar]
        macoefs = newparams[k+k_ar:]
        # AR coeffs
        if k_ar != 0:
            newparams[k:k+k_ar] = _ar_invtransparams(arcoefs)

        # MA coeffs
        if k_ma != 0:
            newparams[k+k_ar:k+k_ar+k_ma] = _ma_invtransparams(macoefs)
        return newparams

    def _get_predict_start(self, start, dynamic):
        # do some defaults
        method = getattr(self, 'method', 'mle')
        k_ar = getattr(self, 'k_ar', 0)
        k_diff = getattr(self, 'k_diff', 0)
        if start is None:
            if 'mle' in method and not dynamic:
                start = 0
            else:
                start = k_ar
        elif isinstance(start, int):
            start = super(ARMA, self)._get_predict_start(start)
        else: # should be on a date
            #elif 'mle' not in method or dynamic: # should be on a date
            start = _validate(start, k_ar, k_diff, self._data.dates,
                              method)
            start = super(ARMA, self)._get_predict_start(start)
        _check_arima_start(start, k_ar, k_diff, method, dynamic)
        return start

    def _get_predict_end(self, start, dynamic=False):
        # pass through so predict works for ARIMA and ARMA
        return super(ARMA, self)._get_predict_end(start)

    def geterrors(self, params):
        """
        Get the errors of the ARMA process.

        Parameters
        ----------
        params : array-like
            The fitted ARMA parameters
        order : array-like
            3 item iterable, with the number of AR, MA, and exogenous
            parameters, including the trend
        """

        #start = self._get_predict_start(start) # will be an index of a date
        #end, out_of_sample = self._get_predict_end(end)
        params = np.asarray(params)
        k_ar, k_ma = self.k_ar, self.k_ma
        k = self.k_exog + self.k_trend


        method = getattr(self, 'method', 'mle')
        if 'mle' in method: # use KalmanFilter to get errors
            (y, k, nobs, k_ar, k_ma, k_lags, newparams, Z_mat, m, R_mat,
            T_mat, paramsdtype) = KalmanFilter._init_kalman_state(params, self)
            errors = KalmanFilter.geterrors(y,k,k_ar,k_ma, k_lags, nobs,
                    Z_mat, m, R_mat, T_mat, paramsdtype)
            if isinstance(errors, tuple):
                errors = errors[0] # non-cython version returns a tuple
        else: # use scipy.signal.lfilter
            y = self.endog.copy()
            k = self.k_exog + self.k_trend
            if k > 0:
                y -= dot(self.exog, params[:k])

            k_ar = self.k_ar
            k_ma = self.k_ma


            (trendparams, exparams,
             arparams, maparams) = _unpack_params(params, (k_ar, k_ma),
                                        self.k_trend, self.k_exog,
                                        reverse=False)
            b,a = np.r_[1,-arparams], np.r_[1,maparams]
            zi = zeros((max(k_ar, k_ma)))
            for i in range(k_ar):
                zi[i] = sum(-b[:i+1][::-1]*y[:i+1])
            e = lfilter(b,a,y,zi=zi)
            errors = e[0][k_ar:]
        return errors.squeeze()

    def predict(self, params, start=None, end=None, exog=None, dynamic=False):
        """
        In-sample and out-of-sample prediction.

        Parameters
        ----------
        params : array-like
            The fitted parameters of the model.
        start : int, str, or datetime
            Zero-indexed observation number at which to start forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type.
        end : int, str, or datetime
            Zero-indexed observation number at which to end forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type.
        exog : array-like, optional
            If the model is an ARMAX and out-of-sample forecasting is
            requested, exog must be given.
        dynamic : bool, optional
            The `dynamic` keyword affects in-sample prediction. If dynamic
            is False, then the in-sample lagged values are used for
            prediction. If `dynamic` is True, then in-sample forecasts are
            used in place of lagged dependent variables. The first forecasted
            value is `start`.

        Notes
        ------
        Consider using the results prediction.
        """
        method = getattr(self, 'method', 'mle') # don't assume fit
        #params = np.asarray(params)

        # will return an index of a date
        start = self._get_predict_start(start, dynamic)
        end, out_of_sample = self._get_predict_end(end, dynamic)
        if out_of_sample and (exog is None and self.k_exog > 0):
            raise ValueError("You must provide exog for ARMAX")

        endog = self.endog
        resid = self.geterrors(params)
        k_ar = self.k_ar

        if dynamic:
            #TODO: now that predict does dynamic in-sample it should
            # also return error estimates and confidence intervals
            # but how? len(endog) is not tot_obs
            out_of_sample += end - start + 1
            return _arma_predict_out_of_sample(params, out_of_sample, resid,
                    k_ar, self.k_ma, self.k_trend, self.k_exog, endog, exog,
                    start, method)

        predictedvalues = _arma_predict_in_sample(start, end, endog, resid,
                            k_ar, method)
        if out_of_sample:
            forecastvalues = _arma_predict_out_of_sample(params, out_of_sample,
                                        resid, k_ar, self.k_ma, self.k_trend,
                                        self.k_exog, endog, exog,
                                        method=method)
            predictedvalues = np.r_[predictedvalues, forecastvalues]
        return predictedvalues

    def loglike(self, params):
        """
        Compute the log-likelihood for ARMA(p,q) model

        Notes
        -----
        Likelihood used depends on the method set in fit
        """
        method = self.method
        if method in ['mle', 'css-mle']:
            return self.loglike_kalman(params)
        elif method == 'css':
            return self.loglike_css(params)
        else:
            raise ValueError("Method %s not understood" % method)

    def loglike_kalman(self, params):
        """
        Compute exact loglikelihood for ARMA(p,q) model using the Kalman Filter.
        """
        return KalmanFilter.loglike(params, self)

    def loglike_css(self, params):
        """
        Conditional Sum of Squares likelihood function.
        """
        k_ar = self.k_ar
        k_ma = self.k_ma
        k = self.k_exog + self.k_trend
        y = self.endog.copy().astype(params.dtype)
        nobs = self.nobs
        # how to handle if empty?
        if self.transparams:
            newparams = self._transparams(params)
        else:
            newparams = params
        if k > 0:
            y -= dot(self.exog, newparams[:k])
        # the order of p determines how many zeros errors to set for lfilter
        b,a = np.r_[1,-newparams[k:k+k_ar]], np.r_[1,newparams[k+k_ar:]]
        zi = np.zeros((max(k_ar,k_ma)), dtype=params.dtype)
        for i in range(k_ar):
            zi[i] = sum(-b[:i+1][::-1] * y[:i+1])
        errors = lfilter(b,a, y, zi=zi)[0][k_ar:]

        ssr = np.dot(errors,errors)
        sigma2 = ssr/nobs
        self.sigma2 = sigma2
        llf = -nobs/2.*(log(2*pi) + log(sigma2)) - ssr/(2*sigma2)
        return llf

    def fit(self, order, start_params=None, trend='c', method = "css-mle",
            transparams=True, solver=None, maxiter=35, full_output=1,
            disp=5, callback=None, **kwargs):
        """
        Fits ARMA(p,q) model using exact maximum likelihood via Kalman filter.

        Parameters
        ----------
        start_params : array-like, optional
            Starting parameters for ARMA(p,q).  If None, the default is given
            by ARMA._fit_start_params.  See there for more information.
        transparams : bool, optional
            Whehter or not to transform the parameters to ensure stationarity.
            Uses the transformation suggested in Jones (1980).  If False,
            no checking for stationarity or invertibility is done.
        method : str {'css-mle','mle','css'}
            This is the loglikelihood to maximize.  If "css-mle", the
            conditional sum of squares likelihood is maximized and its values
            are used as starting values for the computation of the exact
            likelihood via the Kalman filter.  If "mle", the exact likelihood
            is maximized via the Kalman Filter.  If "css" the conditional sum
            of squares likelihood is maximized.  All three methods use
            `start_params` as starting parameters.  See above for more
            information.
        trend : str {'c','nc'}
            Whehter to include a constant or not.  'c' includes constant,
            'nc' no constant.
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
            If True, convergence information is printed.  For the default
            l_bfgs_b solver, disp controls the frequency of the output during
            the iterations. disp < 0 means no output in this case.
        callback : function, optional
            Called after each iteration as callback(xk) where xk is the current
            parameter vector.
        kwargs
            See Notes for keyword arguments that can be passed to fit.

        Returns
        -------
        `statsmodels.tsa.arima.ARMAResults` class

        See also
        --------
        statsmodels.model.LikelihoodModel.fit for more information
        on using the solvers.

        Notes
        ------
        If fit by 'mle', it is assumed for the Kalman Filter that the initial
        unkown state is zero, and that the inital variance is
        P = dot(inv(identity(m**2)-kron(T,T)),dot(R,R.T).ravel('F')).reshape(r,
        r, order = 'F')

        The below is the docstring from
        `statsmodels.LikelihoodModel.fit`
        """
        # enforce invertibility
        self.transparams = transparams

        self.method = method.lower()

        # get model order and constants
        self.k_ar = k_ar = int(order[0])
        self.k_ma = k_ma = int(order[1])
        self.k_lags = k_lags = max(k_ar,k_ma+1)
        endog, exog = self.endog, self.exog
        k_exog = self.k_exog
        self.nobs = len(endog) # this is overwritten if method is 'css'

        # (re)set trend and handle exogenous variables
        # always pass original exog
        k_trend, exog = _make_arma_exog(endog, self._data.exog, trend)

        self.k_trend = k_trend
        self.exog = exog    # overwrites original exog from __init__

        # (re)set names for this model
        self.exog_names = _make_arma_names(self._data, k_trend, order)

        k = k_trend + k_exog


        # choose objective function
        method = method.lower()
        # adjust nobs for css
        if method == 'css':
            self.nobs = len(self.endog) - self.k_ar
        loglike = lambda params: -self.loglike(params)

        if start_params is not None:
            start_params = np.asarray(start_params)

        else: # estimate starting parameters
            start_params = self._fit_start_params((k_ar,k_ma,k), method)

        if transparams: # transform initial parameters to ensure invertibility
            start_params = self._invtransparams(start_params)

        if solver is None:  # use default limited memory bfgs
            bounds = [(None,)*2]*(k_ar+k_ma+k)
            mlefit = optimize.fmin_l_bfgs_b(loglike, start_params,
                    approx_grad=True, m=12, pgtol=1e-8, factr=1e2,
                    bounds=bounds, iprint=disp)
            self.mlefit = mlefit
            params = mlefit[0]

        else:   # call the solver from LikelihoodModel
            mlefit = super(ARMA, self).fit(start_params, method=solver,
                        maxiter=maxiter, full_output=full_output, disp=disp,
                        callback = callback, **kwargs)
            self.mlefit = mlefit
            params = mlefit.params

        if transparams: # transform parameters back
            params = self._transparams(params)

        self.transparams = False # set to false so methods don't expect transf.

        normalized_cov_params = None #TODO: fix this
        armafit = ARMAResults(self, params, normalized_cov_params)
        return ARMAResultsWrapper(armafit)

    fit.__doc__ += base.LikelihoodModel.fit.__doc__

#NOTE: the length of endog changes when we give a difference to fit
#so model methods are not the same on unfit models as fit ones
#starting to think that order of model should be put in instantiation...
class ARIMA(ARMA):

    __doc__ = tsbase._tsa_doc % {"model" : _arima_model,
            "params" : _arima_params, "extra" : ""}

    def __init__(self, endog, order, exog=None, dates=None, freq=None):
        super(ARIMA, self).__init__(endog, exog, dates, freq)
        p,d,q = order
        self.k_diff = d
        self.k_ar = p
        self.k_ma = q
        self.endog = np.diff(self.endog, n=d)
        self._data.ynames = 'D.' + self.endog_names
        # what about exog, should we difference it automatically before
        # super call?

    def _get_predict_start(self, start, dynamic):
        """
        """
        #TODO: remove all these getattr and move order specification to
        # class constructor
        k_diff = getattr(self, 'k_diff', 0)
        method = getattr(self, 'method', 'mle')
        k_ar = getattr(self, 'k_ar', 0)
        if start is None:
            if 'mle' in method and not dynamic:
                start = 0
            else:
                start = k_ar
        elif isinstance(start, int):
                start -= k_diff
                try: # catch when given an integer outside of dates index
                    start = super(ARIMA, self)._get_predict_start(start,
                                                                  dynamic)
                except IndexError as err:
                    raise ValueError("start must be in series. "
                                     "got %d" % (start + k_diff))
        else: # received a date
            start = _validate(start, k_ar, k_diff, self._data.dates,
                              method)
            start = super(ARIMA, self)._get_predict_start(start, dynamic)
        # reset date for k_diff adjustment
        self._set_predict_start_date(start + k_diff)
        return start

    def _get_predict_end(self, end, dynamic=False):
        """
        Returns last index to be forecast of the differenced array.
        Handling of inclusiveness should be done in the predict function.
        """
        end, out_of_sample = super(ARIMA, self)._get_predict_end(end, dynamic)
        if 'mle' not in self.method and not dynamic:
            end -= self.k_ar

        return end - self.k_diff, out_of_sample

    def fit(self, start_params=None, trend='c', method = "css-mle",
            transparams=True, solver=None, maxiter=35, full_output=1,
            disp=5, callback=None, **kwargs):
        """
        Fits ARIMA(p,d,q) model by exact maximum likelihood via Kalman filter.

        Parameters
        ----------
        start_params : array-like, optional
            Starting parameters for ARMA(p,q).  If None, the default is given
            by ARMA._fit_start_params.  See there for more information.
        transparams : bool, optional
            Whehter or not to transform the parameters to ensure stationarity.
            Uses the transformation suggested in Jones (1980).  If False,
            no checking for stationarity or invertibility is done.
        method : str {'css-mle','mle','css'}
            This is the loglikelihood to maximize.  If "css-mle", the
            conditional sum of squares likelihood is maximized and its values
            are used as starting values for the computation of the exact
            likelihood via the Kalman filter.  If "mle", the exact likelihood
            is maximized via the Kalman Filter.  If "css" the conditional sum
            of squares likelihood is maximized.  All three methods use
            `start_params` as starting parameters.  See above for more
            information.
        trend : str {'c','nc'}
            Whether to include a constant or not.  'c' includes constant,
            'nc' no constant.
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
            If True, convergence information is printed.  For the default
            l_bfgs_b solver, disp controls the frequency of the output during
            the iterations. disp < 0 means no output in this case.
        callback : function, optional
            Called after each iteration as callback(xk) where xk is the current
            parameter vector.
        kwargs
            See Notes for keyword arguments that can be passed to fit.

        Returns
        -------
        `statsmodels.tsa.arima.ARMAResults` class

        See also
        --------
        statsmodels.model.LikelihoodModel.fit for more information
        on using the solvers.

        Notes
        ------
        If fit by 'mle', it is assumed for the Kalman Filter that the initial
        unkown state is zero, and that the inital variance is
        P = dot(inv(identity(m**2)-kron(T,T)),dot(R,R.T).ravel('F')).reshape(r,
        r, order = 'F')

        The below is the docstring from
        `statsmodels.LikelihoodModel.fit`
        """
        arima_fit = super(ARIMA, self).fit((self.k_ar, self.k_ma),
                               start_params, trend, method,
                               transparams, solver, maxiter, full_output,
                               disp, callback, **kwargs)
        if self.k_diff == 0:#TODO: what do to here?
            #Overide results methods or just return ARMA?
            return arima_fit
        normalized_cov_params = None #TODO: fix this?
        arima_fit = ARIMAResults(self, arima_fit._results.params,
                                       normalized_cov_params)
        arima_fit.k_diff = self.k_diff
        return ARIMAResultsWrapper(arima_fit)

    def predict(self, params, start=None, end=None, exog=None, typ='linear',
                dynamic=False):
        """
        ARIMA model prediction

        Parameters
        ----------
        params : array
            The parameters of the model
        start : int, str, or datetime
            Zero-indexed observation number at which to start forecasting, ie.,
            the first forecast is start. See notes. Can also be a date string
            to parse or a datetime type.
        end : int, str, or datetime
            Zero-indexed observation number at which to end forecasting, ie.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type.
        exog : array-like, optional
            If the model is an ARMAX and out-of-sample forecasting is
            requestion, exog must be given.
        typ : str {'linear', 'levels'}, optional
            - 'linear' : Linear prediction in terms of the differenced
              endogenous variables.
            - 'levels' : Predict the levels of the original endogenous
              variables.
        dynamic : bool, optional
            The `dynamic` keyword affects in-sample prediction. If dynamic
            is False, then the in-sample lagged values are used for
            prediction. If `dynamic` is True, then in-sample forecasts are
            used in place of lagged dependent variables. The first forecasted
            value is `start`.

        Returns
        -------
        predict : array
            The predicted values.

        Notes
        -----
        It is recommended to use dates with the time-series models, as the
        below will probably make clear. However, if ARIMA is used without
        dates and/or `start` and `end` are given as indices, then these
        indices are in terms of the *original*, undifferenced series. Ie.,
        given some undifferenced observations::

        1970Q1, 1
        1970Q2, 1.5
        1970Q3, 1.25
        1970Q4, 2.25
        1971Q1, 1.2
        1971Q2, 4.1

        1970Q1 is observation 0 in the original series. However, if we fit an
        ARIMA(p,1,q) model then we lose this first observation through
        differencing. Therefore, the first observation we can forecast (if
        using exact MLE) is index 1. In the differenced series this is index
        0, but we refer to it as 1 from the original series.
        """
        # go ahead and convert to an index for easier checking
        if isinstance(start, (basestring, datetime)):
            start = _index_date(start, self._data.dates)
        if typ == 'linear':
            if not dynamic or (start != self.k_ar + self.k_diff and
                                                    start is not None):
                return super(ARIMA, self).predict(params, start, end, exog,
                                              dynamic)
            else:
                # need to assume pre-sample residuals are zero
                # do this by a hack
                q = self.k_ma
                self.k_ma = 0
                predictedvalues = super(ARIMA, self).predict(params, start,
                                                             end, exog,
                                                             dynamic)
                self.k_ma = q
                return predictedvalues
        elif typ == 'levels':
            endog = self._data.endog
            if not dynamic:
                predict = super(ARIMA, self).predict(params, start, end,
                                                     dynamic)

                start = self._get_predict_start(start, dynamic)
                end, out_of_sample = self._get_predict_end(end)
                if 'mle' in self.method:
                    # add each predicted diff to lagged endog
                    if out_of_sample:
                        fv = predict[:-out_of_sample] + endog[start:end+1]
                        fv = np.r_[fv,
                              endog[-1] + np.cumsum(predict[-out_of_sample:])]
                    else:
                        fv = predict + endog[start:end + 1]
                else:
                    k_ar = self.k_ar
                    if out_of_sample:
                        fv = (predict[:-out_of_sample] +
                                endog[max(start, self.k_ar-1):end+k_ar+1])
                        fv = np.r_[fv,
                              endog[-1] + np.cumsum(predict[-out_of_sample:])]
                    else:
                        fv = predict + endog[max(start, k_ar):end+k_ar+1]
            else:
                #IFF we need to use pre-sample values assume pre-sample
                # residuals are zero, do this by a hack
                if start == self.k_ar + self.k_diff or start is None:
                    # do the first k_diff+1 separately
                    p = self.k_ar
                    q = self.k_ma
                    k_exog = self.k_exog
                    k_trend = self.k_trend
                    k_diff = self.k_diff
                    (trendparam, exparams,
                     arparams, maparams) = _unpack_params(params, (p,q),
                                                          k_trend,
                                                          k_exog,
                                                          reverse=True)
                    # this is the hack
                    self.k_ma = 0

                    predict = super(ARIMA, self).predict(params, start, end,
                                                         exog, dynamic)
                    if not start:
                        start = self._get_predict_start(start, dynamic)
                        start += k_diff
                    self.k_ma = q
                    return endog[start-1] + np.cumsum(predict)
                else:
                    predict = super(ARIMA, self).predict(params, start, end,
                                                         exog, dynamic)
                    return endog[start-1] + np.cumsum(predict)
            return fv

        else: # pragma : no cover
            raise ValueError("typ %s not understood" % typ)

class ARMAResults(tsbase.TimeSeriesModelResults):
    """
    Class to hold results from fitting an ARMA model.

    Parameters
    ----------
    model : ARMA instance
        The fitted model instance
    params : array
        Fitted parameters
    normalized_cov_params : array, optional
        The normalized variance covariance matrix
    scale : float, optional
        Optional argument to scale the variance covariance matrix.

    Returns
    --------
    **Attributes**

    aic : float
        Akaike Information Criterion
        :math:`-2*llf+2*(df_model+1)`
    arparams : array
        The parameters associated with the AR coefficients in the model.
    arroots : array
        The roots of the AR coefficients are the solution to
        (1 - arparams[0]*z - arparams[1]*z**2 -...- arparams[p-1]*z**k_ar) = 0
        Stability requires that the roots in modulus lie outside the unit
        circle.
    bic : float
        Bayes Information Criterion
        -2*llf + log(nobs)*(df_model+1)
        Where if the model is fit using conditional sum of squares, the
        number of observations `nobs` does not include the `p` pre-sample
        observations.
    bse : array
        The standard errors of the parameters. These are computed using the
        numerical Hessian.
    df_model : array
        The model degrees of freedom = `k_exog` + `k_trend` + `k_ar` + `k_ma`
    df_resid : array
        The residual degrees of freedom = `nobs` - `df_model`
    fittedvalues : array
        The predicted values of the model.
    hqic : float
        Hannan-Quinn Information Criterion
        -2*llf + 2*(`df_model`)*log(log(nobs))
        Like `bic` if the model is fit using conditional sum of squares then
        the `k_ar` pre-sample observations are not counted in `nobs`.
    k_ar : int
        The number of AR coefficients in the model.
    k_exog : int
        The number of exogenous variables included in the model. Does not
        include the constant.
    k_ma : int
        The number of MA coefficients.
    k_trend : int
        This is 0 for no constant or 1 if a constant is included.
    llf : float
        The value of the log-likelihood function evaluated at `params`.
    maparams : array
        The value of the moving average coefficients.
    maroots : array
        The roots of the MA coefficients are the solution to
        (1 + maparams[0]*z + maparams[1]*z**2 + ... + maparams[q-1]*z**q) = 0
        Stability requires that the roots in modules lie outside the unit
        circle.
    model : ARMA instance
        A reference to the model that was fit.
    nobs : float
        The number of observations used to fit the model. If the model is fit
        using exact maximum likelihood this is equal to the total number of
        observations, `n_totobs`. If the model is fit using conditional
        maximum likelihood this is equal to `n_totobs` - `k_ar`.
    n_totobs : float
        The total number of observations for `endog`. This includes all
        observations, even pre-sample values if the model is fit using `css`.
    params : array
        The parameters of the model. The order of variables is the trend
        coefficients and the `k_exog` exognous coefficients, then the
        `k_ar` AR coefficients, and finally the `k_ma` MA coefficients.
    pvalues : array
        The p-values associated with the t-values of the coefficients. Note
        that the coefficients are assumed to have a Student's T distribution.
    resid : array
        The model residuals. If the model is fit using 'mle' then the
        residuals are created via the Kalman Filter. If the model is fit
        using 'css' then the residuals are obtained via `scipy.signal.lfilter`
        adjusted such that the first `k_ma` residuals are zero. These zero
        residuals are not returned.
    scale : float
        This is currently set to 1.0 and not used by the model or its results.
    sigma2 : float
        The variance of the residuals. If the model is fit by 'css',
        sigma2 = ssr/nobs, where ssr is the sum of squared residuals. If
        the model is fit by 'mle', then sigma2 = 1/nobs * sum(v**2 / F)
        where v is the one-step forecast error and F is the forecast error
        variance. See `nobs` for the difference in definitions depending on the
        fit.
    """
    _cache = {}

    #TODO: use this for docstring when we fix nobs issue


    def __init__(self, model, params, normalized_cov_params=None, scale=1.):
        super(ARMAResults, self).__init__(model, params, normalized_cov_params,
                scale)
        self.sigma2 = model.sigma2
        nobs = model.nobs
        self.nobs = nobs
        k_exog = model.k_exog
        self.k_exog = k_exog
        k_trend = model.k_trend
        self.k_trend = k_trend
        k_ar = model.k_ar
        self.k_ar = k_ar
        self.n_totobs = len(model.endog)
        k_ma = model.k_ma
        self.k_ma = k_ma
        df_model = k_exog + k_trend + k_ar + k_ma
        self.df_model = df_model
        self.df_resid = self.nobs - df_model
        self._cache = resettable_cache()

    @cache_readonly
    def arroots(self):
        return np.roots(np.r_[1,-self.arparams])**-1

    @cache_readonly
    def maroots(self):
        return np.roots(np.r_[1,self.maparams])**-1

    @cache_readonly
    def arfreq(self):
        """
        Returns the frequency of the AR roots.

        This is the solution, x, to z = |z|*exp(2j*np.pi*x) where z are the
        roots.
        """
        z = self.arroots
        if not z.size:
            return
        return np.arctan2(z.imag, z.real) / (2*pi)

    @cache_readonly
    def mafreq(self):
        """
        Returns the frequency of the MA roots.

        This is the solution, x, to z = |z|*exp(2j*np.pi*x) where z are the
        roots.
        """
        z = self.maroots
        if not z.size:
            return
        return np.arctan2(z.imag, z.real) / (2*pi)

    @cache_readonly
    def arparams(self):
        k = self.k_exog + self.k_trend
        return self.params[k:k+self.k_ar]

    @cache_readonly
    def maparams(self):
        k = self.k_exog + self.k_trend
        k_ar = self.k_ar
        return self.params[k+k_ar:]

    @cache_readonly
    def llf(self):
        return self.model.loglike(self.params)

    @cache_readonly
    def bse(self):
        params = self.params
        hess = self.model.hessian(params)
        if len(params) == 1: # can't take an inverse
            return np.sqrt(-1./hess)
        return np.sqrt(np.diag(-inv(hess)))

    def cov_params(self): # add scale argument?
        params = self.params
        hess = self.model.hessian(params)
        return -inv(hess)

    @cache_readonly
    def aic(self):
        return -2*self.llf + 2*(self.df_model+1)

    @cache_readonly
    def bic(self):
        nobs = self.nobs
        return -2*self.llf + np.log(nobs)*(self.df_model+1)

    @cache_readonly
    def hqic(self):
        nobs = self.nobs
        return -2*self.llf + 2*(self.df_model+1)*np.log(np.log(nobs))

    @cache_readonly
    def fittedvalues(self):
        model = self.model
        endog = model.endog.copy()
        k_ar = self.k_ar
        exog = model.exog # this is a copy
        if exog is not None:
            if model.method == "css" and k_ar > 0:
                exog = exog[k_ar:]
        if model.method == "css" and k_ar > 0:
            endog = endog[k_ar:]
        fv = endog - self.resid
        # add deterministic part back in
        k = self.k_exog + self.k_trend
    #TODO: this needs to be commented out for MLE with constant

    #    if k != 0:
    #        fv += dot(exog, self.params[:k])
        return fv

    @cache_readonly
    def resid(self):
        return self.model.geterrors(self.params)

    @cache_readonly
    def pvalues(self):
    #TODO: same for conditional and unconditional?
        df_resid = self.df_resid
        return t.sf(np.abs(self.tvalues), df_resid) * 2

    def predict(self, start=None, end=None, exog=None, dynamic=False):
        """
        In-sample and out-of-sample prediction.

        Parameters
        ----------
        start : int, str, or datetime
            Zero-indexed observation number at which to start forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type.
        end : int, str, or datetime
            Zero-indexed observation number at which to end forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type.
        exog : array-like, optional
            If the model is an ARMAX and out-of-sample forecasting is
            requestion, exog must be given.
        dynamic : bool, optional
            The `dynamic` keyword affects in-sample prediction. If dynamic
            is False, then the in-sample lagged values are used for
            prediction. If `dynamic` is True, then in-sample forecasts are
            used in place of lagged dependent variables. The first forecasted
            value is `start`.

        """
        return self.model.predict(self.params, start, end, exog, dynamic)

    def forecast(self, steps=1, exog=None, alpha=.05):
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : int
            The number of out of sample forecasts from the end of the
            sample.
        exog : array
            If the model is an ARMAX, you must provide out of sample
            values for the exogenous variables. This should not include
            the constant.
        alpha : float
            The confidence intervals for the forecasts are (1 - alpha) %

        Returns
        -------
        forecast : array
            Array of out of sample forecasts
        stderr : array
            Array of the standard error of the forecasts.
        conf_int : array
            2d array of the confidence interval for the forecast
        """

        arparams = self.arparams
        maparams = self.maparams
        forecast = _arma_predict_out_of_sample(self.params,
                steps, self.resid, self.k_ar, self.k_ma, self.k_trend,
                self.k_exog, self.model.endog, exog, method=self.model.method)
        # compute the standard errors
        sigma2 = self.sigma2
        ma_rep = arma2ma(np.r_[1,-arparams],
                         np.r_[1, maparams], nobs=steps)


        fcasterr = np.sqrt(sigma2 * np.cumsum(ma_rep**2))

        const = norm.ppf(1 - alpha/2.)
        conf_int = np.c_[forecast - const*fcasterr, forecast + const*fcasterr]

        return forecast, fcasterr, conf_int

    def summary(self, alpha=.05):
        """Summarize the Model

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence intervals.

        Returns
        -------
        smry : Summary instance
            This holds the summary table and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary
        """
        from statsmodels.iolib.summary import Summary
        model = self.model
        title = model.__class__.__name__ + ' Model Results'
        method = model.method
        # get sample TODO: make better sample machinery for estimation
        k_diff = getattr(self, 'k_diff', 0)
        if 'mle' in method:
            start = k_diff
        else:
            start = k_diff + self.k_ar
        if self._data.dates is not None:
            dates = self._data.dates
            sample = [dates[start].strftime('%m-%d-%Y')]
            sample += ['- ' + dates[-1].strftime('%m-%d-%Y')]
        else:
            sample = str(start) + ' - ' + str(len(self._data._orig_endog))

        k_ar, k_ma = self.k_ar, self.k_ma
        if not k_diff:
            order = str((k_ar, k_ma))
        else:
            order = str((k_ar, k_diff, k_ma))
        top_left = [('Dep. Variable:', None),
                    ('Model:', [model.__class__.__name__ + order]),
                    ('Method:', [method]),
                    ('Date:', None),
                    ('Time:', None),
                    ('Sample:', [sample[0]]),
                    ('', [sample[1]])
                    ]

        top_right = [
                     ('No. Observations:', [str(len(self.model.endog))]),
                     ('Log Likelihood', ["%#5.3f" % self.llf]),
                     ('S.D. of innovations', ["%#5.3f" % self.sigma2**.5]),
                     ('AIC', ["%#5.3f" % self.aic]),
                     ('BIC', ["%#5.3f" % self.bic]),
                     ('HQIC', ["%#5.3f" % self.hqic])]

        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                                   title=title)
        smry.add_table_params(self, alpha=alpha, use_t=False)

        # Make the roots table
        from statsmodels.iolib.table import SimpleTable

        if k_ma and k_ar:
            arstubs = ["AR.%d" % i for i in range(1, k_ar + 1)]
            mastubs = ["MA.%d" % i for i in range(1, k_ma + 1)]
            stubs = arstubs + mastubs
            roots = np.r_[self.arroots, self.maroots]
            freq = np.r_[self.arfreq, self.mafreq]
        elif k_ma:
            mastubs = ["MA.%d" % i for i in range(1, k_ma + 1)]
            stubs = mastubs
            roots = self.maroots
            freq = self.mafreq
        elif k_ar:
            arstubs = ["AR.%d" % i for i in range(1, k_ar + 1)]
            stubs = arstubs
            roots = self.arroots
            freq = self.arfreq
        modulus = np.abs(roots)
        data = np.column_stack((roots.real, roots.imag, modulus, freq))
        roots_table = SimpleTable(data,
                headers=['           Real', '         Imaginary',
                        '         Modulus', '        Frequency'],
                title="Roots",
                stubs=stubs, data_fmts=["%17.4f", "%+17.4fj", "%17.4f",
                    "%17.4f"])

        smry.tables.append(roots_table)
        return smry


class ARMAResultsWrapper(wrap.ResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._wrap_attrs,
                                    _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(
                        tsbase.TimeSeriesResultsWrapper._wrap_methods,
                        _methods)
wrap.populate_wrapper(ARMAResultsWrapper, ARMAResults)

class ARIMAResults(ARMAResults):
    def predict(self, start=None, end=None, exog=None, typ='linear',
                dynamic=False):
        """
        ARIMA model in-sample and out-of-sample prediction

        Parameters
        ----------
        start : int, str, or datetime
            Zero-indexed observation number at which to start forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type.
        end : int, str, or datetime
            Zero-indexed observation number at which to end forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type.
        exog : array-like, optional
            If the model is an ARMAX and out-of-sample forecasting is
            requestion, exog must be given.
        typ : str {'linear', 'levels'}
            - 'linear' : Linear prediction in terms of the differenced
              endogenous variables.
            - 'levels' : Predict the levels of the original endogenous
              variables.

        Returns
        -------
        predict : array
            The predicted values.
        """
        return self.model.predict(self.params, start, end, exog, typ, dynamic)

    def forecast(self, steps=1, exog=None, alpha=.05):
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : int
            The number of out of sample forecasts from the end of the
            sample.
        exog : array
            If the model is an ARIMAX, you must provide out of sample
            values for the exogenous variables. This should not include
            the constant.
        alpha : float
            The confidence intervals for the forecasts are (1 - alpha) %

        Returns
        -------
        forecast : array
            Array of out of sample forecasts
        stderr : array
            Array of the standard error of the forecasts.
        conf_int : array
            2d array of the confidence interval for the forecast

        Notes
        -----
        Prediction is done in the levels of the original endogenous variable.
        If you would like prediction of differences in levels use `predict`.
        """
        forecast = _arma_predict_out_of_sample(self.params, steps, self.resid,
                                        self.k_ar, self.k_ma, self.k_trend,
                                        self.k_exog, self.model.endog,
                                        exog, method=self.model.method)
        forecast = self.model._data.endog[-1] + np.cumsum(forecast)
        # get forecast errors
        arparams = self.arparams
        maparams = self.maparams
        sigma2 = self.sigma2
        ma_rep = arma2ma(np.r_[1, -arparams], np.r_[1, maparams], nobs=steps)
        fcerr = np.sqrt(np.cumsum(np.cumsum(ma_rep)**2)*sigma2)
        const = norm.ppf(1 - alpha/2.)
        conf_int = np.c_[forecast - const*fcerr, forecast + const*fcerr]
        return forecast, fcerr, conf_int

class ARIMAResultsWrapper(ARMAResultsWrapper):
    pass
wrap.populate_wrapper(ARIMAResultsWrapper, ARIMAResults)


if __name__ == "__main__":
    import numpy as np
    import statsmodels.api as sm

    # simulate arma process
    from statsmodels.tsa.arima_process import arma_generate_sample
    y = arma_generate_sample([1., -.75],[1.,.25], nsample=1000)
    arma = ARMA(y)
    res = arma.fit(trend='nc', order=(1,1))

    np.random.seed(12345)
    y_arma22 = arma_generate_sample([1.,-.85,.35],[1,.25,-.9], nsample=1000)
    arma22 = ARMA(y_arma22)
    res22 = arma22.fit(trend = 'nc', order=(2,2))

    # test CSS
    arma22_css = ARMA(y_arma22)
    res22css = arma22_css.fit(trend='nc', order=(2,2), method='css')


    data = sm.datasets.sunspots.load()
    ar = ARMA(data.endog)
    resar = ar.fit(trend='nc', order=(9,0))

    y_arma31 = arma_generate_sample([1,-.75,-.35,.25],[.1], nsample=1000)

    arma31css = ARMA(y_arma31)
    res31css = arma31css.fit(order=(3,1), method="css", trend="nc",
            transparams=True)

    y_arma13 = arma_generate_sample([1., -.75],[1,.25,-.5,.8], nsample=1000)
    arma13css = ARMA(y_arma13)
    res13css = arma13css.fit(order=(1,3), method='css', trend='nc')


# check css for p < q and q < p
    y_arma41 = arma_generate_sample([1., -.75, .35, .25, -.3],[1,-.35],
                    nsample=1000)
    arma41css = ARMA(y_arma41)
    res41css = arma41css.fit(order=(4,1), trend='nc', method='css')

    y_arma14 = arma_generate_sample([1, -.25], [1., -.75, .35, .25, -.3],
                    nsample=1000)
    arma14css = ARMA(y_arma14)
    res14css = arma14css.fit(order=(4,1), trend='nc', method='css')

    # ARIMA Model
    from statsmodels.tools.tools import webuse
    dta = webuse('wpi1')
    wpi = dta['wpi']

    mod = ARIMA(wpi, (1,1,1)).fit()
