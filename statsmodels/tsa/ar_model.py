# -*- coding: utf-8 -*-
import copy
import datetime as dt
from collections.abc import Iterable
from types import SimpleNamespace

import numpy as np
import pandas as pd
from numpy.linalg import inv, slogdet
from scipy.stats import norm, gaussian_kde

import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.compat.pandas import Appender, Substitution
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.docstring import (Docstring, remove_parameters)
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from statsmodels.tools.validation import array_like, string_like, bool_like, \
    int_like
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.base import tsa_model
from statsmodels.tsa.kalmanf.kalmanfilter import KalmanFilter
from statsmodels.tsa.tsatools import (lagmat, add_trend, _ar_transparams,
                                      _ar_invtransparams, freq_to_period)
from statsmodels.tsa.vector_ar import util

__all__ = ['AR', 'AutoReg']

AR_DEPRECATION_WARN = """
statsmodels.tsa.AR has been deprecated in favor of statsmodels.tsa.AutoReg and
statsmodels.tsa.SARIMAX.

AutoReg adds the ability to specify exogenous variables, include time trends,
and add seasonal dummies. The AutoReg API differs from AR since the model is
treated as immutable, and so the entire specification including the lag
length must be specified when creating the model. This change is too
substantial to incorporate into the existing AR api. The function
ar_select_order performs lag length selection for AutoReg models.

AutoReg only estimates parameters using conditional MLE (OLS). Use SARIMAX to
estimate ARX and related models using full MLE via the Kalman Filter.

To silence this warning and continue using AR until it is removed, use:

import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.ar_model.AR', FutureWarning)
"""

REPEATED_FIT_ERROR = """
Model has been fit using maxlag={0}, method={1}, ic={2}, trend={3}. These
cannot be changed in subsequent calls to `fit`. Instead, use a new instance of
AR.
"""


def sumofsq(x, axis=0):
    """Helper function to calculate sum of squares along first axis"""
    return np.sum(x ** 2, axis=axis)


def _ar_predict_out_of_sample(y, params, k_ar, k_trend, steps, start=0):
    mu = params[:k_trend] if k_trend else 0  # only have to worry constant
    arparams = params[k_trend:][::-1]  # reverse for dot

    # dynamic endogenous variable
    endog = np.zeros(k_ar + steps)  # this is one too big but does not matter
    if start:
        endog[:k_ar] = y[start - k_ar:start]
    else:
        endog[:k_ar] = y[-k_ar:]

    forecast = np.zeros(steps)
    for i in range(steps):
        fcast = mu + np.dot(arparams, endog[i:i + k_ar])
        forecast[i] = fcast
        endog[i + k_ar] = fcast

    return forecast


class AutoReg(tsa_model.TimeSeriesModel):
    """
    Autoregressive AR-X(p) model.

    Estimate an AR-X model using Conditional Maximum Likelihood (OLS).

    Parameters
    ----------
    endog : array_like
        A 1-d endogenous response variable. The independent variable.
    lags : {int, list[int]}
        The number of lags to include in the model if an integer or the
        list of lag indices to include.  For example, [1, 4] will only
        include lags 1 and 4 while lags=4 will include lags 1, 2, 3, and 4.
    trend : {'n', 'c', 't', 'ct'}
        The trend to include in the model:

        * 'n' - No trend.
        * 'c' - Constant only.
        * 't' - Time trend only.
        * 'ct' - Constant and time trend.

    seasonal : bool
        Flag indicating whether to include seasonal dummies in the model. If
        seasonal is True and trend includes 'c', then the first period
        is excluded from the seasonal terms.
    exog : array_like, optional
        Exogenous variables to include in the model. Must have the same number
        of observations as endog and should be aligned so that endog[i] is
        regressed on exog[i].
    hold_back : {None, int}
        Initial observations to exclude from the estimation sample.  If None,
        then hold_back is equal to the maximum lag in the model.  Set to a
        non-zero value to produce comparable models with different lag
        length.  For example, to compare the fit of a model with lags=3 and
        lags=1, set hold_back=3 which ensures that both models are estimated
        using observations 3,...,nobs. hold_back must be >= the maximum lag in
        the model.
    period : {None, int}
        The period of the data. Only used if seasonal is True. This parameter
        can be omitted if using a pandas object for endog that contains a
        recognized frequency.
    missing : str
        Available options are 'none', 'drop', and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default is 'none'.

    See Also
    --------
    statsmodels.tsa.statespace.sarimax.SARIMAX
        Estimation of SARIMAX models using exact likelihood and the
        Kalman Filter.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from statsmodels.tsa.ar_model import AutoReg
    >>> data = sm.datasets.sunspots.load_pandas().data['SUNACTIVITY']
    >>> out = 'AIC: {0:0.3f}, HQIC: {1:0.3f}, BIC: {2:0.3f}'

    Start by fitting an unrestricted Seasonal AR model

    >>> res = AutoReg(data, lags = [1, 11, 12]).fit()
    >>> print(out.format(res.aic, res.hqic, res.bic))
    AIC: 5.945, HQIC: 5.970, BIC: 6.007

    An alternative used seasonal dummies

    >>> res = AutoReg(data, lags=1, seasonal=True, period=11).fit()
    >>> print(out.format(res.aic, res.hqic, res.bic))
    AIC: 6.017, HQIC: 6.080, BIC: 6.175

    Finally, both the seasonal AR structure and dummies can be included

    >>> res = AutoReg(data, lags=[1, 11, 12], seasonal=True, period=11).fit()
    >>> print(out.format(res.aic, res.hqic, res.bic))
    AIC: 5.884, HQIC: 5.959, BIC: 6.071
    """

    def __init__(self, endog, lags, trend='c', seasonal=False, exog=None,
                 hold_back=None, period=None, missing='none'):
        super(AutoReg, self).__init__(endog, exog, None, None,
                                      missing=missing)
        self._trend = string_like(trend, 'trend',
                                  options=('n', 'c', 't', 'ct'))
        self._seasonal = bool_like(seasonal, 'seasonal')
        self._period = int_like(period, 'period', optional=True)
        if self._period is None and self._seasonal:
            if self.data.freq:
                self._period = freq_to_period(self._index_freq)
            else:
                err = 'freq cannot be inferred from endog and model includes' \
                      ' seasonal terms.  The number of periods must be ' \
                      'explicitly set when the endog\'s index does not ' \
                      'contain a frequency.'
                raise ValueError(err)
        self._lags = lags
        self._exog_names = []
        self._k_ar = 0
        self._hold_back = int_like(hold_back, 'hold_back', optional=True)
        self._check_lags()
        self._setup_regressors()
        self.nobs = self._y.shape[0]
        self.data.xnames = self.exog_names

    @property
    def ar_lags(self):
        """The autoregressive lags included in the model"""
        return self._lags

    @property
    def hold_back(self):
        """The number of initial obs. excluded from the estimation sample."""
        return self._hold_back

    @property
    def seasonal(self):
        """Flag indicating that the model contains a seasonal component."""
        return self._seasonal

    @property
    def df_model(self):
        """The model degrees of freedom."""
        return self._x.shape[1]

    @property
    def exog_names(self):
        """Names of exogenous variables included in model"""
        return self._exog_names

    def initialize(self):
        """Initialize the model (no-op)."""
        pass

    def _check_lags(self):
        lags = self._lags
        if isinstance(lags, Iterable):
            lags = np.array(sorted([int_like(lag, 'lags') for lag in lags]))
            self._lags = lags
            if np.any(lags < 1) or np.unique(lags).shape[0] != lags.shape[0]:
                raise ValueError('All values in lags must be positive and '
                                 'distinct.')
            self._maxlag = np.max(lags)
        else:
            self._maxlag = int_like(lags, 'lags')
            if self._maxlag < 0:
                raise ValueError('lags must be a positive scalar.')
            self._lags = np.arange(1, self._maxlag + 1)
        if self._hold_back is None:
            self._hold_back = self._maxlag
        if self._hold_back < self._maxlag:
            raise ValueError('hold_back must be >= lags if lags is an int or'
                             'max(lags) if lags is array_like.')

    def _setup_regressors(self):
        maxlag = self._maxlag
        hold_back = self._hold_back
        exog_names = []
        endog_names = self.endog_names
        x, y = lagmat(self.endog, maxlag, original='sep')
        exog_names.extend([endog_names + '.L{0}'.format(lag)
                           for lag in self._lags])
        if len(self._lags) < maxlag:
            x = x[:, self._lags - 1]
        self._k_ar = x.shape[1]
        if self._seasonal:
            nobs, period = self.endog.shape[0], self._period
            season_names = ['seasonal.{0}'.format(i) for i in range(period)]
            dummies = np.zeros((nobs, period))
            for i in range(period):
                dummies[i::period, i] = 1
            if 'c' in self._trend:
                dummies = dummies[:, 1:]
                season_names = season_names[1:]
            x = np.c_[dummies, x]
            exog_names = season_names + exog_names
        x = add_trend(x, trend=self._trend, prepend=True)
        if 't' in self._trend:
            exog_names.insert(0, 'trend')
        if 'c' in self._trend:
            exog_names.insert(0, 'intercept')
        if self.exog is not None:
            x = np.c_[x, self.exog]
            exog_names.extend(self.data.param_names)
        y = y[hold_back:]
        x = x[hold_back:]
        if y.shape[0] < x.shape[1]:
            reg = x.shape[1]
            trend = 0 if self._trend == 'n' else len(self._trend)
            seas = 0 if not self._seasonal else period - ('c' in self._trend)
            lags = self._lags.shape[0]
            nobs = y.shape[0]
            raise ValueError('The model specification cannot be estimated. '
                             'The model contains {0} regressors ({1} trend, '
                             '{2} seasonal, {3} lags) but after adjustment '
                             'for hold_back and creation of the lags, there '
                             'are only {4} data points available to estimate '
                             'parameters.'.format(reg, trend, seas, lags,
                                                  nobs))
        self._y, self._x = y, x
        self._exog_names = exog_names

    def fit(self, cov_type='nonrobust', cov_kwds=None, use_t=False):
        """
        Estimate the model parameters.

        Parameters
        ----------
        cov_type : str
            The covariance estimator to use. The most common choices are listed
            below.  Supports all covariance estimators that are available
            in ``OLS.fit``.

            * 'nonrobust' - The class OLS covariance estimator that assumes
              homoskedasticity.
            * 'HC0', 'HC1', 'HC2', 'HC3' - Variants of White's
              (or Eiker-Huber-White) covariance estimator. `HC0` is the
              standard implementation.  The other make corrections to improve
              the finite sample performance of the heteroskedasticity robust
              covariance estimator.
            * 'HAC' - Heteroskedasticity-autocorrelation robust covariance
              estimation. Supports cov_kwds.

              - `maxlag` integer (required) : number of lags to use.
              - `kernel` callable or str (optional) : kernel
                  currently available kernels are ['bartlett', 'uniform'],
                  default is Bartlett.
              - `use_correction` bool (optional) : If true, use small sample
                  correction.
        cov_kwds : dict, optional
            A dictionary of keyword arguments to pass to the covariance
            estimator. `nonrobust` and `HC#` do not support cov_kwds.
        use_t : bool, optional
            A flag indicating that inference should use the Student's t
            distribution that accounts for model degree of freedom.  If False,
            uses the normal distribution. If None, defers the choice to
            the cov_type. It also removes degree of freedom corrections from
            the covariance estimator when cov_type is 'nonrobust'.

        Returns
        -------
        AutoRegResults
            Estimation results.

        See Also
        --------
        statsmodels.regression.linear_model.OLS
            Ordinary Least Squares estimation.
        statsmodels.regression.linear_model.RegressionResults
            See ``get_robustcov_results`` for a detailed list of available
            covariance estimators and options.

        Notes
        -----
        Use ``OLS`` to estimate model parameters and to estimate parameter
        covariance.
        """
        # TODO: Determine correction for degree-of-freedom
        # Special case parameterless model
        if self._x.shape[1] == 0:
            return AutoRegResultsWrapper(AutoRegResults(self,
                                                        np.empty(0),
                                                        np.empty((0, 0))))

        ols_mod = OLS(self._y, self._x)
        ols_res = ols_mod.fit(cov_type=cov_type, cov_kwds=cov_kwds,
                              use_t=use_t)
        cov_params = ols_res.cov_params()
        use_t = ols_res.use_t
        if cov_type == "nonrobust" and not use_t:
            nobs = self._y.shape[0]
            k = self._x.shape[1]
            scale = nobs / (nobs - k)
            cov_params /= scale
        res = AutoRegResults(self, ols_res.params, cov_params,
                             ols_res.normalized_cov_params)

        return AutoRegResultsWrapper(res)

    def _resid(self, params):
        params = array_like(params, 'params', ndim=2)
        resid = self._y - self._x @ params
        return resid.squeeze()

    def loglike(self, params):
        """
        Log-likelihood of model.

        Parameters
        ----------
        params : ndarray
            The model parameters used to compute the log-likelihood.

        Returns
        -------
        float
            The log-likelihood value.
        """
        nobs = self.nobs
        resid = self._resid(params)
        ssr = resid @ resid
        llf = -(nobs / 2) * (np.log(2 * np.pi) + np.log(ssr / nobs) + 1)
        return llf

    def score(self, params):
        """
        Score vector of model.

        The gradient of logL with respect to each parameter.

        Parameters
        ----------
        params : ndarray
            The parameters to use when evaluating the Hessian.

        Returns
        -------
        ndarray
            The score vector evaluated at the parameters.
        """
        resid = self._resid(params)
        return self._x.T @ resid

    def information(self, params):
        """
        Fisher information matrix of model.

        Returns -1 * Hessian of the log-likelihood evaluated at params.

        Parameters
        ----------
        params : ndarray
            The model parameters.

        Returns
        -------
        ndarray
            The information matrix.
        """
        resid = self._resid(params)
        sigma2 = resid @ resid / self.nobs
        return sigma2 * (self._x.T @ self._x)

    def hessian(self, params):
        """
        The Hessian matrix of the model.

        Parameters
        ----------
        params : ndarray
            The parameters to use when evaluating the Hessian.

        Returns
        -------
        ndarray
            The hessian evaluated at the parameters.
        """
        return -self.information(params)

    def _setup_oos_forecast(self, add_forecasts, exog_oos):
        full_nobs = self.data.endog.shape[0]
        x = np.zeros((add_forecasts, self._x.shape[1]))
        loc = 0
        if 'c' in self._trend:
            x[:, 0] = 1
            loc += 1
        if 't' in self._trend:
            x[:, loc] = np.arange(full_nobs + 1, full_nobs + add_forecasts + 1)
            loc += 1
        if self.seasonal:
            seasonal = np.zeros((add_forecasts, self._period))
            period = self._period
            col = full_nobs % period
            for i in range(period):
                seasonal[i::period, (col + i) % period] = 1
            if 'c' in self._trend:
                x[:, loc:loc + period - 1] = seasonal[:, 1:]
                loc += seasonal.shape[1] - 1
            else:
                x[:, loc:loc + period] = seasonal
                loc += seasonal.shape[1]
        # skip the AR columns
        loc += len(self._lags)
        if self.exog is not None:
            x[:, loc:] = exog_oos[:add_forecasts]
        return x

    def _wrap_prediction(self, prediction, start, end):
        if not isinstance(self.data.orig_endog, (pd.Series, pd.DataFrame)):
            return prediction
        index = self.data.orig_endog.index
        if end > self.endog.shape[0]:
            freq = getattr(index, 'freq', None)
            if freq:
                index = pd.date_range(index[0], freq=freq, periods=end)
            else:
                index = pd.RangeIndex(end)
        index = index[start:end]
        return pd.Series(prediction, index=index)

    def _dynamic_predict(self, params, start, end, dynamic, num_oos, exog,
                         exog_oos):
        """

        :param params:
        :param start:
        :param end:
        :param dynamic:
        :param num_oos:
        :param exog:
        :param exog_oos:
        :return:
        """
        reg = []
        hold_back = self._hold_back
        if (start - hold_back) <= self.nobs:
            is_loc = slice(start - hold_back, end + 1 - hold_back)
            x = self._x[is_loc]
            if exog is not None:
                x = x.copy()
                # Replace final columns
                x[:, -exog.shape[1]:] = exog[start:end + 1]
            reg.append(x)
        if num_oos > 0:
            reg.append(self._setup_oos_forecast(num_oos, exog_oos))
        reg = np.vstack(reg)
        det_col_idx = self._x.shape[1] - len(self._lags)
        det_col_idx -= 0 if self.exog is None else self.exog.shape[1]
        # + 1 is due t0 inclusiveness of predict functions
        adj_dynamic = dynamic - start + 1
        forecasts = np.empty(reg.shape[0])
        forecasts[:adj_dynamic] = reg[:adj_dynamic] @ params
        for h in range(adj_dynamic, reg.shape[0]):
            # Fill in regressor matrix
            for j, lag in enumerate(self._lags):
                fcast_loc = h - lag
                if fcast_loc >= 0:
                    val = forecasts[fcast_loc]
                else:
                    # If before the start of the forecasts, use actual values
                    val = self.endog[start + fcast_loc]
                reg[h, det_col_idx + j] = val
            forecasts[h] = reg[h:h + 1] @ params
        return self._wrap_prediction(forecasts, start, end + 1 + num_oos)

    def _static_oos_predict(self, params, num_oos, exog_oos):
        new_x = self._setup_oos_forecast(num_oos, exog_oos)
        if self._maxlag == 0:
            return new_x @ params
        forecasts = np.empty(num_oos)
        nexog = 0 if self.exog is None else self.exog.shape[1]
        ar_offset = self._x.shape[1] - nexog - self._lags.shape[0]
        for i in range(num_oos):
            for j, lag in enumerate(self._lags):
                loc = i - lag
                val = self._y[loc] if loc < 0 else forecasts[loc]
                new_x[i, ar_offset + j] = val
            forecasts[i] = new_x[i:i + 1] @ params
        return forecasts

    def _static_predict(self, params, start, end, num_oos, exog, exog_oos):
        """
        Path for static predictions

        Parameters
        ----------
        start : int
            Index of first observation
        end : int
            Index of last in-sample observation. Inclusive, so start:end+1
            in slice notation.
        num_oos : int
            Number of out-of-sample observations, so that the returned size is
            num_oos + (end - start + 1).
        exog : ndarray
            Array containing replacement exog values
        exog_oos :  ndarray
            Containing forecast exog values
        """
        hold_back = self._hold_back
        nobs = self.endog.shape[0]

        x = np.empty((0, self._x.shape[1]))
        if start <= nobs:
            is_loc = slice(start - hold_back, end + 1 - hold_back)
            x = self._x[is_loc]
            if exog is not None:
                x = x.copy()
                # Replace final columns
                x[:, -exog.shape[1]:] = exog[start:end + 1]
        in_sample = x @ params
        if num_oos == 0:  # No out of sample
            return self._wrap_prediction(in_sample, start, end + 1)

        out_of_sample = self._static_oos_predict(params, num_oos, exog_oos)

        prediction = np.hstack((in_sample, out_of_sample))
        return self._wrap_prediction(prediction, start, end + 1 + num_oos)

    def predict(self, params, start=None, end=None, dynamic=False, exog=None,
                exog_oos=None):
        """
        In-sample prediction and out-of-sample forecasting.

        Parameters
        ----------
        params : array_like
            The fitted model parameters.
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out-of-sample prediction. Default is the last observation in
            the sample. Unlike standard python slices, end is inclusive so
            that all the predictions [start, start+1, ..., end-1, end] are
            returned.
        dynamic : {bool, int, str, datetime, Timestamp}, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Prior to this observation, true endogenous values
            will be used for prediction; starting with this observation and
            continuing through the end of prediction, forecasted endogenous
            values will be used instead. Datetime-like objects are not
            interpreted as offsets. They are instead used to find the index
            location of `dynamic` which is then used to to compute the offset.
        exog : array_like
            A replacement exogenous array.  Must have the same shape as the
            exogenous data array used when the model was created.
        exog_oos : array_like
            An array containing out-of-sample values of the exogenous variable.
            Must has the same number of columns as the exog used when the
            model was created, and at least as many rows as the number of
            out-of-sample forecasts.

        Returns
        -------
        array_like
            Array of out of in-sample predictions and / or out-of-sample
            forecasts. An (npredict x k_endog) array.
        """
        params = array_like(params, 'params')
        exog = array_like(exog, 'exog', ndim=2, optional=True)
        exog_oos = array_like(exog_oos, 'exog_oos', ndim=2, optional=True)

        start = 0 if start is None else start
        end = self._index[-1] if end is None else end
        start, end, num_oos, _ = self._get_prediction_index(start, end)
        start = max(start, self._hold_back)
        if self.exog is None and (exog is not None or exog_oos is not None):
            raise ValueError('exog and exog_oos cannot be used when the model '
                             'does not contains exogenous regressors.')
        elif self.exog is not None:
            if exog is not None and exog.shape != self.exog.shape:
                msg = ('The shape of exog {0} must match the shape of the '
                       'exog variable used to create the model {1}.')
                raise ValueError(msg.format(exog.shape, self.exog.shape))
            if exog_oos is not None and \
                    exog_oos.shape[1] != self.exog.shape[1]:
                msg = ('The number of columns in exog_oos ({0}) must match '
                       'the number of columns  in the exog variable used to '
                       'create the model ({1}).')
                raise ValueError(msg.format(exog_oos.shape[1],
                                            self.exog.shape[1]))
            if num_oos > 0 and exog_oos is None:
                raise ValueError('exog_oos must be provided when producing '
                                 'out-of-sample forecasts.')
            elif exog_oos is not None and num_oos > exog_oos.shape[0]:
                msg = ('start and end indicate that {0} out-of-sample '
                       'predictions must be computed. exog_oos has {1} rows '
                       'but must have at least {0}.')
                raise ValueError(msg.format(num_oos, exog_oos.shape[0]))

        if (isinstance(dynamic, bool) and not dynamic) or self._maxlag == 0:
            # If model has no lags, static and dynamic are identical
            return self._static_predict(params, start, end, num_oos,
                                        exog, exog_oos)

        if isinstance(dynamic, (str, bytes, pd.Timestamp, dt.datetime)):
            dynamic, _, _ = self._get_index_loc(dynamic)
            offset = dynamic - start
        elif dynamic is True:
            # if True, all forecasts are dynamic, except start
            offset = 0
        else:
            offset = int(dynamic)
        dynamic = start + offset
        if dynamic < 0:
            raise ValueError('Dynamic prediction cannot begin prior to the'
                             ' first observation in the sample.')

        return self._dynamic_predict(params, start, end, dynamic, num_oos,
                                     exog, exog_oos)


class AR(tsa_model.TimeSeriesModel):
    __doc__ = tsa_model._tsa_doc % {"model": "Autoregressive AR(p) model.\n\n"
                                             "    .. deprecated:: 0.11",
                                    "params": """endog : array_like
        A 1-d endogenous response variable. The independent variable.""",
                                    "extra_params": base._missing_param_doc,
                                    "extra_sections": ""}

    def __init__(self, endog, dates=None, freq=None, missing='none'):
        import warnings
        warnings.warn(AR_DEPRECATION_WARN, FutureWarning)
        super(AR, self).__init__(endog, None, dates, freq, missing=missing)
        endog = self.endog  # original might not have been an ndarray
        if endog.ndim == 1:
            endog = endog[:, None]
            self.endog = endog  # to get shapes right
        elif endog.ndim > 1 and endog.shape[1] != 1:
            raise ValueError("Only the univariate case is implemented")
        self._fit_params = None

    def initialize(self):
        """Initialization of the model (no-op)."""
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
        newparams[k:k + p] = _ar_transparams(params[k:k + p].copy())
        return newparams

    def _invtransparams(self, start_params):
        """
        Inverse of the Jones reparameterization
        """
        p = self.k_ar
        k = self.k_trend
        newparams = start_params.copy()
        newparams[k:k + p] = _ar_invtransparams(start_params[k:k + p].copy())
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
        Q_0 = np.dot(inv(np.identity(p ** 2) - np.kron(T_mat, T_mat)),
                     np.dot(R_mat, R_mat.T).ravel('F'))

        Q_0 = Q_0.reshape(p, p, order='F')  # TODO: order might need to be p+k
        P = Q_0
        Z_mat = KalmanFilter.Z(p)
        for i in range(end):  # iterate p-1 times to fit presample
            v_mat = y[i] - np.dot(Z_mat, alpha)
            F_mat = np.dot(np.dot(Z_mat, P), Z_mat.T)
            Finv = 1. / F_mat  # inv. always scalar
            K = np.dot(np.dot(np.dot(T_mat, P), Z_mat.T), Finv)
            # update state
            alpha = np.dot(T_mat, alpha) + np.dot(K, v_mat)
            L = T_mat - np.dot(K, Z_mat)
            P = np.dot(np.dot(T_mat, P), L.T) + np.dot(R_mat, R_mat.T)
            if i >= start - 1:  # only record if we ask for it
                predictedvalues[i + 1 - start] = np.dot(Z_mat, alpha)

    def _get_prediction_index(self, start, end, dynamic, index=None):
        method = getattr(self, 'method', 'mle')
        k_ar = getattr(self, 'k_ar', 0)
        if start is None:
            if method == 'mle' and not dynamic:
                start = 0
            else:  # cannot do presample fit for cmle or dynamic
                start = k_ar
            start = self._index[start]
        if end is None:
            end = self._index[-1]

        start, end, out_of_sample, prediction_index = (
            super(AR, self)._get_prediction_index(start, end, index))

        # Other validation
        if (method == 'cmle' or dynamic) and start < k_ar:
            raise ValueError("Start must be >= k_ar for conditional MLE "
                             "or dynamic forecast. Got %d" % start)

        return start, end, out_of_sample, prediction_index

    def predict(self, params, start=None, end=None, dynamic=False):
        """
        Construct in-sample and out-of-sample prediction.

        Parameters
        ----------
        params : ndarray
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
        array_like
            An array containing the predicted values.

        Notes
        -----
        The linear Gaussian Kalman filter is used to return pre-sample fitted
        values. The exact initial Kalman Filter is used. See Durbin and Koopman
        in the references for more information.
        """
        if not (hasattr(self, 'k_ar') and hasattr(self, 'k_trend')):
            raise RuntimeError('Model must be fit before calling predict')
        # will return an index of a date
        start, end, out_of_sample, _ = (
            self._get_prediction_index(start, end, dynamic))

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
                mu = params[0] / (1 - np.sum(params[k_trend:]))
            else:
                mu = 0

            # modifies predictedvalues in place
            if start < k_ar:
                self._presample_fit(params, start, k_ar, min(k_ar - 1, end),
                                    endog[:k_ar] - mu, predictedvalues)
                predictedvalues[:k_ar - start] += mu

        if end < k_ar:
            return predictedvalues

        # just do the whole thing and truncate
        fittedvalues = np.dot(self.X, params)

        pv_start = max(k_ar - start, 0)
        fv_start = max(start - k_ar, 0)
        fv_end = min(len(fittedvalues), end - k_ar + 1)
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

        # get inv(Vp) Hamilton 5.3.7
        params0 = np.r_[-1, params[k:]]

        Vpinv = np.zeros((p, p), dtype=params.dtype)
        for i in range(1, p + 1):
            Vpinv[i - 1, i - 1:] = np.correlate(params0, params0[:i])[:-1]
            Vpinv[i - 1, i - 1:] -= np.correlate(params0[-i:], params0)[:-1]

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
        sigma2 = ssr / nobs
        return -nobs / 2 * (np.log(2 * np.pi) + np.log(sigma2) + 1)

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
        sigma2 = 1. / nobs * (diffpVpinv + ssr)
        self.sigma2 = sigma2
        logdet = slogdet(Vpinv)[1]  # TODO: add check for singularity
        loglike = -1 / 2. * (nobs * (np.log(2 * np.pi) + np.log(sigma2))
                             - logdet + diffpVpinv / sigma2 + ssr / sigma2)
        return loglike

    def loglike(self, params):
        r"""
        The loglikelihood of an AR(p) process.

        Parameters
        ----------
        params : ndarray
            The fitted parameters of the AR model.

        Returns
        -------
        float
            The loglikelihood evaluated at `params`.

        Notes
        -----
        Contains constant term.  If the model is fit by OLS then this returns
        the conditional maximum likelihood.

        .. math::

           \frac{\left(n-p\right)}{2}\left(\log\left(2\pi\right)
           +\log\left(\sigma^{2}\right)\right)
           -\frac{1}{\sigma^{2}}\sum_{i}\epsilon_{i}^{2}

        If it is fit by MLE then the (exact) unconditional maximum likelihood
        is returned.

        .. math::

           -\frac{n}{2}log\left(2\pi\right)
           -\frac{n}{2}\log\left(\sigma^{2}\right)
           +\frac{1}{2}\left|V_{p}^{-1}\right|
           -\frac{1}{2\sigma^{2}}\left(y_{p}
           -\mu_{p}\right)^{\prime}V_{p}^{-1}\left(y_{p}-\mu_{p}\right)
           -\frac{1}{2\sigma^{2}}\sum_{t=p+1}^{n}\epsilon_{i}^{2}

        where

        :math:`\mu_{p}` is a (`p` x 1) vector with each element equal to the
        mean of the AR process and :math:`\sigma^{2}V_{p}` is the (`p` x `p`)
        variance-covariance matrix of the first `p` observations.
        """
        # Math is on Hamilton ~pp 124-5
        if self.method == "cmle":
            return self._loglike_css(params)

        else:
            return self._loglike_mle(params)

    def score(self, params):
        """
        Compute the gradient of the log-likelihood at params.

        Parameters
        ----------
        params : array_like
            The parameter values at which to evaluate the score function.

        Returns
        -------
        ndarray
            The gradient computed using numerical methods.
        """
        loglike = self.loglike
        return approx_fprime(params, loglike, epsilon=1e-8)

    def information(self, params):
        """
        Not implemented.

        Parameters
        ----------
        params : ndarray
            The model parameters.
        """
        return

    def hessian(self, params):
        """
        Compute the hessian using a numerical approximation.

        Parameters
        ----------
        params : ndarray
            The model parameters.

        Returns
        -------
        ndarray
            The hessian evaluated at params.
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
        ic : {'aic','bic','hqic','t-stat'}
            Criterion used for selecting the optimal lag length.
            See `AR.fit`.
        trend : {'c','nc'}
            Whether to include a constant or not. 'c' - include constant.
            'nc' - no constant.
        method : {'cmle', 'mle'}, optional
            The method to use in estimation.

            * 'cmle' - Conditional maximum likelihood using OLS
            * 'mle' - Unconditional (exact) maximum likelihood.  See `solver`
              and the Notes.

        Returns
        -------
        int
            Best lag according to the information criteria.
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
            for lag in range(k, maxlag + 1):
                # have to reinstantiate the model to keep comparable models
                endog_tmp = endog[maxlag - lag:]
                fit = AR(endog_tmp).fit(maxlag=lag, method=method,
                                        full_output=0, trend=trend,
                                        maxiter=100, disp=0)
                results[lag] = getattr(fit, ic)
            bestic, bestlag = min((res, k) for k, res in results.items())

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
            round(12*(nobs/100.)**(1/4.)).
        method : {'cmle', 'mle'}, optional
            The method to use in estimation.

            * 'cmle' - Conditional maximum likelihood using OLS
            * 'mle' - Unconditional (exact) maximum likelihood.  See `solver`
              and the Notes.
        ic : {'aic','bic','hic','t-stat'}
            Criterion used for selecting the optimal lag length.

            * 'aic' - Akaike Information Criterion
            * 'bic' - Bayes Information Criterion
            * 't-stat' - Based on last lag
            * 'hqic' - Hannan-Quinn Information Criterion

            If any of the information criteria are selected, the lag length
            which results in the lowest value is selected.  If t-stat, the
            model starts with maxlag and drops a lag until the highest lag
            has a t-stat that is significant at the 95 % level.
        trend : {'c','nc'}
            Whether to include a constant or not.

            * 'c' - include constant.
            * 'nc' - no constant.
        transparams : bool, optional
            Whether or not to transform the parameters to ensure stationarity.
            Uses the transformation suggested in Jones (1980).
        start_params : array_like, optional
            A first guess on the parameters.  Default is cmle estimates.
        solver : str or None, optional
            Solver to be used if method is 'mle'.  The default is 'lbfgs'
            (limited memory Broyden-Fletcher-Goldfarb-Shanno).  Other choices
            are 'bfgs', 'newton' (Newton-Raphson), 'nm' (Nelder-Mead),
            'cg' - (conjugate gradient), 'ncg' (non-conjugate gradient),
            and 'powell'.
        maxiter : int, optional
            The maximum number of function evaluations. Default is 35.
        full_output : bool, optional
            If True, all output from solver will be available in
            the Results object's mle_retvals attribute.  Output is dependent
            on the solver.  See Notes for more information.
        disp : bool, optional
            If True, convergence information is output.
        callback : function, optional
            Called after each iteration as callback(xk) where xk is the current
            parameter vector.
        **kwargs
            See LikelihoodModel.fit for keyword arguments that can be passed
            to fit.

        Returns
        -------
        ARResults
            Results instance.

        See Also
        --------
        statsmodels.base.model.LikelihoodModel.fit
            Base fit class with further details about options.

        Notes
        -----
        The parameters after `trend` are only used when method is 'mle'.

        References
        ----------
        .. [*] Jones, R.H. 1980 "Maximum likelihood fitting of ARMA models to
           time series with missing observations."  `Technometrics`.  22.3.
           389-95.
        """
        start_params = array_like(start_params, 'start_params', ndim=1,
                                  optional=True)
        method = method.lower()
        if method not in ['cmle', 'mle']:
            raise ValueError("Method %s not recognized" % method)
        self.method = method
        self.trend = trend
        self.transparams = transparams
        nobs = len(self.endog)  # overwritten if method is 'cmle'
        endog = self.endog
        # The parameters are no longer allowed to change in an instance
        fit_params = (maxlag, method, ic, trend)
        if self._fit_params is not None and self._fit_params != fit_params:
            raise RuntimeError(REPEATED_FIT_ERROR.format(*self._fit_params))
        if maxlag is None:
            maxlag = int(round(12 * (nobs / 100.) ** (1 / 4.)))
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

        if method == "cmle":  # do OLS
            arfit = OLS(Y, X).fit()
            params = arfit.params
            self.nobs = nobs - k_ar
            self.sigma2 = arfit.ssr / arfit.nobs  # needed for predict fcasterr

        else:  # method == "mle"
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

        pinv_exog = np.linalg.pinv(X)
        normalized_cov_params = np.dot(pinv_exog, pinv_exog.T)
        arfit = ARResults(copy.copy(self), params, normalized_cov_params)
        if method == 'mle' and full_output:
            arfit.mle_retvals = mlefit.mle_retvals
            arfit.mle_settings = mlefit.mle_settings
        # Set fit params since completed the fit
        if self._fit_params is None:
            self._fit_params = fit_params
        return ARResultsWrapper(arfit)


class ARResults(tsa_model.TimeSeriesModelResults):
    """
    Class to hold results from fitting an AR model.

    Parameters
    ----------
    model : AR Model instance
        Reference to the model that is fit.
    params : ndarray
        The fitted parameters from the AR Model.
    normalized_cov_params : ndarray
        The array inv(dot(x.T,x)) where x contains the regressors in the
        model.
    scale : float, optional
        An estimate of the scale of the model.

    Attributes
    ----------
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
    params : ndarray
        The fitted parameters of the model.
    scale : float
        Same as sigma2
    sigma2 : float
        The variance of the innovations (residuals).
    trendorder : int
        The polynomial order of the trend. 'nc' = None, 'c' or 't' = 0,
        'ct' = 1, etc.
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
        # TODO: cmle vs mle?
        self.df_model = k_ar + k_trend
        self.df_resid = self.model.df_resid = n_totobs - self.df_model

    @cache_writable()
    def sigma2(self):
        model = self.model
        if model.method == "cmle":  # do DOF correction
            return 1. / self.nobs * sumofsq(self.resid)
        else:
            return self.model.sigma2

    @cache_writable()  # for compatability with RegressionResults
    def scale(self):
        return self.sigma2

    @cache_readonly
    def bse(self):  # allow user to specify?
        """
        The standard errors of the estimated parameters.

        If `method` is 'cmle', then the standard errors that are returned are
        the OLS standard errors of the coefficients. If the `method` is 'mle'
        then they are computed using the numerical Hessian.
        """
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
        """The p values associated with the standard errors."""
        return norm.sf(np.abs(self.tvalues)) * 2

    @cache_readonly
    def aic(self):
        """
        Akaike Information Criterion using Lutkephol's definition.

        :math:`log(sigma) + 2*(1 + k_ar + k_trend)/nobs`
        """
        # TODO: this is based on loglike with dropped constant terms ?
        # Lutkepohl
        # return np.log(self.sigma2) + 1./self.model.nobs * self.k_ar
        # Include constant as estimated free parameter and double the loss
        return np.log(self.sigma2) + 2 * (1 + self.df_model) / self.nobs
        # Stata defintion
        # nobs = self.nobs
        # return -2 * self.llf/nobs + 2 * (self.k_ar+self.k_trend)/nobs

    @cache_readonly
    def hqic(self):
        """Hannan-Quinn Information Criterion."""
        nobs = self.nobs
        # Lutkepohl
        # return np.log(self.sigma2)+ 2 * np.log(np.log(nobs))/nobs * self.k_ar
        # R uses all estimated parameters rather than just lags
        return (np.log(self.sigma2) + 2 * np.log(np.log(nobs))
                / nobs * (1 + self.df_model))
        # Stata
        # nobs = self.nobs
        # return -2 * self.llf/nobs + 2 * np.log(np.log(nobs))/nobs * \
        #        (self.k_ar + self.k_trend)

    @cache_readonly
    def fpe(self):
        """
        Final prediction error using Lütkepohl's definition.

        ((n_totobs+k_trend)/(n_totobs-k_ar-k_trend))*sigma
        """
        nobs = self.nobs
        df_model = self.df_model
        # Lutkepohl
        return ((nobs + df_model) / (nobs - df_model)) * self.sigma2

    @cache_readonly
    def bic(self):
        """
         Bayes Information Criterion

        :math:`\\log(\\sigma) + (1 + k_ar + k_trend)*\\log(nobs)/nobs`
        """
        nobs = self.nobs
        # Lutkepohl
        # return np.log(self.sigma2) + np.log(nobs)/nobs * self.k_ar
        # Include constant as est. free parameter
        return np.log(self.sigma2) + (1 + self.df_model) * np.log(nobs) / nobs
        # Stata
        # return -2 * self.llf/nobs + np.log(nobs)/nobs * (self.k_ar + \
        #       self.k_trend)

    @cache_readonly
    def resid(self):
        """
        The residuals of the model.

        If the model is fit by 'mle' then the pre-sample residuals are
        calculated using fittedvalues from the Kalman Filter.
        """
        # NOTE: uses fittedvalues because it calculate presample values for mle
        model = self.model
        endog = model.endog.squeeze()
        if model.method == "cmle":  # elimate pre-sample
            return endog[self.k_ar:] - self.fittedvalues
        else:
            return model.endog.squeeze() - self.fittedvalues

    @cache_readonly
    def roots(self):
        """
        The roots of the AR process.

        The roots are the solution to
        (1 - arparams[0]*z - arparams[1]*z**2 -...- arparams[p-1]*z**k_ar) = 0.
        Stability requires that the roots in modulus lie outside the unit
        circle.
        """
        k = self.k_trend
        return np.roots(np.r_[1, -self.params[k:]]) ** -1

    @cache_readonly
    def arfreq(self):
        r"""
        Returns the frequency of the AR roots.

        This is the solution, x, to z = abs(z)*exp(2j*np.pi*x) where z are the
        roots.
        """
        z = self.roots
        return np.arctan2(z.imag, z.real) / (2 * np.pi)

    @cache_readonly
    def fittedvalues(self):
        """
        The in-sample predicted values of the fitted AR model.

        The `k_ar` initial values are computed via the Kalman Filter if the
        model is fit by `mle`.
        """
        return self.model.predict(self.params)

    @Appender(remove_parameters(AR.predict.__doc__, 'params'))
    def predict(self, start=None, end=None, dynamic=False):
        params = self.params
        predictedvalues = self.model.predict(params, start, end, dynamic)
        return predictedvalues
        # TODO: consider returning forecast errors and confidence intervals?

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
        model = self.model
        title = model.__class__.__name__ + ' Model Results'
        method = model.method
        # get sample
        start = 0 if 'mle' in method else self.k_ar
        if self.data.dates is not None:
            dates = self.data.dates
            sample = [dates[start].strftime('%m-%d-%Y')]
            sample += ['- ' + dates[-1].strftime('%m-%d-%Y')]
        else:
            sample = str(start) + ' - ' + str(len(self.data.orig_endog))

        k_ar = self.k_ar
        order = '({0})'.format(k_ar)
        dep_name = str(self.model.endog_names)
        top_left = [('Dep. Variable:', dep_name),
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
            ('S.D. of innovations', ["%#5.3f" % self.sigma2 ** .5]),
            ('AIC', ["%#5.3f" % self.aic]),
            ('BIC', ["%#5.3f" % self.bic]),
            ('HQIC', ["%#5.3f" % self.hqic])]

        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                             title=title)
        smry.add_table_params(self, alpha=alpha, use_t=False)

        # Make the roots table
        from statsmodels.iolib.table import SimpleTable

        if k_ar:
            arstubs = ["AR.%d" % i for i in range(1, k_ar + 1)]
            stubs = arstubs
            roots = self.roots
            freq = self.arfreq
        else:  # AR(0) model
            stubs = []
        if len(stubs):  # not AR(0)
            modulus = np.abs(roots)
            data = np.column_stack((roots.real, roots.imag, modulus, freq))
            roots_table = SimpleTable([('%17.4f' % row[0],
                                        '%+17.4fj' % row[1],
                                        '%17.4f' % row[2],
                                        '%17.4f' % row[3]) for row in data],
                                      headers=['            Real',
                                               '         Imaginary',
                                               '         Modulus',
                                               '        Frequency'],
                                      title="Roots",
                                      stubs=stubs)

            smry.tables.append(roots_table)
        return smry


class ARResultsWrapper(wrap.ResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(
        tsa_model.TimeSeriesResultsWrapper._wrap_attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(
        tsa_model.TimeSeriesResultsWrapper._wrap_methods, _methods)


wrap.populate_wrapper(ARResultsWrapper, ARResults)

doc = Docstring(AutoReg.predict.__doc__)
_predict_params = doc.extract_parameters(['start', 'end', 'dynamic',
                                          'exog', 'exog_oos'], 8)


class AutoRegResults(tsa_model.TimeSeriesModelResults):
    """
    Class to hold results from fitting an AutoReg model.

    Parameters
    ----------
    model : AutoReg
        Reference to the model that is fit.
    params : ndarray
        The fitted parameters from the AR Model.
    cov_params : ndarray
        The estimated covariance matrix of the model parameters.
    normalized_cov_params : ndarray
        The array inv(dot(x.T,x)) where x contains the regressors in the
        model.
    scale : float, optional
        An estimate of the scale of the model.
    """

    _cache = {}  # for scale setter

    def __init__(self, model, params, cov_params, normalized_cov_params=None,
                 scale=1.):
        super(AutoRegResults, self).__init__(model, params,
                                             normalized_cov_params, scale)
        self._cache = {}
        self._params = params
        self._nobs = model.nobs
        self._n_totobs = model.endog.shape[0]
        self._df_model = model.df_model
        self._ar_lags = model.ar_lags
        self._max_lag = 0
        if self._ar_lags.shape[0] > 0:
            self._max_lag = self._ar_lags.max()
        self._hold_back = self.model.hold_back
        self.cov_params_default = cov_params

    def initialize(self, model, params, **kwargs):
        """
        Initialize (possibly re-initialize) a Results instance.

        Parameters
        ----------
        model : Model
            The model instance.
        params : ndarray
            The model parameters.
        **kwargs
            Any additional keyword arguments required to initialize the model.
        """
        self._params = params
        self.model = model

    @property
    def ar_lags(self):
        """The autoregressive lags included in the model"""
        return self._ar_lags

    @property
    def params(self):
        """The estimated parameters."""
        return self._params

    @property
    def df_model(self):
        """The degrees of freedom consumed by the model."""
        return self._df_model

    @property
    def df_resid(self):
        """The remaining degrees of freedom in the residuals."""
        return self.nobs - self._df_model

    @property
    def nobs(self):
        """
        The number of observations after adjusting for losses due to lags.
        """
        return self._nobs

    @cache_writable()
    def sigma2(self):
        return 1. / self.nobs * sumofsq(self.resid)

    @cache_writable()  # for compatability with RegressionResults
    def scale(self):
        return self.sigma2

    @cache_readonly
    def bse(self):  # allow user to specify?
        """
        The standard errors of the estimated parameters.

        If `method` is 'cmle', then the standard errors that are returned are
        the OLS standard errors of the coefficients. If the `method` is 'mle'
        then they are computed using the numerical Hessian.
        """
        return np.sqrt(np.diag(self.cov_params()))

    @cache_readonly
    def aic(self):
        """
        Akaike Information Criterion using Lutkephol's definition.

        :math:`log(sigma) + 2*(1 + df_model) / nobs`
        """
        # This is based on loglike with dropped constant terms ?
        # Lutkepohl
        # return np.log(self.sigma2) + 1./self.model.nobs * self.k_ar
        # Include constant as estimated free parameter and double the loss
        # Stata defintion
        # nobs = self.nobs
        # return -2 * self.llf/nobs + 2 * (self.k_ar+self.k_trend)/nobs
        return np.log(self.sigma2) + 2 * (1 + self.df_model) / self.nobs

    @cache_readonly
    def hqic(self):
        """Hannan-Quinn Information Criterion."""
        # Lutkepohl
        # return np.log(self.sigma2)+ 2 * np.log(np.log(nobs))/nobs * self.k_ar
        # R uses all estimated parameters rather than just lags
        # Stata
        # nobs = self.nobs
        # return -2 * self.llf/nobs + 2 * np.log(np.log(nobs))/nobs * \
        #        (self.k_ar + self.k_trend)
        nobs = self.nobs
        loglog_n = np.log(np.log(nobs))
        log_sigma2 = np.log(self.sigma2)
        return (log_sigma2 + 2 * loglog_n / nobs * (1 + self.df_model))

    @cache_readonly
    def fpe(self):
        """
        Final prediction error using Lütkepohl's definition.

        ((n_totobs+k_trend)/(n_totobs-k_ar-k_trend))*sigma
        """
        nobs = self.nobs
        df_model = self.df_model
        # Lutkepohl
        return ((nobs + df_model) / (nobs - df_model)) * self.sigma2

    @cache_readonly
    def bic(self):
        r"""
        Bayes Information Criterion

        :math:`\ln(\sigma) + df_{model} \ln(nobs)/nobs`
        """
        # Lutkepohl
        # np.log(self.sigma2) + np.log(nobs)/nobs * self.k_ar
        # Include constant as est. free parameter
        # Stata
        # -2 * self.llf/nobs + np.log(nobs)/nobs * (self.k_ar + self.k_trend)
        nobs = self.nobs
        return np.log(self.sigma2) + (1 + self.df_model) * np.log(nobs) / nobs

    @cache_readonly
    def resid(self):
        """
        The residuals of the model.
        """
        model = self.model
        endog = model.endog.squeeze()
        return endog[self._hold_back:] - self.fittedvalues

    def _lag_repr(self):
        """Returns poly repr of an AR, (1  -phi1 L -phi2 L^2-...)"""
        k_ar = len(self.ar_lags)
        ar_params = np.zeros(self._max_lag + 1)
        ar_params[0] = 1
        df_model = self._df_model
        exog = self.model.exog
        k_exog = exog.shape[1] if exog is not None else 0
        params = self._params[df_model - k_ar - k_exog:df_model - k_exog]
        for i, lag in enumerate(self._ar_lags):
            ar_params[lag] = -params[i]
        return ar_params

    @cache_readonly
    def roots(self):
        """
        The roots of the AR process.

        The roots are the solution to
        (1 - arparams[0]*z - arparams[1]*z**2 -...- arparams[p-1]*z**k_ar) = 0.
        Stability requires that the roots in modulus lie outside the unit
        circle.
        """
        lag_repr = self._lag_repr()
        if lag_repr.shape[0] == 1:
            return np.empty(0)

        return np.roots(lag_repr) ** -1

    @cache_readonly
    def arfreq(self):
        r"""
        Returns the frequency of the AR roots.

        This is the solution, x, to z = abs(z)*exp(2j*np.pi*x) where z are the
        roots.
        """
        z = self.roots
        return np.arctan2(z.imag, z.real) / (2 * np.pi)

    @cache_readonly
    def fittedvalues(self):
        """
        The in-sample predicted values of the fitted AR model.

        The `k_ar` initial values are computed via the Kalman Filter if the
        model is fit by `mle`.
        """
        return self.model.predict(self.params)

    def test_serial_correlation(self, lags=None, model_df=None):
        """
        Ljung-Box test for residual serial correlation

        Parameters
        ----------
        lags : int
            The maximum number of lags to use in the test. Jointly tests that
            all autocorrelations up to and including lag j are zero for
            j = 1, 2, ..., lags. If None, uses lag=12*(nobs/100)^{1/4}.
        model_df : int
            The model degree of freedom to use when adjusting computing the
            test statistic to account for parameter estimation. If None, uses
            the number of AR lags included in the model.

        Returns
        -------
        output : DataFrame
            DataFrame containing three columns: the test statistic, the
            p-value of the test, and the degree of freedom used in the test.

        Notes
        -----
        Null hypothesis is no serial correlation.

        The the test degree-of-freedom is 0 or negative once accounting for
        model_df, then the test statistic's p-value is missing.

        See Also
        --------
        statsmodels.stats.diagnostic.acorr_ljungbox
            Ljung-Box test for serial correlation.
        """
        # Deferred to prevent circular import
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lags = int_like(lags, 'lags', optional=True)
        model_df = int_like(model_df, 'df_model', optional=True)
        model_df = len(self.ar_lags) if model_df is None else model_df
        nobs_effective = self.resid.shape[0]
        # Default lags for acorr_ljungbox is 40, but may not always have
        # that many observations
        if lags is None:
            lags = int(min(12 * (nobs_effective / 100) ** (1 / 4),
                           nobs_effective - 1))
        test_stats = acorr_ljungbox(self.resid, lags=lags, boxpierce=False,
                                    model_df=model_df, return_df=False)
        cols = ['Ljung-Box', 'LB P-value', 'DF']
        if lags == 1:
            test_stats = [list(test_stats) + [max(0, 1 - model_df)]]
        else:
            df = np.clip(np.arange(1, lags + 1) - model_df, 0, np.inf).astype(
                np.int)
            test_stats = list(test_stats) + [df]
            test_stats = [[test_stats[i][j] for i in range(3)] for j in
                          range(lags)]
        index = pd.RangeIndex(1, lags + 1, name='Lag')
        return pd.DataFrame(test_stats,
                            columns=cols,
                            index=index)

    def test_normality(self):
        """
        Test for normality of standardized residuals.

        Returns
        -------
        Series
            Series containing four values, the test statistic and its p-value,
            the skewness and the kurtosis.

        Notes
        -----
        Null hypothesis is normality.

        See Also
        --------
        statsmodels.stats.stattools.jarque_bera
            The Jarque-Bera test of normality.
        """
        # Deferred to prevent circular import
        from statsmodels.stats.stattools import jarque_bera
        index = ['Jarque-Bera', 'P-value', 'Skewness', 'Kurtosis']
        return pd.Series(jarque_bera(self.resid), index=index)

    def test_heteroskedasticity(self, lags=None):
        """
        ARCH-LM test of residual heteroskedasticity

        Parameters
        ----------
        lags : int
            The maximum number of lags to use in the test. Jointly tests that
            all squared autocorrelations up to and including lag j are zero for
            j = 1, 2, ..., lags. If None, uses lag=12*(nobs/100)^{1/4}.

        Returns
        -------
        Series
            Series containing the test statistic and its p-values.

        See Also
        --------
        statsmodels.stats.diagnostic.het_arch
            ARCH-LM test.
        statsmodels.stats.diagnostic.acorr_lm
            LM test for autocorrelation.
        """
        from statsmodels.stats.diagnostic import het_arch

        lags = int_like(lags, 'lags', optional=True)
        nobs_effective = self.resid.shape[0]
        if lags is None:
            max_lag = (nobs_effective - 1) // 2
            lags = int(min(12 * (nobs_effective / 100) ** (1 / 4), max_lag))
        out = []
        for lag in range(1, lags + 1):
            res = het_arch(self.resid, maxlag=lag, autolag=None)
            out.append([res[0], res[1], lag])
        index = pd.RangeIndex(1, lags + 1, name='Lag')
        cols = ['ARCH-LM', 'P-value', 'DF']
        return pd.DataFrame(out, columns=cols, index=index)

    def diagnostic_summary(self):
        """
        Returns a summary containing standard model diagnostic tests

        Returns
        -------
        Summary
            A summary instance with panels for serial correlation tests,
            normality tests and heteroskedasticity tests.

        See Also
        --------
        test_serial_correlation
            Test models residuals for serial correlation.
        test_normality
            Test models residuals for deviations from normality.
        test_heteroskedasticity
            Test models residuals for conditional heteroskedasticity.
        """
        from statsmodels.iolib.table import SimpleTable
        spacer = SimpleTable([''])
        smry = Summary()
        sc = self.test_serial_correlation()
        sc = sc.loc[sc.DF > 0]
        values = [[i + 1] + row for i, row in enumerate(sc.values.tolist())]
        data_fmts = ('%10d', '%10.3f', '%10.3f', '%10d')
        if sc.shape[0]:
            tab = SimpleTable(values,
                              headers=['Lag'] + list(sc.columns),
                              title='Test of No Serial Correlation',
                              header_align='r', data_fmts=data_fmts)
            smry.tables.append(tab)
            smry.tables.append(spacer)
        jb = self.test_normality()
        data_fmts = ('%10.3f', '%10.3f', '%10.3f', '%10.3f')
        tab = SimpleTable([jb.values], headers=list(jb.index),
                          title='Test of Normality',
                          header_align='r', data_fmts=data_fmts)
        smry.tables.append(tab)
        smry.tables.append(spacer)
        arch_lm = self.test_heteroskedasticity()
        values = [[i + 1] + row for i, row in
                  enumerate(arch_lm.values.tolist())]
        data_fmts = ('%10d', '%10.3f', '%10.3f', '%10d')
        tab = SimpleTable(values,
                          headers=['Lag'] + list(arch_lm.columns),
                          title='Test of Conditional Homoskedasticity',
                          header_align='r', data_fmts=data_fmts)
        smry.tables.append(tab)
        return smry

    @Appender(remove_parameters(AutoReg.predict.__doc__, 'params'))
    def predict(self, start=None, end=None, dynamic=False, exog=None,
                exog_oos=None):
        return self.model.predict(self._params, start=start, end=end,
                                  dynamic=dynamic, exog=exog,
                                  exog_oos=exog_oos)

    @Substitution(predict_params=_predict_params)
    def plot_predict(self, start=None, end=None, dynamic=False, exog=None,
                     exog_oos=None, alpha=.05, in_sample=True, fig=None,
                     figsize=None):
        """
        Plot in- and out-of-sample predictions

        Parameters
        ----------
%(predict_params)s
        alpha : {float, None}
            The tail probability not covered by the confidence interval. Must
            be in (0, 1). Confidence interval is constructed assuming normally
            distributed shocks. If None, figure will not show the confidence
            interval.
        in_sample : bool
            Flag indicating whether to include the in-sample period in the
            plot.
        fig : Figure
            An existing figure handle. If not provided, a new figure is
            created.
        figsize: tuple[float, float]
            Tuple containing the figure size values.

        Returns
        -------
        Figure
            Figure handle containing the plot.
        """
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        _import_mpl()
        fig = create_mpl_fig(fig, figsize)
        predictions = self.predict(start=start, end=end, dynamic=dynamic,
                                   exog=exog, exog_oos=exog_oos)
        start = 0 if start is None else start
        end = self.model._index[-1] if end is None else end
        _, _, oos, _ = self.model._get_prediction_index(start, end)

        ax = fig.add_subplot(111)
        if in_sample:
            ax.plot(predictions)
        elif oos:
            if isinstance(predictions, pd.Series):
                predictions = predictions.iloc[-oos:]
            else:
                predictions = predictions[-oos:]
        else:
            raise ValueError('in_sample is False but there are no'
                             'out-of-sample forecasts to plot.')

        if oos and alpha is not None:
            pred_oos = np.asarray(predictions)[-oos:]
            ar_params = self._lag_repr()
            ma = arma2ma(ar_params, [1], lags=oos)
            fc_error = np.sqrt(self.sigma2) * np.cumsum(ma ** 2)
            quantile = norm.ppf(alpha / 2)
            lower = pred_oos + fc_error * quantile
            upper = pred_oos + fc_error * -quantile
            label = "{0:.0%} confidence interval".format(1 - alpha)
            x = ax.get_lines()[-1].get_xdata()
            ax.fill_between(x[-oos:], lower, upper,
                            color='gray', alpha=.5, label=label)

        ax.legend(loc='best')

        return fig

    def plot_diagnostics(self, lags=10, fig=None, figsize=None):
        """
        Diagnostic plots for standardized residuals

        Parameters
        ----------
        lags : int, optional
            Number of lags to include in the correlogram. Default is 10.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the 2x2 grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----
        Produces a 2x2 plot grid with the following plots (ordered clockwise
        from top left):

        1. Standardized residuals over time
        2. Histogram plus estimated density of standardized residuals, along
           with a Normal(0,1) density plotted for reference.
        3. Normal Q-Q plot, with Normal reference line.
        4. Correlogram

        See Also
        --------
        statsmodels.graphics.gofplots.qqplot
        statsmodels.graphics.tsaplots.plot_acf
        """
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        _import_mpl()
        fig = create_mpl_fig(fig, figsize)
        # Eliminate residuals associated with burned or diffuse likelihoods
        resid = self.resid

        # Top-left: residuals vs time
        ax = fig.add_subplot(221)
        if hasattr(self.model.data, 'dates') and self.data.dates is not None:
            x = self.model.data.dates._mpl_repr()
            x = x[self.model.hold_back:]
        else:
            hold_back = self.model.hold_back
            x = hold_back + np.arange(self.resid.shape[0])
        std_resid = resid / np.sqrt(self.sigma2)
        ax.plot(x, std_resid)
        ax.hlines(0, x[0], x[-1], alpha=0.5)
        ax.set_xlim(x[0], x[-1])
        ax.set_title('Standardized residual')

        # Top-right: histogram, Gaussian kernel density, Normal density
        # Can only do histogram and Gaussian kernel density on the non-null
        # elements
        std_resid_nonmissing = std_resid[~(np.isnan(resid))]
        ax = fig.add_subplot(222)

        # gh5792: Remove  except after support for matplotlib>2.1 required
        try:
            ax.hist(std_resid_nonmissing, density=True, label='Hist')
        except AttributeError:
            ax.hist(std_resid_nonmissing, normed=True, label='Hist')

        kde = gaussian_kde(std_resid)
        xlim = (-1.96 * 2, 1.96 * 2)
        x = np.linspace(xlim[0], xlim[1])
        ax.plot(x, kde(x), label='KDE')
        ax.plot(x, norm.pdf(x), label='N(0,1)')
        ax.set_xlim(xlim)
        ax.legend()
        ax.set_title('Histogram plus estimated density')

        # Bottom-left: QQ plot
        ax = fig.add_subplot(223)
        from statsmodels.graphics.gofplots import qqplot
        qqplot(std_resid, line='s', ax=ax)
        ax.set_title('Normal Q-Q')

        # Bottom-right: Correlogram
        ax = fig.add_subplot(224)
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(resid, ax=ax, lags=lags)
        ax.set_title('Correlogram')

        ax.set_ylim(-1, 1)

        return fig

    def summary(self, alpha=.05):
        """
        Summarize the Model

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
        model = self.model

        title = model.__class__.__name__ + ' Model Results'
        method = 'Conditional MLE'
        # get sample
        start = self._hold_back
        if self.data.dates is not None:
            dates = self.data.dates
            sample = [dates[start].strftime('%m-%d-%Y')]
            sample += ['- ' + dates[-1].strftime('%m-%d-%Y')]
        else:
            sample = [str(start), str(len(self.data.orig_endog))]
        model = model.__class__.__name__
        if self.model.seasonal:
            model = 'Seas. ' + model
        if len(self.ar_lags) < self._max_lag:
            model = 'Restr. ' + model
        if self.model.exog is not None:
            model += '-X'

        order = '({0})'.format(self._max_lag)
        dep_name = str(self.model.endog_names)
        top_left = [('Dep. Variable:', [dep_name]),
                    ('Model:', [model + order]),
                    ('Method:', [method]),
                    ('Date:', None),
                    ('Time:', None),
                    ('Sample:', [sample[0]]),
                    ('', [sample[1]])
                    ]

        top_right = [
            ('No. Observations:', [str(len(self.model.endog))]),
            ('Log Likelihood', ["%#5.3f" % self.llf]),
            ('S.D. of innovations', ["%#5.3f" % self.sigma2 ** .5]),
            ('AIC', ["%#5.3f" % self.aic]),
            ('BIC', ["%#5.3f" % self.bic]),
            ('HQIC', ["%#5.3f" % self.hqic])]

        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                             title=title)
        smry.add_table_params(self, alpha=alpha, use_t=False)

        # Make the roots table
        from statsmodels.iolib.table import SimpleTable

        if self._max_lag:
            arstubs = ["AR.%d" % i for i in range(1, self._max_lag + 1)]
            stubs = arstubs
            roots = self.roots
            freq = self.arfreq
            modulus = np.abs(roots)
            data = np.column_stack((roots.real, roots.imag, modulus, freq))
            roots_table = SimpleTable([('%17.4f' % row[0],
                                        '%+17.4fj' % row[1],
                                        '%17.4f' % row[2],
                                        '%17.4f' % row[3]) for row in data],
                                      headers=['            Real',
                                               '         Imaginary',
                                               '         Modulus',
                                               '        Frequency'],
                                      title="Roots",
                                      stubs=stubs)

            smry.tables.append(roots_table)
        return smry


class AutoRegResultsWrapper(wrap.ResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(
        tsa_model.TimeSeriesResultsWrapper._wrap_attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(
        tsa_model.TimeSeriesResultsWrapper._wrap_methods, _methods)


wrap.populate_wrapper(AutoRegResultsWrapper, AutoRegResults)

doc = Docstring(AutoReg.__doc__)
_auto_reg_params = doc.extract_parameters(['trend', 'seasonal', 'exog',
                                           'hold_back', 'period', 'missing'],
                                          4)


@Substitution(auto_reg_params=_auto_reg_params)
def ar_select_order(endog, maxlag, ic='bic', glob=False, trend='c',
                    seasonal=False, exog=None, hold_back=None, period=None,
                    missing='none'):
    """
    Autoregressive AR-X(p) model order selection.

    Parameters
    ----------
    endog : array_like
        A 1-d endogenous response variable. The independent variable.
    maxlag : int
        The maximum lag to consider.
    ic : {'aic', 'hqic', 'bic'}
        The information criterion to use in the selection.
    glob : bool
        Flag indicating where to use a global search  across all combinations
        of lags.  In practice, this option is not computational feasible when
        maxlag is larger than 15 (or perhaps 20) since the global search
        requires fitting 2**maxlag models.
%(auto_reg_params)s

    Returns
    -------
    AROrderSelectionResults
        A results holder containing the model and the complete set of
        information criteria for all models fit.

    Examples
    --------
    >>> from statsmodels.tsa.ar_model import ar_select_order
    >>> data = sm.datasets.sunspots.load_pandas().data['SUNACTIVITY']

    Determine the optimal lag structure

    >>> mod = ar_select_order(data, maxlag=13)
    >>> mod.ar_lags
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    Determine the optimal lag structure with seasonal terms

    >>> mod = ar_select_order(data, maxlag=13, seasonal=True, period=12)
    >>> mod.ar_lags
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    Globally determine the optimal lag structure

    >>> mod = ar_select_order(data, maxlag=13, glob=True)
    >>> mod.ar_lags
    array([1, 2, 9])
    """
    full_mod = AutoReg(endog, maxlag, trend=trend, seasonal=seasonal,
                       exog=exog, hold_back=hold_back, period=period,
                       missing=missing)
    nexog = full_mod.exog.shape[1] if full_mod.exog is not None else 0
    y, x = full_mod._y, full_mod._x
    base_col = x.shape[1] - nexog - maxlag
    sel = np.ones(x.shape[1], dtype=bool)
    ics = []

    def compute_ics(res):
        nobs = res.nobs
        df_model = res.df_model
        sigma2 = 1. / nobs * sumofsq(res.resid)

        res = SimpleNamespace(nobs=nobs, df_model=df_model, sigma2=sigma2)

        aic = AutoRegResults.aic.func(res)
        bic = AutoRegResults.bic.func(res)
        hqic = AutoRegResults.hqic.func(res)

        return aic, bic, hqic

    def ic_no_data():
        """Fake mod and results to handle no regressor case"""
        mod = SimpleNamespace(nobs=y.shape[0],
                              endog=y,
                              exog=np.empty((y.shape[0], 0)))
        llf = OLS.loglike(mod, np.empty(0))
        res = SimpleNamespace(resid=y, nobs=y.shape[0], llf=llf,
                              df_model=0, k_constant=0)

        return compute_ics(res)

    if not glob:
        sel[base_col: base_col + maxlag] = False
        for i in range(maxlag + 1):
            sel[base_col:base_col + i] = True
            if not np.any(sel):
                ics.append((0, ic_no_data()))
                continue
            res = OLS(y, x[:, sel]).fit()
            lags = tuple(j for j in range(1, i + 1))
            lags = 0 if not lags else lags
            ics.append((lags, compute_ics(res)))
    else:
        bits = np.arange(2 ** maxlag, dtype=np.int32)[:, None]
        bits = bits.view(np.uint8)
        bits = np.unpackbits(bits).reshape(-1, 32)
        for i in range(4):
            bits[:, 8 * i:8 * (i + 1)] = bits[:, 8 * i:8 * (i + 1)][:, ::-1]
        masks = bits[:, :maxlag]
        for mask in masks:
            sel[base_col:base_col + maxlag] = mask
            if not np.any(sel):
                ics.append((0, ic_no_data()))
                continue
            res = OLS(y, x[:, sel]).fit()
            lags = tuple(np.where(mask)[0] + 1)
            lags = 0 if not lags else lags
            ics.append((lags, compute_ics(res)))

    key_loc = {'aic': 0, 'bic': 1, 'hqic': 2}[ic]
    ics = sorted(ics, key=lambda x: x[1][key_loc])
    selected_model = ics[0][0]
    mod = AutoReg(endog, selected_model, trend=trend, seasonal=seasonal,
                  exog=exog, hold_back=hold_back, period=period,
                  missing=missing)
    return AROrderSelectionResults(mod, ics, trend, seasonal, period)


class AROrderSelectionResults(object):
    """
    Results from an AR order selection

    Contains the information criteria for all fitted model orders.
    """

    def __init__(self, model, ics, trend, seasonal, period):
        self._model = model
        self._ics = ics
        self._trend = trend
        self._seasonal = seasonal
        self._period = period
        aic = sorted(ics, key=lambda r: r[1][0])
        self._aic = dict([(key, val[0]) for key, val in aic])
        bic = sorted(ics, key=lambda r: r[1][1])
        self._bic = dict([(key, val[1]) for key, val in bic])
        hqic = sorted(ics, key=lambda r: r[1][2])
        self._hqic = dict([(key, val[2]) for key, val in hqic])

    @property
    def model(self):
        """The model selected using the chosen information criterion."""
        return self._model

    @property
    def seasonal(self):
        """Flag indicating if a seasonal component is included."""
        return self._seasonal

    @property
    def trend(self):
        """The trend included in the model selection."""
        return self._trend

    @property
    def period(self):
        """The period of the seasonal component."""
        return self._period

    @property
    def aic(self):
        """
        The Akaike information criterion for the models fit.

        Returns
        -------
        dict[tuple, float]
        """
        return self._aic

    @property
    def bic(self):
        """
        The Bayesian (Schwarz) information criteria for the models fit.

        Returns
        -------
        dict[tuple, float]
        """
        return self._bic

    @property
    def hqic(self):
        """
        The Hannan-Quinn information criteria for the models fit.

        Returns
        -------
        dict[tuple, float]
        """
        return self._hqic

    @property
    def ar_lags(self):
        """The lags included in the selected model."""
        return self._model.ar_lags
