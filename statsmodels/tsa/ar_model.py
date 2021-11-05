# -*- coding: utf-8 -*-
from __future__ import annotations

from statsmodels.compat.pandas import (
    Appender,
    Substitution,
    call_cached_func,
    to_numpy,
)

from collections.abc import Iterable
import datetime as dt
from types import SimpleNamespace
from typing import List, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm

import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import eval_measures
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.validation import (
    array_like,
    bool_like,
    int_like,
    string_like,
)
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.base import tsa_model
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import (
    DeterministicProcess,
    Seasonality,
    TimeTrend,
)
from statsmodels.tsa.tsatools import freq_to_period, lagmat

__all__ = ["AR", "AutoReg"]

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


def _get_period(data, index_freq):
    """Shared helper to get period from frequenc or raise"""
    if data.freq:
        return freq_to_period(index_freq)
    raise ValueError(
        "freq cannot be inferred from endog and model includes seasonal "
        "terms.  The number of periods must be explicitly set when the "
        "endog's index does not contain a frequency."
    )


class AutoReg(tsa_model.TimeSeriesModel):
    """
    Autoregressive AR-X(p) model

    Estimate an AR-X model using Conditional Maximum Likelihood (OLS).

    Parameters
    ----------
    endog : array_like
        A 1-d endogenous response variable. The dependent variable.
    lags : {None, int, list[int]}
        The number of lags to include in the model if an integer or the
        list of lag indices to include.  For example, [1, 4] will only
        include lags 1 and 4 while lags=4 will include lags 1, 2, 3, and 4.
        None excludes all AR lags, and behave identically to 0.
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
    deterministic : DeterministicProcess
        A deterministic process.  If provided, trend and seasonal are ignored.
        A warning is raised if trend is not "n" and seasonal is not False.
    old_names : bool
        Flag indicating whether to use the v0.11 names or the v0.12+ names.

        .. deprecated:: 0.13

           old_names is deprecated and will be removed after 0.14 is
           released. You must update any code reliant on the old variable
           names to use the new names.

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

    def __init__(
        self,
        endog,
        lags,
        trend="c",
        seasonal=False,
        exog=None,
        hold_back=None,
        period=None,
        missing="none",
        *,
        deterministic=None,
        old_names=False,
    ):
        super().__init__(endog, exog, None, None, missing=missing)
        self._trend = string_like(
            trend, "trend", options=("n", "c", "t", "ct"), optional=False
        )
        self._seasonal = bool_like(seasonal, "seasonal")
        self._period = int_like(period, "period", optional=True)
        if self._period is None and self._seasonal:
            self._period = _get_period(self.data, self._index_freq)
        terms = [TimeTrend.from_string(self._trend)]
        if seasonal:
            terms.append(Seasonality(self._period))
        if hasattr(self.data.orig_endog, "index"):
            index = self.data.orig_endog.index
        else:
            index = np.arange(self.data.endog.shape[0])
        self._user_deterministic = False
        if deterministic is not None:
            if not isinstance(deterministic, DeterministicProcess):
                raise TypeError("deterministic must be a DeterministicProcess")
            self._deterministics = deterministic
            self._user_deterministic = True
        else:
            self._deterministics = DeterministicProcess(
                index, additional_terms=terms
            )
        self._lags = lags
        self._exog_names = []
        self._k_ar = 0
        self._hold_back = int_like(hold_back, "hold_back", optional=True)
        self._old_names = bool_like(old_names, "old_names", optional=False)
        if deterministic is not None and (
            self._trend != "n" or self._seasonal
        ):
            warnings.warn(
                'When using deterministic, trend must be "n" and '
                "seasonal must be False.",
                SpecificationWarning,
            )
        if self._old_names:
            warnings.warn(
                "old_names will be removed after the 0.14 release. You should "
                "stop setting this parameter and use the new names.",
                FutureWarning,
            )
        self._lags, self._hold_back = self._check_lags()
        self._setup_regressors()
        self.nobs = self._y.shape[0]
        self.data.xnames = self.exog_names

    @property
    def ar_lags(self):
        """The autoregressive lags included in the model"""
        lags = list(self._lags)
        return None if not lags else lags

    @property
    def hold_back(self):
        """The number of initial obs. excluded from the estimation sample."""
        return self._hold_back

    @property
    def trend(self):
        """The trend used in the model."""
        return self._trend

    @property
    def seasonal(self):
        """Flag indicating that the model contains a seasonal component."""
        return self._seasonal

    @property
    def deterministic(self):
        """The deterministic used to construct the model"""
        return self._deterministics if self._user_deterministic else None

    @property
    def period(self):
        """The period of the seasonal component."""
        return self._period

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

    def _check_lags(self) -> Tuple[List[int], int]:
        lags = self._lags
        if lags is None:
            lags = []
            self._maxlag = 0
        elif isinstance(lags, Iterable):
            lags = np.array(sorted([int_like(lag, "lags") for lag in lags]))
            if np.any(lags < 1) or np.unique(lags).shape[0] != lags.shape[0]:
                raise ValueError(
                    "All values in lags must be positive and distinct."
                )
            self._maxlag = np.max(lags)
        else:
            self._maxlag = int_like(lags, "lags")
            if self._maxlag < 0:
                raise ValueError("lags must be a non-negative scalar.")
            lags = np.arange(1, self._maxlag + 1)
        hold_back = self._hold_back
        if hold_back is None:
            hold_back = self._maxlag
        if hold_back < self._maxlag:
            raise ValueError(
                "hold_back must be >= lags if lags is an int or"
                "max(lags) if lags is array_like."
            )
        return list(lags), int(hold_back)

    def _setup_regressors(self):
        maxlag = self._maxlag
        hold_back = self._hold_back
        exog_names = []
        endog_names = self.endog_names
        x, y = lagmat(self.endog, maxlag, original="sep")
        exog_names.extend(
            [endog_names + ".L{0}".format(lag) for lag in self._lags]
        )
        if len(self._lags) < maxlag:
            x = x[:, np.asarray(self._lags) - 1]
        self._k_ar = x.shape[1]
        deterministic = self._deterministics.in_sample()
        if deterministic.shape[1]:
            x = np.c_[to_numpy(deterministic), x]
            if self._old_names:
                deterministic_names = []
                if "c" in self._trend:
                    deterministic_names.append("intercept")
                if "t" in self._trend:
                    deterministic_names.append("trend")
                if self._seasonal:
                    period = self._period
                    names = ["seasonal.{0}".format(i) for i in range(period)]
                    if "c" in self._trend:
                        names = names[1:]
                    deterministic_names.extend(names)
            else:
                deterministic_names = list(deterministic.columns)
            exog_names = deterministic_names + exog_names
        if self.exog is not None:
            x = np.c_[x, self.exog]
            exog_names.extend(self.data.param_names)
        y = y[hold_back:]
        x = x[hold_back:]
        if y.shape[0] < x.shape[1]:
            reg = x.shape[1]
            period = self._period
            trend = 0 if self._trend == "n" else len(self._trend)
            seas = 0 if not self._seasonal else period - ("c" in self._trend)
            lags = len(self._lags)
            nobs = y.shape[0]
            raise ValueError(
                "The model specification cannot be estimated. "
                f"The model contains {reg} regressors ({trend} trend, "
                f"{seas} seasonal, {lags} lags) but after adjustment "
                "for hold_back and creation of the lags, there "
                f"are only {nobs} data points available to estimate "
                "parameters."
            )
        self._y, self._x = y, x
        self._exog_names = exog_names

    def fit(self, cov_type="nonrobust", cov_kwds=None, use_t=False):
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

              - `maxlags` integer (required) : number of lags to use.
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
            return AutoRegResultsWrapper(
                AutoRegResults(self, np.empty(0), np.empty((0, 0)))
            )

        ols_mod = OLS(self._y, self._x)
        ols_res = ols_mod.fit(
            cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t
        )
        cov_params = ols_res.cov_params()
        use_t = ols_res.use_t
        if cov_type == "nonrobust" and not use_t:
            nobs = self._y.shape[0]
            k = self._x.shape[1]
            scale = nobs / (nobs - k)
            cov_params /= scale
        res = AutoRegResults(
            self,
            ols_res.params,
            cov_params,
            ols_res.normalized_cov_params,
            use_t=use_t,
        )

        return AutoRegResultsWrapper(res)

    def _resid(self, params):
        params = array_like(params, "params", ndim=2)
        return self._y.squeeze() - (self._x @ params).squeeze()

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
        return (self._x.T @ self._x) * (1 / sigma2)

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
        x = np.zeros((add_forecasts, self._x.shape[1]))
        oos_exog = self._deterministics.out_of_sample(steps=add_forecasts)
        n_deterministic = oos_exog.shape[1]
        x[:, :n_deterministic] = to_numpy(oos_exog)
        # skip the AR columns
        loc = n_deterministic + len(self._lags)
        if self.exog is not None:
            x[:, loc:] = exog_oos[:add_forecasts]
        return x

    def _wrap_prediction(self, prediction, start, end, pad):
        prediction = np.hstack([np.full(pad, np.nan), prediction])
        n_values = end - start + pad
        if not isinstance(self.data.orig_endog, (pd.Series, pd.DataFrame)):
            return prediction[-n_values:]
        index = self._index
        if end > self.endog.shape[0]:
            freq = getattr(index, "freq", None)
            if freq:
                if isinstance(index, pd.PeriodIndex):
                    index = pd.period_range(index[0], freq=freq, periods=end)
                else:
                    index = pd.date_range(index[0], freq=freq, periods=end)
            else:
                index = pd.RangeIndex(end)
        index = index[start - pad : end]
        prediction = prediction[-n_values:]
        return pd.Series(prediction, index=index)

    def _dynamic_predict(
        self, params, start, end, dynamic, num_oos, exog, exog_oos
    ):
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
        adj = 0
        if start < hold_back:
            # Adjust start and dynamic
            adj = hold_back - start
        start += adj
        # New offset shifts, but must remain non-negative
        dynamic = max(dynamic - adj, 0)

        if (start - hold_back) <= self.nobs:
            # _x is missing hold_back observations, which is why
            # it is shifted by this amount
            is_loc = slice(start - hold_back, end + 1 - hold_back)
            x = self._x[is_loc]
            if exog is not None:
                x = x.copy()
                # Replace final columns
                x[:, -exog.shape[1] :] = exog[start : end + 1]
            reg.append(x)
        if num_oos > 0:
            reg.append(self._setup_oos_forecast(num_oos, exog_oos))
        reg = np.vstack(reg)
        det_col_idx = self._x.shape[1] - len(self._lags)
        det_col_idx -= 0 if self.exog is None else self.exog.shape[1]
        # Simple 1-step static forecasts for dynamic observations
        forecasts = np.empty(reg.shape[0])
        forecasts[:dynamic] = reg[:dynamic] @ params
        for h in range(dynamic, reg.shape[0]):
            # Fill in regressor matrix
            for j, lag in enumerate(self._lags):
                fcast_loc = h - lag
                if fcast_loc >= dynamic:
                    val = forecasts[fcast_loc]
                else:
                    # If before the start of the forecasts, use actual values
                    val = self.endog[fcast_loc + start]
                reg[h, det_col_idx + j] = val
            forecasts[h] = reg[h : h + 1] @ params
        return self._wrap_prediction(forecasts, start, end + 1 + num_oos, adj)

    def _static_oos_predict(self, params, num_oos, exog_oos):
        new_x = self._setup_oos_forecast(num_oos, exog_oos)
        if self._maxlag == 0:
            return new_x @ params
        forecasts = np.empty(num_oos)
        nexog = 0 if self.exog is None else self.exog.shape[1]
        ar_offset = self._x.shape[1] - nexog - len(self._lags)
        for i in range(num_oos):
            for j, lag in enumerate(self._lags):
                loc = i - lag
                val = self._y[loc] if loc < 0 else forecasts[loc]
                new_x[i, ar_offset + j] = val
            forecasts[i] = new_x[i : i + 1] @ params
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

        # Adjust start to reflect observations lost
        adj = max(0, hold_back - start)
        start += adj
        if start <= nobs:
            # Use existing regressors
            is_loc = slice(start - hold_back, end + 1 - hold_back)
            x = self._x[is_loc]
            if exog is not None:
                x = x.copy()
                # Replace final columns
                x[:, -exog.shape[1] :] = exog[start : end + 1]
        in_sample = x @ params
        if num_oos == 0:  # No out of sample
            return self._wrap_prediction(in_sample, start, end + 1, adj)

        out_of_sample = self._static_oos_predict(params, num_oos, exog_oos)
        prediction = np.hstack((in_sample, out_of_sample))
        return self._wrap_prediction(prediction, start, end + 1 + num_oos, adj)

    def _prepare_prediction(self, params, exog, exog_oos, start, end):
        params = array_like(params, "params")
        if not isinstance(exog, pd.DataFrame):
            exog = array_like(exog, "exog", ndim=2, optional=True)
        if not isinstance(exog_oos, pd.DataFrame):
            exog_oos = array_like(exog_oos, "exog_oos", ndim=2, optional=True)

        start = 0 if start is None else start
        end = self._index[-1] if end is None else end
        start, end, num_oos, _ = self._get_prediction_index(start, end)
        return params, exog, exog_oos, start, end, num_oos

    def _parse_dynamic(self, dynamic, start):
        if isinstance(
            dynamic, (str, bytes, pd.Timestamp, dt.datetime, pd.Period)
        ):
            dynamic_loc, _, _ = self._get_index_loc(dynamic)
            # Adjust since relative to start
            dynamic_loc -= start
        elif dynamic is True:
            # if True, all forecasts are dynamic
            dynamic_loc = 0
        else:
            dynamic_loc = int(dynamic)
        # At this point dynamic is an offset relative to start
        # and it must be non-negative
        if dynamic_loc < 0:
            raise ValueError(
                "Dynamic prediction cannot begin prior to the "
                "first observation in the sample."
            )
        return dynamic_loc

    def predict(
        self,
        params,
        start=None,
        end=None,
        dynamic=False,
        exog=None,
        exog_oos=None,
    ):
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
        predictions : {ndarray, Series}
            Array of out of in-sample predictions and / or out-of-sample
            forecasts.
        """

        params, exog, exog_oos, start, end, num_oos = self._prepare_prediction(
            params, exog, exog_oos, start, end
        )
        if self.exog is None and (exog is not None or exog_oos is not None):
            raise ValueError(
                "exog and exog_oos cannot be used when the model "
                "does not contains exogenous regressors."
            )
        elif self.exog is not None:
            if exog is not None and exog.shape != self.exog.shape:
                msg = (
                    "The shape of exog {0} must match the shape of the "
                    "exog variable used to create the model {1}."
                )
                raise ValueError(msg.format(exog.shape, self.exog.shape))
            if (
                exog_oos is not None
                and exog_oos.shape[1] != self.exog.shape[1]
            ):
                msg = (
                    "The number of columns in exog_oos ({0}) must match "
                    "the number of columns  in the exog variable used to "
                    "create the model ({1})."
                )
                raise ValueError(
                    msg.format(exog_oos.shape[1], self.exog.shape[1])
                )
            if num_oos > 0 and exog_oos is None:
                raise ValueError(
                    "exog_oos must be provided when producing "
                    "out-of-sample forecasts."
                )
            elif exog_oos is not None and num_oos > exog_oos.shape[0]:
                msg = (
                    "start and end indicate that {0} out-of-sample "
                    "predictions must be computed. exog_oos has {1} rows "
                    "but must have at least {0}."
                )
                raise ValueError(msg.format(num_oos, exog_oos.shape[0]))

        if (isinstance(dynamic, bool) and not dynamic) or self._maxlag == 0:
            # If model has no lags, static and dynamic are identical
            return self._static_predict(
                params, start, end, num_oos, exog, exog_oos
            )
        dynamic = self._parse_dynamic(dynamic, start)

        return self._dynamic_predict(
            params, start, end, dynamic, num_oos, exog, exog_oos
        )


class AR:
    """
    The AR class has been removed and replaced with AutoReg

    See Also
    --------
    AutoReg
        The replacement for AR that improved deterministic modeling
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "AR has been removed from statsmodels and replaced with "
            "statsmodels.tsa.ar_model.AutoReg."
        )


class ARResults:
    """
    Removed and replaced by AutoRegResults.

    See Also
    --------
    AutoReg
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "AR and ARResults have been removed and replaced by "
            "AutoReg And AutoRegResults."
        )


doc = Docstring(AutoReg.predict.__doc__)
_predict_params = doc.extract_parameters(
    ["start", "end", "dynamic", "exog", "exog_oos"], 8
)


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
    use_t : bool, optional
        Whether use_t was set in fit
    """

    _cache = {}  # for scale setter

    def __init__(
        self,
        model,
        params,
        cov_params,
        normalized_cov_params=None,
        scale=1.0,
        use_t=False,
    ):
        super().__init__(model, params, normalized_cov_params, scale)
        self._cache = {}
        self._params = params
        self._nobs = model.nobs
        self._n_totobs = model.endog.shape[0]
        self._df_model = model.df_model
        self._ar_lags = model.ar_lags
        self._use_t = use_t
        if self._ar_lags is not None:
            self._max_lag = max(self._ar_lags)
        else:
            self._max_lag = 0
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
        return 1.0 / self.nobs * sumofsq(self.resid)

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
        r"""
        Akaike Information Criterion using Lutkepohl's definition.

        :math:`-2 llf + \ln(nobs) (1 + df_{model})`
        """
        # This is based on loglike with dropped constant terms ?
        # Lutkepohl
        # return np.log(self.sigma2) + 1./self.model.nobs * self.k_ar
        # Include constant as estimated free parameter and double the loss
        # Stata defintion
        # nobs = self.nobs
        # return -2 * self.llf/nobs + 2 * (self.k_ar+self.k_trend)/nobs
        return eval_measures.aic(self.llf, self.nobs, self.df_model + 1)

    @cache_readonly
    def hqic(self):
        r"""
        Hannan-Quinn Information Criterion using Lutkepohl's definition.

        :math:`-2 llf + 2 \ln(\ln(nobs)) (1 + df_{model})`
        """
        # Lutkepohl
        # return np.log(self.sigma2)+ 2 * np.log(np.log(nobs))/nobs * self.k_ar
        # R uses all estimated parameters rather than just lags
        # Stata
        # nobs = self.nobs
        # return -2 * self.llf/nobs + 2 * np.log(np.log(nobs))/nobs * \
        #        (self.k_ar + self.k_trend)
        return eval_measures.hqic(self.llf, self.nobs, self.df_model + 1)

    @cache_readonly
    def fpe(self):
        r"""
        Final prediction error using Lütkepohl's definition.

        :math:`((nobs+df_{model})/(nobs-df_{model})) \sigma^2`
        """
        nobs = self.nobs
        df_model = self.df_model
        # Lutkepohl
        return self.sigma2 * ((nobs + df_model) / (nobs - df_model))

    @cache_readonly
    def aicc(self):
        r"""
        Akaike Information Criterion with small sample correction

        :math:`2.0 * df_{model} * nobs / (nobs - df_{model} - 1.0)`
        """
        return eval_measures.aicc(self.llf, self.nobs, self.df_model + 1)

    @cache_readonly
    def bic(self):
        r"""
        Bayes Information Criterion

        :math:`-2 llf + \ln(nobs) (1 + df_{model})`
        """
        # Lutkepohl
        # np.log(self.sigma2) + np.log(nobs)/nobs * self.k_ar
        # Include constant as est. free parameter
        # Stata
        # -2 * self.llf/nobs + np.log(nobs)/nobs * (self.k_ar + self.k_trend)
        return eval_measures.bic(self.llf, self.nobs, self.df_model + 1)

    @cache_readonly
    def resid(self):
        """
        The residuals of the model.
        """
        model = self.model
        endog = model.endog.squeeze()
        return endog[self._hold_back :] - self.fittedvalues

    def _lag_repr(self):
        """Returns poly repr of an AR, (1  -phi1 L -phi2 L^2-...)"""
        ar_lags = self._ar_lags if self._ar_lags is not None else []
        k_ar = len(ar_lags)
        ar_params = np.zeros(self._max_lag + 1)
        ar_params[0] = 1
        df_model = self._df_model
        exog = self.model.exog
        k_exog = exog.shape[1] if exog is not None else 0
        params = self._params[df_model - k_ar - k_exog : df_model - k_exog]
        for i, lag in enumerate(ar_lags):
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
        # TODO: Specific to AR
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
        # TODO: Specific to AR
        z = self.roots
        return np.arctan2(z.imag, z.real) / (2 * np.pi)

    @cache_readonly
    def fittedvalues(self):
        """
        The in-sample predicted values of the fitted AR model.

        The `k_ar` initial values are computed via the Kalman Filter if the
        model is fit by `mle`.
        """
        return self.model.predict(self.params)[self._hold_back :]

    def test_serial_correlation(self, lags=None, model_df=None):
        """
        Ljung-Box test for residual serial correlation

        Parameters
        ----------
        lags : int
            The maximum number of lags to use in the test. Jointly tests that
            all autocorrelations up to and including lag j are zero for
            j = 1, 2, ..., lags. If None, uses lag=12*(nobs/100)^{1/4}.
            After 0.12 the number of lags will change to min(10, nobs // 5).
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

        lags = int_like(lags, "lags", optional=True)
        model_df = int_like(model_df, "df_model", optional=True)
        model_df = self.df_model if model_df is None else model_df
        nobs_effective = self.resid.shape[0]
        if lags is None:
            lags = min(nobs_effective // 5, 10)
        test_stats = acorr_ljungbox(
            self.resid,
            lags=lags,
            boxpierce=False,
            model_df=model_df,
            return_df=False,
        )
        cols = ["Ljung-Box", "LB P-value", "DF"]
        if lags == 1:
            test_stats = [list(test_stats) + [max(0, 1 - model_df)]]
        else:
            df = np.clip(np.arange(1, lags + 1) - model_df, 0, np.inf).astype(
                int
            )
            test_stats = list(test_stats) + [df]
            test_stats = [
                [test_stats[i][j] for i in range(3)] for j in range(lags)
            ]
        index = pd.RangeIndex(1, lags + 1, name="Lag")
        return pd.DataFrame(test_stats, columns=cols, index=index)

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

        index = ["Jarque-Bera", "P-value", "Skewness", "Kurtosis"]
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

        lags = int_like(lags, "lags", optional=True)
        nobs_effective = self.resid.shape[0]
        if lags is None:
            lags = min(nobs_effective // 5, 10)
        out = []
        for lag in range(1, lags + 1):
            res = het_arch(self.resid, nlags=lag, autolag=None)
            out.append([res[0], res[1], lag])
        index = pd.RangeIndex(1, lags + 1, name="Lag")
        cols = ["ARCH-LM", "P-value", "DF"]
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

        spacer = SimpleTable([""])
        smry = Summary()
        sc = self.test_serial_correlation()
        sc = sc.loc[sc.DF > 0]
        values = [[i + 1] + row for i, row in enumerate(sc.values.tolist())]
        data_fmts = ("%10d", "%10.3f", "%10.3f", "%10d")
        if sc.shape[0]:
            tab = SimpleTable(
                values,
                headers=["Lag"] + list(sc.columns),
                title="Test of No Serial Correlation",
                header_align="r",
                data_fmts=data_fmts,
            )
            smry.tables.append(tab)
            smry.tables.append(spacer)
        jb = self.test_normality()
        data_fmts = ("%10.3f", "%10.3f", "%10.3f", "%10.3f")
        tab = SimpleTable(
            [jb.values],
            headers=list(jb.index),
            title="Test of Normality",
            header_align="r",
            data_fmts=data_fmts,
        )
        smry.tables.append(tab)
        smry.tables.append(spacer)
        arch_lm = self.test_heteroskedasticity()
        values = [
            [i + 1] + row for i, row in enumerate(arch_lm.values.tolist())
        ]
        data_fmts = ("%10d", "%10.3f", "%10.3f", "%10d")
        tab = SimpleTable(
            values,
            headers=["Lag"] + list(arch_lm.columns),
            title="Test of Conditional Homoskedasticity",
            header_align="r",
            data_fmts=data_fmts,
        )
        smry.tables.append(tab)
        return smry

    @Appender(remove_parameters(AutoReg.predict.__doc__, "params"))
    def predict(
        self, start=None, end=None, dynamic=False, exog=None, exog_oos=None
    ):
        return self.model.predict(
            self._params,
            start=start,
            end=end,
            dynamic=dynamic,
            exog=exog,
            exog_oos=exog_oos,
        )

    def get_prediction(
        self, start=None, end=None, dynamic=False, exog=None, exog_oos=None
    ):
        """
        Predictions and prediction intervals

        Parameters
        ----------
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
        PredictionResults
            Prediction results with mean and prediction intervals
        """
        mean = self.predict(
            start=start, end=end, dynamic=dynamic, exog=exog, exog_oos=exog_oos
        )
        mean_var = np.full_like(mean, self.sigma2)
        mean_var[np.isnan(mean)] = np.nan
        start = 0 if start is None else start
        end = self.model._index[-1] if end is None else end
        _, _, oos, _ = self.model._get_prediction_index(start, end)
        if oos > 0:
            ar_params = self._lag_repr()
            ma = arma2ma(ar_params, np.ones(1), lags=oos)
            mean_var[-oos:] = self.sigma2 * np.cumsum(ma ** 2)
        if isinstance(mean, pd.Series):
            mean_var = pd.Series(mean_var, index=mean.index)

        return PredictionResults(mean, mean_var)

    def forecast(self, steps=1, exog=None):
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : {int, str, datetime}, default 1
            If an integer, the number of steps to forecast from the end of the
            sample. Can also be a date string to parse or a datetime type.
            However, if the dates index does not have a fixed frequency,
            steps must be an integer.
        exog : {ndarray, DataFrame}
            Exogenous values to use out-of-sample. Must have same number of
            columns as original exog data and at least `steps` rows

        Returns
        -------
        array_like
            Array of out of in-sample predictions and / or out-of-sample
            forecasts.

        See Also
        --------
        AutoRegResults.predict
            In- and out-of-sample predictions
        AutoRegResults.get_prediction
            In- and out-of-sample predictions and confidence intervals
        """
        start = self.model.data.orig_endog.shape[0]
        if isinstance(steps, (int, np.integer)):
            end = start + steps - 1
        else:
            end = steps
        return self.predict(start=start, end=end, dynamic=False, exog_oos=exog)

    def _plot_predictions(
        self,
        predictions,
        start,
        end,
        alpha,
        in_sample,
        fig,
        figsize,
    ):
        """Shared helper for plotting predictions"""
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig

        _import_mpl()
        fig = create_mpl_fig(fig, figsize)
        start = 0 if start is None else start
        end = self.model._index[-1] if end is None else end
        _, _, oos, _ = self.model._get_prediction_index(start, end)

        ax = fig.add_subplot(111)
        mean = predictions.predicted_mean
        if not in_sample and oos:
            if isinstance(mean, pd.Series):
                mean = mean.iloc[-oos:]
        elif not in_sample:
            raise ValueError(
                "in_sample is False but there are no"
                "out-of-sample forecasts to plot."
            )
        ax.plot(mean, zorder=2)

        if oos and alpha is not None:
            ci = np.asarray(predictions.conf_int(alpha))
            lower, upper = ci[-oos:, 0], ci[-oos:, 1]
            label = "{0:.0%} confidence interval".format(1 - alpha)
            x = ax.get_lines()[-1].get_xdata()
            ax.fill_between(
                x[-oos:],
                lower,
                upper,
                color="gray",
                alpha=0.5,
                label=label,
                zorder=1,
            )
        ax.legend(loc="best")

        return fig

    @Substitution(predict_params=_predict_params)
    def plot_predict(
        self,
        start=None,
        end=None,
        dynamic=False,
        exog=None,
        exog_oos=None,
        alpha=0.05,
        in_sample=True,
        fig=None,
        figsize=None,
    ):
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
        predictions = self.get_prediction(
            start=start, end=end, dynamic=dynamic, exog=exog, exog_oos=exog_oos
        )
        return self._plot_predictions(
            predictions, start, end, alpha, in_sample, fig, figsize
        )

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
        if hasattr(self.model.data, "dates") and self.data.dates is not None:
            x = self.model.data.dates._mpl_repr()
            x = x[self.model.hold_back :]
        else:
            hold_back = self.model.hold_back
            x = hold_back + np.arange(self.resid.shape[0])
        std_resid = resid / np.sqrt(self.sigma2)
        ax.plot(x, std_resid)
        ax.hlines(0, x[0], x[-1], alpha=0.5)
        ax.set_xlim(x[0], x[-1])
        ax.set_title("Standardized residual")

        # Top-right: histogram, Gaussian kernel density, Normal density
        # Can only do histogram and Gaussian kernel density on the non-null
        # elements
        std_resid_nonmissing = std_resid[~(np.isnan(resid))]
        ax = fig.add_subplot(222)

        ax.hist(std_resid_nonmissing, density=True, label="Hist")

        kde = gaussian_kde(std_resid)
        xlim = (-1.96 * 2, 1.96 * 2)
        x = np.linspace(xlim[0], xlim[1])
        ax.plot(x, kde(x), label="KDE")
        ax.plot(x, norm.pdf(x), label="N(0,1)")
        ax.set_xlim(xlim)
        ax.legend()
        ax.set_title("Histogram plus estimated density")

        # Bottom-left: QQ plot
        ax = fig.add_subplot(223)
        from statsmodels.graphics.gofplots import qqplot

        qqplot(std_resid, line="s", ax=ax)
        ax.set_title("Normal Q-Q")

        # Bottom-right: Correlogram
        ax = fig.add_subplot(224)
        from statsmodels.graphics.tsaplots import plot_acf

        plot_acf(resid, ax=ax, lags=lags)
        ax.set_title("Correlogram")

        ax.set_ylim(-1, 1)

        return fig

    def summary(self, alpha=0.05):
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

        title = model.__class__.__name__ + " Model Results"
        method = "Conditional MLE"
        # get sample
        start = self._hold_back
        if self.data.dates is not None:
            dates = self.data.dates
            sample = [dates[start].strftime("%m-%d-%Y")]
            sample += ["- " + dates[-1].strftime("%m-%d-%Y")]
        else:
            sample = [str(start), str(len(self.data.orig_endog))]
        model = model.__class__.__name__
        if self.model.seasonal:
            model = "Seas. " + model
        if self.ar_lags is not None and len(self.ar_lags) < self._max_lag:
            model = "Restr. " + model
        if self.model.exog is not None:
            model += "-X"

        order = "({0})".format(self._max_lag)
        dep_name = str(self.model.endog_names)
        top_left = [
            ("Dep. Variable:", [dep_name]),
            ("Model:", [model + order]),
            ("Method:", [method]),
            ("Date:", None),
            ("Time:", None),
            ("Sample:", [sample[0]]),
            ("", [sample[1]]),
        ]

        top_right = [
            ("No. Observations:", [str(len(self.model.endog))]),
            ("Log Likelihood", ["%#5.3f" % self.llf]),
            ("S.D. of innovations", ["%#5.3f" % self.sigma2 ** 0.5]),
            ("AIC", ["%#5.3f" % self.aic]),
            ("BIC", ["%#5.3f" % self.bic]),
            ("HQIC", ["%#5.3f" % self.hqic]),
        ]

        smry = Summary()
        smry.add_table_2cols(
            self, gleft=top_left, gright=top_right, title=title
        )
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
            roots_table = SimpleTable(
                [
                    (
                        "%17.4f" % row[0],
                        "%+17.4fj" % row[1],
                        "%17.4f" % row[2],
                        "%17.4f" % row[3],
                    )
                    for row in data
                ],
                headers=[
                    "            Real",
                    "         Imaginary",
                    "         Modulus",
                    "        Frequency",
                ],
                title="Roots",
                stubs=stubs,
            )

            smry.tables.append(roots_table)
        return smry


class AutoRegResultsWrapper(wrap.ResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(
        tsa_model.TimeSeriesResultsWrapper._wrap_attrs, _attrs
    )
    _methods = {}
    _wrap_methods = wrap.union_dicts(
        tsa_model.TimeSeriesResultsWrapper._wrap_methods, _methods
    )


wrap.populate_wrapper(AutoRegResultsWrapper, AutoRegResults)

doc = Docstring(AutoReg.__doc__)
_auto_reg_params = doc.extract_parameters(
    [
        "trend",
        "seasonal",
        "exog",
        "hold_back",
        "period",
        "missing",
        "old_names",
    ],
    4,
)


@Substitution(auto_reg_params=_auto_reg_params)
def ar_select_order(
    endog,
    maxlag,
    ic="bic",
    glob=False,
    trend="c",
    seasonal=False,
    exog=None,
    hold_back=None,
    period=None,
    missing="none",
    old_names=False,
):
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
    full_mod = AutoReg(
        endog,
        maxlag,
        trend=trend,
        seasonal=seasonal,
        exog=exog,
        hold_back=hold_back,
        period=period,
        missing=missing,
        old_names=old_names,
    )
    nexog = full_mod.exog.shape[1] if full_mod.exog is not None else 0
    y, x = full_mod._y, full_mod._x
    base_col = x.shape[1] - nexog - maxlag
    sel = np.ones(x.shape[1], dtype=bool)
    ics = []

    def compute_ics(res):
        nobs = res.nobs
        df_model = res.df_model
        sigma2 = 1.0 / nobs * sumofsq(res.resid)
        llf = -nobs * (np.log(2 * np.pi * sigma2) + 1) / 2
        res = SimpleNamespace(
            nobs=nobs, df_model=df_model, sigma2=sigma2, llf=llf
        )

        aic = call_cached_func(AutoRegResults.aic, res)
        bic = call_cached_func(AutoRegResults.bic, res)
        hqic = call_cached_func(AutoRegResults.hqic, res)

        return aic, bic, hqic

    def ic_no_data():
        """Fake mod and results to handle no regressor case"""
        mod = SimpleNamespace(
            nobs=y.shape[0], endog=y, exog=np.empty((y.shape[0], 0))
        )
        llf = OLS.loglike(mod, np.empty(0))
        res = SimpleNamespace(
            resid=y, nobs=y.shape[0], llf=llf, df_model=0, k_constant=0
        )

        return compute_ics(res)

    if not glob:
        sel[base_col : base_col + maxlag] = False
        for i in range(maxlag + 1):
            sel[base_col : base_col + i] = True
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
            bits[:, 8 * i : 8 * (i + 1)] = bits[:, 8 * i : 8 * (i + 1)][
                :, ::-1
            ]
        masks = bits[:, :maxlag]
        for mask in masks:
            sel[base_col : base_col + maxlag] = mask
            if not np.any(sel):
                ics.append((0, ic_no_data()))
                continue
            res = OLS(y, x[:, sel]).fit()
            lags = tuple(np.where(mask)[0] + 1)
            lags = 0 if not lags else lags
            ics.append((lags, compute_ics(res)))

    key_loc = {"aic": 0, "bic": 1, "hqic": 2}[ic]
    ics = sorted(ics, key=lambda x: x[1][key_loc])
    selected_model = ics[0][0]
    mod = AutoReg(
        endog,
        selected_model,
        trend=trend,
        seasonal=seasonal,
        exog=exog,
        hold_back=hold_back,
        period=period,
        missing=missing,
        old_names=old_names,
    )
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
