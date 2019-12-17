"""
ARIMA model class.

Author: Chad Fulton
License: BSD-3
"""
from statsmodels.compat.pandas import Appender

import warnings

from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.statespace.kalman_filter import MEMORY_CONSERVE
from statsmodels.tsa.statespace.tools import diff
import statsmodels.base.wrapper as wrap

from statsmodels.tsa.arima.estimators.yule_walker import yule_walker
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
    innovations, innovations_mle)
from statsmodels.tsa.arima.estimators.gls import gls as estimate_gls

from statsmodels.tsa.arima.specification import SARIMAXSpecification


class ARIMA(sarimax.SARIMAX):
    """
    Autoregressive Integrated Moving Average (ARIMA) model, and extensions

    This model is the basic interface for ARIMA-type models, including those
    with exogenous regressors and those with seasonal components. The most
    general form of the model is SARIMAX(p, d, q)x(P, D, Q, s). It also allows
    all specialized cases, including

    - autoregressive models: AR(p)
    - moving average models: MA(q)
    - mixed autoregressive moving average models: ARMA(p, q)
    - integration models: ARIMA(p, d, q)
    - seasonal models: SARIMA(P, D, Q, s)
    - regression with errors that follow one of the above ARIMA-type models

    Parameters
    ----------
    endog : array_like, optional
        The observed time-series process :math:`y`.
    exog : array_like, optional
        Array of exogenous regressors.
    order : tuple, optional
        The (p,d,q) order of the model for the autoregressive, differences, and
        moving average components. d is always an integer, while p and q may
        either be integers or lists of integers.
    seasonal_order : tuple, optional
        The (P,D,Q,s) order of the seasonal component of the model for the
        AR parameters, differences, MA parameters, and periodicity. Default
        is (0, 0, 0, 0). D and s are always integers, while P and Q
        may either be integers or lists of positive integers.
    trend : str{'n','c','t','ct'} or iterable, optional
        Parameter controlling the deterministic trend. Can be specified as a
        string where 'c' indicates a constant term, 't' indicates a
        linear trend in time, and 'ct' includes both. Can also be specified as
        an iterable defining a polynomial, as in `numpy.poly1d`, where
        `[1,1,0,1]` would denote :math:`a + bt + ct^3`. Default is 'c' for
        models without integration, and no trend for models with integration.
    enforce_stationarity : bool, optional
        Whether or not to require the autoregressive parameters to correspond
        to a stationarity process.
    enforce_invertibility : bool, optional
        Whether or not to require the moving average parameters to correspond
        to an invertible process.
    concentrate_scale : bool, optional
        Whether or not to concentrate the scale (variance of the error term)
        out of the likelihood. This reduces the number of parameters by one.
        This is only applicable when considering estimation by numerical
        maximum likelihood.
    trend_offset : int, optional
        The offset at which to start time trend values. Default is 1, so that
        if `trend='t'` the trend is equal to 1, 2, ..., nobs. Typically is only
        set when the model created by extending a previous dataset.
    dates : array-like of datetime, optional
        If no index is given by `endog` or `exog`, an array-like object of
        datetime objects can be provided.
    freq : str, optional
        If no index is given by `endog` or `exog`, the frequency of the
        time-series may be specified here as a Pandas offset or offset string.
    missing : str
        Available options are 'none', 'drop', and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default is 'none'.

    Notes
    -----
    This model incorporates both exogenous regressors and trend components
    through "regression with ARIMA errors".

    `enforce_stationarity` and `enforce_invertibility` are specified in the
    constructor because they affect loglikelihood computations, and so should
    not be changed on the fly. This is why they are not instead included as
    arguments to the `fit` method.

    TODO: should we use concentrate_scale=True by default?

    Examples
    --------
    >>> mod = sm.tsa.arima.ARIMA(endog, order=(1, 0, 0))
    >>> res = mod.fit()
    >>> print(res.summary())
    """
    def __init__(self, endog, exog=None, order=(0, 0, 0),
                 seasonal_order=(0, 0, 0, 0), trend=None,
                 enforce_stationarity=True, enforce_invertibility=True,
                 concentrate_scale=False, trend_offset=1, dates=None,
                 freq=None, missing='none'):
        # Default for trend
        # 'c' if there is no integration and 'n' otherwise
        # TODO: if trend='c', then we could alternatively use `demean=True` in
        # the estimation methods rather than setting up `exog` and using GLS.
        # Not sure if it's worth the trouble though.
        integrated = order[1] > 0 or seasonal_order[1] > 0
        if trend is None and not integrated:
            trend = 'c'
        elif trend is None:
            trend = 'n'

        # Construct the specification
        # (don't pass specific values of enforce stationarity/invertibility,
        # because we don't actually want to restrict the estimators based on
        # this criteria. Instead, we'll just make sure that the parameter
        # estimates from those methods satisfy the criteria.)
        self._spec_arima = SARIMAXSpecification(
            endog, exog=exog, order=order, seasonal_order=seasonal_order,
            trend=trend, enforce_stationarity=None, enforce_invertibility=None,
            concentrate_scale=concentrate_scale, trend_offset=trend_offset,
            dates=dates, freq=freq, missing=missing)
        exog = self._spec_arima._model.data.orig_exog

        # Initialize the base SARIMAX class
        # Note: we don't pass in a trend value to the base class, since ARIMA
        # standardizes the trend to always be part of exog, while the base
        # SARIMAX class puts it in the transition equation.
        super(ARIMA, self).__init__(
            endog, exog, trend=None, order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            concentrate_scale=concentrate_scale, dates=dates, freq=freq,
            missing=missing)
        self.trend = trend

        # Override the public attributes for k_exog and k_trend to reflect the
        # distinction here (for the purpose of the superclass, these are both
        # combined as `k_exog`)
        self.k_exog = self._spec_arima.k_exog
        self.k_trend = self._spec_arima.k_trend

        # Remove some init kwargs that aren't used in this model
        unused = ['measurement_error', 'time_varying_regression',
                  'mle_regression', 'simple_differencing',
                  'hamilton_representation']
        self._init_keys = [key for key in self._init_keys if key not in unused]

    @property
    def _res_classes(self):
        return {'fit': (ARIMAResults, ARIMAResultsWrapper)}

    def fit(self, start_params=None, transformed=True, includes_fixed=False,
            method=None, method_kwargs=None, gls=None, gls_kwargs=None,
            cov_type=None, cov_kwds=None, return_params=False,
            low_memory=False):
        """
        Fit (estimate) the parameters of the model.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            If None, the default is given by Model.start_params.
        transformed : bool, optional
            Whether or not `start_params` is already transformed. Default is
            True.
        includes_fixed : bool, optional
            If parameters were previously fixed with the `fix_params` method,
            this argument describes whether or not `start_params` also includes
            the fixed parameters, in addition to the free parameters. Default
            is False.
        method : str, optional
            The method used for estimating the parameters of the model. Valid
            options include 'statespace', 'innovations_mle', 'hannan_rissanen',
            'burg', 'innovations', and 'yule_walker'. Not all options are
            available for every specification (for example 'yule_walker' can
            only be used with AR(p) models).
        method_kwargs : dict, optional
            Arguments to pass to the fit function for the parameter estimator
            described by the `method` argument.
        gls : bool, optional
            Whether or not to use generalized least squares (GLS) to estimate
            regression effects. The default is False if `method='statespace'`
            and is True otherwise.
        gls_kwargs : dict, optional
            Arguments to pass to the GLS estimation fit method. Only applicable
            if GLS estimation is used (see `gls` argument for details).
        cov_type : str, optional
            The `cov_type` keyword governs the method for calculating the
            covariance matrix of parameter estimates. Can be one of:

            - 'opg' for the outer product of gradient estimator
            - 'oim' for the observed information matrix estimator, calculated
              using the method of Harvey (1989)
            - 'approx' for the observed information matrix estimator,
              calculated using a numerical approximation of the Hessian matrix.
            - 'robust' for an approximate (quasi-maximum likelihood) covariance
              matrix that may be valid even in the presence of some
              misspecifications. Intermediate calculations use the 'oim'
              method.
            - 'robust_approx' is the same as 'robust' except that the
              intermediate calculations use the 'approx' method.
            - 'none' for no covariance matrix calculation.

            Default is 'opg' unless memory conservation is used to avoid
            computing the loglikelihood values for each observation, in which
            case the default is 'oim'.
        cov_kwds : dict or None, optional
            A dictionary of arguments affecting covariance matrix computation.

            **opg, oim, approx, robust, robust_approx**

            - 'approx_complex_step' : bool, optional - If True, numerical
              approximations are computed using complex-step methods. If False,
              numerical approximations are computed using finite difference
              methods. Default is True.
            - 'approx_centered' : bool, optional - If True, numerical
              approximations computed using finite difference methods use a
              centered approximation. Default is False.
        return_params : bool, optional
            Whether or not to return only the array of maximizing parameters.
            Default is False.
        low_memory : bool, optional
            If set to True, techniques are applied to substantially reduce
            memory usage. If used, some features of the results object will
            not be available (including smoothed results and in-sample
            prediction), although out-of-sample forecasting is possible.
            Default is False.

        Returns
        -------
        ARIMAResults

        Examples
        --------
        >>> mod = sm.tsa.arima.ARIMA(endog, order=(1, 0, 0))
        >>> res = mod.fit()
        >>> print(res.summary())
        """
        # Determine which method to use
        # 1. If method is specified, make sure it is valid
        if method is not None:
            self._spec_arima.validate_estimator(method)
        # 2. Otherwise, use state space
        # TODO: may want to consider using innovations (MLE) if possible here,
        # (since in some cases it may be faster than state space), but it is
        # less tested.
        else:
            method = 'statespace'

        # Can only use fixed parameters with method='statespace'
        if self._has_fixed_params and method != 'statespace':
            raise ValueError('When parameters have been fixed, only the method'
                             ' "statespace" can be used; got "%s".' % method)

        # Handle kwargs related to the fit method
        if method_kwargs is None:
            method_kwargs = {}
        required_kwargs = []
        if method == 'statespace':
            required_kwargs = ['enforce_stationarity', 'enforce_invertibility',
                               'concentrate_scale']
        elif method == 'innovations_mle':
            required_kwargs = ['enforce_invertibility']
        for name in required_kwargs:
            if name in method_kwargs:
                raise ValueError('Cannot override model level value for "%s"'
                                 ' when method="%s".' % (name, method))
            method_kwargs[name] = getattr(self, name)

        # Handle kwargs related to GLS estimation
        if gls_kwargs is None:
            gls_kwargs = {}

        # Handle starting parameters
        # TODO: maybe should have standard way of computing starting
        # parameters in this class?
        if start_params is not None:
            if method not in ['statespace', 'innovations_mle']:
                raise ValueError('Estimation method "%s" does not use starting'
                                 ' parameters, but `start_params` argument was'
                                 ' given.' % method)

            method_kwargs['start_params'] = start_params
            method_kwargs['transformed'] = transformed
            method_kwargs['includes_fixed'] = includes_fixed

        # Perform estimation, depending on whether we have exog or not
        p = None
        fit_details = None
        has_exog = self._spec_arima.exog is not None
        if has_exog or method == 'statespace':
            # Use GLS if it was explicitly requested (`gls = True`) or if it
            # was left at the default (`gls = None`) and the ARMA estimator is
            # anything but statespace.
            # Note: both GLS and statespace are able to handle models with
            # integration, so we don't need to difference endog or exog here.
            if has_exog and (gls or (gls is None and method != 'statespace')):
                p, fit_details = estimate_gls(
                    self.endog, exog=self.exog, order=self.order,
                    seasonal_order=self.seasonal_order, include_constant=False,
                    arma_estimator=method, arma_estimator_kwargs=method_kwargs,
                    **gls_kwargs)
            elif method != 'statespace':
                raise ValueError('If `exog` is given and GLS is disabled'
                                 ' (`gls=False`), then the only valid'
                                 " method is 'statespace'. Got '%s'."
                                 % method)
            else:
                method_kwargs.setdefault('disp', 0)

                res = super(ARIMA, self).fit(return_params=return_params,
                                             low_memory=low_memory,
                                             **method_kwargs)
                if not return_params:
                    res.fit_details = res.mlefit
        else:
            # Handle differencing if we have an integrated model
            # (these methods do not support handling integration internally,
            # so we need to manually do the differencing)
            endog = self.endog
            order = self._spec_arima.order
            seasonal_order = self._spec_arima.seasonal_order
            if self._spec_arima.is_integrated:
                warnings.warn('Provided `endog` series has been differenced'
                              ' to eliminate integration prior to parameter'
                              ' estimation by method "%s".' % method)
                endog = diff(
                    endog, k_diff=self._spec_arima.diff,
                    k_seasonal_diff=self._spec_arima.seasonal_diff,
                    seasonal_periods=self._spec_arima.seasonal_periods)
                if order[1] > 0:
                    order = (order[0], 0, order[2])
                if seasonal_order[1] > 0:
                    seasonal_order = (seasonal_order[0], 0, seasonal_order[2],
                                      seasonal_order[3])

            # Now, estimate parameters
            if method == 'yule_walker':
                p, fit_details = yule_walker(
                    endog, ar_order=order[0], demean=False,
                    **method_kwargs)
            elif method == 'burg':
                p, fit_details = burg(endog, ar_order=order[0],
                                      demean=False, **method_kwargs)
            elif method == 'hannan_rissanen':
                p, fit_details = hannan_rissanen(
                    endog, ar_order=order[0],
                    ma_order=order[2], demean=False, **method_kwargs)
            elif method == 'innovations':
                p, fit_details = innovations(
                    endog, ma_order=order[2], demean=False,
                    **method_kwargs)
                # innovations computes estimates through the given order, so
                # we want to take the estimate associated with the given order
                p = p[-1]
            elif method == 'innovations_mle':
                p, fit_details = innovations_mle(
                    endog, order=order,
                    seasonal_order=seasonal_order,
                    demean=False, **method_kwargs)

        # In all cases except method='statespace', we now need to extract the
        # parameters and, optionally, create a new results object
        if p is not None:
            # Need to check that fitted parameters satisfy given restrictions
            if (self.enforce_stationarity
                    and self._spec_arima.max_reduced_ar_order > 0
                    and not p.is_stationary):
                raise ValueError('Non-stationary autoregressive parameters'
                                 ' found with `enforce_stationarity=True`.'
                                 ' Consider setting it to False or using a'
                                 ' different estimation method, such as'
                                 ' method="statespace".')

            if (self.enforce_invertibility
                    and self._spec_arima.max_reduced_ma_order > 0
                    and not p.is_invertible):
                raise ValueError('Non-invertible moving average parameters'
                                 ' found with `enforce_invertibility=True`.'
                                 ' Consider setting it to False or using a'
                                 ' different estimation method, such as'
                                 ' method="statespace".')

            # Build the requested results
            if return_params:
                res = p.params
            else:
                # Handle memory conservation option
                if low_memory:
                    conserve_memory = self.ssm.conserve_memory
                    self.ssm.set_conserve_memory(MEMORY_CONSERVE)

                # Perform filtering / smoothing
                if (self.ssm.memory_no_predicted or self.ssm.memory_no_gain
                        or self.ssm.memory_no_smoothing):
                    func = self.filter
                else:
                    func = self.smooth
                res = func(p.params, transformed=True, includes_fixed=True,
                           cov_type=cov_type, cov_kwds=cov_kwds)

                # Save any details from the fit method
                res.fit_details = fit_details

                # Reset memory conservation
                if low_memory:
                    self.ssm.set_conserve_memory(conserve_memory)

        return res


@Appender(sarimax.SARIMAXResults.__doc__)
class ARIMAResults(sarimax.SARIMAXResults):
    pass


class ARIMAResultsWrapper(sarimax.SARIMAXResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(
        sarimax.SARIMAXResultsWrapper._wrap_attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(
        sarimax.SARIMAXResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(ARIMAResultsWrapper, ARIMAResults)  # noqa:E305
