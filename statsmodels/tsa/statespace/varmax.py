# -*- coding: utf-8 -*-
"""
Vector Autoregressive Moving Average with eXogenous regressors model

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

from warnings import warn
from collections import OrderedDict

import pandas as pd
import numpy as np

from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.vector_ar import var_model
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import EstimationWarning, ValueWarning

from .kalman_filter import INVERT_UNIVARIATE, SOLVE_LU
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .tools import (
    is_invertible, prepare_exog,
    constrain_stationary_multivariate, unconstrain_stationary_multivariate,
    prepare_trend_spec, prepare_trend_data
)


class VARMAX(MLEModel):
    r"""
    Vector Autoregressive Moving Average with eXogenous regressors model

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`, , shaped nobs x k_endog.
    exog : array_like, optional
        Array of exogenous regressors, shaped nobs x k.
    order : iterable
        The (p,q) order of the model for the number of AR and MA parameters to
        use.
    trend : str{'n','c','t','ct'} or iterable, optional
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the polynomial as in `numpy.poly1d`, where
        `[1,1,0,1]` would denote :math:`a + bt + ct^3`. Default is a constant
        trend component.
    error_cov_type : {'diagonal', 'unstructured'}, optional
        The structure of the covariance matrix of the error term, where
        "unstructured" puts no restrictions on the matrix and "diagonal"
        requires it to be a diagonal matrix (uncorrelated errors). Default is
        "unstructured".
    measurement_error : boolean, optional
        Whether or not to assume the endogenous observations `endog` were
        measured with error. Default is False.
    enforce_stationarity : boolean, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.
    enforce_invertibility : boolean, optional
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model. Default is True.
    kwargs
        Keyword arguments may be used to provide default values for state space
        matrices or for Kalman filtering options. See `Representation`, and
        `KalmanFilter` for more details.

    Attributes
    ----------
    order : iterable
        The (p,q) order of the model for the number of AR and MA parameters to
        use.
    trend : str{'n','c','t','ct'} or iterable
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the polynomial as in `numpy.poly1d`, where
        `[1,1,0,1]` would denote :math:`a + bt + ct^3`.
    error_cov_type : {'diagonal', 'unstructured'}, optional
        The structure of the covariance matrix of the error term, where
        "unstructured" puts no restrictions on the matrix and "diagonal"
        requires it to be a diagonal matrix (uncorrelated errors). Default is
        "unstructured".
    measurement_error : boolean, optional
        Whether or not to assume the endogenous observations `endog` were
        measured with error. Default is False.
    enforce_stationarity : boolean, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.
    enforce_invertibility : boolean, optional
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model. Default is True.

    Notes
    -----
    Generically, the VARMAX model is specified (see for example chapter 18 of
    [1]_):

    .. math::

        y_t = A(t) + A_1 y_{t-1} + \dots + A_p y_{t-p} + B x_t + \epsilon_t +
        M_1 \epsilon_{t-1} + \dots M_q \epsilon_{t-q}

    where :math:`\epsilon_t \sim N(0, \Omega)`, and where :math:`y_t` is a
    `k_endog x 1` vector. Additionally, this model allows considering the case
    where the variables are measured with error.

    Note that in the full VARMA(p,q) case there is a fundamental identification
    problem in that the coefficient matrices :math:`\{A_i, M_j\}` are not
    generally unique, meaning that for a given time series process there may
    be multiple sets of matrices that equivalently represent it. See Chapter 12
    of [1]_ for more information. Although this class can be used to estimate
    VARMA(p,q) models, a warning is issued to remind users that no steps have
    been taken to ensure identification in this case.

    References
    ----------
    .. [1] Lütkepohl, Helmut. 2007.
       New Introduction to Multiple Time Series Analysis.
       Berlin: Springer.

    """

    def __init__(self, endog, exog=None, order=(1, 0), trend='c',
                 error_cov_type='unstructured', measurement_error=False,
                 enforce_stationarity=True, enforce_invertibility=True,
                 **kwargs):

        # Model parameters
        self.error_cov_type = error_cov_type
        self.measurement_error = measurement_error
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

        # Save the given orders
        self.order = order
        self.trend = trend

        # Model orders
        self.k_ar = int(order[0])
        self.k_ma = int(order[1])

        # Check for valid model
        if error_cov_type not in ['diagonal', 'unstructured']:
            raise ValueError('Invalid error covariance matrix type'
                             ' specification.')
        if self.k_ar == 0 and self.k_ma == 0:
            raise ValueError('Invalid VARMAX(p,q) specification; at least one'
                             ' p,q must be greater than zero.')

        # Warn for VARMA model
        if self.k_ar > 0 and self.k_ma > 0:
            warn('Estimation of VARMA(p,q) models is not generically robust,'
                 ' due especially to identification issues.',
                 EstimationWarning)

        # Trend
        self.trend = trend
        self.polynomial_trend, self.k_trend = prepare_trend_spec(self.trend)
        self._trend_is_const = (self.polynomial_trend.size == 1 and
                                self.polynomial_trend[0] == 1)

        # Exogenous data
        (self.k_exog, exog) = prepare_exog(exog)

        # Note: at some point in the future might add state regression, as in
        # SARIMAX.
        self.mle_regression = self.k_exog > 0

        # We need to have an array or pandas at this point
        if not _is_using_pandas(endog, None):
            endog = np.asanyarray(endog)

        # Model order
        # Used internally in various places
        _min_k_ar = max(self.k_ar, 1)
        self._k_order = _min_k_ar + self.k_ma

        # Number of states
        k_endog = endog.shape[1]
        k_posdef = k_endog
        k_states = k_endog * self._k_order

        # By default, initialize as stationary
        kwargs.setdefault('initialization', 'stationary')

        # By default, use LU decomposition
        kwargs.setdefault('inversion_method', INVERT_UNIVARIATE | SOLVE_LU)

        # Initialize the state space model
        super(VARMAX, self).__init__(
            endog, exog=exog, k_states=k_states, k_posdef=k_posdef, **kwargs
        )

        # Set as time-varying model if we have time-trend or exog
        if self.k_exog > 0 or (self.k_trend > 0 and not self._trend_is_const):
            self.ssm._time_invariant = False

        # Initialize the parameters
        self.parameters = OrderedDict()
        self.parameters['trend'] = self.k_endog * self.k_trend
        self.parameters['ar'] = self.k_endog**2 * self.k_ar
        self.parameters['ma'] = self.k_endog**2 * self.k_ma
        self.parameters['regression'] = self.k_endog * self.k_exog
        if self.error_cov_type == 'diagonal':
            self.parameters['state_cov'] = self.k_endog
        # These parameters fill in a lower-triangular matrix which is then
        # dotted with itself to get a positive definite matrix.
        elif self.error_cov_type == 'unstructured':
            self.parameters['state_cov'] = (
                int(self.k_endog * (self.k_endog + 1) / 2)
            )
        self.parameters['obs_cov'] = self.k_endog * self.measurement_error
        self.k_params = sum(self.parameters.values())

        # Initialize trend data
        self._trend_data = prepare_trend_data(
            self.polynomial_trend, self.k_trend, self.nobs, offset=1)

        # Initialize known elements of the state space matrices

        # If we have exog effects, then the state intercept needs to be
        # time-varying
        if (self.k_trend > 0 and not self._trend_is_const) or self.k_exog > 0:
            self.ssm['state_intercept'] = np.zeros((self.k_states, self.nobs))
            # self.ssm['obs_intercept'] = np.zeros((self.k_endog, self.nobs))

        # The design matrix is just an identity for the first k_endog states
        idx = np.diag_indices(self.k_endog)
        self.ssm[('design',) + idx] = 1

        # The transition matrix is described in four blocks, where the upper
        # left block is in companion form with the autoregressive coefficient
        # matrices (so it is shaped k_endog * k_ar x k_endog * k_ar) ...
        if self.k_ar > 0:
            idx = np.diag_indices((self.k_ar - 1) * self.k_endog)
            idx = idx[0] + self.k_endog, idx[1]
            self.ssm[('transition',) + idx] = 1
        # ... and the  lower right block is in companion form with zeros as the
        # coefficient matrices (it is shaped k_endog * k_ma x k_endog * k_ma).
        idx = np.diag_indices((self.k_ma - 1) * self.k_endog)
        idx = (idx[0] + (_min_k_ar + 1) * self.k_endog,
               idx[1] + _min_k_ar * self.k_endog)
        self.ssm[('transition',) + idx] = 1

        # The selection matrix is described in two blocks, where the upper
        # block selects the all k_posdef errors in the first k_endog rows
        # (the upper block is shaped k_endog * k_ar x k) and the lower block
        # also selects all k_posdef errors in the first k_endog rows (the lower
        # block is shaped k_endog * k_ma x k).
        idx = np.diag_indices(self.k_endog)
        self.ssm[('selection',) + idx] = 1
        idx = idx[0] + _min_k_ar * self.k_endog, idx[1]
        if self.k_ma > 0:
            self.ssm[('selection',) + idx] = 1

        # Cache some indices
        if self._trend_is_const and self.k_exog == 0:
            self._idx_state_intercept = np.s_['state_intercept', :k_endog, :]
        elif self.k_trend > 0 or self.k_exog > 0:
            self._idx_state_intercept = np.s_['state_intercept', :k_endog, :-1]
        if self.k_ar > 0:
            self._idx_transition = np.s_['transition', :k_endog, :]
        else:
            self._idx_transition = np.s_['transition', :k_endog, k_endog:]
        if self.error_cov_type == 'diagonal':
            self._idx_state_cov = (
                ('state_cov',) + np.diag_indices(self.k_endog))
        elif self.error_cov_type == 'unstructured':
            self._idx_lower_state_cov = np.tril_indices(self.k_endog)
        if self.measurement_error:
            self._idx_obs_cov = ('obs_cov',) + np.diag_indices(self.k_endog)

        # Cache some slices
        def _slice(key, offset):
            length = self.parameters[key]
            param_slice = np.s_[offset:offset + length]
            offset += length
            return param_slice, offset

        offset = 0
        self._params_trend, offset = _slice('trend', offset)
        self._params_ar, offset = _slice('ar', offset)
        self._params_ma, offset = _slice('ma', offset)
        self._params_regression, offset = _slice('regression', offset)
        self._params_state_cov, offset = _slice('state_cov', offset)
        self._params_obs_cov, offset = _slice('obs_cov', offset)

    @property
    def _res_classes(self):
        return {'fit': (VARMAXResults, VARMAXResultsWrapper)}

    @property
    def start_params(self):
        params = np.zeros(self.k_params, dtype=np.float64)

        # A. Run a multivariate regression to get beta estimates
        endog = pd.DataFrame(self.endog.copy())
        # Pandas < 0.13 didn't support the same type of DataFrame interpolation
        # TODO remove this now that we have dropped support for Pandas < 0.13
        try:
            endog = endog.interpolate()
        except TypeError:
            pass
        endog = endog.fillna(method='backfill').values
        exog = None
        if self.k_trend > 0 and self.k_exog > 0:
            exog = np.c_[self._trend_data, self.exog]
        elif self.k_trend > 0:
            exog = self._trend_data
        elif self.k_exog > 0:
            exog = self.exog

        # Although the Kalman filter can deal with missing values in endog,
        # conditional sum of squares cannot
        if np.any(np.isnan(endog)):
            mask = ~np.any(np.isnan(endog), axis=1)
            endog = endog[mask]
            if exog is not None:
                exog = exog[mask]

        # Regression and trend effects via OLS
        trend_params = np.zeros(0)
        exog_params = np.zeros(0)
        if self.k_trend > 0 or self.k_exog > 0:
            trendexog_params = np.linalg.pinv(exog).dot(endog)
            endog -= np.dot(exog, trendexog_params)
            if self.k_trend > 0:
                trend_params = trendexog_params[:self.k_trend].T
            if self.k_endog > 0:
                exog_params = trendexog_params[self.k_trend:].T

        # B. Run a VAR model on endog to get trend, AR parameters
        ar_params = []
        k_ar = self.k_ar if self.k_ar > 0 else 1
        mod_ar = var_model.VAR(endog)
        res_ar = mod_ar.fit(maxlags=k_ar, ic=None, trend='nc')
        if self.k_ar > 0:
            ar_params = np.array(res_ar.params).T.ravel()
        endog = res_ar.resid

        # Test for stationarity
        if self.k_ar > 0 and self.enforce_stationarity:
            coefficient_matrices = (
                ar_params.reshape(
                    self.k_endog * self.k_ar, self.k_endog
                ).T
            ).reshape(self.k_endog, self.k_endog, self.k_ar).T

            stationary = is_invertible([1] + list(-coefficient_matrices))

            if not stationary:
                warn('Non-stationary starting autoregressive parameters'
                     ' found. Using zeros as starting parameters.')
                ar_params *= 0

        # C. Run a VAR model on the residuals to get MA parameters
        ma_params = []
        if self.k_ma > 0:
            mod_ma = var_model.VAR(endog)
            res_ma = mod_ma.fit(maxlags=self.k_ma, ic=None, trend='nc')
            ma_params = np.array(res_ma.params.T).ravel()

            # Test for invertibility
            if self.enforce_invertibility:
                coefficient_matrices = (
                    ma_params.reshape(
                        self.k_endog * self.k_ma, self.k_endog
                    ).T
                ).reshape(self.k_endog, self.k_endog, self.k_ma).T

                invertible = is_invertible([1] + list(-coefficient_matrices))

                if not invertible:
                    warn('Non-stationary starting moving-average parameters'
                         ' found. Using zeros as starting parameters.')
                    ma_params *= 0

        # Transform trend / exog params from mean form to intercept form
        if self.k_ar > 0 and self.k_trend > 0 or self.mle_regression:
            coefficient_matrices = (
                ar_params.reshape(
                    self.k_endog * self.k_ar, self.k_endog
                ).T
            ).reshape(self.k_endog, self.k_endog, self.k_ar).T

            tmp = np.eye(self.k_endog) - np.sum(coefficient_matrices, axis=0)

            if self.k_trend > 0:
                trend_params = np.dot(tmp, trend_params)
            if self.mle_regression > 0:
                exog_params = np.dot(tmp, exog_params)

        # 1. Intercept terms
        if self.k_trend > 0:
            params[self._params_trend] = trend_params.ravel()

        # 2. AR terms
        if self.k_ar > 0:
            params[self._params_ar] = ar_params

        # 3. MA terms
        if self.k_ma > 0:
            params[self._params_ma] = ma_params

        # 4. Regression terms
        if self.mle_regression:
            params[self._params_regression] = exog_params.ravel()

        # 5. State covariance terms
        if self.error_cov_type == 'diagonal':
            params[self._params_state_cov] = res_ar.sigma_u.diagonal()
        elif self.error_cov_type == 'unstructured':
            cov_factor = np.linalg.cholesky(res_ar.sigma_u)
            params[self._params_state_cov] = (
                cov_factor[self._idx_lower_state_cov].ravel())

        # 5. Measurement error variance terms
        if self.measurement_error:
            if self.k_ma > 0:
                params[self._params_obs_cov] = res_ma.sigma_u.diagonal()
            else:
                params[self._params_obs_cov] = res_ar.sigma_u.diagonal()

        return params

    @property
    def param_names(self):
        param_names = []

        # 1. Intercept terms
        if self.k_trend > 0:
            for i in self.polynomial_trend.nonzero()[0]:
                if i == 0:
                    param_names += ['intercept.%s' % self.endog_names[j]
                                    for j in range(self.k_endog)]
                elif i == 1:
                    param_names += ['drift.%s' % self.endog_names[j]
                                    for j in range(self.k_endog)]
                else:
                    param_names += ['trend.%d.%s' % (i, self.endog_names[j])
                                    for j in range(self.k_endog)]

        # 2. AR terms
        param_names += [
            'L%d.%s.%s' % (i+1, self.endog_names[k], self.endog_names[j])
            for j in range(self.k_endog)
            for i in range(self.k_ar)
            for k in range(self.k_endog)
        ]

        # 3. MA terms
        param_names += [
            'L%d.e(%s).%s' % (i+1, self.endog_names[k], self.endog_names[j])
            for j in range(self.k_endog)
            for i in range(self.k_ma)
            for k in range(self.k_endog)
        ]

        # 4. Regression terms
        param_names += [
            'beta.%s.%s' % (self.exog_names[j], self.endog_names[i])
            for i in range(self.k_endog)
            for j in range(self.k_exog)
        ]

        # 5. State covariance terms
        if self.error_cov_type == 'diagonal':
            param_names += [
                'sigma2.%s' % self.endog_names[i]
                for i in range(self.k_endog)
            ]
        elif self.error_cov_type == 'unstructured':
            param_names += [
                ('sqrt.var.%s' % self.endog_names[i] if i == j else
                 'sqrt.cov.%s.%s' % (self.endog_names[j], self.endog_names[i]))
                for i in range(self.k_endog)
                for j in range(i+1)
            ]

        # 5. Measurement error variance terms
        if self.measurement_error:
            param_names += [
                'measurement_variance.%s' % self.endog_names[i]
                for i in range(self.k_endog)
            ]

        return param_names

    def transform_params(self, unconstrained):
        """
        Transform unconstrained parameters used by the optimizer to constrained
        parameters used in likelihood evaluation

        Parameters
        ----------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer, to be
            transformed.

        Returns
        -------
        constrained : array_like
            Array of constrained parameters which may be used in likelihood
            evalation.

        Notes
        -----
        Constrains the factor transition to be stationary and variances to be
        positive.
        """
        unconstrained = np.array(unconstrained, ndmin=1)
        constrained = np.zeros(unconstrained.shape, dtype=unconstrained.dtype)

        # 1. Intercept terms: nothing to do
        constrained[self._params_trend] = unconstrained[self._params_trend]

        # 2. AR terms: optionally force to be stationary
        if self.k_ar > 0 and self.enforce_stationarity:
            # Create the state covariance matrix
            if self.error_cov_type == 'diagonal':
                state_cov = np.diag(unconstrained[self._params_state_cov]**2)
            elif self.error_cov_type == 'unstructured':
                state_cov_lower = np.zeros(self.ssm['state_cov'].shape,
                                           dtype=unconstrained.dtype)
                state_cov_lower[self._idx_lower_state_cov] = (
                    unconstrained[self._params_state_cov])
                state_cov = np.dot(state_cov_lower, state_cov_lower.T)

            # Transform the parameters
            coefficients = unconstrained[self._params_ar].reshape(
                self.k_endog, self.k_endog * self.k_ar)
            coefficient_matrices, variance = (
                constrain_stationary_multivariate(coefficients, state_cov))
            constrained[self._params_ar] = coefficient_matrices.ravel()
        else:
            constrained[self._params_ar] = unconstrained[self._params_ar]

        # 3. MA terms: optionally force to be invertible
        if self.k_ma > 0 and self.enforce_invertibility:
            # Transform the parameters, using an identity variance matrix
            state_cov = np.eye(self.k_endog, dtype=unconstrained.dtype)
            coefficients = unconstrained[self._params_ma].reshape(
                self.k_endog, self.k_endog * self.k_ma)
            coefficient_matrices, variance = (
                constrain_stationary_multivariate(coefficients, state_cov))
            constrained[self._params_ma] = coefficient_matrices.ravel()
        else:
            constrained[self._params_ma] = unconstrained[self._params_ma]

        # 4. Regression terms: nothing to do
        constrained[self._params_regression] = (
            unconstrained[self._params_regression])

        # 5. State covariance terms
        # If we have variances, force them to be positive
        if self.error_cov_type == 'diagonal':
            constrained[self._params_state_cov] = (
                unconstrained[self._params_state_cov]**2)
        # Otherwise, nothing needs to be done
        elif self.error_cov_type == 'unstructured':
            constrained[self._params_state_cov] = (
                unconstrained[self._params_state_cov])

        # 5. Measurement error variance terms
        if self.measurement_error:
            # Force these to be positive
            constrained[self._params_obs_cov] = (
                unconstrained[self._params_obs_cov]**2)

        return constrained

    def untransform_params(self, constrained):
        """
        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer.

        Parameters
        ----------
        constrained : array_like
            Array of constrained parameters used in likelihood evalution, to be
            transformed.

        Returns
        -------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer.
        """
        constrained = np.array(constrained, ndmin=1)
        unconstrained = np.zeros(constrained.shape, dtype=constrained.dtype)

        # 1. Intercept terms: nothing to do
        unconstrained[self._params_trend] = constrained[self._params_trend]

        # 2. AR terms: optionally were forced to be stationary
        if self.k_ar > 0 and self.enforce_stationarity:
            # Create the state covariance matrix
            if self.error_cov_type == 'diagonal':
                state_cov = np.diag(constrained[self._params_state_cov])
            elif self.error_cov_type == 'unstructured':
                state_cov_lower = np.zeros(self.ssm['state_cov'].shape,
                                           dtype=constrained.dtype)
                state_cov_lower[self._idx_lower_state_cov] = (
                    constrained[self._params_state_cov])
                state_cov = np.dot(state_cov_lower, state_cov_lower.T)

            # Transform the parameters
            coefficients = constrained[self._params_ar].reshape(
                self.k_endog, self.k_endog * self.k_ar)
            unconstrained_matrices, variance = (
                unconstrain_stationary_multivariate(coefficients, state_cov))
            unconstrained[self._params_ar] = unconstrained_matrices.ravel()
        else:
            unconstrained[self._params_ar] = constrained[self._params_ar]

        # 3. MA terms: optionally were forced to be invertible
        if self.k_ma > 0 and self.enforce_invertibility:
            # Transform the parameters, using an identity variance matrix
            state_cov = np.eye(self.k_endog, dtype=constrained.dtype)
            coefficients = constrained[self._params_ma].reshape(
                self.k_endog, self.k_endog * self.k_ma)
            unconstrained_matrices, variance = (
                unconstrain_stationary_multivariate(coefficients, state_cov))
            unconstrained[self._params_ma] = unconstrained_matrices.ravel()
        else:
            unconstrained[self._params_ma] = constrained[self._params_ma]

        # 4. Regression terms: nothing to do
        unconstrained[self._params_regression] = (
            constrained[self._params_regression])

        # 5. State covariance terms
        # If we have variances, then these were forced to be positive
        if self.error_cov_type == 'diagonal':
            unconstrained[self._params_state_cov] = (
                constrained[self._params_state_cov]**0.5)
        # Otherwise, nothing needs to be done
        elif self.error_cov_type == 'unstructured':
            unconstrained[self._params_state_cov] = (
                constrained[self._params_state_cov])

        # 5. Measurement error variance terms
        if self.measurement_error:
            # These were forced to be positive
            unconstrained[self._params_obs_cov] = (
                constrained[self._params_obs_cov]**0.5)

        return unconstrained

    def update(self, params, **kwargs):
        params = super(VARMAX, self).update(params, **kwargs)

        # 1. State intercept
        # - Exog
        if self.mle_regression:
            exog_params = params[self._params_regression].reshape(
                self.k_endog, self.k_exog).T
            intercept = np.dot(self.exog[1:], exog_params)
            self.ssm[self._idx_state_intercept] = intercept.T

        # - Trend
        if self.k_trend > 0:
            # If we didn't set the intercept above, zero it out so we can
            # just += later
            if not self.mle_regression:
                zero = np.array(0, dtype=params.dtype)
                self.ssm[self._idx_state_intercept] = zero

            trend_params = params[self._params_trend].reshape(
                self.k_endog, self.k_trend).T
            if self._trend_is_const:
                intercept = trend_params
            else:
                intercept = np.dot(self._trend_data[1:], trend_params)
            self.ssm[self._idx_state_intercept] += intercept.T

        # Need to set the last state intercept to np.nan (with appropriate
        # dtype)
        if self.mle_regression:
            nan = np.array(np.nan, dtype=params.dtype)
            self.ssm['state_intercept', :self.k_endog, -1] = nan

        # 2. Transition
        ar = params[self._params_ar].reshape(
            self.k_endog, self.k_endog * self.k_ar)
        ma = params[self._params_ma].reshape(
            self.k_endog, self.k_endog * self.k_ma)
        self.ssm[self._idx_transition] = np.c_[ar, ma]

        # 3. State covariance
        if self.error_cov_type == 'diagonal':
            self.ssm[self._idx_state_cov] = (
                params[self._params_state_cov]
            )
        elif self.error_cov_type == 'unstructured':
            state_cov_lower = np.zeros(self.ssm['state_cov'].shape,
                                       dtype=params.dtype)
            state_cov_lower[self._idx_lower_state_cov] = (
                params[self._params_state_cov])
            self.ssm['state_cov'] = np.dot(state_cov_lower, state_cov_lower.T)

        # 4. Observation covariance
        if self.measurement_error:
            self.ssm[self._idx_obs_cov] = params[self._params_obs_cov]


class VARMAXResults(MLEResults):
    """
    Class to hold results from fitting an VARMAX model.

    Parameters
    ----------
    model : VARMAX instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the VARMAX model instance.
    coefficient_matrices_var : array
        Array containing autoregressive lag polynomial coefficient matrices,
        ordered from lowest degree to highest.
    coefficient_matrices_vma : array
        Array containing moving average lag polynomial coefficients,
        ordered from lowest degree to highest.

    See Also
    --------
    statsmodels.tsa.statespace.kalman_filter.FilterResults
    statsmodels.tsa.statespace.mlemodel.MLEResults
    """
    def __init__(self, model, params, filter_results, cov_type='opg',
                 cov_kwds=None, **kwargs):
        super(VARMAXResults, self).__init__(model, params, filter_results,
                                            cov_type, cov_kwds, **kwargs)

        self.specification = Bunch(**{
            # Set additional model parameters
            'error_cov_type': self.model.error_cov_type,
            'measurement_error': self.model.measurement_error,
            'enforce_stationarity': self.model.enforce_stationarity,
            'enforce_invertibility': self.model.enforce_invertibility,

            'order': self.model.order,

            # Model order
            'k_ar': self.model.k_ar,
            'k_ma': self.model.k_ma,

            # Trend / Regression
            'trend': self.model.trend,
            'k_trend': self.model.k_trend,
            'k_exog': self.model.k_exog,
        })

        # Polynomials / coefficient matrices
        self.coefficient_matrices_var = None
        self.coefficient_matrices_vma = None
        if self.model.k_ar > 0:
            ar_params = np.array(self.params[self.model._params_ar])
            k_endog = self.model.k_endog
            k_ar = self.model.k_ar
            self.coefficient_matrices_var = (
                ar_params.reshape(k_endog * k_ar, k_endog).T
            ).reshape(k_endog, k_endog, k_ar).T
        if self.model.k_ma > 0:
            ma_params = np.array(self.params[self.model._params_ma])
            k_endog = self.model.k_endog
            k_ma = self.model.k_ma
            self.coefficient_matrices_vma = (
                ma_params.reshape(k_endog * k_ma, k_endog).T
            ).reshape(k_endog, k_endog, k_ma).T

    def get_prediction(self, start=None, end=None, dynamic=False, index=None,
                       exog=None, **kwargs):
        """
        In-sample prediction and out-of-sample forecasting

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.
        exog : array_like, optional
            If the model includes exogenous regressors, you must provide
            exactly enough out-of-sample values for the exogenous variables if
            end is beyond the last observation in the sample.
        dynamic : boolean, int, str, or datetime, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Can also be an absolute date string to parse or a
            datetime type (these are not interpreted as offsets).
            Prior to this observation, true endogenous values will be used for
            prediction; starting with this observation and continuing through
            the end of prediction, forecasted endogenous values will be used
            instead.
        kwargs
            Additional arguments may required for forecasting beyond the end
            of the sample. See `FilterResults.predict` for more details.

        Returns
        -------
        forecast : array
            Array of out of sample forecasts.
        """
        if start is None:
            start = self.model._index[0]

        # Handle end (e.g. date)
        _start, _end, _out_of_sample, prediction_index = (
            self.model._get_prediction_index(start, end, index, silent=True))

        # Handle exogenous parameters
        last_intercept = None
        if _out_of_sample and (self.model.k_exog + self.model.k_trend > 0):
            # Create a new faux VARMAX model for the extended dataset
            nobs = self.model.data.orig_endog.shape[0] + _out_of_sample
            endog = np.zeros((nobs, self.model.k_endog))

            if self.model.k_exog > 0:
                if exog is None:
                    raise ValueError('Out-of-sample forecasting in a model'
                                     ' with a regression component requires'
                                     ' additional exogenous values via the'
                                     ' `exog` argument.')
                exog = np.array(exog)
                required_exog_shape = (_out_of_sample, self.model.k_exog)
                if not exog.shape == required_exog_shape:
                    raise ValueError('Provided exogenous values are not of the'
                                     ' appropriate shape. Required %s, got %s.'
                                     % (str(required_exog_shape),
                                        str(exog.shape)))
                exog = np.c_[self.model.data.orig_exog.T, exog.T].T

            # TODO replace with init_kwds or specification or similar
            model = VARMAX(
                endog,
                exog=exog,
                order=self.model.order,
                trend=self.model.trend,
                error_cov_type=self.model.error_cov_type,
                measurement_error=self.model.measurement_error,
                enforce_stationarity=self.model.enforce_stationarity,
                enforce_invertibility=self.model.enforce_invertibility
            )
            model.update(self.params)
            if model['state_intercept'].ndim > 1:
                last_intercept = model['state_intercept', :, self.nobs - 1]
            else:
                last_intercept = model['state_intercept', :, 0]

            # Set the kwargs with the update time-varying state space
            # representation matrices
            for name in self.filter_results.shapes.keys():
                if name == 'obs':
                    continue
                mat = getattr(model.ssm, name)
                if mat.shape[-1] > 1:
                    if len(mat.shape) == 2:
                        kwargs[name] = mat[:, -_out_of_sample:]
                    else:
                        kwargs[name] = mat[:, :, -_out_of_sample:]
        elif self.model.k_exog == 0 and exog is not None:
            warn('Exogenous array provided to predict, but additional data not'
                 ' required. `exog` argument ignored.', ValueWarning)

        # If we had exog, then the last predicted_state has been set to NaN
        # since we didn't have the appropriate exog to create it. Then, if
        # we are forecasting, we now have new exog that we need to put into
        # the existing state_intercept array (and we will take it out, below)
        if last_intercept is not None:
            self.filter_results.state_intercept[:, -1] = last_intercept

        res = super(VARMAXResults, self).get_prediction(
            start=start, end=end, dynamic=dynamic, index=index, exog=exog,
            **kwargs)

        if last_intercept is not None:
            self.filter_results.state_intercept[:, -1] = np.nan

        return res

    def summary(self, alpha=.05, start=None, separate_params=True):
        from statsmodels.iolib.summary import summary_params

        # Create the model name
        spec = self.specification
        if spec.k_ar > 0 and spec.k_ma > 0:
            model_name = 'VARMA'
            order = '(%s,%s)' % (spec.k_ar, spec.k_ma)
        elif spec.k_ar > 0:
            model_name = 'VAR'
            order = '(%s)' % (spec.k_ar)
        else:
            model_name = 'VMA'
            order = '(%s)' % (spec.k_ma)
        if spec.k_exog > 0:
            model_name += 'X'
        model_name = [model_name + order]

        if spec.k_trend > 0:
            model_name.append('intercept')

        if spec.measurement_error:
            model_name.append('measurement error')

        summary = super(VARMAXResults, self).summary(
            alpha=alpha, start=start, model_name=model_name,
            display_params=not separate_params
        )

        if separate_params:
            indices = np.arange(len(self.params))

            def make_table(self, mask, title, strip_end=True):
                res = (self, self.params[mask], self.bse[mask],
                       self.zvalues[mask], self.pvalues[mask],
                       self.conf_int(alpha)[mask])

                param_names = [
                    '.'.join(name.split('.')[:-1]) if strip_end else name
                    for name in
                    np.array(self.data.param_names)[mask].tolist()
                ]

                return summary_params(res, yname=None, xname=param_names,
                                      alpha=alpha, use_t=False, title=title)

            # Add parameter tables for each endogenous variable
            k_endog = self.model.k_endog
            k_ar = self.model.k_ar
            k_ma = self.model.k_ma
            k_trend = self.model.k_trend
            k_exog = self.model.k_exog
            endog_masks = []
            for i in range(k_endog):
                masks = []
                offset = 0

                # 1. Intercept terms
                if k_trend > 0:
                    masks.append(np.arange(i, i + k_endog * k_trend, k_endog))
                    offset += k_endog * k_trend

                # 2. AR terms
                if k_ar > 0:
                    start = i * k_endog * k_ar
                    end = (i + 1) * k_endog * k_ar
                    masks.append(
                        offset + np.arange(start, end))
                    offset += k_ar * k_endog**2

                # 3. MA terms
                if k_ma > 0:
                    start = i * k_endog * k_ma
                    end = (i + 1) * k_endog * k_ma
                    masks.append(
                        offset + np.arange(start, end))
                    offset += k_ma * k_endog**2

                # 4. Regression terms
                if k_exog > 0:
                    masks.append(
                        offset + np.arange(i * k_exog, (i + 1) * k_exog))
                    offset += k_endog * k_exog

                # 5. Measurement error variance terms
                if self.model.measurement_error:
                    masks.append(
                        np.array(self.model.k_params - i - 1, ndmin=1))

                # Create the table
                mask = np.concatenate(masks)
                endog_masks.append(mask)

                title = "Results for equation %s" % self.model.endog_names[i]
                table = make_table(self, mask, title)
                summary.tables.append(table)

            # State covariance terms
            state_cov_mask = (
                np.arange(len(self.params))[self.model._params_state_cov])
            table = make_table(self, state_cov_mask, "Error covariance matrix",
                               strip_end=False)
            summary.tables.append(table)

            # Add a table for all other parameters
            masks = []
            for m in (endog_masks, [state_cov_mask]):
                m = np.array(m).flatten()
                if len(m) > 0:
                    masks.append(m)
            masks = np.concatenate(masks)
            inverse_mask = np.array(list(set(indices).difference(set(masks))))
            if len(inverse_mask) > 0:
                table = make_table(self, inverse_mask, "Other parameters",
                                   strip_end=False)
                summary.tables.append(table)

        return summary
    summary.__doc__ = MLEResults.summary.__doc__


class VARMAXResultsWrapper(MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(MLEResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(MLEResultsWrapper._wrap_methods,
                                     _methods)
wrap.populate_wrapper(VARMAXResultsWrapper, VARMAXResults)  # noqa:E305
