"""
Vector Autoregressive Moving Average with eXogenous regressors model

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

from warnings import warn
from statsmodels.compat.collections import OrderedDict

import pandas as pd
import numpy as np
from .kalman_filter import (
    KalmanFilter, FilterResults, INVERT_UNIVARIATE, SOLVE_LU
)
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .tools import (
    companion_matrix, diff, is_invertible,
    constrain_stationary_multivariate, unconstrain_stationary_multivariate
)
from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.vector_ar import var_model


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
    trend : {'nc', 'c'}, optional
        Parameter controlling the deterministic trend polynomial.
        Can be specified as a string where 'c' indicates a constant intercept
        and 'nc' indicates no intercept term.
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
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices or for Kalman filtering options. See `Representation`, and
        `KalmanFilter` for more details.
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
        self.k_trend = int(self.trend == 'c')

        # Check for valid model
        if trend not in ['c', 'nc']:
            raise ValueError('Invalid trend specification.')
        if error_cov_type not in ['diagonal', 'unstructured']:
            raise ValueError('Invalid error covariance matrix type'
                             ' specification.')
        if self.k_ar == 0 and self.k_ma == 0:
            raise ValueError('Invalid VARMAX(p,q) specification; at least one'
                             ' p,q must be greater than zero.')

        # Warn for VARMA model
        if self.k_ar > 0 and self.k_ma > 0:
            warn('Estimation of VARMA(p,q) models is not generically robust,'
                 ' due especially to identification issues.')

        # Exogenous data
        self.k_exog = 0
        if exog is not None:
            exog_is_using_pandas = _is_using_pandas(exog, None)
            if not exog_is_using_pandas:
                exog = np.asarray(exog)

            # Make sure we have 2-dimensional array
            if exog.ndim == 1:
                if not exog_is_using_pandas:
                    exog = exog[:, None]
                else:
                    exog = pd.DataFrame(exog)

            self.k_exog = exog.shape[1]

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

        # Initialize known elements of the state space matrices

        # If we have exog effects, then the state intercept needs to be
        # time-varying
        if self.k_exog > 0:
            self.ssm['state_intercept'] = np.zeros((self.k_states, self.nobs))

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
        idx = idx[0] + (_min_k_ar + 1) * self.k_endog, idx[1] + _min_k_ar * self.k_endog
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
        if self.trend == 'c' and self.k_exog == 0:
            self._idx_state_intercept = np.s_['state_intercept', :k_endog]
        elif self.k_exog > 0:
            self._idx_state_intercept = np.s_['state_intercept', :k_endog, :]
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
    def start_params(self):
        params = np.zeros(self.k_params, dtype=np.float64)

        # A. Run a multivariate regression to get beta estimates
        endog = self.endog.copy()
        exog = self.exog.copy() if self.k_exog > 0 else None

        # Although the Kalman filter can deal with missing values in endog,
        # conditional sum of squares cannot
        if np.any(np.isnan(endog)):
            endog = endog[~np.isnan(endog)]
            if exog is not None:
                exog = exog[~np.isnan(endog)]

        # Regression effects via OLS
        exog_params = np.zeros(0)
        if self.k_exog > 0:
            exog_params = np.linalg.pinv(exog).dot(endog)
            endog -= np.dot(exog, exog_params)

        # B. Run a VAR model on endog to get trend, AR parameters
        ar_params = []
        k_ar = self.k_ar if self.k_ar > 0 else 1
        mod_ar = var_model.VAR(endog)
        res_ar = mod_ar.fit(maxlags=k_ar, ic=None, trend=self.trend)
        ar_params = np.array(res_ar.params.T)
        if self.trend == 'c':
            trend_params = ar_params[:, 0]
            if self.k_ar > 0:
                ar_params = ar_params[:, 1:].ravel()
            else:
                ar_params = []
        elif self.k_ar > 0:
            ar_params = ar_params.ravel()
        else:
            ar_params = []
        endog = res_ar.resid

        # C. Run a VAR model on the residuals to get MA parameters
        ma_params = []
        if self.k_ma > 0:
            mod_ma = var_model.VAR(endog)
            res_ma = mod_ma.fit(maxlags=self.k_ma, ic=None, trend='nc')
            ma_params = np.array(res_ma.params.T).ravel()

        # 1. Intercept terms
        if self.trend == 'c':
            params[self._params_trend] = trend_params

        # 2. AR terms
        params[self._params_ar] = ar_params

        # 3. MA terms
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
        if self.trend == 'c':
            param_names += ['const.y%d' % (i+1) for i in range(self.k_endog)]

        # 2. AR terms
        param_names += [
            'L%dy%d.y%d' % (i+1, k+1, j+1)
            for i in range(self.k_ar)
            for j in range(self.k_endog)
            for k in range(self.k_endog)
        ]

        # 3. MA terms
        param_names += [
            'L%de%d.y%d' % (i+1, k+1, j+1)
            for i in range(self.k_ma)
            for j in range(self.k_endog)
            for k in range(self.k_endog)
        ]

        # 4. Regression terms
        # TODO replace "beta" with the name of the exog, when possible
        param_names += [
            'beta%d.y%d' % (j+1, i+1)
            for i in range(self.k_endog)
            for j in range(self.k_exog)
        ]

        # 5. State covariance terms
        if self.error_cov_type == 'diagonal':
            param_names += ['sigma2.y%d' % i for i in range(self.k_endog)]
        elif self.error_cov_type == 'unstructured':
            param_names += [
                'sqrt.cov.y%d.y%d' % (j+1, i+1)
                for i in range(self.k_endog)
                for j in range(i+1)
            ]

        # 5. Measurement error variance terms
        if self.measurement_error:
            param_names += [
                'measurement_variance.y%d' % (i+1)
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

    def update(self, params, *args, **kwargs):
        params = super(VARMAX, self).update(params, *args, **kwargs)

        # 1. State intercept
        if self.mle_regression:
            exog_params = params[self._params_regression].reshape(
                self.k_exog, self.k_endog)
            intercept = np.dot(self.exog, exog_params)
            if self.trend == 'c':
                intercept += params[self._params_trend]
            self.ssm[self._idx_state_intercept] = intercept.T
        elif self.trend == 'c':
            self.ssm[self._idx_state_intercept] = params[self._params_trend]

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
