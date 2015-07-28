"""
Dynamic factor model

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

from warnings import warn
from statsmodels.compat.collections import OrderedDict

import numpy as np
import pandas as pd
from .kalman_filter import KalmanFilter, FilterResults
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .tools import (
    companion_matrix, diff, is_invertible,
    constrain_stationary_multivariate, unconstrain_stationary_multivariate
)
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tools.pca import PCA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap


class StaticFactors(MLEModel):
    r"""
    Static form of the dynamic factor model

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    k_factors : int
        The number of unobserved factors.
    factor_order : int
        The order of the vector autoregression followed by the factors.
    enforce_stationarity : boolean, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices or for Kalman filtering options. See `Representation`, and
        `KalmanFilter` for more details.

    Attributes
    ----------

    Notes
    -----
    The static form of the dynamic factor model is specified:

    .. math::

        X_t & = \Lambda F_t + \varepsilon_t \\
        F_t & = \Psi(L) F_{t-1} + \eta_t

    where there are :math:`N` observed series and :math:`q` unobserved factors
    so that :math:`X_t, \varepsilon_t` are :math:`(N x 1)` and
    :math:`F_t, \eta_t` are  :math:`(q x 1)`.

    We assume that :math:`\varepsilon_{it} \sim N(0, \sigma_{\varepsilon_i}^2)`
    and :math:`\eta_t \sim N(0, I)`
    
    References
    ----------
    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
       Time Series Analysis by State Space Methods: Second Edition.
       Oxford University Press.
    .. [2] Lutkepohl, Helmut. 2007.
       New Introduction to Multiple Time Series Analysis.
       Berlin: Springer.

    """

    def __init__(self, endog, k_factors, factor_order,
                 enforce_stationarity=True, **kwargs):

        # Model parameters
        self.k_factors = k_factors
        self.enforce_stationarity = enforce_stationarity

        # Save given orders
        self.factor_order = factor_order

        # Calculate the number of states
        k_states = self.factor_order * self.k_factors
        k_posdef = self.k_factors

        # By default, initialize as stationary
        kwargs.setdefault('initialization', 'stationary')

        # Initialize the state space model
        super(StaticFactors, self).__init__(
            endog, k_states=k_states, k_posdef=k_posdef, **kwargs
        )

        # Initialize the parameters
        self.parameters = OrderedDict()
        self.parameters['factor_loadings'] = self.k_endog * self.k_factors
        self.parameters['idiosyncratic'] = self.k_endog
        self.parameters['transition'] = self.k_factors**2 * self.factor_order
        self.k_params = sum(self.parameters.values())

        # Setup fixed components of state space matrices
        self.ssm['selection', :k_posdef, :k_posdef] = np.eye(k_posdef)
        self.ssm['transition', k_factors:, :k_factors * (factor_order - 1)] = (
            np.eye(k_factors * (factor_order - 1)))
        self.ssm['state_cov'] = np.eye(self.k_factors)

        # Setup indices of state space matrices
        idx = np.diag_indices(self.k_endog)
        self._idx_obs_cov = ('obs_cov', idx[0], idx[1])
        self._idx_transition = np.s_['transition', :self.k_factors, :]

        # Cache some slices
        def _slice(key, offset):
            length = self.parameters[key]
            param_slice = np.s_[offset:offset + length]
            offset += length
            return param_slice, offset
        
        offset = 0
        self._params_loadings, offset = _slice('factor_loadings', offset)
        self._params_idiosyncratic, offset = _slice('idiosyncratic', offset)
        self._params_transition, offset = _slice('transition', offset)

    @property
    def start_params(self):
        params = np.zeros(self.k_params, dtype=np.float64)

        # Use principal components as starting values
        res_pca = PCA(self.endog, ncomp=self.k_factors)

        # 1. Factor loadings (estimated via PCA)
        params[self._params_loadings] = res_pca.loadings.ravel()

        # 2. Idiosyncratic variances
        resid = self.endog - np.dot(res_pca.factors, res_pca.loadings.T)
        params[self._params_idiosyncratic] = resid.var(axis=0)

        if self.k_factors > 1:
            # 3a. VAR transition (OLS on factors estimated via PCA)
            mod_var = VAR(res_pca.factors)
            res_var = mod_var.fit(maxlags=self.factor_order, ic=None,
                                  trend='nc')
            params[self._params_transition] = res_var.params.T.ravel()
        else:
            # 3b. AR transition
            Y = res_pca.factors[self.factor_order:]
            X = lagmat(res_pca.factors, self.factor_order, trim='both')
            params_ar = np.linalg.pinv(X).dot(Y)
            resid = Y - np.dot(X, params_ar)
            params[self._params_transition] = params_ar[:,0]

        return params

    @property
    def param_names(self):
        param_names = []

        # 1. Factor loadings (estimated via PCA)
        param_names += [
            'loading%d.y%d' % (j+1, i+1)
            for i in range(self.k_endog)
            for j in range(self.k_factors)
        ]

        # 2. Idiosyncratic variances
        param_names += ['sigma2.y%d' % (i+1) for i in range(self.k_endog)]

        # 3. VAR transition (OLS on factors estimated via PCA)
        param_names += [
            'L%df%d.f%d' % (i+1, k+1, j+1)
            for i in range(self.factor_order)
            for j in range(self.k_factors)
            for k in range(self.k_factors)
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

        # The factor loadings do not need to be adjusted
        constrained[self._params_loadings] = (
            unconstrained[self._params_loadings])

        # The observation variances must be positive
        constrained[self._params_idiosyncratic] = (
            unconstrained[self._params_idiosyncratic]**2)

        # VAR transition: optionally force to be stationary
        if self.enforce_stationarity:
            # Transform the parameters
            coefficients = unconstrained[self._params_transition].reshape(self.k_factors, self.k_factors * self.factor_order)
            unconstrained_matrices = [coefficients[:,i*self.k_factors:(i+1)*self.k_factors] for i in range(self.factor_order)]
            coefficient_matrices, variance = (
                constrain_stationary_multivariate(unconstrained_matrices, self.ssm['state_cov']))
            constrained[self._params_transition] = np.concatenate(coefficient_matrices, axis=1).ravel()
        else:
            constrained[self._params_transition] = unconstrained[self._params_transition]

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

        # The factor loadings do not need to be adjusted
        unconstrained[self._params_loadings] = (
            constrained[self._params_loadings])

        # The observation variances must have been positive
        unconstrained[self._params_idiosyncratic] = (
            constrained[self._params_idiosyncratic]**0.5)

        # VAR transition: optionally were forced to be stationary
        if self.enforce_stationarity:
            coefficients = constrained[self._params_transition].reshape(self.k_factors, self.k_factors * self.factor_order)
            coefficient_matrices = [coefficients[:,i*self.k_factors:(i+1)*self.k_factors] for i in range(self.factor_order)]
            unconstrained_matrices, variance = (
                unconstrain_stationary_multivariate(coefficient_matrices, self.ssm['state_cov']))
            unconstrained[self._params_transition] = np.concatenate(unconstrained_matrices, axis=1).ravel()
        else:
            unconstrained[self._params_transition] = constrained[self._params_transition]

        return unconstrained

    def update(self, params, transformed=True):
        """
        Update the parameters of the model

        Updates the representation matrices to fill in the new parameter
        values.

        Parameters
        ----------
        params : array_like
            Array of new parameters.
        transformed : boolean, optional
            Whether or not `params` is already transformed. If set to False,
            `transform_params` is called. Default is True..

        Returns
        -------
        params : array_like
            Array of parameters.

        Notes
        -----
        Let `n = k_endog`, `m = k_factors`, and `p = factor_order`. Then the
        `params` vector has length
        :math:`[n \times m] + [n] + [m^2 \times p] + [m \times (m + 1) / 2]`.
        It is expanded in the following way:

        - The first :math:`n \times m` parameters fill out the factor loading
          matrix, starting from the [0,0] entry and then proceeding along rows.
          These parameters are not modified in `transform_params`.
        - The next :math:`n` parameters provide variances for the idiosyncratic
          errors in the observation equation. They fill in the diagonal of the
          observation covariance matrix, and are constrained to be positive by
          `transofrm_params`.
        - The next :math:`m^2 \times p` parameters are used to create the `p`
          coefficient matrices for the vector autoregression describing the
          factor transition. They are transformed in `transform_params` to
          enforce stationarity of the VAR(p). They are placed so as to make
          the transition matrix a companion matrix for the VAR. In particular,
          we assume that the first :math:`m^2` parameters fill the first
          coefficient matrix (starting at [0,0] and filling along rows), the
          second :math:`m^2` parameters fill the second matrix, etc.
        - The last :math:`m \times (m + 1) / 2` parameters are used to fill in
          a lower-triangular matrix, which is multipled with its transpose to
          create a positive definite variance / covariance matrix for the
          factor's VAR. The are not transformed in `transform_params` because
          the matrix multiplication procedure ensures the variance terms are
          positive and the matrix is positive definite.

        """
        params = super(StaticFactors, self).update(params, transformed)

        # Update the design / factor loading matrix
        self.ssm['design', :, :self.k_factors] = (
            params[self._params_loadings].reshape(self.k_endog, self.k_factors)
        )

        # Update the observation covariance
        self.ssm[self._idx_obs_cov] = params[self._params_idiosyncratic]

        # Update the transition matrix
        self.ssm[self._idx_transition] = params[self._params_transition].reshape(self.k_factors, self.k_factors * self.factor_order)
