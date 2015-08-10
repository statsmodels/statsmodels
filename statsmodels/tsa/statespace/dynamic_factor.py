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

        # Test for too many factors
        if self.k_factors >= self.k_endog:
            raise ValueError('Number of factors must be less than the number'
                             ' of endogenous variables.')

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

    def filter(self, params, transformed=True, cov_type=None, return_ssm=False,
               **kwargs):
        params = np.array(params, ndmin=1)

        # Transform parameters if necessary
        if not transformed:
            params = self.transform_params(params)
            transformed = True

        # Get the state space output
        result = super(StaticFactors, self).filter(params, transformed,
                                                   cov_type, return_ssm=True,
                                                   **kwargs)

        # Wrap in a results object
        if not return_ssm:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            result = StaticFactorsResultsWrapper(
                StaticFactorsResults(self, params, result, **result_kwargs)
            )

        return result

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
            params[self._params_transition] = params_ar[:, 0]

        return params

    @property
    def param_names(self):
        param_names = []

        # 1. Factor loadings (estimated via PCA)
        param_names += [
            'loading.f%d.%s' % (j+1, self.endog_names[i])
            for i in range(self.k_endog)
            for j in range(self.k_factors)
        ]

        # 2. Idiosyncratic variances
        param_names += [
            'sigma2.%s' % self.endog_names[i]
            for i in range(self.k_endog)
        ]

        # 3. VAR transition (OLS on factors estimated via PCA)
        param_names += [
            'L%d.f%d.f%d' % (i+1, k+1, j+1)
            for j in range(self.k_factors)
            for i in range(self.factor_order)
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
            unconstrained_matrices = (
                unconstrained[self._params_transition].reshape(
                    self.k_factors, self.k_factors * self.factor_order))
            coefficient_matrices, variance = (
                constrain_stationary_multivariate(unconstrained_matrices,
                                                  self.ssm['state_cov']))
            constrained[self._params_transition] = (
                coefficient_matrices.ravel())
        else:
            constrained[self._params_transition] = (
                unconstrained[self._params_transition])

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
            coefficients = constrained[self._params_transition].reshape(
                self.k_factors, self.k_factors * self.factor_order)
            unconstrained_matrices, variance = (
                unconstrain_stationary_multivariate(coefficients,
                                                    self.ssm['state_cov']))
            unconstrained[self._params_transition] = (
                unconstrained_matrices.ravel())
        else:
            unconstrained[self._params_transition] = (
                constrained[self._params_transition])

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
        self.ssm[self._idx_transition] = (
            params[self._params_transition].reshape(
                self.k_factors, self.k_factors * self.factor_order))


class StaticFactorsResults(MLEResults):
    """
    Class to hold results from fitting an StaticFactors model.

    Parameters
    ----------
    model : StaticFactors instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the StaticFactors model
        instance.
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
                 **kwargs):
        super(StaticFactorsResults, self).__init__(model, params,
                                                   filter_results, cov_type,
                                                   **kwargs)

        self.df_resid = np.inf  # attribute required for wald tests

        self.specification = Bunch(**{
            # Set additional model parameters
            'enforce_stationarity': self.model.enforce_stationarity,

            # Factors
            'k_factors': self.model.k_factors,
            'factor_order': self.model.factor_order
        })

        # Polynomials / coefficient matrices
        self.coefficient_matrices_var = None
        if self.model.factor_order > 0:
            ar_params = np.array(self.params[self.model._params_transition])
            k_factors = self.model.k_factors
            factor_order = self.model.factor_order
            self.coefficient_matrices_var = (
                ar_params.reshape(k_factors * factor_order, k_factors).T
            ).reshape(k_factors, k_factors, factor_order).T

    def summary(self, alpha=.05, start=None):
        from statsmodels.iolib.summary import summary_params
        # Create the model name

        # See if we have an ARIMA component
        order = '(factors=%d, order=%d)' % (self.specification.k_factors,
                                            self.specification.factor_order)

        model_name = (
            '%s%s' % (self.model.__class__.__name__, order)
            )
        summary = super(StaticFactorsResults, self).summary(
            alpha=alpha, start=start, model_name=model_name,
            display_params=False
        )

        # Add parameter tables for each endogenous variable
        k_endog = self.model.k_endog
        k_factors = self.model.k_factors
        factor_order = self.model.factor_order
        for i in range(k_endog):
            mask = []
            offset = 0

            # 1. Loadings
            offset = k_factors * i
            mask.append(np.arange(offset, offset + k_factors))

            # 2. Idiosyncratic variance
            offset = k_factors * k_endog
            mask.append(np.array(offset + i, ndmin=1))

            # Create the table
            mask = np.concatenate(mask)
            res = (self, self.params[mask], self.bse[mask], self.zvalues[mask],
                   self.pvalues[mask], self.conf_int(alpha)[mask])

            param_names = [
                '.'.join(name.split('.')[:-1])
                for name in
                np.array(self.data.param_names)[mask].tolist()
            ]
            title = "Results for equation %s" % self.model.endog_names[i]

            table = summary_params(res, yname=None, xname=param_names,
                                   alpha=alpha, use_t=False, title=title)

            summary.tables.append(table)

        # Add parameter tables for each factor
        offset = k_endog * (k_factors + 1)
        for i in range(k_factors):
            mask = []
            start = i * k_factors * factor_order
            end = (i + 1) * k_factors * factor_order
            mask.append(
                offset + np.arange(start, end))

            # Create the table
            mask = np.concatenate(mask)
            res = (self, self.params[mask], self.bse[mask], self.zvalues[mask],
                   self.pvalues[mask], self.conf_int(alpha)[mask])

            param_names = [
                '.'.join(name.split('.')[:-1])
                for name in
                np.array(self.data.param_names)[mask].tolist()
            ]
            title = "Results for factor equation f%d" % (i+1)

            table = summary_params(res, yname=None, xname=param_names,
                                   alpha=alpha, use_t=False, title=title)

            summary.tables.append(table)

        return summary
    summary.__doc__ = MLEResults.summary.__doc__


class StaticFactorsResultsWrapper(MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(MLEResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(MLEResultsWrapper._wrap_methods,
                                     _methods)
wrap.populate_wrapper(StaticFactorsResultsWrapper, StaticFactorsResults)
