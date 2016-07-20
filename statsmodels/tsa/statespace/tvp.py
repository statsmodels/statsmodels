"""
Time varying parameters model

Author: Valery Likhosherstov
License: Simplified-BSD
"""
import numpy as np
from .mlemodel import MLEModel


class TVPModel(MLEModel):
    r"""
    Time varying parameters model

    Parameters
    ----------
    endog : array_like
        The observed time series process :math:`y`.
    exog : array_like
        Array of exogenous regressors for the observation equation, shaped
        nobs x k_exog. Required for this model.
    kwargs
        Keyword arguments, passed to superclass (`MLEModel`) initializer.

    Attributes
    ----------
    k_exog : int
        The number of exog variables.

    Notes
    -----
    The specification of the model is as follows:

    .. math::

        y_t = \beta_{1t} x_{1t} + \beta_{2t} x_{2t} + ... +
        \beta_{kt} x_{kt} + e_t \\
        \beta_{it} = \beta_{i,t-1} + v_{it} \\
        e_t \sim N(0, \sigma^2) \\
        v_{it} \sim N(0, \sigma_i^2)

    where :math:`\beta_{it}` are time varying parameters. The number of
    time varying parameters (:math:`k`) is defined by `exog` second dimension.

    References
    ----------
    Kim, Chang-Jin, and Charles R. Nelson. 1999.
    "State-Space Models with Regime Switching:
    Classical and Gibbs-Sampling Approaches with Applications".
    MIT Press Books. The MIT Press.
    """

    _obs_var_idx = np.s_[:1]
    _tv_params_cov_idx = np.s_[1:]

    def __init__(self, endog, exog, **kwargs):

        # Delete exog in optional arguments
        if 'exog' in kwargs:
            del kwargs['exog']

        # Transform to numpy array
        exog = np.asarray(exog)

        # Reshape exog, if it is one-dimensional array
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)

        # The number of time varying parameters is defined by `exog` second
        # dimension
        self.k_exog = exog.shape[1]
        k_exog = self.k_exog

        # Superclass initialization
        super(TVPModel, self).__init__(endog, k_exog, exog=exog, **kwargs)

        # Every observation is a single value
        if self.k_endog != 1:
            raise ValueError('Endogenous vector must be univariate.')

        # The dimension of parameters space
        self.k_params = k_exog + 1

        # Parameter names
        self._param_names = ['obs_var'] + ['tvp_var{0}'.format(i) for i in \
                range(k_exog)]

        # Setting up constant representation matrices

        self['design'] = exog.T.reshape(1, k_exog, -1)

        self['transition'] = np.identity(k_exog)

        self['selection'] = np.identity(k_exog)

    @property
    def start_params(self):

        # Default start params is a vector of ones
        return np.ones(self.k_params)

    def transform_params(self, unconstrained):

        # All parameters are error variances
        # Keeping them positive
        return unconstrained**2

    def untransform_params(self, constrained):

        # All parameters are error variances
        # Keeping them positive
        return constrained**0.5

    def update(self, params, **kwargs):

        # Transorm params, if they are untransformed
        params = super(TVPModel, self).update(params, **kwargs)

        k_exog = self.k_exog

        # Observation covariance matrix is a single value
        self['obs_cov'] = np.array(params[self._obs_var_idx]).reshape(1, 1)

        # State covariance matrix is diagonal
        self['state_cov'] = np.diag(params[self._tv_params_cov_idx])
