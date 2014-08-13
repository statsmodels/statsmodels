"""
State Space Representation

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

from warnings import warn

import numpy as np
from .tools import (
    find_best_blas_type, prefix_dtype_map, prefix_statespace_map,
    prefix_kalman_filter_map, validate_matrix_shape, validate_vector_shape
)

# Define constants
FILTER_CONVENTIONAL = 0x01     # Durbin and Koopman (2012), Chapter 4
FILTER_EXACT_INITIAL = 0x02    # ibid., Chapter 5.6
FILTER_AUGMENTED = 0x04        # ibid., Chapter 5.7
FILTER_SQUARE_ROOT = 0x08      # ibid., Chapter 6.3
FILTER_UNIVARIATE = 0x10       # ibid., Chapter 6.4
FILTER_COLLAPSED = 0x20        # ibid., Chapter 6.5
FILTER_EXTENDED = 0x40         # ibid., Chapter 10.2
FILTER_UNSCENTED = 0x80        # ibid., Chapter 10.3

INVERT_UNIVARIATE = 0x01
SOLVE_LU = 0x02
INVERT_LU = 0x04
SOLVE_CHOLESKY = 0x08
INVERT_CHOLESKY = 0x10

STABILITY_FORCE_SYMMETRY = 0x01

MEMORY_STORE_ALL = 0
MEMORY_NO_FORECAST = 0x01
MEMORY_NO_PREDICTED = 0x02
MEMORY_NO_FILTERED = 0x04
MEMORY_NO_LIKELIHOOD = 0x08
MEMORY_CONSERVE = (
    MEMORY_NO_FORECAST | MEMORY_NO_PREDICTED | MEMORY_NO_FILTERED |
    MEMORY_NO_LIKELIHOOD
)


class Representation(object):
    r"""
    State space representation of a time series process

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    k_states : int
        The dimension of the unobserved state process.
    k_posdef : int, optional
        The dimension of a guaranteed positive definite covariance matrix
        describing the shocks in the measurement equation. Must be less than
        or equal to `k_states`. Default is `k_states`.
    time-invariant : bool, optional
        In initializing model matrices, whether to assume the model is
        time-invariant (all matrices can be resized later). Default is True.
    design : array_like, optional
        The design matrix, :math:`Z`. Default is set to zeros.
    obs_intercept : array_like, optional
        The intercept for the observation equation, :math:`d`. Default is set
        to zeros.
    obs_cov : array_like, optional
        The covariance matrix for the observation equation :math:`H`. Default
        is set to zeros.
    transition : array_like, optional
        The transition matrix, :math:`T`. Default is set to zeros.
    state_intercept : array_like, optional
        The intercept for the transition equation, :math:`c`. Default is set to
        zeros.
    selection : array_like, optional
        The selection matrix, :math:`R`. Default is set to zeros.
    state_cov : array_like, optional
        The covariance matrix for the state equation :math:`Q`. Default is set
        to zeros.

    Notes
    -----

    A general state space model is of the form

    .. math::

        y_t = Z_t \alpha_t + d_t + \varepsilon_t \\
        \alpha_t = T_t \alpha_{t-1} + c_t + R_t \eta_t \\

    where :math:`y_t` refers to the observation vector at time :math:`t`,
    :math:`\alpha_t` refers to the (unobserved) state vector at time
    :math:`t`, and where the irregular components are defined as

    .. math::

        \varepsilon_t \sim N(0, H_t) \\
        \eta_t \sim N(0, Q_t) \\

    The remaining variables (:math:`Z_t, d_t, H_t, T_t, c_t, R_t, Q_t`) in the
    equations are matrices describing the process. Their variable names and
    dimensions are as follows

    Z : `design`          :math:`(k_endog \times k_states \times nobs)`

    d : `obs_intercept`   :math:`(k_endog \times nobs)`

    H : `obs_cov`         :math:`(k_endog \times k_endog \times nobs)`

    T : `transition`      :math:`(k_states \times k_states \times nobs)`

    c : `state_intercept` :math:`(k_states \times nobs)`

    R : `selection`       :math:`(k_states \times k_posdef \times nobs)`

    Q : `state_cov`       :math:`(k_posdef \times k_posdef \times nobs)`

    In the case that one of the matrices is time-invariant (so that, for
    example, :math:`Z_t = Z_{t+1} ~ \forall ~ t`), its last dimension may
    be of size :math:`1` rather than size `nobs`.

    References
    ----------

    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
       Time Series Analysis by State Space Methods: Second Edition.
       Oxford University Press.
    """
    def __init__(self, endog, k_states, k_posdef=None,
                 design=None, obs_intercept=None, obs_cov=None,
                 transition=None, state_intercept=None, selection=None,
                 state_cov=None, *args, **kwargs):

        # Explicitly copy the endogenous array
        # Note: we assume that the given endog array is nobs x k_endog, but
        # _statespace assumes it is k_endog x nobs. Thus we create it in the
        # transposed shape as order "C" and then transpose to get order "F".
        if np.ndim(endog) == 1:
            self.endog = np.array(endog, ndmin=2, copy=True, order="F")
        else:
            self.endog = np.array(endog, copy=True, order="C").T
        dtype = self.endog.dtype

        # Dimensions
        self.k_endog, self.nobs = self.endog.shape
        if k_states < 1:
            raise ValueError('Number of states in statespace model must be a'
                             ' positive number.')
        self.k_states = k_states
        self.k_posdef = k_posdef if k_posdef is not None else k_states

        # Parameters
        self.initialization = None

        # Record the shapes of all of our matrices
        # Note: these are time-invariant shapes; in practice the last dimension
        # may also be `self.nobs` for any or all of these.
        self.shapes = {
            'obs': self.endog.shape,
            'design': (
                (self.k_endog, self.k_states, 1)
                if design is None else design.shape
            ),
            'obs_intercept': (
                (self.k_endog, 1)
                if obs_intercept is None else obs_intercept.shape
            ),
            'obs_cov': (
                (self.k_endog, self.k_endog, 1)
                if obs_cov is None else obs_cov.shape
            ),
            'transition': (
                (self.k_states, self.k_states, 1)
                if transition is None else transition.shape
            ),
            'state_intercept': (
                (self.k_states, 1)
                if state_intercept is None else state_intercept.shape
            ),
            'selection': (
                (self.k_states, self.k_posdef, 1)
                if selection is None else selection.shape
            ),
            'state_cov': (
                (self.k_posdef, self.k_posdef, 1)
                if state_cov is None else state_cov.shape
            )
        }

        # Representation matrices
        # These matrices are only used in the Python object as containers,
        # which will be copied to the appropriate _statespace object if a
        # filter is called.
        self._design = np.zeros(
            self.shapes['design'], dtype=dtype, order="F"
        )
        self._obs_intercept = np.zeros(
            self.shapes['obs_intercept'], dtype=dtype, order="F"
        )
        self._obs_cov = np.zeros(
            self.shapes['obs_cov'], dtype=dtype, order="F"
        )
        self._transition = np.zeros(
            self.shapes['transition'], dtype=dtype, order="F"
        )
        self._state_intercept = np.zeros(
            self.shapes['state_intercept'], dtype=dtype, order="F"
        )
        self._selection = np.zeros(
            self.shapes['selection'], dtype=dtype, order="F"
        )
        self._state_cov = np.zeros(
            self.shapes['state_cov'], dtype=dtype, order="F"
        )

        # Initialize with provided matrices
        if design is not None:
            self.design = design
        if obs_intercept is not None:
            self.obs_intercept = obs_intercept
        if obs_cov is not None:
            self.obs_cov = obs_cov
        if transition is not None:
            self.transition = transition
        if state_intercept is not None:
            self.state_intercept = state_intercept
        if selection is not None:
            self.selection = selection
        if state_cov is not None:
            self.state_cov = state_cov

        # State-space initialization data
        self._initial_state = None
        self._initial_state_cov = None
        self._initial_variance = None

        # Matrix representations storage
        self._representations = {}

        # Setup the underlying statespace object storage
        self._statespaces = {}

        # Setup the underlying Kalman filter storage
        self._kalman_filters = {}

        # Options
        self.initial_variance = kwargs.get('initial_variance', 1e6)
        self.loglikelihood_burn = kwargs.get('loglikelihood_burn', 0)
        self.filter_results_class = kwargs.get('filter_results_class', FilterResults)

        self.filter_method = kwargs.get(
            'filter_method', FILTER_CONVENTIONAL
        )
        self.inversion_method = kwargs.get(
            'inversion_method', INVERT_UNIVARIATE | SOLVE_CHOLESKY
        )
        self.stability_method = kwargs.get(
            'stability_method', STABILITY_FORCE_SYMMETRY
        )
        self.conserve_memory = kwargs.get('conserve_memory', 0)
        self.tolerance = kwargs.get('tolerance', 1e-19)

    @property
    def prefix(self):
        return find_best_blas_type((
            self.endog, self._design, self._obs_intercept, self._obs_cov,
            self._transition, self._state_intercept, self._selection,
            self._state_cov
        ))[0]

    @property
    def dtype(self):
        return prefix_dtype_map[self.prefix]

    @property
    def time_invariant(self):
        return (
            self._design.shape[2] == self._obs_intercept.shape[1] ==
            self._obs_cov.shape[2] == self._transition.shape[2] ==
            self._state_intercept.shape[1] == self._selection.shape[2] ==
            self._state_cov.shape[2]
        )

    @property
    def _statespace(self):
        prefix = self.prefix
        if prefix in self._statespaces:
            return self._statespaces[prefix]
        return None

    @property
    def _kalman_filter(self):
        prefix = self.prefix
        if prefix in self._kalman_filters:
            return self._kalman_filters[prefix]
        return None

    @property
    def obs(self):
        return self.endog

    @property
    def design(self):
        return self._design

    @design.setter
    def design(self, value):
        design = np.asarray(value, order="F")

        # Expand 1-dimensional array if possible
        if (design.ndim == 1 and self.k_endog == 1
                and design.shape[0] == self.k_states):
            design = design[None, :]

        # Enforce that the design matrix is k_endog by k_states
        validate_matrix_shape(
            'design', design.shape, self.k_endog, self.k_states, self.nobs
        )

        # Expand time-invariant design matrix
        if design.ndim == 2:
            design = np.array(design[:, :, None], order="F")

        # Set the array elements
        self._design = design

    @property
    def obs_intercept(self):
        return self._obs_intercept

    @obs_intercept.setter
    def obs_intercept(self, value):
        obs_intercept = np.asarray(value, order="F")

        # Enforce that the observation intercept has length k_endog
        validate_vector_shape(
            'observation intercept', obs_intercept.shape, self.k_endog,
            self.nobs
        )

        # Expand the time-invariant observation intercept vector
        if obs_intercept.ndim == 1:
            obs_intercept = np.array(obs_intercept[:, None], order="F")

        # Set the array
        self._obs_intercept = obs_intercept

    @property
    def obs_cov(self):
        return self._obs_cov

    @obs_cov.setter
    def obs_cov(self, value):
        obs_cov = np.asarray(value, order="F")

        # Expand 1-dimensional array if possible
        if (obs_cov.ndim == 1 and self.k_endog == 1
                and obs_cov.shape[0] == self.k_endog):
            obs_cov = obs_cov[None, :]

        # Enforce that the observation covariance matrix is k_endog by k_endog
        validate_matrix_shape(
            'observation covariance', obs_cov.shape, self.k_endog,
            self.k_endog, self.nobs
        )

        # Expand time-invariant obs_cov matrix
        if obs_cov.ndim == 2:
            obs_cov = np.array(obs_cov[:, :, None], order="F")
        # Set the array
        self._obs_cov = obs_cov

    @property
    def transition(self):
        return self._transition

    @transition.setter
    def transition(self, value):
        transition = np.asarray(value, order="F")

        # Expand 1-dimensional array if possible
        if (transition.ndim == 1 and self.k_states == 1
                and transition.shape[0] == self.k_states):
            transition = transition[None, :]

        # Enforce that the transition matrix is k_states by k_states
        validate_matrix_shape(
            'transition', transition.shape, self.k_states, self.k_states,
            self.nobs
        )

        # Expand time-invariant transition matrix
        if transition.ndim == 2:
            transition = np.array(transition[:, :, None], order="F")

        # Set the array
        self._transition = transition

    @property
    def state_intercept(self):
        return self._state_intercept

    @state_intercept.setter
    def state_intercept(self, value):
        state_intercept = np.asarray(value, order="F")

        # Enforce dimension (1 is later expanded to time-invariant 2-dim)
        if state_intercept.ndim > 2:
            raise ValueError('Invalid state intercept vector. Requires a'
                             ' 1- or 2-dimensional array, got %d dimensions'
                             % state_intercept.ndim)

        # Enforce that the state intercept has length k_endog
        validate_vector_shape(
            'state intercept', state_intercept.shape, self.k_states,
            self.nobs
        )

        # Expand the time-invariant state intercept vector
        if state_intercept.ndim == 1:
            state_intercept = np.array(state_intercept[:, None], order="F")

        # Set the array
        self._state_intercept = state_intercept

    @property
    def selection(self):
        return self._selection

    @selection.setter
    def selection(self, value):
        selection = np.asarray(value, order="F")

        # Expand 1-dimensional array if possible
        if (selection.ndim == 1 and self.k_states == 1
                and selection.shape[0] == self.k_states):
            selection = selection[None, :]

        # Enforce that the selection matrix is k_states by k_posdef
        validate_matrix_shape(
            'selection', selection.shape, self.k_states, self.k_posdef,
            self.nobs
        )

        # Expand time-invariant selection matrix
        if selection.ndim == 2:
            selection = np.array(selection[:, :, None], order="F")
        # Set the array
        self._selection = selection

    @property
    def state_cov(self):
        return self._state_cov

    @state_cov.setter
    def state_cov(self, value):
        state_cov = np.asarray(value, order="F")

        # Expand 1-dimensional array if possible
        if (state_cov.ndim == 1 and self.k_posdef == 1
                and state_cov.shape[0] == self.k_posdef):
            state_cov = state_cov[None, :]

        # Enforce that the state covariance matrix is k_states by k_states
        validate_matrix_shape(
            'state covariance', state_cov.shape, self.k_posdef, self.k_posdef,
            self.nobs
        )

        # Expand time-invariant state_cov matrix
        if state_cov.ndim == 2:
            state_cov = np.array(state_cov[:, :, None], order="F")
        # Set the array
        self._state_cov = state_cov

    def initialize_known(self, initial_state, initial_state_cov):
        """
        Initialize the statespace model with known distribution for initial
        state.

        These values are assumed to be known with certainty or else
        filled with parameters during, for example, maximum likelihood
        estimation.

        Parameters
        ----------
        initial_state : array_like
            Known mean of the initial state vector.
        initial_state_cov : array_like
            Known covariance matrix of the initial state vector.
        """
        initial_state = np.asarray(initial_state, order="F")
        initial_state_cov = np.asarray(initial_state_cov, order="F")

        if not initial_state.shape == (self.k_states,):
            raise ValueError('Invalid dimensions for initial state vector.'
                             ' Requires shape (%d,), got %s' %
                             (self.k_states, str(initial_state.shape)))
        if not initial_state_cov.shape == (self.k_states, self.k_states):
            raise ValueError('Invalid dimensions for initial covariance'
                             ' matrix. Requires shape (%d,%d), got %s' %
                             (self.k_states, self.k_states,
                              str(initial_state.shape)))

        self._initial_state = initial_state
        self._initial_state_cov = initial_state_cov
        self.initialization = 'known'

    def initialize_approximate_diffuse(self, variance=None):
        """
        Initialize the statespace model with approximate diffuse values.

        Rather than following the exact diffuse treatment (which is developed
        for the case that the variance becomes infinitely large), this assigns
        an arbitrary large number for the variance.

        Parameters
        ----------
        variance : float, optional
            The variance for approximating diffuse initial conditions. Default
            is 1e3.
        """
        if variance is None:
            variance = self.initial_variance

        self._initial_variance = variance
        self.initialization = 'approximate_diffuse'

    def initialize_stationary(self):
        """
        Initialize the statespace model as stationary.
        """
        self.initialization = 'stationary'

    def _initialize_filter(self, filter_method=None, inversion_method=None,
                           stability_method=None, conserve_memory=None,
                           tolerance=None, loglikelihood_burn=None,
                           recreate=True, return_loglike=False,
                           *args, **kwargs):
        if filter_method is None:
            filter_method = self.filter_method
        if inversion_method is None:
            inversion_method = self.inversion_method
        if stability_method is None:
            stability_method = self.stability_method
        if conserve_memory is None:
            conserve_memory = self.conserve_memory
        if loglikelihood_burn is None:
            loglikelihood_burn = self.loglikelihood_burn
        if tolerance is None:
            tolerance = self.tolerance

        # Determine which filter to call
        prefix = self.prefix
        dtype = self.dtype

        # If the dtype-specific representation matrices do not exist, create
        # them
        if prefix not in self._representations:
            # Copy the statespace representation matrices
            self._representations[prefix] = {}
            for matrix in self.shapes.keys():
                if matrix == 'obs':
                    self._representations[prefix][matrix] = self.obs.astype(dtype)
                else:
                    # Note: this always makes a copy
                    self._representations[prefix][matrix] = (
                        getattr(self, '_' + matrix).astype(dtype)
                    )
        # If they do exist, update them
        else:
            for matrix in self.shapes.keys():
                if matrix == 'obs':
                    self._representations[prefix][matrix] = self.obs.astype(dtype)[:]
                else:
                    self._representations[prefix][matrix][:] = (
                        getattr(self, '_' + matrix).astype(dtype)[:]
                    )

        # Determine if we need to re-create the _statespace models
        # (if time-varying matrices changed)
        recreate_statespace = False
        if recreate and prefix in self._statespaces:
            ss = self._statespaces[prefix]
            recreate_statespace = (
                not ss.obs.shape[1] == self.endog.shape[1] or
                not ss.design.shape[2] == self.design.shape[2] or
                not ss.obs_intercept.shape[1] == self.obs_intercept.shape[1] or
                not ss.obs_cov.shape[2] == self.obs_cov.shape[2] or
                not ss.transition.shape[2] == self.transition.shape[2] or
                not (ss.state_intercept.shape[1] ==
                     self.state_intercept.shape[1]) or
                not ss.selection.shape[2] == self.selection.shape[2] or
                not ss.state_cov.shape[2] == self.state_cov.shape[2]
            )

        # If the dtype-specific _statespace model does not exist, create it
        if prefix not in self._statespaces or recreate_statespace:
            # Setup the base statespace object
            cls = prefix_statespace_map[prefix]
            self._statespaces[prefix] = cls(
                self._representations[prefix]['obs'],
                self._representations[prefix]['design'],
                self._representations[prefix]['obs_intercept'],
                self._representations[prefix]['obs_cov'],
                self._representations[prefix]['transition'],
                self._representations[prefix]['state_intercept'],
                self._representations[prefix]['selection'],
                self._representations[prefix]['state_cov']
            )

        # Determine if we need to re-create the filter
        # (definitely need to recreate if we recreated the _statespace object)
        recreate_filter = recreate_statespace
        if recreate and not recreate_filter and prefix in self._kalman_filters:
            kalman_filter = self._kalman_filters[prefix]

            recreate_filter = (
                not kalman_filter.k_endog == self.k_endog or
                not kalman_filter.k_states == self.k_states or
                not kalman_filter.k_posdef == self.k_posdef or
                not kalman_filter.k_posdef == self.k_posdef or
                not kalman_filter.conserve_memory == conserve_memory or
                not kalman_filter.loglikelihood_burn == loglikelihood_burn
            )

        # If the dtype-specific _kalman_filter does not exist (or if we need
        # to recreate it), create it
        if prefix not in self._kalman_filters or recreate_filter:
            if recreate_filter:
                # Delete the old filter
                del self._kalman_filters[prefix]
            # Setup the filter
            cls = prefix_kalman_filter_map[prefix]
            self._kalman_filters[prefix] = cls(
                self._statespaces[prefix], filter_method, inversion_method,
                stability_method, conserve_memory, tolerance,
                loglikelihood_burn
            )
        # Otherwise, update the filter parameters
        else:
            self._kalman_filters[prefix].filter_method = filter_method
            self._kalman_filters[prefix].inversion_method = inversion_method
            self._kalman_filters[prefix].stability_method = stability_method
            self._kalman_filters[prefix].tolerance = tolerance

        # (Re-)initialize the statespace model
        if self.initialization == 'known':
            self._statespaces[prefix].initialize_known(
                self._initial_state.astype(dtype),
                self._initial_state_cov.astype(dtype)
            )
        elif self.initialization == 'approximate_diffuse':
            self._statespaces[prefix].initialize_approximate_diffuse(
                self._initial_variance
            )
        elif self.initialization == 'stationary':
            self._statespaces[prefix].initialize_stationary()
        else:
            raise RuntimeError('Statespace model not initialized.')

        return prefix, dtype

    def filter(self, filter_method=None, inversion_method=None,
               stability_method=None, conserve_memory=None, tolerance=None,
               loglikelihood_burn=None,
               recreate=True, return_loglike=False, results_class=None,
               *args, **kwargs):
        """
        Apply the Kalman filter to the statespace model.

        Parameters
        ----------
        filter_method : int, optional
            Determines which Kalman filter to use. Default is conventional.
        inversion_method : int, optional
            Determines which inversion technique to use. Default is by Cholesky
            decomposition.
        stability_method : int, optional
            Determines which numerical stability techniques to use. Default is
            to enforce symmetry of the predicted state covariance matrix.
        conserve_memory : int, optional
            Determines what output from the filter to store. Default is to
            store everything.
        tolerance : float, optional
            The tolerance at which the Kalman filter determines convergence to
            steady-state. Default is 1e-19.
        loglikelihood_burn : int, optional
            The number of initial periods during which the loglikelihood is not
            recorded. Default is 0.
        recreate : bool, optional
            Whether or not to consider re-creating the underlying _statespace
            or filter objects (e.g. due to changing parameters, etc.). Often
            set to false during maximum likelihood estimation. Default is true.
        return_loglike : bool, optional
            Whether to only return the loglikelihood rather than a full
            `FilterResults` object. Default is False.
        """

        if results_class is None:
            results_class = self.filter_results_class

        # Initialize the filter
        prefix, dtype = self._initialize_filter(
            filter_method, inversion_method, stability_method, conserve_memory,
            tolerance, loglikelihood_burn, recreate, return_loglike,
            *args, **kwargs
        )

        # Run the filter
        self._kalman_filters[prefix]()

        if return_loglike:
            return np.array(self._kalman_filters[prefix].loglikelihood)
        else:
            return results_class(self, self._kalman_filters[prefix])

    def loglike(self, loglikelihood_burn=None, *args, **kwargs):
        """
        Calculate the loglikelihood associated with the statespace model.

        Parameters
        ----------
        loglikelihood_burn : int, optional
            The number of initial periods during which the loglikelihood is not
            recorded. Default is 0.
        """
        if loglikelihood_burn is None:
            loglikelihood_burn = self.loglikelihood_burn
        kwargs['return_loglike'] = True
        return np.sum(self.filter(*args, **kwargs)[loglikelihood_burn:])


class FilterResults(object):
    """
    Results from applying the Kalman filter to a state space model.

    Takes a snapshot of a Statespace model and accompanying Kalman filter,
    saving the model representation and filter output.

    Parameters
    ----------
    model : Representation
        A Statespace representation
    kalman_filter : _statespace.{'s','c','d','z'}KalmanFilter
        A Kalman filter object.
    """
    def __init__(self, model, kalman_filter):
        # Data type
        self.prefix = model.prefix
        self.dtype = model.dtype

        # Copy the model dimensions
        self.nobs = model.nobs
        self.k_endog = model.k_endog
        self.k_states = model.k_states
        self.k_posdef = model.k_posdef
        self.time_invariant = model.time_invariant

        # Save the state space representation at the time
        self.endog = model.endog
        self.design = model._design.copy()
        self.obs_intercept = model._obs_intercept.copy()
        self.obs_cov = model._obs_cov.copy()
        self.transition = model._transition.copy()
        self.state_intercept = model._state_intercept.copy()
        self.selection = model._selection.copy()
        self.state_cov = model._state_cov.copy()

        self.missing = np.array(model._statespaces[self.prefix].missing,
                                copy=True)
        self.nmissing = np.array(model._statespaces[self.prefix].nmissing,
                                 copy=True)

        # Save the final shapes of the matrices
        self.shapes = dict(model.shapes)
        for name in self.shapes.keys():
            if name == 'obs':
                continue
            self.shapes[name] = getattr(self, name).shape
        self.shapes['obs'] = self.endog.shape

        # Save the state space initialization
        self.initialization = model.initialization
        self.initial_state = np.array(kalman_filter.model.initial_state,
                                      copy=True)
        self.initial_state_cov = np.array(
            kalman_filter.model.initial_state_cov, copy=True
        )

        # Save Kalman filter parameters
        self.filter_method = kalman_filter.filter_method
        self.inversion_method = kalman_filter.inversion_method
        self.stability_method = kalman_filter.stability_method
        self.conserve_memory = kalman_filter.conserve_memory
        self.tolerance = kalman_filter.tolerance
        self.loglikelihood_burn = kalman_filter.loglikelihood_burn

        # Save Kalman filter output
        self.converged = bool(kalman_filter.converged)
        self.period_converged = kalman_filter.period_converged

        self.filtered_state = np.array(kalman_filter.filtered_state, copy=True)
        self.filtered_state_cov = np.array(kalman_filter.filtered_state_cov, copy=True)
        self.predicted_state = np.array(kalman_filter.predicted_state, copy=True)
        self.predicted_state_cov = np.array(
            kalman_filter.predicted_state_cov, copy=True
        )
        # Note: use forecasts rather than forecast, so as not to interfer
        # with the `forecast` methods in subclasses
        self.forecasts = np.array(kalman_filter.forecast, copy=True)
        self.forecasts_error = np.array(kalman_filter.forecast_error, copy=True)
        self.forecasts_error_cov = np.array(kalman_filter.forecast_error_cov, copy=True)
        self.loglikelihood = np.array(kalman_filter.loglikelihood, copy=True)

        # Fill in missing values in the forecast, forecast error, and
        # forecast error covariance matrix (this is required due to how the
        # Kalman filter implements observations that are completely missing)
        # Construct the predictions, forecasts
        if not (self.conserve_memory & MEMORY_NO_FORECAST or
                self.conserve_memory & MEMORY_NO_PREDICTED):
            for t in range(self.nobs):
                design_t = 0 if self.design.shape[2] == 1 else t
                obs_cov_t = 0 if self.obs_cov.shape[2] == 1 else t
                obs_intercept_t = 0 if self.obs_intercept.shape[1] == 1 else t

                # Skip anything that is less than completely missing
                if self.nmissing[t] < self.k_endog:
                    continue

                self.forecasts[:, t] = np.dot(
                    self.design[:, :, design_t], self.predicted_state[:, t]
                ) + self.obs_intercept[:, obs_intercept_t]
                self.forecasts_error[:, t] = np.nan
                self.forecasts_error_cov[:, :, t] = np.dot(
                    np.dot(self.design[:, :, design_t],
                           self.predicted_state_cov[:, :, t]),
                    self.design[:, :, design_t].T
                ) + self.obs_cov[:, :, obs_cov_t]

    @property
    def standardized_forecast_error(self):
        standardized_forecast_error = np.zeros(self.forecasts_error.shape,
                                               dtype=self.dtype)

        for t in range(self.forecasts_error_cov.shape[2]):
            upper = np.linalg.cholesky(self.forecasts_error_cov[:, :, t]).T
            standardized_forecast_error[:, t] = np.dot(
                upper, self.forecasts_error[:, t]
            )

        return standardized_forecast_error

    def predict(self, start=None, end=None, dynamic=None, full_results=False,
                *args, **kwargs):
        """
        Statespace model in-sample and out-of-sample prediction.

        Parameters
        ----------
        start : int, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast will be at start.
        end : int, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast will be at end.
        dynamic : int, optional
            Specifies the number of steps ahead for each in-sample prediction.
            If not specified, then in-sample predictions are one-step-ahead.

        Returns
        -------
        predict : array
            The predicted values.

        Notes
        -----
        All prediction is performed by applying the deterministic part of the
        measurement equation using the predicted state variables.

        Out-of-sample prediction first applies the Kalman filter to missing
        data for the number of periods desired to obtain the predicted states.
        """
        # Cannot predict if we do not have appropriate arrays
        if (self.conserve_memory & MEMORY_NO_FORECAST or
           self.conserve_memory & MEMORY_NO_PREDICTED):
            raise ValueError('Predict is not possible if memory conservation'
                             ' has been used to avoid storing forecasts or'
                             ' predicted values.')

        # Get the start and the end of the entire prediction range
        if start is None:
            start = 0
        elif start < 0:
            raise ValueError('Cannot predict values period to the sample.')
        if end is None:
            end = self.nobs

        # Total number of predicted values
        npredicted = end - start

        # Short-circuit if end is before start
        if npredicted < 0:
            return (np.zeros((self.k_endog, 0)),
                    np.zeros((self.k_endog, self.k_endog, 0)))

        # Get the number of forecasts to make after the end of the sample
        # Note: this may be larger than npredicted if the predict command was
        # called, for example, to calculate forecasts for nobs+10 through
        # nobs+20, because the operations below will need to start forecasts no
        # later than the end of the sample and go through `end`. Any
        # calculations prior to `start` will be ignored.
        nforecast = max(0, end - self.nobs)

        # Get the total size of the in-sample prediction component (whether via
        # one-step-ahead or dynamic prediction)
        nsample = npredicted - nforecast

        # Get the number of periods until dynamic forecasting is used
        if dynamic > nsample:
            warn('Dynamic prediction specified for more steps-ahead (%d) than'
                 ' there are observations in the specified range (%d).'
                 ' `dynamic` has been automatically adjusted to %d. If'
                 ' possible, you may want to set `start` to be earlier.'
                 % (dynamic, nsample, nsample))
            dynamic = nsample

        if dynamic is None or dynamic is False:
            dynamic = nsample
        ndynamic = nsample - dynamic

        if dynamic < 0:
            raise ValueError('Prediction cannot be specified with a negative'
                             ' dynamic prediction offset.')

        # Get the number of in-sample, one-step-ahead predictions
        ninsample = nsample - ndynamic

        # Total numer of left-padded zeros
        # Two cases would have this as non-zero. Here are some examples:
        # - If start = 4 and dynamic = 4, then npadded >= 4 so that even the
        #   `start` observation has dynamic of 4
        # - If start = 10 and nobs = 5, then npadded >= 5 because the
        #   intermediate forecasts are required for the desired forecasts.
        npadded = max(0, start - dynamic, start - self.nobs)

        # Construct the design and observation intercept and covariance
        # matrices for start-npadded:end. If not time-varying in the original
        # model, then they will be copied over if none are provided in
        # `kwargs`. Otherwise additional matrices must be provided in `kwargs`.
        representation = {}
        for name, shape in self.shapes.items():
            if name == 'obs':
                continue
            mat = getattr(self, name)
            if shape[-1] == 1:
                representation[name] = mat
            elif len(shape) == 3:
                representation[name] = mat[:, :, start-npadded:]
            else:
                representation[name] = mat[:, start-npadded:]

        # Update the matrices from kwargs for forecasts
        warning = ('Model has time-invariant %s matrix, so the %s'
                   ' argument to `predict` has been ignored.')
        exception = ('Forecasting for models with time-varying %s matrix'
                     ' requires an updated time-varying matrix for the'
                     ' period to be forecasted.')
        if nforecast > 0:
            for name, shape in self.shapes.items():
                if name == 'obs':
                    continue
                if representation[name].shape[-1] == 1:
                    if name in kwargs:
                        warn(warning % (name, name))
                elif name not in kwargs:
                    raise ValueError(exception % name)
                else:
                    mat = np.asarray(kwargs[name])
                    if len(shape) == 2:
                        validate_vector_shape('obs_intercept', mat.shape,
                                              shape[0], nforecast)
                        if mat.ndim < 2 or not mat.shape[1] == nforecast:
                            raise ValueError(exception % name)
                        representation[name] = np.c_[representation[name], mat]
                    else:
                        validate_matrix_shape(name, mat.shape, shape[0],
                                              shape[1], nforecast)
                        if mat.ndim < 3 or not mat.shape[2] == nforecast:
                            raise ValueError(exception % name)
                        representation[name] = np.c_[representation[name], mat]

        # Construct the predicted state and covariance matrix for each time
        # period depending on whether that time period corresponds to
        # one-step-ahead prediction, dynamic prediction, or out-of-sample
        # forecasting.

        # If we only have simple prediction, then we can use the already saved
        # Kalman filter output
        if ndynamic == 0 and nforecast == 0:
            result = self
        else:
            # Construct the new endogenous array - notice that it has
            # npredicted + npadded values (rather than the entire start array,
            # in case the number of observations is large and we don't want to
            # re-run the entire filter just for forecasting)
            endog = np.empty((self.k_endog, nforecast))
            endog.fill(np.nan)
            endog = np.c_[self.endog[:, start-npadded:], endog]

            # Setup the new statespace representation
            model_kwargs = {
                'filter_method': self.filter_method,
                'inversion_method': self.inversion_method,
                'stability_method': self.stability_method,
                'conserve_memory': self.conserve_memory,
                'tolerance': self.tolerance,
                'loglikelihood_burn': self.loglikelihood_burn
            }
            model_kwargs.update(representation)
            model = Representation(
                endog.T, self.k_states, self.k_posdef, **model_kwargs
            )
            model.initialize_known(
                self.predicted_state[:, 0],
                self.predicted_state_cov[:, :, 0]
            )
            model._initialize_filter(*args, **kwargs)

            result = self._predict(ninsample, ndynamic, nforecast, model)

        if full_results:
            return result
        else:
            return (
                result.forecasts[:, npadded:],
                result.forecasts_error[:, npadded:],
                result.forecasts_error_cov[:, :, npadded:]
            )

    def _predict(self, ninsample, ndynamic, nforecast, model, *args, **kwargs):
        # Get the underlying filter
        kfilter = model._kalman_filter

        # Save this (which shares memory with the memoryview on which the
        # Kalman filter will be operating) so that we can replace actual data
        # with predicted data during dynamic forecasting
        endog = model._representations[model.prefix]['obs']

        for t in range(kfilter.model.nobs):
            # Run the Kalman filter for the first `ninsample` periods (for
            # which dynamic computation will not be performed)
            if t < ninsample:
                next(kfilter)
            # Perform dynamic prediction
            elif t < ninsample+ndynamic:
                design_t = 0 if model.design.shape[2] == 1 else t
                obs_intercept_t = 0 if model.obs_intercept.shape[1] == 1 else t

                # Predict endog[:, t] given `predicted_state` calculated in
                # previous iteration (i.e. t-1)
                endog[:, t] = np.dot(
                    model.design[:, :, design_t],
                    kfilter.predicted_state[:, t]
                ) + model.obs_intercept[:, obs_intercept_t]

                # Advance Kalman filter
                next(kfilter)
            # Perform any (one-step-ahead) forecasting
            else:
                next(kfilter)

        # Return the predicted state and predicted state covariance matrices
        return FilterResults(model, kfilter)
