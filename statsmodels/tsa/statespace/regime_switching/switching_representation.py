"""
Markov Switching State Space Representation

Author: Valery Likhosherstov
License: Simplified-BSD
"""
import numpy as np
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.statespace.representation import FrozenRepresentation
from .tools import _is_left_stochastic

class SwitchingRepresentation(object):
    r"""
    Markov switching state space representation of a time series process.

    Parameters
    ----------
    k_endog : int
        The number of variables in the process.
    k_states : int
        The dimension of the unobserved state process.
    k_regimes : int
        The number of switching regimes.
    dtype : dtype, optional
        Default datatype of the state space matrices. Default is `np.float64`.
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
    regime_transition : array_like, optional
        Left stochastic matrix of regime transition probabilities. Default are
        equal switch probabilites.
    **kwargs
        Additional keyword arguments, passed to each KalmanFilter instance,
        representing corresponding regime.

    Attributes
    ----------
    nobs : int
        The number of observations. Initialized after data binding.
    k_endog : int
        The dimension of the observation series.
    k_states : int
        The dimension of the unobserved state process.
    k_regimes : int
        The number of switching regimes.
    regime_filters: array
        `KalmanFilter` instances, corresponding to each regime.
    regime_transition: array
        Left stochastic matrix of regime transition probabilities.

    Notes
    -----
    This class stores `k_regimes` `KalmanFilter` instances and left stochastic
    `regime_transition` matrix. Each Kalman Filter stores representation of its
    regime. Interface is partially copied from `Representation` class.

    A general Markov switching state space model is of the form

    .. math::

        y_t & = Z_{S_t} \alpha_t + d_{S_t} + \varepsilon_t \\
        \alpha_t & = T_{S_t} \alpha_{t-1} + c_{S_t} + R_{S_t} \eta_t \\

    where :math:`y_t` refers to the observation vector at time :math:`t`,
    :math:`\alpha_t` refers to the (unobserved) state vector at time
    :math:`t`, :math:`S_t` refers to the (unobserved) regime at time :math:`t`
    and where the irregular components are defined as

    .. math::

        \varepsilon_t \sim N(0, H_{S_t}) \\
        \eta_t \sim N(0, Q_{S_t}) \\

    The remaining variables (:math:`Z_{S_t}, d_{S_t}, H_{S_t}, T_{S_t},
    c_{S_t}, R_{S_t}, Q_{S_t}`) in the equations are matrices describing
    the process. Their variable names and dimensions are as follows

    Z : `design`          :math:`(k\_regimes \times k\_endog \times k\_states
        \times nobs)`

    d : `obs_intercept`   :math:`(k\_regimes \times k\_endog \times nobs)`

    H : `obs_cov`         :math:`(k\_regimes \times k\_endog \times k\_endog
        \times nobs)`

    T : `transition`      :math:`(k\_regimes \times k\_states \times k\_states
        \times nobs)`

    c : `state_intercept` :math:`(k\_regimes \times k\_states \times nobs)`

    R : `selection`       :math:`(k\_regimes \times k\_states \times k\_posdef
        \times nobs)`

    Q : `state_cov`       :math:`(k\_regimes \times k\_posdef \times k\_posdef
        \times nobs)`

    In the case that one of the matrices is time-invariant (so that, for
    example, :math:`Z_t = Z_{t+1} ~ \forall ~ t`), its last dimension may
    be of size :math:`1` rather than size `nobs`.
    In the case that one of the matrices is the same for all regimes, it
    shouldn't have the first `k_regimes` dimension.

    Unobserved process of switching regimes is described by Markovian law:

    .. math::

        Pr[S_t = i | S_{t-1} = j] = p_{ij} \\

    where :math:`p` is a left stochastic matrix (that is, all its elements are
    non-negative and the sum of elements of each row is 1), referenced as
    `regime_transition`.

    References
    ----------
    .. [1] Kim, Chang-Jin, and Charles R. Nelson. 1999.
        "State-Space Models with Regime Switching:
        Classical and Gibbs-Sampling Approaches with Applications".
        MIT Press Books. The MIT Press.
    """

    # Dimensions of state space matrices in KalmanFilter class (without first
    # `k_regimes` dimension). This is used to detect whether provided matrix is
    # not switching and has to be broadcasted to every regime.
    _per_regime_dims = {
            'design': 3,
            'obs_intercept': 2,
            'obs_cov': 3,
            'transition': 3,
            'state_intercept': 2,
            'selection': 3,
            'state_cov': 3
    }

    def __init__(self, k_endog, k_states, k_regimes, dtype=np.float64,
            design=None, obs_intercept=None, obs_cov=None, transition=None,
            state_intercept=None, selection=None, state_cov=None,
            regime_transition=None, **kwargs):

        # All checks for correct types and dimensions, etc. are delegated to
        # KalmanFilter instances. This class is supposed to accomplish only
        # the high-level logic.

        if k_regimes < 1:
            raise ValueError('Only multiple regimes are available.'
                    'Consider using regular KalmanFilter.')

        self._k_endog = k_endog
        self._k_states = k_states
        self._k_regimes = k_regimes
        self._dtype = dtype

        # Broadcasting matrices, if they are provided in non-switching shape
        design = self._broadcast_per_regime(design,
                self._per_regime_dims['design'])
        obs_intercept = self._broadcast_per_regime(obs_intercept,
                self._per_regime_dims['obs_intercept'])
        obs_cov = self._broadcast_per_regime(obs_cov,
                self._per_regime_dims['obs_cov'])
        transition = self._broadcast_per_regime(transition,
                self._per_regime_dims['transition'])
        state_intercept = self._broadcast_per_regime(state_intercept,
                self._per_regime_dims['state_intercept'])
        selection = self._broadcast_per_regime(selection,
                self._per_regime_dims['selection'])
        state_cov = self._broadcast_per_regime(state_cov,
                self._per_regime_dims['state_cov'])

        # Using Kim and Nelson timing convention is required for Kim filter
        kwargs['alternate_timing'] = True

        # Kalman filters for each regime
        # Also, all low-level checks are carried in these initializations
        self._regime_kalman_filters = [KalmanFilter(k_endog, k_states,
                dtype=dtype, design=design[i], obs_intercept=obs_intercept[i],
                obs_cov=obs_cov[i], transition=transition[i],
                state_intercept=state_intercept[i], selection=selection[i],
                state_cov=state_cov[i], **kwargs) for i in range(k_regimes)]

        # Check and store transition matrix
        self._set_regime_transition(regime_transition)

    def _set_regime_transition(self, regime_transition):
        # This method does some checks of regime transition matrix and stores
        # its value.

        dtype = self._dtype
        k_regimes = self._k_regimes

        # All probabilites are stored and treated by their logarithms
        if regime_transition is not None:

            # If regime transition is provided as list
            regime_transition = np.asarray(regime_transition, dtype=dtype)

            if regime_transition.shape != (k_regimes, k_regimes):
                raise ValueError('Regime transition matrix should have shape'
                        ' (k_regimes, k_regimes)')

            # Regime transition matrix is required to be left stochastic
            if not _is_left_stochastic(regime_transition):
                raise ValueError(
                        'Provided regime transition matrix is not stochastic')

            self._log_regime_transition = np.log(regime_transition)
        else:
            # If regime transition matrix is not provided by user, set all
            # probabilites to one value - `1/k_regimes`.
            self._log_regime_transition = -np.log(k_regimes)

    def _broadcast_per_regime(self, matrix, per_regime_dims):
        # This method checks if provided state space representation matrix is in
        # a non-switching shape and then duplicates matrix for every regime.
        # After that, a version of a matrix for the corresponding regime is
        # passed to `KalmanFilter` initializer.

        k_regimes = self._k_regimes

        # If matrix is not provided, pass `None` to every `KalmanFilter`
        if matrix is None:
            return [None for _ in range(k_regimes)]

        matrix = np.asarray(matrix, dtype=self._dtype)

        # `per_regime_dims` contains a number of dimensions for corresponding
        # matrix in a non-switching case
        if len(matrix.shape) == per_regime_dims:
            return [matrix for _ in range(k_regimes)]

        # In the switching case, matrix has one more (first) dimension of size
        # `k_regimes`
        if matrix.shape[0] != k_regimes:
            raise ValueError('First dimension is not k_regimes')

        return matrix

    def __getitem__(self, key):

        # Check if matrix name or slice is provided
        if type(key) == str:
            get_slice = False
            matrix_name = key
        elif type(key) == tuple:
            get_slice = True
            matrix_name = key[0]
            slice_ = key[1:]
        else:
            raise ValueError('First index must be the name of a valid state' \
                    'space matrix.')

        # Regime transition matrix is stored in this class, so it is treated
        # differently from other matrices.
        if matrix_name == 'regime_transition':
            if get_slice:
                return np.exp(self._log_regime_transition)[slice_]
            else:
                return np.exp(self._log_regime_transition)

        # Check if matrix name belongs to state space matrices set
        if matrix_name not in self._per_regime_dims:
            raise IndexError('"%s" is an invalid state space matrix name.' \
                    % matrix_name)

        # Combine corresponding matrices from all regimes together and return
        # result.
        return np.asarray([regime_filter[key] for regime_filter in \
                self._regime_kalman_filters])

    def __setitem__(self, key, value):
        # When no slice is provided, value can contain a battery with values
        # for every regime. Otherwise, the same value is set to matrix slice
        # of every regime.

        # Check if matrix name or slice is provided
        if type(key) == str:
            set_slice = False
            matrix_name = key
        elif type(key) == tuple:
            set_slice = True
            matrix_name = key[0]
            slice_ = key[1:]
        else:
            raise ValueError('First index must be the name of a valid state' \
                    ' space matrix.')

        # Regime transition matrix is stored in this class, so it is treated
        # differently from other matrices.
        if matrix_name == 'regime_transition':
            if set_slice:

                self._log_regime_transition[slice_] = np.log(value)

                # Check if value doesn't violate left-stochastic feature
                if not _is_left_stochastic(
                        np.exp(self._log_regime_transition)):
                    raise ValueError('Regime transition matrix is not' \
                            ' left-stochastic anymore')
            else:
                self._set_regime_transition(value)
            return

        # Check if matrix name belongs to state space matrices set
        if matrix_name not in self._per_regime_dims:
            raise IndexError('"%s" is an invalid state space matrix name.' \
                    % matrix_name)

        if set_slice:
            # Broadcast one value to every regime
            for regime_filter in self._regime_kalman_filters:
                regime_filter[key] = value
        else:
            # Prepare data to set to every regime
            value = self._broadcast_per_regime(value,
                self._per_regime_dims[matrix_name])

            for regime_filter, regime_value in zip(self._regime_kalman_filters,
                    value):
                regime_filter[key] = regime_value

    @property
    def nobs(self):
        """
        (int) The number of observations. Initialized after data binding.
        Otherwise, will throw `AttributeError` exception.
        """
        return self._nobs

    @property
    def k_regimes(self):
        """
        (int) The number of switching regimes.
        """
        return self._k_regimes

    @property
    def k_endog(self):
        """
        (int) The dimension of the observation series.
        """
        return self._k_endog

    @property
    def k_states(self):
        """
        (int) The dimension of the unobserved state process.
        """
        return self._k_states

    @property
    def dtype(self):
        """
        (dtype) Datatype of currently active representation matrices.
        """
        return self._dtype

    @property
    def regime_filters(self):
        """
        (array) `KalmanFilter` instances, corresponding to each regime.
        """
        return self._regime_kalman_filters

    @property
    def regime_transition(self):
        """
        (array) Left stochastic matrix of regime transition probabilities.
        """
        return np.exp(self._log_regime_transition)

    @regime_transition.setter
    def regime_transition(self, value):
        """
        (array) Left stochastic matrix of regime transition probabilities.
        """
        self._set_regime_transition(value)

    @property
    def initialization(self):
        """
        (str) Kalman filter initalization method. This property handling is
        delegated to `KalmanFilter`. Since diffuse approximation is not
        available for Kim Filter, `initialization` can be `None`, 'stationary'
        or 'known'.
        """
        return self.regime_filters[0].initialization

    @property
    def _complex_endog(self):
        # A flag for complex data, handled by `Representation` class.
        # This is used internally in `MLEModel` class.
        return self.regime_filters[0]._complex_endog

    def bind(self, endog):
        """
        This method propagates the action of the same name among regime filters.
        See `Representation.bind` documentation for details.
        """

        # Copying for the case 
        for regime_filter in self._regime_kalman_filters:
            regime_filter.bind(endog)

        # `endog` is the same for every regime filter.
        self.endog = self._regime_kalman_filters[0].endog
        # `nobs` is the same for every regime filter.
        self._nobs = self._regime_kalman_filters[0].nobs

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
            If its shape is `(k_states,)`, then this value is broadcasted to
            all regimes.
            If its shape is `(k_regimes, k_states)`, then it is assumed that
            every regime has its own `initial_state`.
        initial_state_cov : array_like
            Known covariance matrix of the initial state vector.
            If its shape is `(k_states, k_states)`, then this value is
            broadcasted to all regimes.
            If its shape is `(k_regimes, k_states, k_states)`, then it is
            assumed that every regime has its own `initial_state_cov`.
        """

        k_regimes = self._k_regimes
        regime_filters = self._regime_kalman_filters

        # Broadcast matrices, if they are provided in non-switching shape
        initial_state = self._broadcast_per_regime(initial_state, 1)
        initial_state_cov = self._broadcast_per_regime(initial_state_cov, 2)

        # Delegating initialization to `KalmanFilter`
        for i in range(k_regimes):
            regime_filters[i].initialize_known(initial_state[i],
                    initial_state_cov[i])

    def initialize_stationary(self):
        """
        This method propagates the action of the same name among regime filters.
        See `Representation.initialize_stationary` documentation for details.
        """

        for regime_filter in self._regime_kalman_filters:
            regime_filter.initialize_stationary()

    def initialize_known_regime_probs(self, initial_regime_probs):
        """
        Initialize marginal regime distribution at t=0.

        Parameters
        ----------
        initial_regime_probs : array_like
            Array of shape `(k_regimes,)`, representating marginal regime
            distribution at t=0.
        """

        self._initial_regime_logprobs = np.log(initial_regime_probs)

    def initialize_uniform_regime_probs(self):
        """
        Initialize marginal regime distribution at t=0 with uniform
        distribution.
        """

        k_regimes = self._k_regimes
        # All probabilities are set to `1/k_regimes`
        self._initial_regime_logprobs = -np.log(k_regimes)

    def initialize_stationary_regime_probs(self):
        """
        Initialize marginal regime distribution at t=0 with stationary
        distribution of the Markov chain, described by `regime_transition`
        matrix.
        """

        k_regimes = self._k_regimes
        dtype = self._dtype

        # Switch from logarithms to actual values
        regime_transition = np.exp(self._log_regime_transition)

        # Stationary distribution is calculated by solving the system of linear
        # equations.

        # This matrix, multiplied by stationary distribution (in a column
        # shape), gives a column of `k_regimes` zeros and one on the bottom.
        coefficient_matrix = np.vstack((regime_transition - \
                np.identity(k_regimes, dtype=dtype),
                np.ones((1, k_regimes), dtype=dtype)))

        # Solve the system by OLS approach. Multiplication by "dependent
        # variable" column is performed implicitly by selecting the last column.
        candidate = np.linalg.pinv(coefficient_matrix)[:, -1]

        eps = 1e-8

        # Check whether all values are non-negative (by eps)
        if np.any(candidate < -eps):
            raise RuntimeError('Regime switching chain doesn\'t have ' \
                    'a stationary distribution')

        # Set very small negative values to zero
        candidate[candidate < 0] = 0

        # This is required to perform a normalization
        if candidate.sum() < eps:
            raise RuntimeError('Regime switching chain doesn\'t have ' \
                    'a stationary distribution')

        # Normalization, since OLS solution can be imprecise
        candidate /= candidate.sum()

        # Store distribution in the class
        self._initial_regime_logprobs = np.log(candidate)

    @property
    def initial_regime_probs(self):
        """
        Marginal regime distribution at t=0.
        """
        return np.exp(self._initial_regime_logprobs)


class FrozenSwitchingRepresentation(object):
    """
    Frozen Markov switching state space model

    Takes a snapshot of a Markov switching state space model

    Parameters
    ----------
    model : SwitchingRepresentation
        A Markov switching state space representation

    Attributes
    ----------
    dtype : dtype
        Datatype of representation matrices
    nobs : int
        The number of observations. Initialized after data binding.
    k_endog : int
        The dimension of the observation series.
    k_states : int
        The dimension of the unobserved state process.
    k_regimes : int
        The number of switching regimes.
    endog : array
        The observation vector.
    log_regime_transition: array
        Square `k_regimes` x `k_regimes` matrix of regime transition
        log-probabilities.
    regime_representations: array
        `FrozenRepresentation` instances, corresponding to each regime.
    """
    _model_attributes = [
        'model', 'dtype', 'nobs', 'k_endog', 'k_states', 'k_regimes',
        'endog', 'log_regime_transition', 'regime_representations'
    ]
    _attributes = _model_attributes

    def __init__(self, model):

        # Initialize all attributes to None
        for name in self._attributes:
            setattr(self, name, None)

    def update_representation(self, model):
        # Model
        self.model = model

        # Data type
        self.dtype = model.dtype

        # Copy the model dimensions
        self.nobs = model._nobs
        self.k_endog = model._k_endog
        self.k_states = model._k_states
        self.k_regimes = model._k_regimes

        # Save endog data
        self.endog = model.endog

        # Save regime transition matrix
        self.log_regime_transition = model._log_regime_transition

        # Create and store frozen representations of each regime
        self.regime_representations = \
                [FrozenRepresentation(regime_kalman_filter) for \
                regime_kalman_filter in model._regime_kalman_filters]

    def __getitem__(self, key):

        # regime transition matrix is treated differently from other matrices
        if key == 'regime_transition':
            return np.exp(self.log_regime_transition)

        # Check if matrix name belongs to state space matrices set
        if key not in SwitchingRepresentation._per_regime_dims:
            raise IndexError('"%s" is an invalid state space matrix name.' \
                    % key)

        # Combine corresponding matrices from all regimes together and return
        # result.
        return np.asarray([getattr(representation, key) for representation in \
                self.regime_representations])
