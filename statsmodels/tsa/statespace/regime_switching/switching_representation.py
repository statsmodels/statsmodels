import numpy as np
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.statespace.representation import FrozenRepresentation


class SwitchingRepresentation(object):
    '''
    Kim Filter representation
    '''

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

        if k_regimes < 1:
            raise ValueError('Only multiple regimes are available.'
                    'Consider using regular KalmanFilter.')

        self._k_endog = k_endog
        self._k_states = k_states
        self._k_regimes = k_regimes
        self._dtype = dtype

        design = self._prepare_data_for_regimes(design,
                self._per_regime_dims['design'])
        obs_intercept = self._prepare_data_for_regimes(obs_intercept,
                self._per_regime_dims['obs_intercept'])
        obs_cov = self._prepare_data_for_regimes(obs_cov,
                self._per_regime_dims['obs_cov'])
        transition = self._prepare_data_for_regimes(transition,
                self._per_regime_dims['transition'])
        state_intercept = self._prepare_data_for_regimes(state_intercept,
                self._per_regime_dims['state_intercept'])
        selection = self._prepare_data_for_regimes(selection,
                self._per_regime_dims['selection'])
        state_cov = self._prepare_data_for_regimes(state_cov,
                self._per_regime_dims['state_cov'])

        # Kalman filters for each regime

        kwargs['alternate_timing'] = True
        self._regime_kalman_filters = [KalmanFilter(k_endog, k_states,
                dtype=dtype, design=design[i], obs_intercept=obs_intercept[i],
                obs_cov=obs_cov[i], transition=transition[i],
                state_intercept=state_intercept[i], selection=selection[i],
                state_cov=state_cov[i], **kwargs) for i in range(k_regimes)]

        self.set_regime_transition(regime_transition)

    def set_regime_transition(self, regime_transition):

        dtype = self._dtype
        k_regimes = self._k_regimes

        if regime_transition is not None:
            regime_transition = np.asarray(regime_transition, dtype=dtype)
            if regime_transition.shape != (k_regimes, k_regimes):
                raise ValueError('Regime transition matrix should have shape'
                        ' (k_regimes, k_regimes)')
            if not self._is_left_stochastic(regime_transition):
                raise ValueError(
                        'Provided regime transition matrix is not stochastic')
            self._log_regime_transition = np.log(regime_transition)
        else:
            self._log_regime_transition = -np.log(k_regimes)

    def _prepare_data_for_regimes(self, data, per_regime_dims):
        '''
        This duplicates representation matrix for every regime, if it's
        provided in the only example.
        '''

        k_regimes = self._k_regimes

        if data is None:
            return [None for _ in range(k_regimes)]

        data = np.asarray(data, dtype=self._dtype)

        if len(data.shape) == per_regime_dims:
            return [data for _ in range(k_regimes)]

        if data.shape[0] != k_regimes:
            raise ValueError('First dimension is not k_regimes')

        return data

    def __getitem__(self, key):

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

        if matrix_name == 'regime_transition':
            if get_slice:
                return np.exp(self._log_regime_transition)[slice_]
            else:
                return np.exp(self._log_regime_transition)

        if matrix_name not in self._per_regime_dims:
            raise IndexError('"%s" is an invalid state space matrix name.' \
                    % matrix_name)

        return np.asarray([regime_filter[matrix_name] for regime_filter in \
                self._regime_kalman_filters])

    def __setitem__(self, key, value):
        '''
        When slice is provided, `__setitem__` is forced to be broadcasted to
        every regime's filter.
        '''

        if type(key) == str:
            set_slice = False
            matrix_name = key
        elif type(key) == tuple:
            set_slice = True
            matrix_name = key[0]
            slice_ = key[1:]
        else:
            raise ValueError('First index must be the name of a valid state' \
                    'space matrix.')

        if matrix_name == 'regime_transition':
            if set_slice:
                self._log_regime_transition[slice_] = np.log(value)
            else:
                self.set_regime_transition(value)
            return

        if matrix_name not in self._per_regime_dims:
            raise IndexError('"%s" is an invalid state space matrix name.' \
                    % matrix_name)

        if set_slice:
            for regime_filter in self._regime_kalman_filters:
                regime_filter[key] = value
        else:
            value = self._prepare_data_for_regimes(value,
                self._per_regime_dims[matrix_name])

            for regime_filter, regime_value in zip(self._regime_kalman_filters,
                    value):
                regime_filter[key] = regime_value

    def _is_left_stochastic(self, matrix):

        eps = 1e-8

        if np.any(matrix < -eps):
            return False
        matrix[matrix < 0] = 0
        if not np.all(np.fabs(matrix.sum(axis=0) - 1) < eps):
            return False
        return True

    @property
    def k_regimes(self):
        return self._k_regimes

    @property
    def k_endog(self):
        return self._k_endog

    @property
    def dtype(self):
        return self._dtype

    @property
    def regime_filters(self):
        return self._regime_kalman_filters

    @property
    def initialization(self):
        return self.regime_filters[0].initialization

    @property
    def initial_variance(self):
        return self.regime_filters[0].initial_variance

    @initial_variance.setter
    def initial_variance(self, value):
        for regime_filter in self.regime_filters:
            regime_filter.initial_variance = value

    @property
    def loglikelihood_burn(self):
        return self._loglikelihood_burn

    @loglikelihood_burn.setter
    def loglikelihood_burn(self, value):
        self._loglikelihood_burn = value
        for regime_filter in self.regime_filters:
            regime_filter.loglikelihood_burn = value

    @property
    def tolerance(self):
        return self.regime_filters[0].tolerance

    @tolerance.setter
    def tolerance(self, value):
        for regime_filter in self.regime_filters:
            regime_filter.tolerance = value

    @property
    def _complex_endog(self):
        return self.regime_filters[0]._complex_endog

    def _initialize_filter(self, **kwargs):

        kwargs['filter_timing'] = 1

        for regime_filter in self._regime_kalman_filters:
            regime_filter._initialize_filter(**kwargs)

    def set_filter_method(self, **kwargs):

        for regime_filter in self._regime_kalman_filters:
            regime_filter.set_filter_method(**kwargs)

    def set_inversion_method(self, **kwargs):

        for regime_filter in self._regime_kalman_filters:
            regime_filter.set_inversion_method(**kwargs)

    def set_stability_method(self, **kwargs):

        for regime_filter in self._regime_kalman_filters:
            regime_filter.set_stability_method(**kwargs)

    def set_conserve_memory(self, **kwargs):

        for regime_filter in self._regime_kalman_filters:
            regime_filter.set_conserve_memory(**kwargs)

    def bind(self, endog):

        self.endog = endog

        for regime_filter in self._regime_kalman_filters:
            regime_filter.bind(endog)

        self._nobs = self._regime_kalman_filters[0].nobs

    def initialize_known(self, initial_state, initial_state_cov):

        k_regimes = self._k_regimes
        regime_filters = self._regime_kalman_filters

        initial_state = self._prepare_data_for_regimes(initial_state, 1)
        initial_state_cov = self._prepare_data_for_regimes(initial_state_cov, 2)

        for i in range(k_regimes):
            regime_filters[i].initialize_known(initial_state[i],
                    initial_state_cov[i])

    def initialize_stationary(self):

        for regime_filter in self._regime_kalman_filters:
            regime_filter.initialize_stationary()

    def initialize_known_regime_probs(self, initial_regime_probs):
        '''
        Initialization of marginal regime distribution at t=0.
        '''

        self._initial_regime_logprobs = np.log(initial_regime_probs)

    def initialize_uniform_regime_probs(self):
        '''
        Initialization of marginal regime distribution at t=0.
        '''

        k_regimes = self._k_regimes
        self._initial_regime_logprobs = -np.log(k_regimes)

    def initialize_stationary_regime_probs(self):
        '''
        Initialization of marginal regime distribution at t=0.
        '''

        k_regimes = self._k_regimes
        dtype = self._dtype

        regime_transition = np.exp(self._log_regime_transition)

        constraint_matrix = np.vstack((regime_transition - \
                np.identity(k_regimes, dtype=dtype),
                np.ones((1, k_regimes), dtype=dtype)))

        candidate = np.linalg.pinv(constraint_matrix)[:, -1]

        eps = 1e-8

        if np.any(candidate < -eps):
            raise RuntimeError('Regime switching chain doesn\'t have ' \
                'a stationary distribution')

        candidate[candidate < 0] = 0

        self._initial_regime_logprobs = np.log(candidate)

    @property
    def initial_regime_probs():
        '''
        Marginal regime distribution at t=0.
        '''

        return np.exp(self._initial_regime_logprobs)

    def _initialize_filters(self, filter_method=None, inversion_method=None,
            stability_method=None, conserve_memory=None, tolerance=None,
            complex_step=False):

        kfilters = []
        state_init_kwargs = []

        for regime_filter in self._regime_kalman_filters:
            prefix = regime_filter._initialize_filter(
                    filter_method=filter_method,
                    inversion_method=inversion_method,
                    stability_method=stability_method,
                    conserve_memory=conserve_memory, tolerance=tolerance)[0]
            kfilters.append(regime_filter._kalman_filters[prefix])

            state_init_kwargs.append({'prefix': prefix,
                    'complex_step': complex_step})
            #regime_filter._initialize_state(prefix=prefix,
            #        complex_step=complex_step)

        self._kfilters = kfilters
        self._state_init_kwargs = state_init_kwargs

class FrozenSwitchingRepresentation(object):

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

        # Save the state space representation
        self.endog = model.endog
        self.log_regime_transition = model._log_regime_transition

        self.regime_representations = \
                [FrozenRepresentation(regime_kalman_filter) for \
                regime_kalman_filter in model._regime_kalman_filters]

    def __getitem__(self, key):

        if key == 'regime_transition':
            return np.exp(self.log_regime_transition)

        if key not in SwitchingRepresentation._per_regime_dims:
            raise IndexError('"%s" is an invalid state space matrix name.' \
                    % key)

        return np.asarray([getattr(representation, key) for representation in \
                self.regime_representations])
