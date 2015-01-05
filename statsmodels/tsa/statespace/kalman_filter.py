"""
State Space Representation and Kalman Filter

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

from warnings import warn

import numpy as np
from .representation import Representation, FrozenRepresentation
from .tools import (
    prefix_kalman_filter_map, validate_vector_shape, validate_matrix_shape
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
INVERT_NUMPY = 0x20

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

class KalmanFilter(Representation):
    r"""
    State space representation of a time series process, with Kalman filter
    """

    def __init__(self, *args, **kwargs):
        super(KalmanFilter, self).__init__(*args, **kwargs)

        # Setup the underlying Kalman filter storage
        self._kalman_filters = {}

        # Filter options
        self.loglikelihood_burn = kwargs.get('loglikelihood_burn', 0)
        self.results_class = kwargs.get('results_class', FilterResults)

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
    def _kalman_filter(self):
        prefix = self.prefix
        if prefix in self._kalman_filters:
            return self._kalman_filters[prefix]
        return None

    def _initialize_filter(self, filter_method=None, inversion_method=None,
                           stability_method=None, conserve_memory=None,
                           tolerance=None, loglikelihood_burn=None,
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

        # Make sure we have endog
        if self.endog is None:
            raise RuntimeError('Must bind a dataset to the model before'
                               ' filtering or smoothing.')

        # Initialize the representation matrices
        prefix, dtype, create_statespace = self._initialize_representation()

        # Determine if we need to (re-)create the filter
        # (definitely need to recreate if we recreated the _statespace object)
        create_filter = create_statespace or prefix not in self._kalman_filters
        if not create_filter:
            kalman_filter = self._kalman_filters[prefix]

            create_filter = (
                not kalman_filter.conserve_memory == conserve_memory or
                not kalman_filter.loglikelihood_burn == loglikelihood_burn
            )

        # If the dtype-specific _kalman_filter does not exist (or if we need
        # to re-create it), create it
        if create_filter:
            if prefix in self._kalman_filters:
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
            self._kalman_filters[prefix].set_filter_method(filter_method, False)
            self._kalman_filters[prefix].inversion_method = inversion_method
            self._kalman_filters[prefix].stability_method = stability_method
            self._kalman_filters[prefix].tolerance = tolerance
            # conserve_memory and loglikelihood_burn changes always lead to
            # re-created filters

        return prefix, dtype, create_filter, create_statespace

    def filter(self, filter_method=None, inversion_method=None,
               stability_method=None, conserve_memory=None, tolerance=None,
               loglikelihood_burn=None, results=None,
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
        results : class, object, or {'loglikelihood'}, optional
            If a class which is a subclass of FilterResults, then that class is
            instantiated and returned with the result of filtering. Classes
            must subclass FilterResults.
            If an object, then that object is updated with the new filtering
            results.
            If the string 'loglikelihood', then only the loglikelihood is
            returned as an ndarray.
            If None, then the default results object is updated with the
            result of filtering.
        """
        # Set the class to be the default results class, if None provided
        if results is None:
            results = self.results_class

        # Initialize the filter
        prefix, dtype, create_filter, create_statespace = (
            self._initialize_filter(
                filter_method, inversion_method, stability_method,
                conserve_memory, tolerance, loglikelihood_burn,
                *args, **kwargs
            )
        )
        kfilter = self._kalman_filters[prefix]

        # Instantiate a new results object, if required
        if isinstance(results, type):
            if not issubclass(results, FilterResults):
                raise ValueError
            results = results(self)

        # Initialize the state
        self._initialize_state(prefix=prefix)

        # Run the filter
        kfilter()

        # We may just want the loglikelihood
        if results == 'loglikelihood':
            results = np.array(self._kalman_filters[prefix].loglikelihood,
                               copy=True)
        # Otherwise update the results object
        else:
            # Update the model features if we had to recreate the statespace
            if create_statespace:
                results.update_representation(self)
            results.update_filter(kfilter)

        return results

    def loglike(self, loglikelihood_burn=None, *args, **kwargs):
        """
        Calculate the loglikelihood associated with the statespace model.

        Parameters
        ----------
        loglikelihood_burn : int, optional
            The number of initial periods during which the loglikelihood is not
            recorded. Default is 0.

        Returns
        -------
        loglike : float
            The joint loglikelihood.
        """
        if self.filter_method & MEMORY_NO_LIKELIHOOD:
            raise RuntimeError('Cannot compute loglikelihood if'
                               ' MEMORY_NO_LIKELIHOOD option is selected.')
        if loglikelihood_burn is None:
            loglikelihood_burn = self.loglikelihood_burn
        kwargs['results'] = 'loglikelihood'
        return np.sum(self.filter(*args, **kwargs)[loglikelihood_burn:])

class FilterResults(FrozenRepresentation):
    """
    Results from applying the Kalman filter to a state space model.

    Parameters
    ----------
    model : Representation
        A Statespace representation

    Attributes
    ----------
    nobs : int
        Number of observations.
    k_endog : int
        The dimension of the observation series.
    k_states : int
        The dimension of the unobserved state process.
    k_posdef : int
        The dimension of a guaranteed positive definite covariance matrix
        describing the shocks in the measurement equation.
    dtype : dtype
        Datatype of representation matrices
    prefix : str
        BLAS prefix of representation matrices
    shapes : dictionary of name:tuple
        A dictionary recording the shapes of each of the representation
        matrices as tuples.
    endog : array
        The observation vector.
    design : array
        The design matrix, :math:`Z`.
    obs_intercept : array
        The intercept for the observation equation, :math:`d`.
    obs_cov : array
        The covariance matrix for the observation equation :math:`H`.
    transition : array
        The transition matrix, :math:`T`.
    state_intercept : array
        The intercept for the transition equation, :math:`c`.
    selection : array
        The selection matrix, :math:`R`.
    state_cov : array
        The covariance matrix for the state equation :math:`Q`.
    missing : array of bool
        An array of the same size as `endog`, filled with boolean values that
        are True if the corresponding entry in `endog` is NaN and False
        otherwise.
    nmissing : array of int
        An array of size `nobs`, where the ith entry is the number (between 0
        and k_endog) of NaNs in the ith row of the `endog` array.
    time_invariant : bool
        Whether or not the representation matrices are time-invariant
    initialization : str
        Kalman filter initialization method.
    initial_state : array_like
        The state vector used to initialize the Kalamn filter.
    initial_state_cov : array_like
        The state covariance matrix used to initialize the Kalamn filter.
    filter_method : int
        Bitmask representing the Kalman filtering method
    inversion_method : int
        Bitmask representing the method used to invert the forecast error
        covariance matrix.
    stability_method : int
        Bitmask representing the methods used to promote numerical stability in
        the Kalman filter recursions.
    conserve_memory : int
        Bitmask representing the selected memory conservation method.
    tolerance : float
        The tolerance at which the Kalman filter determines convergence to
        steady-state.
    loglikelihood_burn : int
        The number of initial periods during which the loglikelihood is not
        recorded.
    converged : bool
        Whether or not the Kalman filter converged.
    period_converged : int
        The time period in which the Kalman filter converged.
    filtered_state : array
        The filtered state vector at each time period.
    filtered_state_cov : array
        The filtered state covariance matrix at each time period.
    predicted_state : array
        The predicted state vector at each time period.
    predicted_state_cov : array
        The predicted state covariance matrix at each time period.
    kalman_gain : array
        The Kalman gain at each time period.
    forecasts : array
        The one-step-ahead forecasts of observations at each time period.
    forecasts_error : array
        The forecast errors at each time period.
    forecasts_error_cov : array
        The forecast error covariance matrices at each time period.
    loglikelihood : array
        The loglikelihood values at each time period.
    collapsed_forecasts : array
        If filtering using collapsed observations, stores the one-step-ahead
        forecasts of collapsed observations at each time period.
    collapsed_forecasts_error : array
        If filtering using collapsed observations, stores the one-step-ahead
        forecast errors of collapsed observations at each time period.
    collapsed_forecasts_error_cov : array
        If filtering using collapsed observations, stores the one-step-ahead
        forecast error covariance matrices of collapsed observations at each
        time period.
    standardized_forecast_error : array
        The standardized forecast errors
    """
    _filter_attributes = [
        'filter_method', 'inversion_method', 'stability_method',
        'conserve_memory', 'tolerance', 'loglikelihood_burn', 'converged',
        'period_converged', 'filtered_state', 'filtered_state_cov',
        'predicted_state', 'predicted_state_cov',
        'forecasts', 'forecasts_error', 'forecasts_error_cov',
        'loglikelihood'
    ]

    _attributes = FrozenRepresentation._model_attributes + _filter_attributes

    def __init__(self, model):
        super(FilterResults, self).__init__(model)

        # Setup caches for uninitialized objects
        self._kalman_gain = None
        self._standardized_forecasts_error = None

    def update_filter(self, kalman_filter):
        # State initialization
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
                if self.nmissing[t] == self.k_endog:
                    self.forecasts_error[:, t] = np.nan
                else:
                    self.forecasts_error[:, t] = self.endog[:, t] - self.forecasts[:, t]
                self.forecasts_error_cov[:, :, t] = np.dot(
                    np.dot(self.design[:, :, design_t],
                           self.predicted_state_cov[:, :, t]),
                    self.design[:, :, design_t].T
                ) + self.obs_cov[:, :, obs_cov_t]

    @property
    def kalman_gain(self):
        if self._kalman_gain is None:
            # k x n
            self._kalman_gain = np.zeros(
                (self.k_states, self.k_endog, self.nobs), dtype=self.dtype)
            for t in range(self.nobs):
                design_t = 0 if self.design.shape[2] == 1 else t
                transition_t = 0 if self.transition.shape[2] == 1 else t
                self._kalman_gain[:, :, t] = np.dot(
                    np.dot(
                        self.transition[:, :, transition_t],
                        self.predicted_state_cov[:, :, t]
                    ),
                    np.dot(
                        np.transpose(self.design[:, :, design_t]),
                        np.linalg.inv(self.forecasts_error_cov[:, :, t])
                    )
                )
        return self._kalman_gain

    @property
    def standardized_forecasts_error(self):
        if self._standardized_forecasts_error is None:
            from scipy import linalg
            self._standardized_forecasts_error = np.zeros(
                self.forecasts_error.shape, dtype=self.dtype)

            for t in range(self.forecasts_error_cov.shape[2]):
                upper, _ = linalg.cho_factor(self.forecasts_error_cov[:, :, t],
                                         check_finite=False)
                self._standardized_forecasts_error[:, t] = (
                    linalg.solve_triangular(upper, self.forecasts_error[:, t],
                                            check_finite=False))

        return self._standardized_forecasts_error

    def predict(self, start=None, end=None, dynamic=None, full_results=False,
                *args, **kwargs):
        """
        In-sample and out-of-sample prediction for state space models generally

        Parameters
        ----------
        start : int, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast will be at start.
        end : int, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast will be at end.
        dynamic : int or boolean or None, optional
            Specifies the number of steps ahead for each in-sample prediction.
            If not specified, then in-sample predictions are one-step-ahead.
            False and None are interpreted as 0. Default is False.
        full_results : boolean, optional
            If True, returns a FilterResults instance; if False returns a
            tuple with forecasts, the forecast errors, and the forecast error
            covariance matrices. Default is False.

        Returns
        -------
        results : FilterResults or tuple
            Either a FilterResults object (if `full_results=True`) or else a
            tuple of forecasts, the forecast errors, and the forecast error
            covariance matrices otherwise.

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
            raise ValueError('Cannot predict values previous to the sample.')
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
            endog = np.asfortranarray(endog)

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
            model = KalmanFilter(
                endog, self.k_states, self.k_posdef, **model_kwargs
            )
            model.initialize_known(
                self.predicted_state[:, 0],
                self.predicted_state_cov[:, :, 0]
            )
            model._initialize_filter(*args, **kwargs)
            model._initialize_state(*args, **kwargs)

            result = self._predict(ninsample, ndynamic, nforecast, model)

        if full_results:
            return result
        else:
            return result.forecasts[:, npadded:]

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
        results = FilterResults(model)
        results.update_filter(kfilter)
        return results
