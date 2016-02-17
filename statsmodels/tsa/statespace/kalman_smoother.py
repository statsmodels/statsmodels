"""
Kalman Smoother

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

from collections import namedtuple
import numpy as np
from .representation import OptionWrapper
from .kalman_filter import KalmanFilter, FilterResults

SMOOTHER_STATE = 0x01          # Durbin and Koopman (2012), Chapter 4.4.2
SMOOTHER_STATE_COV = 0x02      # ibid., Chapter 4.4.3
SMOOTHER_DISTURBANCE = 0x04    # ibid., Chapter 4.5
SMOOTHER_DISTURBANCE_COV = 0x08    # ibid., Chapter 4.5
SMOOTHER_ALL = (
    SMOOTHER_STATE | SMOOTHER_STATE_COV | SMOOTHER_DISTURBANCE |
    SMOOTHER_DISTURBANCE_COV
)

_SmootherOutput = namedtuple('_SmootherOutput', (
    'tmp_L'
    ' scaled_smoothed_estimator scaled_smoothed_estimator_cov'
    ' smoothing_error'
    ' smoothed_state smoothed_state_cov'
    ' smoothed_state_disturbance smoothed_state_disturbance_cov'
    ' smoothed_measurement_disturbance smoothed_measurement_disturbance_cov'
))


def _kalman_smooth(model, kfilter, smoother_output):
    # Create storage
    scaled_smoothed_estimator = None
    scaled_smoothed_estimator_cov = None
    smoothing_error = None
    smoothed_state = None
    smoothed_state_cov = None
    smoothed_state_disturbance = None
    smoothed_state_disturbance_cov = None
    smoothed_measurement_disturbance = None
    smoothed_measurement_disturbance_cov = None

    # Intermediate values
    tmp_L = np.zeros((model.k_states, model.k_states, model.nobs),
                     dtype=kfilter.dtype)

    if smoother_output & (SMOOTHER_STATE | SMOOTHER_DISTURBANCE):
        scaled_smoothed_estimator = np.zeros((model.k_states, model.nobs + 1),
                                             dtype=kfilter.dtype)
        smoothing_error = np.zeros((model.k_endog, model.nobs),
                                   dtype=kfilter.dtype)
    if smoother_output & (SMOOTHER_STATE_COV | SMOOTHER_DISTURBANCE_COV):
        scaled_smoothed_estimator_cov = (
            np.zeros((model.k_states, model.k_states, model.nobs + 1),
                     dtype=kfilter.dtype)
        )

    # State smoothing
    if smoother_output & SMOOTHER_STATE:
        smoothed_state = np.zeros((model.k_states, model.nobs),
                                  dtype=kfilter.dtype)
    if smoother_output & SMOOTHER_STATE_COV:
        smoothed_state_cov = (
            np.zeros((model.k_states, model.k_states, model.nobs),
                     dtype=kfilter.dtype)
        )

    # Disturbance smoothing
    if smoother_output & SMOOTHER_DISTURBANCE:
        smoothed_state_disturbance = np.zeros((model.k_posdef, model.nobs),
                                              dtype=kfilter.dtype)
        smoothed_measurement_disturbance = (
            np.zeros((model.k_endog, model.nobs),
                     dtype=kfilter.dtype)
        )
    if smoother_output & SMOOTHER_DISTURBANCE_COV:
        smoothed_state_disturbance_cov = (
            np.zeros((model.k_posdef, model.k_posdef, model.nobs),
                     dtype=kfilter.dtype)
        )
        smoothed_measurement_disturbance_cov = (
            np.zeros((model.k_endog, model.k_endog, model.nobs),
                     dtype=kfilter.dtype)
        )

    # Smoothing recursions
    for t in range(model.nobs - 1, -1, -1):
        # Get the appropriate (possibly time-varying) indices
        design_t = 0 if kfilter.design.shape[2] == 1 else t
        obs_cov_t = 0 if kfilter.obs_cov.shape[2] == 1 else t
        transition_t = 0 if kfilter.transition.shape[2] == 1 else t
        selection_t = 0 if kfilter.selection.shape[2] == 1 else t
        state_cov_t = 0 if kfilter.state_cov.shape[2] == 1 else t

        # Get references to representation matrices and Kalman filter output
        design = model.design[:, :, design_t]
        obs_cov = model.obs_cov[:, :, obs_cov_t]
        transition = model.transition[:, :, transition_t]
        selection = model.selection[:, :, selection_t]
        state_cov = model.state_cov[:, :, state_cov_t]

        predicted_state = kfilter.predicted_state[:, t]
        predicted_state_cov = kfilter.predicted_state_cov[:, :, t]
        kalman_gain = kfilter.kalman_gain[:, :, t]
        forecasts_error_cov = kfilter.forecasts_error_cov[:, :, t]
        forecasts_error = kfilter.forecasts_error[:, t]

        # Create a temporary matrix
        tmp_L[:, :, t] = transition - kalman_gain.dot(design)
        L = tmp_L[:, :, t]

        # Perform the recursion

        # Intermediate values
        F_inv = np.linalg.inv(forecasts_error_cov)
        if smoother_output & (SMOOTHER_STATE | SMOOTHER_DISTURBANCE):
            smoothing_error[:, t] = (
                F_inv.dot(forecasts_error) -
                kalman_gain.transpose().dot(scaled_smoothed_estimator[:, t])
            )
            scaled_smoothed_estimator[:, t - 1] = (
                design.transpose().dot(smoothing_error[:, t]) +
                transition.transpose().dot(scaled_smoothed_estimator[:, t])
            )
        if smoother_output & (SMOOTHER_STATE_COV | SMOOTHER_DISTURBANCE_COV):
            scaled_smoothed_estimator_cov[:, :, t - 1] = (
                design.transpose().dot(F_inv).dot(design) +
                L.transpose().dot(
                    scaled_smoothed_estimator_cov[:, :, t]
                ).dot(L)
            )

        # State smoothing
        if smoother_output & SMOOTHER_STATE:
            smoothed_state[:, t] = (
                predicted_state +
                predicted_state_cov.dot(scaled_smoothed_estimator[:, t - 1])
            )
        if smoother_output & SMOOTHER_STATE_COV:
            smoothed_state_cov[:, :, t] = (
                predicted_state_cov -
                predicted_state_cov.dot(
                    scaled_smoothed_estimator_cov[:, :, t - 1]
                ).dot(predicted_state_cov)
            )

        # Disturbance smoothing
        if smoother_output & (SMOOTHER_DISTURBANCE | SMOOTHER_DISTURBANCE_COV):
            QR = state_cov.dot(selection.transpose())

        if smoother_output & SMOOTHER_DISTURBANCE:
            smoothed_state_disturbance[:, t] = (
                QR.dot(scaled_smoothed_estimator[:, t])
            )
            smoothed_measurement_disturbance[:, t] = (
                obs_cov.dot(smoothing_error[:, t])
            )

        if smoother_output & SMOOTHER_DISTURBANCE_COV:
            smoothed_state_disturbance_cov[:, :, t] = (
                state_cov -
                QR.dot(
                    scaled_smoothed_estimator_cov[:, :, t]
                ).dot(QR.transpose())
            )

            smoothed_measurement_disturbance_cov[:, :, t] = (
                obs_cov - obs_cov.dot(
                    F_inv + kalman_gain.transpose().dot(
                        scaled_smoothed_estimator_cov[:, :, t]
                    ).dot(kalman_gain)
                ).dot(obs_cov)
            )

    output = _SmootherOutput(
        tmp_L=tmp_L,
        scaled_smoothed_estimator=scaled_smoothed_estimator,
        scaled_smoothed_estimator_cov=scaled_smoothed_estimator_cov,
        smoothing_error=smoothing_error,
        smoothed_state=smoothed_state,
        smoothed_state_cov=smoothed_state_cov,
        smoothed_state_disturbance=smoothed_state_disturbance,
        smoothed_state_disturbance_cov=smoothed_state_disturbance_cov,
        smoothed_measurement_disturbance=smoothed_measurement_disturbance,
        smoothed_measurement_disturbance_cov=(
            smoothed_measurement_disturbance_cov
        ),
    )

    return output


class KalmanSmoother(KalmanFilter):
    r"""
    State space representation of a time series process, with Kalman filter
    and smoother.

    Parameters
    ----------
    k_endog : array_like or integer
        The observed time-series process :math:`y` if array like or the
        number of variables in the process if an integer.
    k_states : int
        The dimension of the unobserved state process.
    k_posdef : int, optional
        The dimension of a guaranteed positive definite covariance matrix
        describing the shocks in the measurement equation. Must be less than
        or equal to `k_states`. Default is `k_states`.
    results_class : class, optional
        Default results class to use to save filtering output. Default is
        `SmootherResults`. If specified, class must extend from
        `SmootherResults`.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices, for Kalman filtering options, or for Kalman smoothing
        options. See `Representation` for more details.
    """

    smoother_outputs = [
        'smoother_state', 'smoother_state_cov', 'smoother_disturbance',
        'smoother_disturbance_cov', 'smoother_all',
    ]

    smoother_state = OptionWrapper('smoother_output', SMOOTHER_STATE)
    smoother_state_cov = OptionWrapper('smoother_output', SMOOTHER_STATE_COV)
    smoother_disturbance = (
        OptionWrapper('smoother_output', SMOOTHER_DISTURBANCE)
    )
    smoother_disturbance_cov = (
        OptionWrapper('smoother_output', SMOOTHER_DISTURBANCE_COV)
    )
    smoother_all = OptionWrapper('smoother_output', SMOOTHER_ALL)

    # Default smoother options
    smoother_output = SMOOTHER_ALL

    def __init__(self, k_endog, k_states, k_posdef=None, results_class=None,
                 **kwargs):
        # Set the default results class
        if results_class is None:
            results_class = SmootherResults

        super(KalmanSmoother, self).__init__(
            k_endog, k_states, k_posdef, results_class=results_class, **kwargs
        )

        # Set the smoother output
        self.set_smoother_output(**kwargs)

    def set_smoother_output(self, smoother_output=None, **kwargs):
        """
        Set the smoother output

        The smoother can produce several types of results. The smoother output
        variable controls which are calculated and returned.

        Parameters
        ----------
        smoother_output : integer, optional
            Bitmask value to set the smoother output to. See notes for details.
        **kwargs
            Keyword arguments may be used to influence the smoother output by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        The smoother output is defined by a collection of boolean flags, and
        is internally stored as a bitmask. The methods available are:

        SMOOTHER_STATE = 0x01
            Calculate and return the smoothed states.
        SMOOTHER_STATE_COV = 0x02
            Calculate and return the smoothed state covariance matrices.
        SMOOTHER_DISTURBANCE = 0x04
            Calculate and return the smoothed state and observation
            disturbances.
        SMOOTHER_DISTURBANCE_COV = 0x08
            Calculate and return the covariance matrices for the smoothed state
            and observation disturbances.
        SMOOTHER_ALL
            Calculate and return all results.

        If the bitmask is set directly via the `smoother_output` argument, then
        the full method must be provided.

        If keyword arguments are used to set individual boolean flags, then
        the lowercase of the method must be used as an argument name, and the
        value is the desired value of the boolean flag (True or False).

        Note that the smoother output may also be specified by directly
        modifying the class attributes which are defined similarly to the
        keyword arguments.

        The default smoother output is SMOOTHER_ALL.

        If performance is a concern, only those results which are needed should
        be specified as any results that are not specified will not be
        calculated. For example, if the smoother output is set to only include
        SMOOTHER_STATE, the smoother operates much more quickly than if all
        output is required.

        Examples
        --------
        >>> mod = sm.tsa.statespace.KalmanSmoother(1,1)
        >>> mod.smoother_output
        15
        >>> mod.set_smoother_output(smoother_output=0)
        >>> mod.smoother_state = True
        >>> mod.smoother_output
        1
        >>> mod.smoother_state
        True
        """
        if smoother_output is not None:
            self.smoother_output = smoother_output
        for name in KalmanSmoother.smoother_outputs:
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def smooth(self, smoother_output=None, results=None):
        """
        Apply the Kalman smoother to the statespace model.

        Parameters
        ----------
        smoother_output : int, optional
            Determines which Kalman smoother output calculate. Default is all
            (including state, disturbances, and all covariances).
        results : class or object, optional
            If a class, then that class is instantiated and returned with the
            result of both filtering and smoothing.
            If an object, then that object is updated with the smoothing data.
            If None, then a SmootherResults object is returned with both
            filtering and smoothing results.
        prefix : string
            The prefix of the datatype. Usually only used internally.

        Returns
        -------
        SmootherResults object
        """
        if smoother_output is None:
            smoother_output = self.smoother_output

        new_results = not isinstance(results, SmootherResults)

        # Set the class to be the default results class, if None provided
        if results is None:
            results = self.results_class

        # Initialize the filter and statespace object if necessary
        prefix, dtype, create_filter, create_statespace = (
            self._initialize_filter()
        )

        # Instantiate a new results object, if required
        if isinstance(results, type):
            if not issubclass(results, SmootherResults):
                raise ValueError('Invalid results class provided.')
            results = results(self)

        # Run the filter
        kfilter = self._kalman_filters[prefix]
        if not kfilter.t == self.nobs:
            self._initialize_state()
            kfilter()

        # Update the results object with filtered output
        # Update the model features; unless we had to recreate the
        # statespace, only update the filter options
        results.update_representation(self, only_options=not create_statespace)
        if new_results or create_filter:
            results.update_filter(kfilter)

        # For now we can't handle missing observations
        if np.sum(results.nmissing) > 0:
            raise RuntimeError('Kalman smoother does not currently support'
                               ' smoothing with missing values.')

        # Run the smoother and update the output
        output = _kalman_smooth(self, results, smoother_output)
        results.update_smoother(output)

        return results


class SmootherResults(FilterResults):
    """
    Results from applying the Kalman smoother and/or filter to a state space
    model.

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
    smoother_output : int
        Bitmask representing the generated Kalman smoothing output
    scaled_smoothed_estimator : array
        The scaled smoothed estimator at each time period.
    scaled_smoothed_estimator_cov : array
        The scaled smoothed estimator covariance matrices at each time period.
    smoothing_error : array
        The smoothing error covariance matrices at each time period.
    smoothed_state : array
        The smoothed state at each time period.
    smoothed_state_cov : array
        The smoothed state covariance matrices at each time period.
    smoothed_measurement_disturbance : array
        The smoothed measurement at each time period.
    smoothed_state_disturbance : array
        The smoothed state at each time period.
    smoothed_measurement_disturbance_cov : array
        The smoothed measurement disturbance covariance matrices at each time
        period.
    smoothed_state_disturbance_cov : array
        The smoothed state disturbance covariance matrices at each time period.
    """

    _smoother_attributes = [
        'smoother_output', 'scaled_smoothed_estimator',
        'scaled_smoothed_estimator_cov', 'smoothing_error',
        'smoothed_state', 'smoothed_state_cov',
        'smoothed_measurement_disturbance', 'smoothed_state_disturbance',
        'smoothed_measurement_disturbance_cov',
        'smoothed_state_disturbance_cov'
    ]

    _smoother_options = KalmanSmoother.smoother_outputs

    _attributes = FilterResults._model_attributes + _smoother_attributes

    def update_representation(self, model, only_options=False):
        """
        Update the results to match a given model

        Parameters
        ----------
        model : Representation
            The model object from which to take the updated values.
        only_options : boolean, optional
            If set to true, only the smoother and filter options are updated,
            and the state space representation is not updated. Default is
            False.

        Notes
        -----
        This method is rarely required except for internal usage.
        """
        super(SmootherResults, self).update_representation(model, only_options)

        # Save the options as boolean variables
        for name in self._smoother_options:
            setattr(self, name, getattr(model, name, None))

        # Initialize holders for smoothed forecasts
        self._smoothed_forecasts = None
        self._smoothed_forecasts_error = None
        self._smoothed_forecasts_error_cov = None

    def update_smoother(self, smoother):
        """
        Update the smoother results

        Parameters
        ----------
        smoother : KalmanSmoother
            The model object from which to take the updated values.

        Notes
        -----
        This method is rarely required except for internal usage.
        """
        # Copy the appropriate output
        attributes = []

        # Since update_representation will already have been called, we can
        # use the boolean options smoother_* and know they match the smoother
        # itself
        if self.smoother_state or self.smoother_disturbance:
            attributes.append('scaled_smoothed_estimator')
        if self.smoother_state_cov or self.smoother_disturbance_cov:
            attributes.append('scaled_smoothed_estimator_cov')
        if self.smoother_state:
            attributes.append('smoothed_state')
        if self.smoother_state_cov:
            attributes.append('smoothed_state_cov')
        if self.smoother_disturbance:
            attributes += [
                'smoothing_error',
                'smoothed_measurement_disturbance',
                'smoothed_state_disturbance'
            ]
        if self.smoother_disturbance_cov:
            attributes += [
                'smoothed_measurement_disturbance_cov',
                'smoothed_state_disturbance_cov'
            ]

        for name in self._smoother_attributes:
            if name == 'smoother_output':
                pass
            elif name in attributes:
                setattr(
                    self, name,
                    np.array(getattr(smoother, name, None), copy=True)
                )
            else:
                setattr(self, name, None)

        # Adjustments

        # For r_t (and similarly for N_t), what was calculated was
        # r_T, ..., r_{-1}, and stored such that
        # scaled_smoothed_estimator[0] == r_{-1}. We only want r_0, ..., r_T
        # so exclude the zeroth element so that the time index is consistent
        # with the other returned output
        if 'scaled_smoothed_estimator' in attributes:
            self.scaled_smoothed_estimator = (
                self.scaled_smoothed_estimator[:, 1:]
            )
        if 'scaled_smoothed_estimator_cov' in attributes:
            self.scaled_smoothed_estimator_cov = (
                self.scaled_smoothed_estimator_cov[:, 1:]
            )

        # Clear the smoothed forecasts
        self._smoothed_forecasts = None
        self._smoothed_forecasts_error = None
        self._smoothed_forecasts_error_cov = None

    def _get_smoothed_forecasts(self):
        if self._smoothed_forecasts is None:
            # Initialize empty arrays
            self._smoothed_forecasts = np.zeros(self.forecasts.shape,
                                                dtype=self.dtype)
            self._smoothed_forecasts_error = (
                np.zeros(self.forecasts_error.shape, dtype=self.dtype)
            )
            self._smoothed_forecasts_error_cov = (
                np.zeros(self.forecasts_error_cov.shape, dtype=self.dtype)
            )

            for t in range(self.nobs):
                design_t = 0 if self.design.shape[2] == 1 else t
                obs_cov_t = 0 if self.obs_cov.shape[2] == 1 else t
                obs_intercept_t = 0 if self.obs_intercept.shape[1] == 1 else t

                # For completely missing observations
                if self.nmissing[t] == self.k_endog:
                    # We can recover forecasts
                    self._smoothed_forecasts[:, t] = np.dot(
                        self.design[:, :, design_t], self.smoothed_state[:, t]
                    ) + self.obs_intercept[:, obs_intercept_t]
                    self._smoothed_forecasts_error[:, t] = np.nan
                    self._smoothed_forecasts_error_cov[:, :, t] = np.dot(
                        np.dot(self.design[:, :, design_t],
                               self.smoothed_state_cov[:, :, t]),
                        self.design[:, :, design_t].T
                    ) + self.obs_cov[:, :, obs_cov_t]
                # For partially missing observations
                elif self.nmissing[t] > 0:
                    self._smoothed_forecasts[:, t] = np.nan
                    self._smoothed_forecasts_error[:, t] = np.nan
                    self._smoothed_forecasts_error_cov[:, :, t] = np.nan
                else:
                    self._smoothed_forecasts[:, t] = np.dot(
                        self.design[:, :, design_t], self.smoothed_state[:, t]
                    ) + self.obs_intercept[:, obs_intercept_t]

                    self._smoothed_forecasts_error[:, t] = (
                        self.endog[:, t] - self._smoothed_forecasts[:, t]
                    )

                    self._smoothed_forecasts_error_cov[:, :, t] = np.dot(
                        np.dot(self.design[:, :, design_t],
                               self.smoothed_state_cov[:, :, t]),
                        self.design[:, :, design_t].T
                    ) + self.obs_cov[:, :, obs_cov_t]
        return (
            self._smoothed_forecasts,
            self._smoothed_forecasts_error,
            self._smoothed_forecasts_error_cov
        )

    @property
    def smoothed_forecasts(self):
        return self._get_smoothed_forecasts()[0]

    @property
    def smoothed_forecasts_error(self):
        return self._get_smoothed_forecasts()[1]

    @property
    def smoothed_forecasts_error_cov(self):
        return self._get_smoothed_forecasts()[2]
