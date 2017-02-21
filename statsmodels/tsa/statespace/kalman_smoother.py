"""
State Space Representation and Kalman Filter, Smoother

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np

from statsmodels.tsa.statespace.representation import OptionWrapper
from statsmodels.tsa.statespace.kalman_filter import (KalmanFilter,
                                                      FilterResults)
from statsmodels.tsa.statespace.tools import (
    reorder_missing_matrix, reorder_missing_vector, copy_index_matrix)
from statsmodels.tsa.statespace import tools

SMOOTHER_STATE = 0x01              # Durbin and Koopman (2012), Chapter 4.4.2
SMOOTHER_STATE_COV = 0x02          # ibid., Chapter 4.4.3
SMOOTHER_DISTURBANCE = 0x04        # ibid., Chapter 4.5
SMOOTHER_DISTURBANCE_COV = 0x08    # ibid., Chapter 4.5
SMOOTHER_STATE_AUTOCOV = 0x10      # ibid., Chapter 4.7
SMOOTHER_ALL = (
    SMOOTHER_STATE | SMOOTHER_STATE_COV | SMOOTHER_DISTURBANCE |
    SMOOTHER_DISTURBANCE_COV | SMOOTHER_STATE_AUTOCOV
)

SMOOTH_CONVENTIONAL = 0x01
SMOOTH_CLASSICAL = 0x02
SMOOTH_ALTERNATIVE = 0x04
SMOOTH_UNIVARIATE = 0x08


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
        'smoother_state', 'smoother_state_cov', 'smoother_state_autocov',
        'smoother_disturbance', 'smoother_disturbance_cov', 'smoother_all',
    ]

    smoother_state = OptionWrapper('smoother_output', SMOOTHER_STATE)
    smoother_state_cov = OptionWrapper('smoother_output', SMOOTHER_STATE_COV)
    smoother_disturbance = (
        OptionWrapper('smoother_output', SMOOTHER_DISTURBANCE)
    )
    smoother_disturbance_cov = (
        OptionWrapper('smoother_output', SMOOTHER_DISTURBANCE_COV)
    )
    smoother_state_autocov = (
        OptionWrapper('smoother_output', SMOOTHER_STATE_AUTOCOV)
    )
    smoother_all = OptionWrapper('smoother_output', SMOOTHER_ALL)

    smooth_methods = [
        'smooth_conventional', 'smooth_alternative', 'smooth_classical'
    ]

    smooth_conventional = OptionWrapper('smooth_method', SMOOTH_CONVENTIONAL)
    """
    (bool) Flag for conventional (Durbin and Koopman, 2012) Kalman smoothing.
    """
    smooth_alternative = OptionWrapper('smooth_method', SMOOTH_ALTERNATIVE)
    """
    (bool) Flag for alternative (modified Bryson-Frazier) smoothing.
    """
    smooth_classical = OptionWrapper('smooth_method', SMOOTH_CLASSICAL)
    """
    (bool) Flag for classical (see e.g. Anderson and Moore, 1979) smoothing.
    """
    smooth_univariate = OptionWrapper('smooth_method', SMOOTH_UNIVARIATE)
    """
    (bool) Flag for univariate smoothing (uses modified Bryson-Frazier timing).
    """

    # Default smoother options
    smoother_output = SMOOTHER_ALL
    smooth_method = 0

    def __init__(self, k_endog, k_states, k_posdef=None, results_class=None,
                 kalman_smoother_classes=None, **kwargs):
        # Set the default results class
        if results_class is None:
            results_class = SmootherResults

        super(KalmanSmoother, self).__init__(
            k_endog, k_states, k_posdef, results_class=results_class, **kwargs
        )

        # Options
        self.prefix_kalman_smoother_map = (
            kalman_smoother_classes
            if kalman_smoother_classes is not None
            else tools.prefix_kalman_smoother_map.copy())

        # Setup the underlying Kalman smoother storage
        self._kalman_smoothers = {}

        # Set the smoother options
        self.set_smoother_output(**kwargs)
        self.set_smooth_method(**kwargs)

    @property
    def _kalman_smoother(self):
        prefix = self.prefix
        if prefix in self._kalman_smoothers:
            return self._kalman_smoothers[prefix]
        return None

    def _initialize_smoother(self, smoother_output=None, smooth_method=None,
                             prefix=None, **kwargs):
        if smoother_output is None:
            smoother_output = self.smoother_output
        if smooth_method is None:
            smooth_method = self.smooth_method

        # Make sure we have the required Kalman filter
        prefix, dtype, create_filter, create_statespace = (
            self._initialize_filter(prefix, **kwargs)
        )

        # Determine if we need to (re-)create the smoother
        # (definitely need to recreate if we recreated the filter)
        create_smoother = (create_filter or
                           prefix not in self._kalman_smoothers)
        if not create_smoother:
            kalman_smoother = self._kalman_smoothers[prefix]

            create_smoother = (kalman_smoother.kfilter is not
                               self._kalman_filters[prefix])

        # If the dtype-specific _kalman_smoother does not exist (or if we
        # need to re-create it), create it
        if create_smoother:
            # Setup the smoother
            cls = self.prefix_kalman_smoother_map[prefix]
            self._kalman_smoothers[prefix] = cls(
                self._statespaces[prefix], self._kalman_filters[prefix],
                smoother_output, smooth_method
            )
        # Otherwise, update the smoother parameters
        else:
            self._kalman_smoothers[prefix].set_smoother_output(
                smoother_output, False)
            self._kalman_smoothers[prefix].set_smooth_method(smooth_method)

        return prefix, dtype, create_smoother, create_filter, create_statespace

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
        SMOOTHER_STATE_AUTOCOV = 0x10
            Calculate and return the smoothed state lag-one autocovariance
            matrices.
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
        >>> import statsmodels.tsa.statespace.kalman_smoother as ks
        >>> mod = ks.KalmanSmoother(1,1)
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

    def set_smooth_method(self, smooth_method=None, **kwargs):
        r"""
        Set the smoothing method

        The smoothing method can be used to override the Kalman smoother
        approach used. By default, the Kalman smoother used depends on the
        Kalman filter method.

        Parameters
        ----------
        smooth_method : integer, optional
            Bitmask value to set the filter method to. See notes for details.
        **kwargs
            Keyword arguments may be used to influence the filter method by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        The smoothing method is defined by a collection of boolean flags, and
        is internally stored as a bitmask. The methods available are:

        SMOOTH_CONVENTIONAL = 0x01
            Default Kalman smoother, as presented in Durbin and Koopman, 2012
            chapter 4.
        SMOOTH_CLASSICAL = 0x02
            Classical Kalman smoother, as presented in Anderson and Moore, 1979
            or Durbin and Koopman, 2012 chapter 4.6.1.
        SMOOTH_ALTERNATIVE = 0x04
            Modified Bryson-Frazier Kalman smoother method; this is identical
            to the conventional method of Durbin and Koopman, 2012, except that
            an additional intermediate step is included.
        SMOOTH_UNIVARIATE = 0x08
            Univariate Kalman smoother, as presented in Durbin and Koopman,
            2012 chapter 6, except with modified Bryson-Frazier timing.

        Practically speaking, these methods should all produce the same output
        but different computational implications, numerical stability
        implications, or internal timing assumptions.

        Note that only the first method is available if using a Scipy version
        older than 0.16.

        If the bitmask is set directly via the `smooth_method` argument, then
        the full method must be provided.

        If keyword arguments are used to set individual boolean flags, then
        the lowercase of the method must be used as an argument name, and the
        value is the desired value of the boolean flag (True or False).

        Note that the filter method may also be specified by directly modifying
        the class attributes which are defined similarly to the keyword
        arguments.

        The default filtering method is SMOOTH_CONVENTIONAL.

        Examples
        --------
        >>> mod = sm.tsa.statespace.SARIMAX(range(10))
        >>> mod.smooth_method
        1
        >>> mod.filter_conventional
        True
        >>> mod.filter_univariate = True
        >>> mod.smooth_method
        17
        >>> mod.set_smooth_method(filter_univariate=False,
                                  filter_collapsed=True)
        >>> mod.smooth_method
        33
        >>> mod.set_smooth_method(smooth_method=1)
        >>> mod.filter_conventional
        True
        >>> mod.filter_univariate
        False
        >>> mod.filter_collapsed
        False
        >>> mod.filter_univariate = True
        >>> mod.smooth_method
        17
        """
        if smooth_method is not None:
            self.smooth_method = smooth_method
        for name in KalmanSmoother.smooth_methods:
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def _smooth(self, smoother_output=None, smooth_method=None, prefix=None,
                complex_step=False, results=None, **kwargs):
        # Initialize the smoother
        prefix, dtype, create_smoother, create_filter, create_statespace = (
            self._initialize_smoother(
                smoother_output, smooth_method, prefix=prefix, **kwargs
            ))

        # Check that the filter and statespace weren't just recreated
        if create_filter or create_statespace:
            raise ValueError('Passed settings forced re-creation of the'
                             ' Kalman filter. Please run `_filter` before'
                             ' running `_smooth`.')

        # Get the appropriate smoother
        smoother = self._kalman_smoothers[prefix]

        # Run the smoother
        smoother()

        return smoother

    def smooth(self, smoother_output=None, smooth_method=None, results=None,
               run_filter=True, prefix=None, complex_step=False,
               **kwargs):
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
        run_filter : bool, optional
            Whether or not to run the Kalman filter prior to smoothing. Default
            is True.
        prefix : string
            The prefix of the datatype. Usually only used internally.

        Returns
        -------
        SmootherResults object
        """

        # Run the filter
        kfilter = self._filter(**kwargs)

        # Create the results object
        results = self.results_class(self)
        results.update_representation(self)
        results.update_filter(kfilter)

        # Run the smoother
        if smoother_output is None:
            smoother_output = self.smoother_output
        smoother = self._smooth(smoother_output, results=results, **kwargs)

        # Update the results
        results.update_smoother(smoother)

        return results


class SmootherResults(FilterResults):
    r"""
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
    smoothed_state_autocov : array
        The smoothed state lago-one autocovariance matrices at each time
        period: :math:`Cov(\alpha_{t+1}, \alpha_t)`.
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
        'smoothed_state', 'smoothed_state_cov', 'smoothed_state_autocov',
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
        if self.smoother_state_autocov:
            attributes.append('smoothed_state_autocov')
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

        has_missing = np.sum(self.nmissing) > 0
        for name in self._smoother_attributes:
            if name == 'smoother_output':
                pass
            elif name in attributes:
                if name in ['smoothing_error',
                            'smoothed_measurement_disturbance']:
                    vector = getattr(smoother, name, None)
                    if vector is not None and has_missing:
                        vector = np.array(reorder_missing_vector(
                            vector, self.missing, prefix=self.prefix))
                    else:
                        vector = np.array(vector, copy=True)
                    setattr(self, name, vector)
                elif name == 'smoothed_measurement_disturbance_cov':
                    matrix = getattr(smoother, name, None)
                    if matrix is not None and has_missing:
                        matrix = reorder_missing_matrix(
                            matrix, self.missing, reorder_rows=True,
                            reorder_cols=True, prefix=self.prefix)
                        # In the missing data case, we want to set the missing
                        # components equal to their unconditional distribution
                        copy_index_matrix(
                            self.obs_cov, matrix, self.missing,
                            index_rows=True, index_cols=True, inplace=True,
                            prefix=self.prefix)
                    else:
                        matrix = np.array(matrix, copy=True)
                    setattr(self, name, matrix)
                else:
                    setattr(self, name,
                            np.array(getattr(smoother, name, None), copy=True))
            else:
                setattr(self, name, None)

        # Diffuse objects
        self.scaled_smoothed_diffuse_estimator = None
        self.scaled_smoothed_diffuse1_estimator_cov = None
        self.scaled_smoothed_diffuse2_estimator_cov = None
        if self.nobs_diffuse > 0:
            self.scaled_smoothed_diffuse_estimator = np.array(
                smoother.scaled_smoothed_diffuse_estimator, copy=True)
            self.scaled_smoothed_diffuse1_estimator_cov = np.array(
                smoother.scaled_smoothed_diffuse1_estimator_cov, copy=True)
            self.scaled_smoothed_diffuse2_estimator_cov = np.array(
                smoother.scaled_smoothed_diffuse2_estimator_cov, copy=True)

        # Adjustments

        # For r_t (and similarly for N_t), what was calculated was
        # r_T, ..., r_{-1}. We only want r_0, ..., r_T
        # so exclude the appropriate element so that the time index is
        # consistent with the other returned output
        # r_t stored such that scaled_smoothed_estimator[0] == r_{-1}
        start = 1
        end = None
        if 'scaled_smoothed_estimator' in attributes:
            self.scaled_smoothed_estimator = (
                self.scaled_smoothed_estimator[:, start:end]
            )
        if 'scaled_smoothed_estimator_cov' in attributes:
            self.scaled_smoothed_estimator_cov = (
                self.scaled_smoothed_estimator_cov[:, :, start:end]
            )

        # Clear the smoothed forecasts
        self._smoothed_forecasts = None
        self._smoothed_forecasts_error = None
        self._smoothed_forecasts_error_cov = None

        # Note: if we concentrated out the scale, need to adjust the
        # loglikelihood values and all of the covariance matrices and the
        # values that depend on the covariance matrices
        if self.filter_concentrated and self.model._scale is None:
            self.smoothed_state_cov *= self.scale
            self.smoothed_state_autocov *= self.scale
            self.smoothed_state_disturbance_cov *= self.scale
            self.smoothed_measurement_disturbance_cov *= self.scale
            self.scaled_smoothed_estimator /= self.scale
            self.scaled_smoothed_estimator_cov /= self.scale
            self.smoothing_error /= self.scale

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

                mask = ~self.missing[:, t].astype(bool)
                # We can recover forecasts
                self._smoothed_forecasts[:, t] = np.dot(
                    self.design[:, :, design_t], self.smoothed_state[:, t]
                ) + self.obs_intercept[:, obs_intercept_t]
                if self.nmissing[t] > 0:
                    self._smoothed_forecasts_error[:, t] = np.nan
                self._smoothed_forecasts_error[mask, t] = (
                    self.endog[mask, t] - self._smoothed_forecasts[mask, t]
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
