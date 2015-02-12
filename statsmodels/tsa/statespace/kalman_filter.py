"""
State Space Representation and Kalman Filter

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

from warnings import warn

import numpy as np
from .representation import OptionWrapper, Representation, FrozenRepresentation
from .tools import (
    prefix_kalman_filter_map, validate_vector_shape, validate_matrix_shape
)

# Define constants
FILTER_CONVENTIONAL = 0x01     # Durbin and Koopman (2012), Chapter 4

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
    loglikelihood_burn : int, optional
        The number of initial periods during which the loglikelihood is not
        recorded. Default is 0.
    tolerance : float, optional
        The tolerance at which the Kalman filter determines convergence to
        steady-state. Default is 1e-19.
    results_class : class, optional
        Default results class to use to save filtering output. Default is
        `FilterResults`. If specified, class must extend from `FilterResults`.
    **kwargs
        Keyword arguments may be used to provide values for the filter,
        inversion, and stability methods. See `set_filter_method`,
        `set_inversion_method`, and `set_stability_method`.
        Keyword arguments may be used to provide default values for state space
        matrices. See `Representation` for more details.

    Notes
    -----
    There are several types of options available for controlling the Kalman
    filter operation. All options are internally held as bitmasks, but can be
    manipulated by setting class attributes, which act like boolean flags. For
    more information, see the `set_*` class method documentation. The options
    are:

    filter_method
        The filtering method controls aspects of which
        Kalman filtering approach will be used.
    inversion_method
        The Kalman filter may contain one matrix inversion: that of the
        forecast error covariance matrix. The inversion method controls how and
        if that inverse is performed.
    stability_method
        The Kalman filter is a recursive algorithm that may in some cases
        suffer issues with numerical stability. The stability method controls
        what, if any, measures are taken to promote stability.
    conserve_memory
        By default, the Kalman filter computes a number of intermediate
        matrices at each iteration. The memory conservation options control
        which of those matrices are stored.

    The `filter_method` and `inversion_method` options intentionally allow
    the possibility that multiple methods will be indicated. In the case that
    multiple methods are selected, the underlying Kalman filter will attempt to
    select the optional method given the input data.

    For example, it may be that INVERT_UNIVARIATE and SOLVE_CHOLESKY are
    indicated (this is in fact the default case). In this case, if the
    endogenous vector is 1-dimensional (`k_endog` = 1), then INVERT_UNIVARIATE
    is used and inversion reduces to simple division, and if it has a larger
    dimension, the Cholesky decomposition along with linear solving (rather
    than explicit matrix inversion) is used. If only SOLVE_CHOLESKY had been
    set, then the Cholesky decomposition method would *always* be used, even in
    the case of 1-dimensional data.

    See Also
    --------
    FilterResults
    statsmodels.tsa.statespace.representation.Representation
    """

    filter_methods = [
        'filter_conventional'
    ]

    filter_conventional = OptionWrapper('filter_method', FILTER_CONVENTIONAL)
    """
    (bool) Flag for conventional Kalman filtering.
    """

    inversion_methods = [
        'invert_univariate', 'solve_lu', 'invert_lu', 'solve_cholesky',
        'invert_cholesky', 'invert_numpy',
    ]

    invert_univariate = OptionWrapper('inversion_method', INVERT_UNIVARIATE)
    """
    (bool) Flag for univariate inversion method (recommended).
    """
    solve_lu = OptionWrapper('inversion_method', SOLVE_LU)
    """
    (bool) Flag for LU and linear solver inversion method.
    """
    invert_lu = OptionWrapper('inversion_method', INVERT_LU)
    """
    (bool) Flag for LU inversion method.
    """
    solve_cholesky = OptionWrapper('inversion_method', SOLVE_CHOLESKY)
    """
    (bool) Flag for Cholesky and linear solver inversion method (recommended).
    """
    invert_cholesky = OptionWrapper('inversion_method', INVERT_CHOLESKY)
    """
    (bool) Flag for Cholesky inversion method.
    """
    invert_numpy = OptionWrapper('inversion_method', INVERT_NUMPY)
    """
    (bool) Flag for inversion using numpy (not recommended).
    """

    stability_methods = ['stability_force_symmetry']

    stability_force_symmetry = (
        OptionWrapper('stability_method', STABILITY_FORCE_SYMMETRY)
    )
    """
    (bool) Flag for enforcing covariance matrix symmetry
    """

    memory_options = [
        'memory_store_all', 'memory_no_forecast', 'memory_no_predicted',
        'memory_no_filtered', 'memory_no_likelihood', 'memory_conserve'
    ]

    memory_store_all = OptionWrapper('conserve_memory', MEMORY_STORE_ALL)
    """
    (bool) Flag for storing all intermediate results in memory (default).
    """
    memory_no_forecast = OptionWrapper('conserve_memory', MEMORY_NO_FORECAST)
    """
    (bool) Flag to prevent storing forecasts.
    """
    memory_no_predicted = OptionWrapper('conserve_memory', MEMORY_NO_PREDICTED)
    """
    (bool) Flag to prevent storing predicted state and covariance matrices.
    """
    memory_no_filtered = OptionWrapper('conserve_memory', MEMORY_NO_FILTERED)
    """
    (bool) Flag to prevent storing filtered state and covariance matrices.
    """
    memory_no_likelihood = (
        OptionWrapper('conserve_memory', MEMORY_NO_LIKELIHOOD)
    )
    """
    (bool) Flag to prevent storing likelihood values for each observation.
    """
    memory_conserve = OptionWrapper('conserve_memory', MEMORY_CONSERVE)
    """
    (bool) Flag to conserve the maximum amount of memory.
    """

    # Default filter options
    filter_method = FILTER_CONVENTIONAL
    """
    (int) Filtering method bitmask.
    """
    inversion_method = INVERT_UNIVARIATE | SOLVE_CHOLESKY
    """
    (int) Inversion method bitmask.
    """
    stability_method = STABILITY_FORCE_SYMMETRY
    """
    (int) Stability method bitmask.
    """
    conserve_memory = MEMORY_STORE_ALL
    """
    (int) Memory conservation bitmask.
    """

    def __init__(self, k_endog, k_states, k_posdef=None,
                 loglikelihood_burn=0, tolerance=1e-19, results_class=None,
                 **kwargs):
        super(KalmanFilter, self).__init__(
            k_endog, k_states, k_posdef, **kwargs
        )

        # Setup the underlying Kalman filter storage
        self._kalman_filters = {}

        # Filter options
        self.loglikelihood_burn = loglikelihood_burn
        self.results_class = (
            results_class if results_class is not None else FilterResults
        )

        self.set_filter_method(**kwargs)
        self.set_inversion_method(**kwargs)
        self.set_stability_method(**kwargs)
        self.set_conserve_memory(**kwargs)

        self.tolerance = tolerance

    @property
    def _kalman_filter(self):
        prefix = self.prefix
        if prefix in self._kalman_filters:
            return self._kalman_filters[prefix]
        return None

    def _initialize_filter(self, filter_method=None, inversion_method=None,
                           stability_method=None, conserve_memory=None,
                           tolerance=None, loglikelihood_burn=None):
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
            kalman_filter = self._kalman_filters[prefix]
            kalman_filter.set_filter_method(filter_method, False)
            kalman_filter.inversion_method = inversion_method
            kalman_filter.stability_method = stability_method
            kalman_filter.tolerance = tolerance
            # conserve_memory and loglikelihood_burn changes always lead to
            # re-created filters

        return prefix, dtype, create_filter, create_statespace

    def set_filter_method(self, filter_method=None, **kwargs):
        """
        Set the filtering method

        The filtering method controls aspects of which Kalman filtering
        approach will be used.

        Parameters
        ----------
        filter_method : integer, optional
            Bitmask value to set the filter method to. See notes for details.
        **kwargs
            Keyword arguments may be used to influence the filter method by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        The filtering method is defined by a collection of boolean flags, and
        is internally stored as a bitmask. Only one method is currently
        available:

        FILTER_CONVENTIONAL = 0x01
            Conventional Kalman filter.
        

        If the bitmask is set directly via the `filter_method` argument, then
        the full method must be provided.

        If keyword arguments are used to set individual boolean flags, then
        the lowercase of the method must be used as an argument name, and the
        value is the desired value of the boolean flag (True or False).

        Note that the filter method may also be specified by directly modifying
        the class attributes which are defined similarly to the keyword
        arguments.

        The default filtering method is FILTER_CONVENTIONAL.

        Examples
        --------
        >>> mod = sm.tsa.statespace.SARIMAX(range(10))
        >>> mod.filter_method
        1
        >>> mod.filter_conventional
        True
        >>> mod.set_filter_method(filter_method=1)
        >>> mod.filter_method
        1
        >>> mod.set_filter_method(filter_conventional=True)
        >>> mod.filter_method
        1
        >>> mod.filter_conventional = True
        >>> mod.filter_method
        1
        """
        if filter_method is not None:
            self.filter_method = filter_method
        for name in KalmanFilter.filter_methods:
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def set_inversion_method(self, inversion_method=None, **kwargs):
        """
        Set the inversion method

        The Kalman filter may contain one matrix inversion: that of the
        forecast error covariance matrix. The inversion method controls how and
        if that inverse is performed.

        Parameters
        ----------
        inversion_method : integer, optional
            Bitmask value to set the inversion method to. See notes for
            details.
        **kwargs
            Keyword arguments may be used to influence the inversion method by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        The inversion method is defined by a collection of boolean flags, and
        is internally stored as a bitmask. The methods available are:

        INVERT_UNIVARIATE = 0x01
            If the endogenous time series is univariate, then inversion can be
            performed by simple division. If this flag is set and the time
            series is univariate, then division will always be used even if
            other flags are also set.
        SOLVE_LU = 0x02
            Use an LU decomposition along with a linear solver (rather than
            ever actually inverting the matrix).
        INVERT_LU = 0x04
            Use an LU decomposition along with typical matrix inversion.
        SOLVE_CHOLESKY = 0x08
            Use a Cholesky decomposition along with a linear solver.
        INVERT_CHOLESKY = 0x10
            Use an Cholesky decomposition along with typical matrix inversion.
        INVERT_NUMPY = 0x20
            Use the numpy inversion function. This is not recommended except
            for testing as it will be substantially slower than the other
            methods.

        If the bitmask is set directly via the `inversion_method` argument,
        then the full method must be provided.

        If keyword arguments are used to set individual boolean flags, then
        the lowercase of the method must be used as an argument name, and the
        value is the desired value of the boolean flag (True or False).

        Note that the inversion method may also be specified by directly
        modifying the class attributes which are defined similarly to the
        keyword arguments.

        The default inversion method is `INVERT_UNIVARIATE | SOLVE_CHOLESKY`

        Several things to keep in mind are:

        - Cholesky decomposition is about twice as fast as LU decomposition,
          but it requires that the matrix be positive definite. While this
          should generally be true, it may not be in every case.
        - Using a linear solver rather than true matrix inversion is generally
          faster and is numerically more stable.

        Examples
        --------
        >>> mod = sm.tsa.statespace.SARIMAX(range(10))
        >>> mod.inversion_method
        1
        >>> mod.solve_cholesky
        True
        >>> mod.invert_univariate
        True
        >>> mod.invert_lu
        False
        >>> mod.invert_univariate = False
        >>> mod.inversion_method
        8
        >>> mod.set_inversion_method(solve_cholesky=False,
                                     invert_cholesky=True)
        >>> mod.inversion_method
        16
        """
        if inversion_method is not None:
            self.inversion_method = inversion_method
        for name in KalmanFilter.inversion_methods:
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def set_stability_method(self, stability_method=None, **kwargs):
        """
        Set the numerical stability method

        The Kalman filter is a recursive algorithm that may in some cases
        suffer issues with numerical stability. The stability method controls
        what, if any, measures are taken to promote stability.

        Parameters
        ----------
        stability_method : integer, optional
            Bitmask value to set the stability method to. See notes for
            details.
        **kwargs
            Keyword arguments may be used to influence the stability method by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        The stability method is defined by a collection of boolean flags, and
        is internally stored as a bitmask. The methods available are:

        STABILITY_FORCE_SYMMETRY = 0x01
            If this flag is set, symmetry of the predicted state covariance
            matrix is enforced at each iteration of the filter, where each
            element is set to the average of the corresponding elements in the
            upper and lower triangle.

        If the bitmask is set directly via the `stability_method` argument,
        then the full method must be provided.

        If keyword arguments are used to set individual boolean flags, then
        the lowercase of the method must be used as an argument name, and the
        value is the desired value of the boolean flag (True or False).

        Note that the stability method may also be specified by directly
        modifying the class attributes which are defined similarly to the
        keyword arguments.

        The default stability method is `STABILITY_FORCE_SYMMETRY`

        Examples
        --------
        >>> mod = sm.tsa.statespace.SARIMAX(range(10))
        >>> mod.stability_method
        1
        >>> mod.stability_force_symmetry
        True
        >>> mod.stability_force_symmetry = False
        >>> mod.stability_method
        0
        """
        if stability_method is not None:
            self.stability_method = stability_method
        for name in KalmanFilter.stability_methods:
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def set_conserve_memory(self, conserve_memory=None, **kwargs):
        """
        Set the memory conservation method

        By default, the Kalman filter computes a number of intermediate
        matrices at each iteration. The memory conservation options control
        which of those matrices are stored.

        Parameters
        ----------
        conserve_memory : integer, optional
            Bitmask value to set the memory conservation method to. See notes
            for details.
        **kwargs
            Keyword arguments may be used to influence the memory conservation
            method by setting individual boolean flags. See notes for details.

        Notes
        -----
        The memory conservation method is defined by a collection of boolean
        flags, and is internally stored as a bitmask. The methods available
        are:

        MEMORY_STORE_ALL = 0
            Store all intermediate matrices. This is the default value.
        MEMORY_NO_FORECAST = 0x01
            Do not store the forecast, forecast error, or forecast error
            covariance matrices. If this option is used, the `predict` method
            from the results class is unavailable.
        MEMORY_NO_PREDICTED = 0x02
            Do not store the predicted state or predicted state covariance
            matrices.
        MEMORY_NO_FILTERED = 0x04
            Do not store the filtered state or filtered state covariance
            matrices.
        MEMORY_NO_LIKELIHOOD = 0x08
            Do not store the vector of loglikelihood values for each
            observation. Only the sum of the loglikelihood values is stored.
        MEMORY_CONSERVE
            Do not store any intermediate matrices.

        If the bitmask is set directly via the `conserve_memory` argument,
        then the full method must be provided.

        If keyword arguments are used to set individual boolean flags, then
        the lowercase of the method must be used as an argument name, and the
        value is the desired value of the boolean flag (True or False).

        Note that the memory conservation method may also be specified by
        directly modifying the class attributes which are defined similarly to
        the keyword arguments.

        The default memory conservation method is `MEMORY_STORE_ALL`, so that
        all intermediate matrices are stored.

        Examples
        --------
        >>> mod = sm.tsa.statespace.SARIMAX(range(10))
        >>> mod.conserve_memory
        0
        >>> mod.memory_no_predicted
        False
        >>> mod.memory_no_predicted = True
        >>> mod.conserve_memory
        2
        >>> mod.set_conserve_memory(memory_no_filtered=True,
                                    memory_no_forecast=True)
        >>> mod.conserve_memory
        7
        """
        if conserve_memory is not None:
            self.conserve_memory = conserve_memory
        for name in KalmanFilter.memory_options:
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def filter(self, filter_method=None, inversion_method=None,
               stability_method=None, conserve_memory=None, tolerance=None,
               loglikelihood_burn=None, results=None):
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
                conserve_memory, tolerance, loglikelihood_burn
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
            results = np.array(
                self._kalman_filters[prefix].loglikelihood, copy=True
            )
        # Otherwise update the results object
        else:
            # Update the model features; unless we had to recreate the
            # statespace, only update the filter options
            results.update_representation(
                self, only_options=not create_statespace
            )
            results.update_filter(kfilter)

        return results

    def loglike(self, loglikelihood_burn=None, **kwargs):
        """
        Calculate the loglikelihood associated with the statespace model.

        Parameters
        ----------
        loglikelihood_burn : int, optional
            The number of initial periods during which the loglikelihood is not
            recorded. Default is 0.
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.

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
        return np.sum(self.filter(**kwargs)[loglikelihood_burn:])


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
        The dimension of a guaranteed positive definite
        covariance matrix describing the shocks in the
        measurement equation.
    dtype : dtype
        Datatype of representation matrices
    prefix : str
        BLAS prefix of representation matrices
    shapes : dictionary of name,tuple
        A dictionary recording the shapes of each of the
        representation matrices as tuples.
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
        An array of the same size as `endog`, filled
        with boolean values that are True if the
        corresponding entry in `endog` is NaN and False
        otherwise.
    nmissing : array of int
        An array of size `nobs`, where the ith entry
        is the number (between 0 and `k_endog`) of NaNs in
        the ith row of the `endog` array.
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
        Bitmask representing the method used to
        invert the forecast error covariance matrix.
    stability_method : int
        Bitmask representing the methods used to promote
        numerical stability in the Kalman filter
        recursions.
    conserve_memory : int
        Bitmask representing the selected memory conservation method.
    tolerance : float
        The tolerance at which the Kalman filter
        determines convergence to steady-state.
    loglikelihood_burn : int
        The number of initial periods during which
        the loglikelihood is not recorded.
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
    forecasts : array
        The one-step-ahead forecasts of observations at each time period.
    forecasts_error : array
        The forecast errors at each time period.
    forecasts_error_cov : array
        The forecast error covariance matrices at each time period.
    llf_obs : array
        The loglikelihood values at each time period.
    """
    _filter_attributes = [
        'filter_method', 'inversion_method', 'stability_method',
        'conserve_memory', 'tolerance', 'loglikelihood_burn', 'converged',
        'period_converged', 'filtered_state', 'filtered_state_cov',
        'predicted_state', 'predicted_state_cov',
        'forecasts', 'forecasts_error', 'forecasts_error_cov',
        'llf_obs'
    ]

    _filter_options = (
        KalmanFilter.filter_methods + KalmanFilter.stability_methods +
        KalmanFilter.inversion_methods + KalmanFilter.memory_options
    )

    _attributes = FrozenRepresentation._model_attributes + _filter_attributes

    def __init__(self, model):
        super(FilterResults, self).__init__(model)

        # Setup caches for uninitialized objects
        self._kalman_gain = None
        self._standardized_forecasts_error = None

    def update_representation(self, model, only_options=False):
        """
        Update the results to match a given model

        Parameters
        ----------
        model : Representation
            The model object from which to take the updated values.
        only_options : boolean, optional
            If set to true, only the filter options are updated, and the state
            space representation is not updated. Default is False.

        Notes
        -----
        This method is rarely required except for internal usage.
        """
        if not only_options:
            super(FilterResults, self).update_representation(model)

        # Save the options as boolean variables
        for name in self._filter_options:
            setattr(self, name, getattr(model, name, None))

    def update_filter(self, kalman_filter):
        """
        Update the filter results

        Parameters
        ----------
        kalman_filter : KalmanFilter
            The model object from which to take the updated values.

        Notes
        -----
        This method is rarely required except for internal usage.
        """
        # State initialization
        self.initial_state = np.array(
            kalman_filter.model.initial_state, copy=True
        )
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
        self.filtered_state_cov = np.array(
            kalman_filter.filtered_state_cov, copy=True
        )
        self.predicted_state = np.array(
            kalman_filter.predicted_state, copy=True
        )
        self.predicted_state_cov = np.array(
            kalman_filter.predicted_state_cov, copy=True
        )

        # Note: use forecasts rather than forecast, so as not to interfer
        # with the `forecast` methods in subclasses
        self.forecasts = np.array(kalman_filter.forecast, copy=True)
        self.forecasts_error = np.array(
            kalman_filter.forecast_error, copy=True
        )
        self.forecasts_error_cov = np.array(
            kalman_filter.forecast_error_cov, copy=True
        )
        self.llf_obs = np.array(kalman_filter.loglikelihood, copy=True)

        # If there was missing data, save the original values from the Kalman
        # filter output, since below will set the values corresponding to
        # the missing observations to nans.
        self.missing_forecasts = None
        self.missing_forecasts_error = None
        self.missing_forecasts_error_cov = None
        if np.sum(self.nmissing) > 0:
            # Copy the provided arrays (which are as the Kalman filter dataset)
            # into new variables
            self.missing_forecasts = np.copy(self.forecasts)
            self.missing_forecasts_error = np.copy(self.forecasts_error)
            self.missing_forecasts_error_cov = (
                np.copy(self.forecasts_error_cov)
            )

        # Fill in missing values in the forecast, forecast error, and
        # forecast error covariance matrix (this is required due to how the
        # Kalman filter implements observations that are either partly or
        # completely missing)
        # Construct the predictions, forecasts
        if not (self.memory_no_forecast or self.memory_no_predicted):
            for t in range(self.nobs):
                design_t = 0 if self.design.shape[2] == 1 else t
                obs_cov_t = 0 if self.obs_cov.shape[2] == 1 else t
                obs_intercept_t = 0 if self.obs_intercept.shape[1] == 1 else t

                # For completely missing observations, the Kalman filter will
                # produce forecasts, but forecast errors and the forecast
                # error covariance matrix will be zeros - make them nan to
                # improve clarity of results.
                if self.nmissing[t] == self.k_endog:
                    # We can recover forecasts
                    self.forecasts[:, t] = np.dot(
                        self.design[:, :, design_t], self.predicted_state[:, t]
                    ) + self.obs_intercept[:, obs_intercept_t]
                    self.forecasts_error[:, t] = np.nan
                    self.forecasts_error_cov[:, :, t] = np.dot(
                        np.dot(self.design[:, :, design_t],
                               self.predicted_state_cov[:, :, t]),
                        self.design[:, :, design_t].T
                    ) + self.obs_cov[:, :, obs_cov_t]
                # For partially missing observations, the Kalman filter
                # will produce all elements (forecasts, forecast errors,
                # forecast error covariance matrices) as usual, but their
                # dimension will only be equal to the number of non-missing
                # elements, and their location in memory will be in the first
                # blocks (e.g. for the forecasts_error, the first
                # k_endog - nmissing[t] columns will be filled in), regardless
                # of which endogenous variables they refer to (i.e. the non-
                # missing endogenous variables for that observation).
                # Furthermore, the forecast error covariance matrix is only
                # valid for those elements. What is done is to set all elements
                # to nan for these observations so that they are flagged as
                # missing. The variables missing_forecasts, etc. then provide
                # the forecasts, etc. provided by the Kalman filter, from which
                # the data can be retrieved if desired.
                elif self.nmissing[t] > 0:
                    self.forecasts[:, t] = np.nan
                    self.forecasts_error[:, t] = np.nan
                    self.forecasts_error_cov[:, :, t] = np.nan

    @property
    def kalman_gain(self):
        """
        Kalman gain matrices
        """
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
        """
        Standardized forecast errors
        """
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
                **kwargs):
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
        dynamic : int, optional
            Offset relative to `start` at which to begin dynamic prediction.
            Prior to this observation, true endogenous values will be used for
            prediction; starting with this observation and continuing through
            the end of prediction, forecasted endogenous values will be used
            instead.
        full_results : boolean, optional
            If True, returns a FilterResults instance; if False returns a
            tuple with forecasts, the forecast errors, and the forecast error
            covariance matrices. Default is False.
        **kwargs
            If the prediction range is outside of the sample range, any
            of the state space representation matrices that are time-varying
            must have updated values provided for the out-of-sample range.
            For example, of `obs_intercept` is a time-varying component and
            the prediction range extends 10 periods beyond the end of the
            sample, a (`k_endog` x 10) matrix must be provided with the new
            intercept values.

        Returns
        -------
        results : FilterResults or array
            Either a FilterResults object (if `full_results=True`) or an
            array of forecasts otherwise.

        Notes
        -----
        All prediction is performed by applying the deterministic part of the
        measurement equation using the predicted state variables.

        Out-of-sample prediction first applies the Kalman filter to missing
        data for the number of periods desired to obtain the predicted states.
        """
        # Cannot predict if we do not have appropriate arrays
        if self.memory_no_forecast or self.memory_no_predicted:
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

        # Prediction and forecasting is performed by iterating the Kalman
        # Kalman filter through the entire range [0, end]
        # Then, unless `full_results=True`, forecasts are returned
        # corresponding to the range [start, end].
        # In order to perform the calculations, the range is separately split
        # up into the following categories:
        # - static:   (in-sample) the Kalman filter is run as usual
        # - dynamic:  (in-sample) the Kalman filter is run, but on missing data
        # - forecast: (out-of-sample) the Kalman filter is run, but on missing
        #             data

        # Short-circuit if end is before start
        if end <= start:
            raise ValueError('End of prediction must be after start.')

        # Get the number of forecasts to make after the end of the sample
        nforecast = max(0, end - self.nobs)

        # Get the number of dynamic prediction periods

        # If `dynamic=True`, then assume that we want to begin dynamic
        # prediction at the start of the sample prediction.
        if dynamic is True:
            dynamic = 0
        # If `dynamic=False`, then assume we want no dynamic prediction
        if dynamic is False:
            dynamic = None

        ndynamic = 0
        if dynamic is not None:
            # Replace the relative dynamic offset with an absolute offset
            dynamic = start + dynamic

            # Validate the `dynamic` parameter
            if dynamic < 0:
                raise ValueError('Dynamic prediction cannot begin prior to the'
                                 ' first observation in the sample.')
            if dynamic > end:
                warn('Dynamic prediction specified to begin after the end of'
                     ' prediction, and so has no effect.')
                dynamic = None
            if dynamic > self.nobs:
                warn('Dynamic prediction specified to begin during'
                     ' out-of-sample forecasting period, and so has no'
                     ' effect.')
                dynamic = None

            # Get the total size of the desired dynamic forecasting component
            # Note: the first `dynamic` periods of prediction are actually
            # *not* dynamic, because dynamic prediction begins at observation
            # `dynamic`.
            if dynamic is not None:
                ndynamic = max(0, min(end, self.nobs) - dynamic)

        # Get the number of in-sample static predictions
        nstatic = min(end, self.nobs) if dynamic is None else dynamic

        # Construct the design and observation intercept and covariance
        # matrices for start-npadded:end. If not time-varying in the original
        # model, then they will be copied over if none are provided in
        # `kwargs`. Otherwise additional matrices must be provided in `kwargs`.
        representation = {}
        for name, shape in self.shapes.items():
            if name == 'obs':
                continue
            representation[name] = getattr(self, name)

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
            # Construct the new endogenous array.
            endog = np.empty((self.k_endog, ndynamic + nforecast))
            endog.fill(np.nan)
            endog = np.asfortranarray(np.c_[self.endog[:, :nstatic], endog])

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
                self.initial_state,
                self.initial_state_cov
            )
            model._initialize_filter()
            model._initialize_state()

            result = self._predict(nstatic, ndynamic, nforecast, model)

        if full_results:
            return result
        else:
            return result.forecasts[:, start:end]

    def _predict(self, nstatic, ndynamic, nforecast, model):
        # TODO: this doesn't use self, and can either be a static method or
        #       moved outside the class altogether.

        # Get the underlying filter
        kfilter = model._kalman_filter

        # Save this (which shares memory with the memoryview on which the
        # Kalman filter will be operating) so that we can replace actual data
        # with predicted data during dynamic forecasting
        endog = model._representations[model.prefix]['obs']

        # print(nstatic, ndynamic, nforecast, model.nobs)

        for t in range(kfilter.model.nobs):
            # Run the Kalman filter for the first `nstatic` periods (for
            # which dynamic computation will not be performed)
            if t < nstatic:
                next(kfilter)
            # Perform dynamic prediction
            elif t < nstatic + ndynamic:
                design_t = 0 if model.design.shape[2] == 1 else t
                obs_intercept_t = 0 if model.obs_intercept.shape[1] == 1 else t

                # Unconditional value is the intercept (often zeros)
                endog[:, t] = model.obs_intercept[:, obs_intercept_t]
                # If t > 0, then we can condition the forecast on the state
                if t > 0:
                    # Predict endog[:, t] given `predicted_state` calculated in
                    # previous iteration (i.e. t-1)
                    endog[:, t] += np.dot(
                        model.design[:, :, design_t],
                        kfilter.predicted_state[:, t]
                    )

                # Advance Kalman filter
                next(kfilter)
            # Perform any (one-step-ahead) forecasting
            else:
                next(kfilter)

        # Return the predicted state and predicted state covariance matrices
        results = FilterResults(model)
        results.update_filter(kfilter)
        return results
