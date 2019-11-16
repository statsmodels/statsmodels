#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
Kalman Filter declarations

Author: Chad Fulton  
License: Simplified-BSD
"""

# ## Constants

# ### Filters
cdef int FILTER_CONVENTIONAL     # Durbin and Koopman (2012), Chapter 4
cdef int FILTER_EXACT_INITIAL    # ibid., Chapter 5.6
cdef int FILTER_AUGMENTED        # ibid., Chapter 5.7
cdef int FILTER_SQUARE_ROOT      # ibid., Chapter 6.3
cdef int FILTER_UNIVARIATE       # ibid., Chapter 6.4
cdef int FILTER_COLLAPSED        # ibid., Chapter 6.5
cdef int FILTER_EXTENDED         # ibid., Chapter 10.2
cdef int FILTER_UNSCENTED        # ibid., Chapter 10.3
cdef int FILTER_CONCENTRATED     # Harvey (1989), Chapter 3.4

# ### Inversion methods
# Methods by which the terms using the inverse of the forecast error
# covariance matrix are solved.
cdef int INVERT_UNIVARIATE
cdef int SOLVE_LU
cdef int INVERT_LU
cdef int SOLVE_CHOLESKY
cdef int INVERT_CHOLESKY

# ### Numerical Stability methods
# Methods to improve numerical stability
cdef int STABILITY_FORCE_SYMMETRY

# ### Memory conservation options
cdef int MEMORY_STORE_ALL
cdef int MEMORY_NO_FORECAST_MEAN
cdef int MEMORY_NO_FORECAST_COV
cdef int MEMORY_NO_FORECAST
cdef int MEMORY_NO_PREDICTED_MEAN
cdef int MEMORY_NO_PREDICTED_COV
cdef int MEMORY_NO_PREDICTED
cdef int MEMORY_NO_FILTERED_MEAN
cdef int MEMORY_NO_FILTERED_COV
cdef int MEMORY_NO_FILTERED
cdef int MEMORY_NO_LIKELIHOOD
cdef int MEMORY_NO_GAIN
cdef int MEMORY_NO_SMOOTHING
cdef int MEMORY_NO_STD_FORECAST
cdef int MEMORY_CONSERVE

# ### Timing options
cdef int TIMING_INIT_FILTERED
cdef int TIMING_INIT_PREDICTED

# Typical imports
cimport numpy as np

from statsmodels.tsa.statespace._representation cimport (
    sStatespace, dStatespace, cStatespace, zStatespace
)

# Single precision
cdef class sKalmanFilter(object):
    # Statespace object
    cdef readonly sStatespace model

    # ### Filter parameters
    cdef readonly int t
    cdef public np.float64_t tolerance, tolerance_diffuse
    cdef readonly int nobs_diffuse
    cdef readonly int nobs_kendog_diffuse_nonsingular
    cdef readonly int nobs_kendog_univariate_singular
    cdef readonly int converged
    cdef readonly int period_converged
    cdef readonly int time_invariant
    cdef readonly int filter_method
    cdef public int inversion_method
    cdef public int stability_method
    cdef readonly int conserve_memory
    cdef public int filter_timing
    cdef readonly int loglikelihood_burn

    # ### Kalman filter properties
    cdef readonly np.float32_t [:] loglikelihood, scale
    cdef readonly np.float32_t [::1,:] filtered_state, predicted_state, forecast, forecast_error, standardized_forecast_error
    cdef readonly np.float32_t [::1,:,:] filtered_state_cov, predicted_state_cov, forecast_error_cov, predicted_diffuse_state_cov, forecast_error_diffuse_cov, M, M_inf
    cdef readonly np.float32_t [::1,:,:] kalman_gain

    # ### Steady State Values
    cdef readonly np.float32_t [::1,:] converged_forecast_error_cov
    cdef readonly np.float32_t [::1,:] converged_filtered_state_cov
    cdef readonly np.float32_t [::1,:] converged_predicted_state_cov
    cdef readonly np.float32_t [::1,:] converged_kalman_gain
    cdef readonly np.float32_t [::1,:] converged_M
    cdef readonly np.float32_t converged_determinant

    # ### Temporary arrays
    cdef readonly np.float32_t [::1,:] forecast_error_fac
    cdef readonly int [:] forecast_error_ipiv
    cdef readonly np.float32_t [::1,:] forecast_error_work
    cdef readonly np.float32_t [::1,:] tmp0, tmp00
    cdef readonly np.float32_t [::1,:] tmp2
    cdef readonly np.float32_t [::1,:,:] tmp1, tmp3, tmp4
    cdef readonly np.float32_t [:] tmpK0, tmpK1
    cdef readonly np.float32_t [::1,:] tmpL0, tmpL1

    cdef readonly np.float32_t determinant

    # ### Pointers to current-iteration arrays
    # cdef np.float32_t * _obs
    # cdef np.float32_t * _design
    # cdef np.float32_t * _obs_intercept
    # cdef np.float32_t * _obs_cov
    # cdef np.float32_t * _transition
    # cdef np.float32_t * _state_intercept
    # cdef np.float32_t * _selection
    # cdef np.float32_t * _state_cov
    # cdef np.float32_t * _selected_state_cov
    # cdef np.float32_t * _initial_state
    # cdef np.float32_t * _initial_state_cov

    cdef np.float32_t * _input_state
    cdef np.float32_t * _input_state_cov
    cdef np.float32_t * _input_diffuse_state_cov

    cdef np.float32_t * _forecast
    cdef np.float32_t * _forecast_error
    cdef np.float32_t * _standardized_forecast_error
    cdef np.float32_t * _forecast_error_cov
    cdef np.float32_t * _filtered_state
    cdef np.float32_t * _filtered_state_cov
    cdef np.float32_t * _predicted_state
    cdef np.float32_t * _predicted_state_cov

    cdef np.float32_t * _forecast_error_diffuse_cov
    cdef np.float32_t * _predicted_diffuse_state_cov
    cdef np.float32_t * _M
    cdef np.float32_t * _M_inf

    cdef np.float32_t * _kalman_gain
    cdef np.float32_t * _loglikelihood
    cdef np.float32_t * _scale

    cdef np.float32_t * _converged_forecast_error_cov
    cdef np.float32_t * _converged_filtered_state_cov
    cdef np.float32_t * _converged_predicted_state_cov
    cdef np.float32_t * _converged_kalman_gain
    cdef np.float32_t * _converged_M

    cdef np.float32_t * _forecast_error_fac
    cdef int * _forecast_error_ipiv
    cdef np.float32_t * _forecast_error_work

    cdef np.float32_t * _tmp0
    cdef np.float32_t * _tmp00
    cdef np.float32_t * _tmp1
    cdef np.float32_t * _tmp2
    cdef np.float32_t * _tmp3
    cdef np.float32_t * _tmp4

    cdef np.float32_t * _tmpM_inf
    cdef np.float32_t * _tmpK0
    cdef np.float32_t * _tmpK1
    cdef np.float32_t * _tmpL0
    cdef np.float32_t * _tmpL1

    # ### Pointers to current-iteration Kalman filtering functions
    cdef int (*forecasting)(
        sKalmanFilter, sStatespace
    )
    cdef np.float32_t (*inversion)(
        sKalmanFilter, sStatespace, np.float32_t
    ) except *
    cdef int (*updating)(
        sKalmanFilter, sStatespace
    )
    cdef np.float32_t (*calculate_loglikelihood)(
        sKalmanFilter, sStatespace, np.float32_t
    )
    cdef np.float32_t (*calculate_scale)(
        sKalmanFilter, sStatespace
    )
    cdef int (*prediction)(
        sKalmanFilter, sStatespace
    )

    # ### Define some constants
    cdef readonly int k_endog, k_states, k_posdef, k_endog2, k_states2, k_posdef2, k_endogstates, k_statesposdef
    cdef readonly int ldwork
    
    cdef allocate_arrays(self)
    cdef void set_dimensions(self)
    cpdef set_filter_method(self, int filter_method, int force_reset=*)
    cpdef seek(self, unsigned int t, int reset=*)

    cdef void initialize_statespace_object_pointers(self) except *
    cdef void initialize_filter_object_pointers(self)
    cdef void initialize_function_pointers(self) except *
    cdef void post_convergence(self)
    cdef void numerical_stability(self)
    cdef int check_diffuse(self)
    cdef void check_convergence(self)
    cdef void migrate_storage(self)
    cdef void _reinitialize_pointers(self) except *

    cdef void _forecasting(self)
    cdef np.float32_t _inversion(self)
    cdef void _updating(self)
    cdef np.float32_t _calculate_loglikelihood(self)
    cdef void _prediction(self)

# Double precision
cdef class dKalmanFilter(object):
    # Statespace object
    cdef readonly dStatespace model

    # ### Filter parameters
    cdef readonly int t
    cdef public np.float64_t tolerance, tolerance_diffuse
    cdef readonly int nobs_diffuse
    cdef readonly int nobs_kendog_diffuse_nonsingular
    cdef readonly int nobs_kendog_univariate_singular
    cdef readonly int converged
    cdef readonly int period_converged
    cdef readonly int time_invariant
    cdef readonly int filter_method
    cdef public int inversion_method
    cdef public int stability_method
    cdef readonly int conserve_memory
    cdef public int filter_timing
    cdef readonly int loglikelihood_burn

    # ### Kalman filter properties
    cdef readonly np.float64_t [:] loglikelihood, scale
    cdef readonly np.float64_t [::1,:] filtered_state, predicted_state, forecast, forecast_error, standardized_forecast_error
    cdef readonly np.float64_t [::1,:,:] filtered_state_cov, predicted_state_cov, forecast_error_cov, predicted_diffuse_state_cov, forecast_error_diffuse_cov, M, M_inf
    cdef readonly np.float64_t [::1,:,:] kalman_gain

    # ### Steady State Values
    cdef readonly np.float64_t [::1,:] converged_forecast_error_cov
    cdef readonly np.float64_t [::1,:] converged_filtered_state_cov
    cdef readonly np.float64_t [::1,:] converged_predicted_state_cov
    cdef readonly np.float64_t [::1,:] converged_kalman_gain
    cdef readonly np.float64_t [::1,:] converged_M
    cdef readonly np.float64_t converged_determinant

    # ### Temporary arrays
    cdef readonly np.float64_t [::1,:] forecast_error_fac
    cdef readonly int [:] forecast_error_ipiv
    cdef readonly np.float64_t [::1,:] forecast_error_work
    cdef readonly np.float64_t [::1,:] tmp0, tmp00
    cdef readonly np.float64_t [::1,:] tmp2
    cdef readonly np.float64_t [::1,:,:] tmp1, tmp3, tmp4
    cdef readonly np.float64_t [:] tmpM_inf, tmpK0, tmpK1
    cdef readonly np.float64_t [::1,:] tmpL0, tmpL1

    cdef readonly np.float64_t determinant

    # ### Pointers to current-iteration arrays
    # cdef np.float64_t * _obs
    # cdef np.float64_t * _design
    # cdef np.float64_t * _obs_intercept
    # cdef np.float64_t * _obs_cov
    # cdef np.float64_t * _transition
    # cdef np.float64_t * _state_intercept
    # cdef np.float64_t * _selection
    # cdef np.float64_t * _state_cov
    # cdef np.float64_t * _selected_state_cov
    # cdef np.float64_t * _initial_state
    # cdef np.float64_t * _initial_state_cov

    cdef np.float64_t * _input_state
    cdef np.float64_t * _input_state_cov
    cdef np.float64_t * _input_diffuse_state_cov

    cdef np.float64_t * _forecast
    cdef np.float64_t * _forecast_error
    cdef np.float64_t * _standardized_forecast_error
    cdef np.float64_t * _forecast_error_cov
    cdef np.float64_t * _filtered_state
    cdef np.float64_t * _filtered_state_cov
    cdef np.float64_t * _predicted_state
    cdef np.float64_t * _predicted_state_cov

    cdef np.float64_t * _forecast_error_diffuse_cov
    cdef np.float64_t * _predicted_diffuse_state_cov
    cdef np.float64_t * _M
    cdef np.float64_t * _M_inf

    cdef np.float64_t * _kalman_gain
    cdef np.float64_t * _loglikelihood
    cdef np.float64_t * _scale

    cdef np.float64_t * _converged_forecast_error_cov
    cdef np.float64_t * _converged_filtered_state_cov
    cdef np.float64_t * _converged_predicted_state_cov
    cdef np.float64_t * _converged_kalman_gain
    cdef np.float64_t * _converged_M

    cdef np.float64_t * _forecast_error_fac
    cdef int * _forecast_error_ipiv
    cdef np.float64_t * _forecast_error_work

    cdef np.float64_t * _tmp0
    cdef np.float64_t * _tmp00
    cdef np.float64_t * _tmp1
    cdef np.float64_t * _tmp2
    cdef np.float64_t * _tmp3
    cdef np.float64_t * _tmp4

    cdef np.float64_t * _tmpM_inf
    cdef np.float64_t * _tmpK0
    cdef np.float64_t * _tmpK1
    cdef np.float64_t * _tmpL0
    cdef np.float64_t * _tmpL1

    # ### Pointers to current-iteration Kalman filtering functions
    cdef int (*forecasting)(
        dKalmanFilter, dStatespace
    )
    cdef np.float64_t (*inversion)(
        dKalmanFilter, dStatespace, np.float64_t
    ) except *
    cdef int (*updating)(
        dKalmanFilter, dStatespace
    )
    cdef np.float64_t (*calculate_loglikelihood)(
        dKalmanFilter, dStatespace, np.float64_t
    )
    cdef np.float64_t (*calculate_scale)(
        dKalmanFilter, dStatespace
    )
    cdef int (*prediction)(
        dKalmanFilter, dStatespace
    )

    # ### Define some constants
    cdef readonly int k_endog, k_states, k_posdef, k_endog2, k_states2, k_posdef2, k_endogstates, k_statesposdef
    cdef readonly int ldwork
    
    cdef allocate_arrays(self)
    cdef void set_dimensions(self)
    cpdef set_filter_method(self, int filter_method, int force_reset=*)
    cpdef seek(self, unsigned int t, int reset=*)

    cdef void initialize_statespace_object_pointers(self) except *
    cdef void initialize_filter_object_pointers(self)
    cdef void initialize_function_pointers(self) except *
    cdef void post_convergence(self)
    cdef void numerical_stability(self)
    cdef int check_diffuse(self)
    cdef void check_convergence(self)
    cdef void migrate_storage(self)
    cdef void _reinitialize_pointers(self) except *

    cdef void _forecasting(self)
    cdef np.float64_t _inversion(self)
    cdef void _updating(self)
    cdef np.float64_t _calculate_loglikelihood(self)
    cdef void _prediction(self)

# Single precision complex
cdef class cKalmanFilter(object):
    # Statespace object
    cdef readonly cStatespace model

    # ### Filter parameters
    cdef readonly int t
    cdef public np.float64_t tolerance, tolerance_diffuse
    cdef readonly int nobs_diffuse
    cdef readonly int nobs_kendog_diffuse_nonsingular
    cdef readonly int nobs_kendog_univariate_singular
    cdef readonly int converged
    cdef readonly int period_converged
    cdef readonly int time_invariant
    cdef readonly int filter_method
    cdef public int inversion_method
    cdef public int stability_method
    cdef readonly int conserve_memory
    cdef public int filter_timing
    cdef readonly int loglikelihood_burn

    # ### Kalman filter properties
    cdef readonly np.complex64_t [:] loglikelihood, scale
    cdef readonly np.complex64_t [::1,:] filtered_state, predicted_state, forecast, forecast_error, standardized_forecast_error
    cdef readonly np.complex64_t [::1,:,:] filtered_state_cov, predicted_state_cov, forecast_error_cov, predicted_diffuse_state_cov, forecast_error_diffuse_cov, M, M_inf
    cdef readonly np.complex64_t [::1,:,:] kalman_gain

    # ### Steady State Values
    cdef readonly np.complex64_t [::1,:] converged_forecast_error_cov
    cdef readonly np.complex64_t [::1,:] converged_filtered_state_cov
    cdef readonly np.complex64_t [::1,:] converged_predicted_state_cov
    cdef readonly np.complex64_t [::1,:] converged_kalman_gain
    cdef readonly np.complex64_t [::1,:] converged_M
    cdef readonly np.complex64_t converged_determinant

    # ### Temporary arrays
    cdef readonly np.complex64_t [::1,:] forecast_error_fac
    cdef readonly int [:] forecast_error_ipiv
    cdef readonly np.complex64_t [::1,:] forecast_error_work
    cdef readonly np.complex64_t [::1,:] tmp0, tmp00
    cdef readonly np.complex64_t [::1,:] tmp2
    cdef readonly np.complex64_t [::1,:,:] tmp1, tmp3, tmp4
    cdef readonly np.complex64_t [:] tmpM_inf, tmpK0, tmpK1
    cdef readonly np.complex64_t [::1,:] tmpL0, tmpL1

    cdef readonly np.complex64_t determinant

    # ### Pointers to current-iteration arrays
    # cdef np.complex64_t * _obs
    # cdef np.complex64_t * _design
    # cdef np.complex64_t * _obs_intercept
    # cdef np.complex64_t * _obs_cov
    # cdef np.complex64_t * _transition
    # cdef np.complex64_t * _state_intercept
    # cdef np.complex64_t * _selection
    # cdef np.complex64_t * _state_cov
    # cdef np.complex64_t * _selected_state_cov
    # cdef np.complex64_t * _initial_state
    # cdef np.complex64_t * _initial_state_cov

    cdef np.complex64_t * _input_state
    cdef np.complex64_t * _input_state_cov
    cdef np.complex64_t * _input_diffuse_state_cov

    cdef np.complex64_t * _forecast
    cdef np.complex64_t * _forecast_error
    cdef np.complex64_t * _standardized_forecast_error
    cdef np.complex64_t * _forecast_error_cov
    cdef np.complex64_t * _filtered_state
    cdef np.complex64_t * _filtered_state_cov
    cdef np.complex64_t * _predicted_state
    cdef np.complex64_t * _predicted_state_cov

    cdef np.complex64_t * _forecast_error_diffuse_cov
    cdef np.complex64_t * _predicted_diffuse_state_cov
    cdef np.complex64_t * _M
    cdef np.complex64_t * _M_inf

    cdef np.complex64_t * _kalman_gain
    cdef np.complex64_t * _loglikelihood
    cdef np.complex64_t * _scale

    cdef np.complex64_t * _converged_forecast_error_cov
    cdef np.complex64_t * _converged_filtered_state_cov
    cdef np.complex64_t * _converged_predicted_state_cov
    cdef np.complex64_t * _converged_kalman_gain
    cdef np.complex64_t * _converged_M

    cdef np.complex64_t * _forecast_error_fac
    cdef int * _forecast_error_ipiv
    cdef np.complex64_t * _forecast_error_work

    cdef np.complex64_t * _tmp0
    cdef np.complex64_t * _tmp00
    cdef np.complex64_t * _tmp1
    cdef np.complex64_t * _tmp2
    cdef np.complex64_t * _tmp3
    cdef np.complex64_t * _tmp4

    cdef np.complex64_t * _tmpM_inf
    cdef np.complex64_t * _tmpK0
    cdef np.complex64_t * _tmpK1
    cdef np.complex64_t * _tmpL0
    cdef np.complex64_t * _tmpL1

    # ### Pointers to current-iteration Kalman filtering functions
    cdef int (*forecasting)(
        cKalmanFilter, cStatespace
    )
    cdef np.complex64_t (*inversion)(
        cKalmanFilter, cStatespace, np.complex64_t
    ) except *
    cdef int (*updating)(
        cKalmanFilter, cStatespace
    )
    cdef np.complex64_t (*calculate_loglikelihood)(
        cKalmanFilter, cStatespace, np.complex64_t
    )
    cdef np.complex64_t (*calculate_scale)(
        cKalmanFilter, cStatespace
    )
    cdef int (*prediction)(
        cKalmanFilter, cStatespace
    )

    # ### Define some constants
    cdef readonly int k_endog, k_states, k_posdef, k_endog2, k_states2, k_posdef2, k_endogstates, k_statesposdef
    cdef readonly int ldwork
    
    cdef allocate_arrays(self)
    cdef void set_dimensions(self)
    cpdef set_filter_method(self, int filter_method, int force_reset=*)
    cpdef seek(self, unsigned int t, int reset=*)

    cdef void initialize_statespace_object_pointers(self) except *
    cdef void initialize_filter_object_pointers(self)
    cdef void initialize_function_pointers(self) except *
    cdef void post_convergence(self)
    cdef void numerical_stability(self)
    cdef int check_diffuse(self)
    cdef void check_convergence(self)
    cdef void migrate_storage(self)
    cdef void _reinitialize_pointers(self) except *

    cdef void _forecasting(self)
    cdef np.complex64_t _inversion(self)
    cdef void _updating(self)
    cdef np.complex64_t _calculate_loglikelihood(self)
    cdef void _prediction(self)

# Double precision complex
cdef class zKalmanFilter(object):
    # Statespace object
    cdef readonly zStatespace model

    # ### Filter parameters
    cdef readonly int t
    cdef public np.float64_t tolerance, tolerance_diffuse
    cdef readonly int nobs_diffuse
    cdef readonly int nobs_kendog_diffuse_nonsingular
    cdef readonly int nobs_kendog_univariate_singular
    cdef readonly int converged
    cdef readonly int period_converged
    cdef readonly int time_invariant
    cdef readonly int filter_method
    cdef public int inversion_method
    cdef public int stability_method
    cdef readonly int conserve_memory
    cdef public int filter_timing
    cdef readonly int loglikelihood_burn

    # ### Kalman filter properties
    cdef readonly np.complex128_t [:] loglikelihood, scale
    cdef readonly np.complex128_t [::1,:] filtered_state, predicted_state, forecast, forecast_error, standardized_forecast_error
    cdef readonly np.complex128_t [::1,:,:] filtered_state_cov, predicted_state_cov, forecast_error_cov, predicted_diffuse_state_cov, forecast_error_diffuse_cov, M, M_inf
    cdef readonly np.complex128_t [::1,:,:] kalman_gain

    # ### Steady State Values
    cdef readonly np.complex128_t [::1,:] converged_forecast_error_cov
    cdef readonly np.complex128_t [::1,:] converged_filtered_state_cov
    cdef readonly np.complex128_t [::1,:] converged_predicted_state_cov
    cdef readonly np.complex128_t [::1,:] converged_kalman_gain
    cdef readonly np.complex128_t [::1,:] converged_M
    cdef readonly np.complex128_t converged_determinant

    # ### Temporary arrays
    cdef readonly np.complex128_t [::1,:] forecast_error_fac
    cdef readonly int [:] forecast_error_ipiv
    cdef readonly np.complex128_t [::1,:] forecast_error_work
    cdef readonly np.complex128_t [::1,:] tmp0, tmp00
    cdef readonly np.complex128_t [::1,:] tmp2
    cdef readonly np.complex128_t [::1,:,:] tmp1, tmp3, tmp4
    cdef readonly np.complex128_t [:] tmpM_inf, tmpK0, tmpK1
    cdef readonly np.complex128_t [::1,:] tmpL0, tmpL1

    cdef readonly np.complex128_t determinant

    # ### Pointers to current-iteration arrays
    # cdef np.complex128_t * _obs
    # cdef np.complex128_t * _design
    # cdef np.complex128_t * _obs_intercept
    # cdef np.complex128_t * _obs_cov
    # cdef np.complex128_t * _transition
    # cdef np.complex128_t * _state_intercept
    # cdef np.complex128_t * _selection
    # cdef np.complex128_t * _state_cov
    # cdef np.complex128_t * _selected_state_cov
    # cdef np.complex128_t * _initial_state
    # cdef np.complex128_t * _initial_state_cov

    cdef np.complex128_t * _input_state
    cdef np.complex128_t * _input_state_cov
    cdef np.complex128_t * _input_diffuse_state_cov

    cdef np.complex128_t * _forecast
    cdef np.complex128_t * _forecast_error
    cdef np.complex128_t * _standardized_forecast_error
    cdef np.complex128_t * _forecast_error_cov
    cdef np.complex128_t * _filtered_state
    cdef np.complex128_t * _filtered_state_cov
    cdef np.complex128_t * _predicted_state
    cdef np.complex128_t * _predicted_state_cov

    cdef np.complex128_t * _forecast_error_diffuse_cov
    cdef np.complex128_t * _predicted_diffuse_state_cov
    cdef np.complex128_t * _M
    cdef np.complex128_t * _M_inf

    cdef np.complex128_t * _kalman_gain
    cdef np.complex128_t * _loglikelihood
    cdef np.complex128_t * _scale

    cdef np.complex128_t * _converged_forecast_error_cov
    cdef np.complex128_t * _converged_filtered_state_cov
    cdef np.complex128_t * _converged_predicted_state_cov
    cdef np.complex128_t * _converged_kalman_gain
    cdef np.complex128_t * _converged_M

    cdef np.complex128_t * _forecast_error_fac
    cdef int * _forecast_error_ipiv
    cdef np.complex128_t * _forecast_error_work

    cdef np.complex128_t * _tmp0
    cdef np.complex128_t * _tmp00
    cdef np.complex128_t * _tmp1
    cdef np.complex128_t * _tmp2
    cdef np.complex128_t * _tmp3
    cdef np.complex128_t * _tmp4

    cdef np.complex128_t * _tmpM_inf
    cdef np.complex128_t * _tmpK0
    cdef np.complex128_t * _tmpK1
    cdef np.complex128_t * _tmpL0
    cdef np.complex128_t * _tmpL1

    # ### Pointers to current-iteration Kalman filtering functions
    cdef int (*forecasting)(
        zKalmanFilter, zStatespace
    )
    cdef np.complex128_t (*inversion)(
        zKalmanFilter, zStatespace, np.complex128_t
    ) except *
    cdef int (*updating)(
        zKalmanFilter, zStatespace
    )
    cdef np.complex128_t (*calculate_loglikelihood)(
        zKalmanFilter, zStatespace, np.complex128_t
    )
    cdef np.complex128_t (*calculate_scale)(
        zKalmanFilter, zStatespace
    )
    cdef int (*prediction)(
        zKalmanFilter, zStatespace
    )

    # ### Define some constants
    cdef readonly int k_endog, k_states, k_posdef, k_endog2, k_states2, k_posdef2, k_endogstates, k_statesposdef
    cdef readonly int ldwork
    
    cdef allocate_arrays(self)
    cdef void set_dimensions(self)
    cpdef set_filter_method(self, int filter_method, int force_reset=*)
    cpdef seek(self, unsigned int t, int reset=*)

    cdef void initialize_statespace_object_pointers(self) except *
    cdef void initialize_filter_object_pointers(self)
    cdef void initialize_function_pointers(self) except *
    cdef void post_convergence(self)
    cdef void numerical_stability(self)
    cdef int check_diffuse(self)
    cdef void check_convergence(self)
    cdef void migrate_storage(self)
    cdef void _reinitialize_pointers(self) except *

    cdef void _forecasting(self)
    cdef np.complex128_t _inversion(self)
    cdef void _updating(self)
    cdef np.complex128_t _calculate_loglikelihood(self)
    cdef void _prediction(self)
