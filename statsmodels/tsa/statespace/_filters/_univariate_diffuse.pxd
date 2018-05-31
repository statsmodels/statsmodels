#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
State Space Models - Conventional Kalman Filter declarations

Author: Chad Fulton  
License: Simplified-BSD
"""

cimport numpy as np
from statsmodels.tsa.statespace._representation cimport (
    sStatespace, dStatespace, cStatespace, zStatespace
)
from statsmodels.tsa.statespace._kalman_filter cimport (
    sKalmanFilter, dKalmanFilter, cKalmanFilter, zKalmanFilter
)

# Single precision
cdef int sforecast_univariate_diffuse(sKalmanFilter kfilter, sStatespace model)
cdef int supdating_univariate_diffuse(sKalmanFilter kfilter, sStatespace model)
cdef int sprediction_univariate_diffuse(sKalmanFilter kfilter, sStatespace model)
cdef np.float32_t sinverse_noop_univariate_diffuse(sKalmanFilter kfilter, sStatespace model, np.float32_t determinant) except *
cdef np.float32_t sloglikelihood_univariate_diffuse(sKalmanFilter kfilter, sStatespace model, np.float32_t determinant)

cdef np.float32_t sforecast_error_diffuse_cov(sKalmanFilter kfilter, sStatespace model, int i)
cdef void spredicted_diffuse_state_cov(sKalmanFilter kfilter, sStatespace model)

# Double precision
cdef int dforecast_univariate_diffuse(dKalmanFilter kfilter, dStatespace model)
cdef int dupdating_univariate_diffuse(dKalmanFilter kfilter, dStatespace model)
cdef int dprediction_univariate_diffuse(dKalmanFilter kfilter, dStatespace model)
cdef np.float64_t dinverse_noop_univariate_diffuse(dKalmanFilter kfilter, dStatespace model, np.float64_t determinant) except *
cdef np.float64_t dloglikelihood_univariate_diffuse(dKalmanFilter kfilter, dStatespace model, np.float64_t determinant)

cdef np.float64_t dforecast_error_diffuse_cov(dKalmanFilter kfilter, dStatespace model, int i)
cdef void dpredicted_diffuse_state_cov(dKalmanFilter kfilter, dStatespace model)

# Single precision complex
cdef int cforecast_univariate_diffuse(cKalmanFilter kfilter, cStatespace model)
cdef int cupdating_univariate_diffuse(cKalmanFilter kfilter, cStatespace model)
cdef int cprediction_univariate_diffuse(cKalmanFilter kfilter, cStatespace model)
cdef np.complex64_t cinverse_noop_univariate_diffuse(cKalmanFilter kfilter, cStatespace model, np.complex64_t determinant) except *
cdef np.complex64_t cloglikelihood_univariate_diffuse(cKalmanFilter kfilter, cStatespace model, np.complex64_t determinant)

cdef np.complex64_t cforecast_error_diffuse_cov(cKalmanFilter kfilter, cStatespace model, int i)
cdef void cpredicted_diffuse_state_cov(cKalmanFilter kfilter, cStatespace model)

# Double precision complex
cdef int zforecast_univariate_diffuse(zKalmanFilter kfilter, zStatespace model)
cdef int zupdating_univariate_diffuse(zKalmanFilter kfilter, zStatespace model)
cdef int zprediction_univariate_diffuse(zKalmanFilter kfilter, zStatespace model)
cdef np.complex128_t zinverse_noop_univariate_diffuse(zKalmanFilter kfilter, zStatespace model, np.complex128_t determinant) except *
cdef np.complex128_t zloglikelihood_univariate_diffuse(zKalmanFilter kfilter, zStatespace model, np.complex128_t determinant)

cdef np.complex128_t zforecast_error_diffuse_cov(zKalmanFilter kfilter, zStatespace model, int i)
cdef void zpredicted_diffuse_state_cov(zKalmanFilter kfilter, zStatespace model)
