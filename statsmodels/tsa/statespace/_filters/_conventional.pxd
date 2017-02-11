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
cdef int sforecast_missing_conventional(sKalmanFilter kfilter, sStatespace model)
cdef int supdating_missing_conventional(sKalmanFilter kfilter, sStatespace model)
cdef np.float32_t sinverse_missing_conventional(sKalmanFilter kfilter, sStatespace model, np.float32_t determinant)  except *
cdef np.float32_t sloglikelihood_missing_conventional(sKalmanFilter kfilter, sStatespace model, np.float32_t determinant)
cdef np.float32_t sscale_missing_conventional(sKalmanFilter kfilter, sStatespace model)

cdef int sforecast_conventional(sKalmanFilter kfilter, sStatespace model)
cdef int supdating_conventional(sKalmanFilter kfilter, sStatespace model)
cdef int sprediction_conventional(sKalmanFilter kfilter, sStatespace model)
cdef np.float32_t sloglikelihood_conventional(sKalmanFilter kfilter, sStatespace model, np.float32_t determinant)
cdef np.float32_t sscale_conventional(sKalmanFilter kfilter, sStatespace model)

# Double precision
cdef int dforecast_missing_conventional(dKalmanFilter kfilter, dStatespace model)
cdef int dupdating_missing_conventional(dKalmanFilter kfilter, dStatespace model)
cdef np.float64_t dinverse_missing_conventional(dKalmanFilter kfilter, dStatespace model, np.float64_t determinant)  except *
cdef np.float64_t dloglikelihood_missing_conventional(dKalmanFilter kfilter, dStatespace model, np.float64_t determinant)
cdef np.float64_t dscale_missing_conventional(dKalmanFilter kfilter, dStatespace model)

cdef int dforecast_conventional(dKalmanFilter kfilter, dStatespace model)
cdef int dupdating_conventional(dKalmanFilter kfilter, dStatespace model)
cdef int dprediction_conventional(dKalmanFilter kfilter, dStatespace model)
cdef np.float64_t dloglikelihood_conventional(dKalmanFilter kfilter, dStatespace model, np.float64_t determinant)
cdef np.float64_t dscale_conventional(dKalmanFilter kfilter, dStatespace model)

# Single precision complex
cdef int cforecast_missing_conventional(cKalmanFilter kfilter, cStatespace model)
cdef int cupdating_missing_conventional(cKalmanFilter kfilter, cStatespace model)
cdef np.complex64_t cinverse_missing_conventional(cKalmanFilter kfilter, cStatespace model, np.complex64_t determinant)  except *
cdef np.complex64_t cloglikelihood_missing_conventional(cKalmanFilter kfilter, cStatespace model, np.complex64_t determinant)
cdef np.complex64_t cscale_missing_conventional(cKalmanFilter kfilter, cStatespace model)

cdef int cforecast_conventional(cKalmanFilter kfilter, cStatespace model)
cdef int cupdating_conventional(cKalmanFilter kfilter, cStatespace model)
cdef int cprediction_conventional(cKalmanFilter kfilter, cStatespace model)
cdef np.complex64_t cloglikelihood_conventional(cKalmanFilter kfilter, cStatespace model, np.complex64_t determinant)
cdef np.complex64_t cscale_conventional(cKalmanFilter kfilter, cStatespace model)

# Double precision complex
cdef int zforecast_missing_conventional(zKalmanFilter kfilter, zStatespace model)
cdef int zupdating_missing_conventional(zKalmanFilter kfilter, zStatespace model)
cdef np.complex128_t zinverse_missing_conventional(zKalmanFilter kfilter, zStatespace model, np.complex128_t determinant)  except *
cdef np.complex128_t zloglikelihood_missing_conventional(zKalmanFilter kfilter, zStatespace model, np.complex128_t determinant)
cdef np.complex128_t zscale_missing_conventional(zKalmanFilter kfilter, zStatespace model)

cdef int zforecast_conventional(zKalmanFilter kfilter, zStatespace model)
cdef int zupdating_conventional(zKalmanFilter kfilter, zStatespace model)
cdef int zprediction_conventional(zKalmanFilter kfilter, zStatespace model)
cdef np.complex128_t zloglikelihood_conventional(zKalmanFilter kfilter, zStatespace model, np.complex128_t determinant)
cdef np.complex128_t zscale_conventional(zKalmanFilter kfilter, zStatespace model)
