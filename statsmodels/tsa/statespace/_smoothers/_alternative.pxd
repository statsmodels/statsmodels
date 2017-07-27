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
from statsmodels.tsa.statespace._kalman_smoother cimport (
    sKalmanSmoother, dKalmanSmoother, cKalmanSmoother, zKalmanSmoother
)

# Single precision
cdef int ssmoothed_estimators_measurement_alternative(sKalmanSmoother smoother, sKalmanFilter kfilter, sStatespace model) except *
cdef int ssmoothed_estimators_time_alternative(sKalmanSmoother smoother, sKalmanFilter kfilter, sStatespace model)
cdef int ssmoothed_state_alternative(sKalmanSmoother smoother, sKalmanFilter kfilter, sStatespace model)
cdef int ssmoothed_disturbances_alternative(sKalmanSmoother smoother, sKalmanFilter kfilter, sStatespace model)

# Double precision
cdef int dsmoothed_estimators_measurement_alternative(dKalmanSmoother smoother, dKalmanFilter kfilter, dStatespace model) except *
cdef int dsmoothed_estimators_time_alternative(dKalmanSmoother smoother, dKalmanFilter kfilter, dStatespace model)
cdef int dsmoothed_state_alternative(dKalmanSmoother smoother, dKalmanFilter kfilter, dStatespace model)
cdef int dsmoothed_disturbances_alternative(dKalmanSmoother smoother, dKalmanFilter kfilter, dStatespace model)

# Single precision complex
cdef int csmoothed_estimators_measurement_alternative(cKalmanSmoother smoother, cKalmanFilter kfilter, cStatespace model) except *
cdef int csmoothed_estimators_time_alternative(cKalmanSmoother smoother, cKalmanFilter kfilter, cStatespace model)
cdef int csmoothed_state_alternative(cKalmanSmoother smoother, cKalmanFilter kfilter, cStatespace model)
cdef int csmoothed_disturbances_alternative(cKalmanSmoother smoother, cKalmanFilter kfilter, cStatespace model)

# Double precision complex
cdef int zsmoothed_estimators_measurement_alternative(zKalmanSmoother smoother, zKalmanFilter kfilter, zStatespace model) except *
cdef int zsmoothed_estimators_time_alternative(zKalmanSmoother smoother, zKalmanFilter kfilter, zStatespace model)
cdef int zsmoothed_state_alternative(zKalmanSmoother smoother, zKalmanFilter kfilter, zStatespace model)
cdef int zsmoothed_disturbances_alternative(zKalmanSmoother smoother, zKalmanFilter kfilter, zStatespace model)
