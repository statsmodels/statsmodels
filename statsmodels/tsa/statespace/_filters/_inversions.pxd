#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
State Space Models - Inversion methods declarations

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
cdef np.float32_t sinverse_univariate(sKalmanFilter kfilter, sStatespace model, np.float32_t determinant) except *
cdef np.float32_t sfactorize_cholesky(sKalmanFilter kfilter, sStatespace model, np.float32_t determinant) except *
cdef np.float32_t sfactorize_lu(sKalmanFilter kfilter, sStatespace model, np.float32_t determinant) except *
cdef np.float32_t sinverse_cholesky(sKalmanFilter kfilter, sStatespace model, np.float32_t determinant) except *
cdef np.float32_t sinverse_lu(sKalmanFilter kfilter, sStatespace model, np.float32_t determinant) except *
cdef np.float32_t ssolve_cholesky(sKalmanFilter kfilter, sStatespace model, np.float32_t determinant) except *
cdef np.float32_t ssolve_lu(sKalmanFilter kfilter, sStatespace model, np.float32_t determinant) except *

# Double precision
cdef np.float64_t dinverse_univariate(dKalmanFilter kfilter, dStatespace model, np.float64_t determinant) except *
cdef np.float64_t dfactorize_cholesky(dKalmanFilter kfilter, dStatespace model, np.float64_t determinant) except *
cdef np.float64_t dfactorize_lu(dKalmanFilter kfilter, dStatespace model, np.float64_t determinant) except *
cdef np.float64_t dinverse_cholesky(dKalmanFilter kfilter, dStatespace model, np.float64_t determinant) except *
cdef np.float64_t dinverse_lu(dKalmanFilter kfilter, dStatespace model, np.float64_t determinant) except *
cdef np.float64_t dsolve_cholesky(dKalmanFilter kfilter, dStatespace model, np.float64_t determinant) except *
cdef np.float64_t dsolve_lu(dKalmanFilter kfilter, dStatespace model, np.float64_t determinant) except *

# Single precision complex
cdef np.complex64_t cinverse_univariate(cKalmanFilter kfilter, cStatespace model, np.complex64_t determinant) except *
cdef np.complex64_t cfactorize_cholesky(cKalmanFilter kfilter, cStatespace model, np.complex64_t determinant) except *
cdef np.complex64_t cfactorize_lu(cKalmanFilter kfilter, cStatespace model, np.complex64_t determinant) except *
cdef np.complex64_t cinverse_cholesky(cKalmanFilter kfilter, cStatespace model, np.complex64_t determinant) except *
cdef np.complex64_t cinverse_lu(cKalmanFilter kfilter, cStatespace model, np.complex64_t determinant) except *
cdef np.complex64_t csolve_cholesky(cKalmanFilter kfilter, cStatespace model, np.complex64_t determinant) except *
cdef np.complex64_t csolve_lu(cKalmanFilter kfilter, cStatespace model, np.complex64_t determinant) except *

# Double precision complex
cdef np.complex128_t zinverse_univariate(zKalmanFilter kfilter, zStatespace model, np.complex128_t determinant) except *
cdef np.complex128_t zfactorize_cholesky(zKalmanFilter kfilter, zStatespace model, np.complex128_t determinant) except *
cdef np.complex128_t zfactorize_lu(zKalmanFilter kfilter, zStatespace model, np.complex128_t determinant) except *
cdef np.complex128_t zinverse_cholesky(zKalmanFilter kfilter, zStatespace model, np.complex128_t determinant) except *
cdef np.complex128_t zinverse_lu(zKalmanFilter kfilter, zStatespace model, np.complex128_t determinant) except *
cdef np.complex128_t zsolve_cholesky(zKalmanFilter kfilter, zStatespace model, np.complex128_t determinant) except *
cdef np.complex128_t zsolve_lu(zKalmanFilter kfilter, zStatespace model, np.complex128_t determinant) except *
