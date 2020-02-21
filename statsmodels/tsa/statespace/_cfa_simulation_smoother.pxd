"""
State Space Model CFA simulation smoother declarations

Author: Chad Fulton
License: BSD-3
"""

# Typical imports
cimport numpy as np

from statsmodels.tsa.statespace._representation cimport (
    sStatespace, dStatespace, cStatespace, zStatespace)

cdef class sCFASimulationSmoother(object):
    # Statespace object
    cdef readonly sStatespace model
    cdef readonly int order, lower_bandwidth
    cdef readonly int k_states, k_states2

    cdef readonly np.float32_t [::1, :] prior_mean, posterior_mean
    cdef readonly np.float32_t [::1, :] posterior_cov_inv_chol, K
    cdef readonly np.float32_t [::1, :] initial_state_cov_inv, obs_cov_fac, selected_state_cov_inv
    cdef readonly np.float32_t [::1, :] QiT, TQiT, TQiTpQ, ZHiZ, HiZ
    cdef readonly np.float32_t [:] ymd

    cdef np.float32_t * _K
    cdef np.float32_t * _initial_state_cov_inv
    cdef np.float32_t * _obs_cov_fac
    cdef np.float32_t * _selected_state_cov_inv

    cdef np.float32_t * _QiT
    cdef np.float32_t * _TQiT
    cdef np.float32_t * _TQiTpQ
    cdef np.float32_t * _ZHiZ
    cdef np.float32_t * _HiZ
    cdef np.float32_t * _ymd

    cdef void _reinitialize_pointers(self) except *

    cpdef int update_sparse_posterior_moments(self) except *
    cpdef simulate(self, variates=*)

cdef class dCFASimulationSmoother(object):
    # Statespace object
    cdef readonly dStatespace model
    cdef readonly int order, lower_bandwidth
    cdef readonly int k_states, k_states2

    cdef readonly np.float64_t [::1, :] prior_mean, posterior_mean
    cdef readonly np.float64_t [::1, :] posterior_cov_inv_chol, K
    cdef readonly np.float64_t [::1, :] initial_state_cov_inv, obs_cov_fac, selected_state_cov_inv
    cdef readonly np.float64_t [::1, :] QiT, TQiT, TQiTpQ, ZHiZ, HiZ
    cdef readonly np.float64_t [:] ymd

    cdef np.float64_t * _K
    cdef np.float64_t * _initial_state_cov_inv
    cdef np.float64_t * _obs_cov_fac
    cdef np.float64_t * _selected_state_cov_inv

    cdef np.float64_t * _QiT
    cdef np.float64_t * _TQiT
    cdef np.float64_t * _TQiTpQ
    cdef np.float64_t * _ZHiZ
    cdef np.float64_t * _HiZ
    cdef np.float64_t * _ymd

    cdef void _reinitialize_pointers(self) except *

    cpdef int update_sparse_posterior_moments(self) except *
    cpdef simulate(self, variates=*)


cdef class cCFASimulationSmoother(object):
    # Statespace object
    cdef readonly cStatespace model
    cdef readonly int order, lower_bandwidth
    cdef readonly int k_states, k_states2

    cdef readonly np.complex64_t [::1, :] prior_mean, posterior_mean
    cdef readonly np.complex64_t [::1, :] posterior_cov_inv_chol, K
    cdef readonly np.complex64_t [::1, :] initial_state_cov_inv, obs_cov_fac, selected_state_cov_inv
    cdef readonly np.complex64_t [::1, :] QiT, TQiT, TQiTpQ, ZHiZ, HiZ
    cdef readonly np.complex64_t [:] ymd

    cdef np.complex64_t * _K
    cdef np.complex64_t * _initial_state_cov_inv
    cdef np.complex64_t * _obs_cov_fac
    cdef np.complex64_t * _selected_state_cov_inv

    cdef np.complex64_t * _QiT
    cdef np.complex64_t * _TQiT
    cdef np.complex64_t * _TQiTpQ
    cdef np.complex64_t * _ZHiZ
    cdef np.complex64_t * _HiZ
    cdef np.complex64_t * _ymd

    cdef void _reinitialize_pointers(self) except *

    cpdef int update_sparse_posterior_moments(self) except *
    cpdef simulate(self, variates=*)


cdef class zCFASimulationSmoother(object):
    # Statespace object
    cdef readonly zStatespace model
    cdef readonly int order, lower_bandwidth
    cdef readonly int k_states, k_states2

    cdef readonly np.complex128_t [::1, :] prior_mean, posterior_mean
    cdef readonly np.complex128_t [::1, :] posterior_cov_inv_chol, K
    cdef readonly np.complex128_t [::1, :] initial_state_cov_inv, obs_cov_fac, selected_state_cov_inv
    cdef readonly np.complex128_t [::1, :] QiT, TQiT, TQiTpQ, ZHiZ, HiZ
    cdef readonly np.complex128_t [:] ymd

    cdef np.complex128_t * _K
    cdef np.complex128_t * _initial_state_cov_inv
    cdef np.complex128_t * _obs_cov_fac
    cdef np.complex128_t * _selected_state_cov_inv

    cdef np.complex128_t * _QiT
    cdef np.complex128_t * _TQiT
    cdef np.complex128_t * _TQiTpQ
    cdef np.complex128_t * _ZHiZ
    cdef np.complex128_t * _HiZ
    cdef np.complex128_t * _ymd

    cdef void _reinitialize_pointers(self) except *

    cpdef int update_sparse_posterior_moments(self) except *
    cpdef simulate(self, variates=*)
