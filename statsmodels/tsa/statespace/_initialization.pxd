#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
State Space Models - Initialization declarations

Author: Chad Fulton  
License: Simplified-BSD
"""

cimport numpy as np

from statsmodels.tsa.statespace._representation cimport (
    sStatespace, dStatespace, cStatespace, zStatespace
)

cdef class sInitialization(object):
    cdef readonly int k_states
    cdef public np.float64_t approximate_diffuse_variance

    cdef np.float32_t [:] constant
    cdef np.float32_t [::1, :] stationary_cov
    cdef np.float32_t [::1, :] _tmp_transition
    cdef np.float32_t [::1, :] _tmp_selected_state_cov

    cpdef int initialize(self, inititialization_type, int offset,
                         sStatespace model,
                         np.float32_t [:] initial_state_mean,
                         np.float32_t [::1, :] initial_diffuse_state_cov,
                         np.float32_t [::1, :] initial_stationary_state_cov,
                         int complex_step=*) except 1
    cdef int clear_constant(self, int offset, np.float32_t [:] initial_state_mean) except 1
    cdef int clear_cov(self, int offset, np.float32_t [::1, :] cov) except 1
    cdef int initialize_known_constant(self, int offset, np.float32_t [:] initial_state_mean) except 1
    cdef int initialize_known_stationary_cov(self, int offset, np.float32_t [::1, :] initial_stationary_state_cov) except 1
    cdef int initialize_diffuse(self, int offset, np.float32_t [::1, :] initial_diffuse_state_cov) except 1
    cdef int initialize_approximate_diffuse(self, int offset, np.float32_t [::1, :] initial_stationary_state_cov) except 1
    cdef int initialize_stationary_constant(self, int offset, sStatespace model, np.float32_t [:] initial_state_mean, int complex_step=*) except 1
    cdef int initialize_stationary_stationary_cov(self, int offset, sStatespace model, np.float32_t [::1, :] initial_stationary_state_cov, int complex_step=*) except 1

cdef class dInitialization(object):
    cdef readonly int k_states
    cdef public np.float64_t approximate_diffuse_variance

    cdef np.float64_t [:] constant
    cdef np.float64_t [::1, :] stationary_cov
    cdef np.float64_t [::1, :] _tmp_transition
    cdef np.float64_t [::1, :] _tmp_selected_state_cov

    cpdef int initialize(self, inititialization_type, int offset,
                         dStatespace model,
                         np.float64_t [:] initial_state_mean,
                         np.float64_t [::1, :] initial_diffuse_state_cov,
                         np.float64_t [::1, :] initial_stationary_state_cov,
                         int complex_step=*) except 1
    cdef int clear_constant(self, int offset, np.float64_t [:] initial_state_mean) except 1
    cdef int clear_cov(self, int offset, np.float64_t [::1, :] cov) except 1
    cdef int initialize_known_constant(self, int offset, np.float64_t [:] initial_state_mean) except 1
    cdef int initialize_known_stationary_cov(self, int offset, np.float64_t [::1, :] initial_stationary_state_cov) except 1
    cdef int initialize_diffuse(self, int offset, np.float64_t [::1, :] initial_diffuse_state_cov) except 1
    cdef int initialize_approximate_diffuse(self, int offset, np.float64_t [::1, :] initial_stationary_state_cov) except 1
    cdef int initialize_stationary_constant(self, int offset, dStatespace model, np.float64_t [:] initial_state_mean, int complex_step=*) except 1
    cdef int initialize_stationary_stationary_cov(self, int offset, dStatespace model, np.float64_t [::1, :] initial_stationary_state_cov, int complex_step=*) except 1

cdef class cInitialization(object):
    cdef readonly int k_states
    cdef public np.float64_t approximate_diffuse_variance

    cdef np.complex64_t [:] constant
    cdef np.complex64_t [::1, :] stationary_cov
    cdef np.complex64_t [::1, :] _tmp_transition
    cdef np.complex64_t [::1, :] _tmp_selected_state_cov

    cpdef int initialize(self, inititialization_type, int offset,
                         cStatespace model,
                         np.complex64_t [:] initial_state_mean,
                         np.complex64_t [::1, :] initial_diffuse_state_cov,
                         np.complex64_t [::1, :] initial_stationary_state_cov,
                         int complex_step=*) except 1
    cdef int clear_constant(self, int offset, np.complex64_t [:] initial_state_mean) except 1
    cdef int clear_cov(self, int offset, np.complex64_t [::1, :] cov) except 1
    cdef int initialize_known_constant(self, int offset, np.complex64_t [:] initial_state_mean) except 1
    cdef int initialize_known_stationary_cov(self, int offset, np.complex64_t [::1, :] initial_stationary_state_cov) except 1
    cdef int initialize_diffuse(self, int offset, np.complex64_t [::1, :] initial_diffuse_state_cov) except 1
    cdef int initialize_approximate_diffuse(self, int offset, np.complex64_t [::1, :] initial_stationary_state_cov) except 1
    cdef int initialize_stationary_constant(self, int offset, cStatespace model, np.complex64_t [:] initial_state_mean, int complex_step=*) except 1
    cdef int initialize_stationary_stationary_cov(self, int offset, cStatespace model, np.complex64_t [::1, :] initial_stationary_state_cov, int complex_step=*) except 1

cdef class zInitialization(object):
    cdef readonly int k_states
    cdef public np.float64_t approximate_diffuse_variance

    cdef np.complex128_t [:] constant
    cdef np.complex128_t [::1, :] stationary_cov
    cdef np.complex128_t [::1, :] _tmp_transition
    cdef np.complex128_t [::1, :] _tmp_selected_state_cov

    cpdef int initialize(self, inititialization_type, int offset,
                         zStatespace model,
                         np.complex128_t [:] initial_state_mean,
                         np.complex128_t [::1, :] initial_diffuse_state_cov,
                         np.complex128_t [::1, :] initial_stationary_state_cov,
                         int complex_step=*) except 1
    cdef int clear_constant(self, int offset, np.complex128_t [:] initial_state_mean) except 1
    cdef int clear_cov(self, int offset, np.complex128_t [::1, :] cov) except 1
    cdef int initialize_known_constant(self, int offset, np.complex128_t [:] initial_state_mean) except 1
    cdef int initialize_known_stationary_cov(self, int offset, np.complex128_t [::1, :] initial_stationary_state_cov) except 1
    cdef int initialize_diffuse(self, int offset, np.complex128_t [::1, :] initial_diffuse_state_cov) except 1
    cdef int initialize_approximate_diffuse(self, int offset, np.complex128_t [::1, :] initial_stationary_state_cov) except 1
    cdef int initialize_stationary_constant(self, int offset, zStatespace model, np.complex128_t [:] initial_state_mean, int complex_step=*) except 1
    cdef int initialize_stationary_stationary_cov(self, int offset, zStatespace model, np.complex128_t [::1, :] initial_stationary_state_cov, int complex_step=*) except 1
