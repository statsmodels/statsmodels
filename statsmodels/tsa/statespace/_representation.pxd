#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
State Space Models declarations

Author: Chad Fulton  
License: Simplified-BSD
"""

cimport numpy as np

cdef class sStatespace(object):
    # Statespace dimensions
    cdef readonly int nobs, k_endog, k_states, k_posdef
    
    # Statespace representation matrices
    cdef readonly np.float32_t [::1,:] obs, obs_intercept, state_intercept
    cdef readonly np.float32_t [:] initial_state
    cdef readonly np.float32_t [::1,:] initial_state_cov, initial_diffuse_state_cov
    cdef readonly np.float32_t [::1,:,:] design, obs_cov, transition, selection, state_cov, selected_state_cov

    cdef readonly int [::1,:] missing
    cdef readonly int [:] nmissing
    cdef readonly int has_missing

    # Flags
    cdef readonly int time_invariant
    cdef readonly int initialized
    cdef public int diagonal_obs_cov
    cdef readonly int _diagonal_obs_cov
    cdef public int subset_design
    cdef public int companion_transition

    # Temporary arrays
    cdef np.float32_t [::1,:] tmp

    # Temporary selection arrays
    cdef readonly np.float32_t [:] selected_obs
    cdef readonly np.float32_t [:] selected_obs_intercept
    cdef readonly np.float32_t [:] selected_design
    cdef readonly np.float32_t [:] selected_obs_cov

    # Temporary transformation arrays
    cdef readonly np.float32_t [::1,:] transform_cholesky
    cdef readonly np.float32_t [::1,:] transform_obs_cov
    cdef readonly np.float32_t [::1,:] transform_design
    cdef readonly np.float32_t [:] transform_obs_intercept
    cdef readonly np.float32_t transform_determinant

    cdef readonly np.float32_t [:] collapse_obs
    cdef readonly np.float32_t [:] collapse_obs_tmp
    cdef readonly np.float32_t [::1,:] collapse_design
    cdef readonly np.float32_t [::1,:] collapse_obs_cov
    cdef readonly np.float32_t [::1,:] collapse_cholesky
    cdef readonly np.float32_t collapse_loglikelihood

    # Pointers
    cdef np.float32_t * _obs
    cdef np.float32_t * _design
    cdef np.float32_t * _obs_intercept
    cdef np.float32_t * _obs_cov
    cdef np.float32_t * _transition
    cdef np.float32_t * _state_intercept
    cdef np.float32_t * _selection
    cdef np.float32_t * _state_cov
    cdef np.float32_t * _selected_state_cov
    cdef np.float32_t * _initial_state
    cdef np.float32_t * _initial_state_cov
    cdef np.float32_t * _initial_diffuse_state_cov

    # Current location
    cdef int t
    cdef readonly int _previous_t
    cdef int _k_endog, _k_states, _k_posdef, _k_endog2, _k_states2, _k_posdef2, _k_endogstates, _k_statesposdef
    cdef int _nmissing

    # Functions
    cpdef seek(self, unsigned int t, unsigned int transform_diagonalize, unsigned int transform_generalized_collapse)

    cdef void set_dimensions(self, unsigned int k_endog, unsigned int k_states, unsigned int k_posdef)
    cdef void select_state_cov(self, unsigned int t)
    cdef int select_missing(self, unsigned int t)
    cdef void _select_missing_entire_obs(self, unsigned int t)
    cdef void _select_missing_partial_obs(self, unsigned int t)
    cdef void transform(self, unsigned int t, unsigned int previous_t, unsigned int transform_diagonalize, unsigned int transform_generalized_collapse) except *
    cdef void transform_diagonalize(self, unsigned int t, unsigned int previous_t) except *
    cdef int transform_generalized_collapse(self, unsigned int t, unsigned int previous_t) except *

cdef class dStatespace(object):
    # Statespace dimensions
    cdef readonly int nobs, k_endog, k_states, k_posdef
    
    # Statespace representation matrices
    cdef readonly np.float64_t [::1,:] obs, obs_intercept, state_intercept
    cdef readonly np.float64_t [:] initial_state
    cdef readonly np.float64_t [::1,:] initial_state_cov, initial_diffuse_state_cov
    cdef readonly np.float64_t [::1,:,:] design, obs_cov, transition, selection, state_cov, selected_state_cov

    cdef readonly int [::1,:] missing
    cdef readonly int [:] nmissing
    cdef readonly int has_missing

    # Flags
    cdef readonly int time_invariant
    cdef readonly int initialized
    cdef public int diagonal_obs_cov
    cdef readonly int _diagonal_obs_cov
    cdef public int subset_design
    cdef public int companion_transition

    # Temporary arrays
    cdef np.float64_t [::1,:] tmp

    # Temporary selection arrays
    cdef readonly np.float64_t [:] selected_obs
    cdef readonly np.float64_t [:] selected_obs_intercept
    cdef readonly np.float64_t [:] selected_design
    cdef readonly np.float64_t [:] selected_obs_cov

    # Temporary transformation arrays
    cdef readonly np.float64_t [::1,:] transform_cholesky
    cdef readonly np.float64_t [::1,:] transform_obs_cov
    cdef readonly np.float64_t [::1,:] transform_design
    cdef readonly np.float64_t [:] transform_obs_intercept
    cdef readonly np.float64_t transform_determinant

    cdef readonly np.float64_t [:] collapse_obs
    cdef readonly np.float64_t [:] collapse_obs_tmp
    cdef readonly np.float64_t [::1,:] collapse_design
    cdef readonly np.float64_t [::1,:] collapse_obs_cov
    cdef readonly np.float64_t [::1,:] collapse_cholesky
    cdef readonly np.float64_t collapse_loglikelihood

    # Pointers
    cdef np.float64_t * _obs
    cdef np.float64_t * _design
    cdef np.float64_t * _obs_intercept
    cdef np.float64_t * _obs_cov
    cdef np.float64_t * _transition
    cdef np.float64_t * _state_intercept
    cdef np.float64_t * _selection
    cdef np.float64_t * _state_cov
    cdef np.float64_t * _selected_state_cov
    cdef np.float64_t * _initial_state
    cdef np.float64_t * _initial_state_cov
    cdef np.float64_t * _initial_diffuse_state_cov

    # Current location
    cdef int t
    cdef readonly int _previous_t
    cdef int _k_endog, _k_states, _k_posdef, _k_endog2, _k_states2, _k_posdef2, _k_endogstates, _k_statesposdef
    cdef int _nmissing

    # Functions
    cpdef seek(self, unsigned int t, unsigned int transform_diagonalize, unsigned int transform_generalized_collapse)

    cdef void set_dimensions(self, unsigned int k_endog, unsigned int k_states, unsigned int k_posdef)
    cdef void select_state_cov(self, unsigned int t)
    cdef int select_missing(self, unsigned int t)
    cdef void _select_missing_entire_obs(self, unsigned int t)
    cdef void _select_missing_partial_obs(self, unsigned int t)
    cdef void transform(self, unsigned int t, unsigned int previous_t, unsigned int transform_diagonalize, unsigned int transform_generalized_collapse) except *
    cdef void transform_diagonalize(self, unsigned int t, unsigned int previous_t) except *
    cdef int transform_generalized_collapse(self, unsigned int t, unsigned int previous_t) except *

cdef class cStatespace(object):
    # Statespace dimensions
    cdef readonly int nobs, k_endog, k_states, k_posdef
    
    # Statespace representation matrices
    cdef readonly np.complex64_t [::1,:] obs, obs_intercept, state_intercept
    cdef readonly np.complex64_t [:] initial_state
    cdef readonly np.complex64_t [::1,:] initial_state_cov, initial_diffuse_state_cov
    cdef readonly np.complex64_t [::1,:,:] design, obs_cov, transition, selection, state_cov, selected_state_cov

    cdef readonly int [::1,:] missing
    cdef readonly int [:] nmissing
    cdef readonly int has_missing

    # Flags
    cdef readonly int time_invariant
    cdef readonly int initialized
    cdef public int diagonal_obs_cov
    cdef readonly int _diagonal_obs_cov
    cdef public int subset_design
    cdef public int companion_transition

    # Temporary arrays
    cdef np.complex64_t [::1,:] tmp

    # Temporary selection arrays
    cdef readonly np.complex64_t [:] selected_obs
    cdef readonly np.complex64_t [:] selected_obs_intercept
    cdef readonly np.complex64_t [:] selected_design
    cdef readonly np.complex64_t [:] selected_obs_cov

    # Temporary transformation arrays
    cdef readonly np.complex64_t [::1,:] transform_cholesky
    cdef readonly np.complex64_t [::1,:] transform_obs_cov
    cdef readonly np.complex64_t [::1,:] transform_design
    cdef readonly np.complex64_t [:] transform_obs_intercept
    cdef readonly np.complex64_t transform_determinant

    cdef readonly np.complex64_t [:] collapse_obs
    cdef readonly np.complex64_t [:] collapse_obs_tmp
    cdef readonly np.complex64_t [::1,:] collapse_design
    cdef readonly np.complex64_t [::1,:] collapse_obs_cov
    cdef readonly np.complex64_t [::1,:] collapse_cholesky
    cdef readonly np.complex64_t collapse_loglikelihood

    # Pointers
    cdef np.complex64_t * _obs
    cdef np.complex64_t * _design
    cdef np.complex64_t * _obs_intercept
    cdef np.complex64_t * _obs_cov
    cdef np.complex64_t * _transition
    cdef np.complex64_t * _state_intercept
    cdef np.complex64_t * _selection
    cdef np.complex64_t * _state_cov
    cdef np.complex64_t * _selected_state_cov
    cdef np.complex64_t * _initial_state
    cdef np.complex64_t * _initial_state_cov
    cdef np.complex64_t * _initial_diffuse_state_cov

    # Current location
    cdef int t
    cdef readonly int _previous_t
    cdef int _k_endog, _k_states, _k_posdef, _k_endog2, _k_states2, _k_posdef2, _k_endogstates, _k_statesposdef
    cdef int _nmissing

    # Functions
    cpdef seek(self, unsigned int t, unsigned int transform_diagonalize, unsigned int transform_generalized_collapse)

    cdef void set_dimensions(self, unsigned int k_endog, unsigned int k_states, unsigned int k_posdef)
    cdef void select_state_cov(self, unsigned int t)
    cdef int select_missing(self, unsigned int t)
    cdef void _select_missing_entire_obs(self, unsigned int t)
    cdef void _select_missing_partial_obs(self, unsigned int t)
    cdef void transform(self, unsigned int t, unsigned int previous_t, unsigned int transform_diagonalize, unsigned int transform_generalized_collapse) except *
    cdef void transform_diagonalize(self, unsigned int t, unsigned int previous_t) except *
    cdef int transform_generalized_collapse(self, unsigned int t, unsigned int previous_t) except *

cdef class zStatespace(object):
    # Statespace dimensions
    cdef readonly int nobs, k_endog, k_states, k_posdef
    
    # Statespace representation matrices
    cdef readonly np.complex128_t [::1,:] obs, obs_intercept, state_intercept
    cdef readonly np.complex128_t [:] initial_state
    cdef readonly np.complex128_t [::1,:] initial_state_cov, initial_diffuse_state_cov
    cdef readonly np.complex128_t [::1,:,:] design, obs_cov, transition, selection, state_cov, selected_state_cov

    cdef readonly int [::1,:] missing
    cdef readonly int [:] nmissing
    cdef readonly int has_missing

    # Flags
    cdef readonly int time_invariant
    cdef readonly int initialized
    cdef public int diagonal_obs_cov
    cdef readonly int _diagonal_obs_cov
    cdef public int subset_design
    cdef public int companion_transition

    # Temporary arrays
    cdef np.complex128_t [::1,:] tmp

    # Temporary selection arrays
    cdef readonly np.complex128_t [:] selected_obs
    cdef readonly np.complex128_t [:] selected_obs_intercept
    cdef readonly np.complex128_t [:] selected_design
    cdef readonly np.complex128_t [:] selected_obs_cov

    # Temporary transformation arrays
    cdef readonly np.complex128_t [::1,:] transform_cholesky
    cdef readonly np.complex128_t [::1,:] transform_obs_cov
    cdef readonly np.complex128_t [::1,:] transform_design
    cdef readonly np.complex128_t [:] transform_obs_intercept
    cdef readonly np.complex128_t transform_determinant

    cdef readonly np.complex128_t [:] collapse_obs
    cdef readonly np.complex128_t [:] collapse_obs_tmp
    cdef readonly np.complex128_t [::1,:] collapse_design
    cdef readonly np.complex128_t [::1,:] collapse_obs_cov
    cdef readonly np.complex128_t [::1,:] collapse_cholesky
    cdef readonly np.complex128_t collapse_loglikelihood

    # Pointers
    cdef np.complex128_t * _obs
    cdef np.complex128_t * _design
    cdef np.complex128_t * _obs_intercept
    cdef np.complex128_t * _obs_cov
    cdef np.complex128_t * _transition
    cdef np.complex128_t * _state_intercept
    cdef np.complex128_t * _selection
    cdef np.complex128_t * _state_cov
    cdef np.complex128_t * _selected_state_cov
    cdef np.complex128_t * _initial_state
    cdef np.complex128_t * _initial_state_cov
    cdef np.complex128_t * _initial_diffuse_state_cov

    # Current location
    cdef int t
    cdef readonly int _previous_t
    cdef int _k_endog, _k_states, _k_posdef, _k_endog2, _k_states2, _k_posdef2, _k_endogstates, _k_statesposdef
    cdef int _nmissing

    # Functions
    cpdef seek(self, unsigned int t, unsigned int transform_diagonalize, unsigned int transform_generalized_collapse)

    cdef void set_dimensions(self, unsigned int k_endog, unsigned int k_states, unsigned int k_posdef)
    cdef void select_state_cov(self, unsigned int t)
    cdef int select_missing(self, unsigned int t)
    cdef void _select_missing_entire_obs(self, unsigned int t)
    cdef void _select_missing_partial_obs(self, unsigned int t)
    cdef void transform(self, unsigned int t, unsigned int previous_t, unsigned int transform_diagonalize, unsigned int transform_generalized_collapse) except *
    cdef void transform_diagonalize(self, unsigned int t, unsigned int previous_t) except *
    cdef int transform_generalized_collapse(self, unsigned int t, unsigned int previous_t) except *

cdef int sselect_cov(int k, int k_posdef,
                           np.float32_t * tmp,
                           np.float32_t * selection,
                           np.float32_t * cov,
                           np.float32_t * selected_cov)

cdef int dselect_cov(int k, int k_posdef,
                           np.float64_t * tmp,
                           np.float64_t * selection,
                           np.float64_t * cov,
                           np.float64_t * selected_cov)

cdef int cselect_cov(int k, int k_posdef,
                           np.complex64_t * tmp,
                           np.complex64_t * selection,
                           np.complex64_t * cov,
                           np.complex64_t * selected_cov)

cdef int zselect_cov(int k, int k_posdef,
                           np.complex128_t * tmp,
                           np.complex128_t * selection,
                           np.complex128_t * cov,
                           np.complex128_t * selected_cov)
