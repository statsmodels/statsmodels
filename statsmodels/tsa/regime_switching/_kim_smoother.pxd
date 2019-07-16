#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
Kim smoother declarations

Author: Chad Fulton  
License: Simplified-BSD
"""

cimport numpy as np

cpdef skim_smoother_log(int nobs, int k_regimes, int order,
                    np.float32_t [:, :, :] regime_transition,
                    np.float32_t [:, :] predicted_joint_probabilities,
                    np.float32_t [:, :] filtered_joint_probabilities,
                    np.float32_t [:, :] smoothed_joint_probabilities)

cpdef dkim_smoother_log(int nobs, int k_regimes, int order,
                    np.float64_t [:, :, :] regime_transition,
                    np.float64_t [:, :] predicted_joint_probabilities,
                    np.float64_t [:, :] filtered_joint_probabilities,
                    np.float64_t [:, :] smoothed_joint_probabilities)

cpdef ckim_smoother_log(int nobs, int k_regimes, int order,
                    np.complex64_t [:, :, :] regime_transition,
                    np.complex64_t [:, :] predicted_joint_probabilities,
                    np.complex64_t [:, :] filtered_joint_probabilities,
                    np.complex64_t [:, :] smoothed_joint_probabilities)

cpdef zkim_smoother_log(int nobs, int k_regimes, int order,
                    np.complex128_t [:, :, :] regime_transition,
                    np.complex128_t [:, :] predicted_joint_probabilities,
                    np.complex128_t [:, :] filtered_joint_probabilities,
                    np.complex128_t [:, :] smoothed_joint_probabilities)

cdef void skim_smoother_log_iteration(int tt, int k_regimes, int order,
                    np.float32_t [:] tmp_joint_probabilities,
                    np.float32_t [:] tmp_probabilities_fraction,
                    np.float32_t [:, :] regime_transition,
                    np.float32_t [:] predicted_joint_probabilities,
                    np.float32_t [:] filtered_joint_probabilities,
                    np.float32_t [:] prev_smoothed_joint_probabilities,
                    np.float32_t [:] next_smoothed_joint_probabilities) nogil

cdef void dkim_smoother_log_iteration(int tt, int k_regimes, int order,
                    np.float64_t [:] tmp_joint_probabilities,
                    np.float64_t [:] tmp_probabilities_fraction,
                    np.float64_t [:, :] regime_transition,
                    np.float64_t [:] predicted_joint_probabilities,
                    np.float64_t [:] filtered_joint_probabilities,
                    np.float64_t [:] prev_smoothed_joint_probabilities,
                    np.float64_t [:] next_smoothed_joint_probabilities) nogil

cdef void ckim_smoother_log_iteration(int tt, int k_regimes, int order,
                    np.complex64_t [:] tmp_joint_probabilities,
                    np.complex64_t [:] tmp_probabilities_fraction,
                    np.complex64_t [:, :] regime_transition,
                    np.complex64_t [:] predicted_joint_probabilities,
                    np.complex64_t [:] filtered_joint_probabilities,
                    np.complex64_t [:] prev_smoothed_joint_probabilities,
                    np.complex64_t [:] next_smoothed_joint_probabilities) nogil

cdef void zkim_smoother_log_iteration(int tt, int k_regimes, int order,
                    np.complex128_t [:] tmp_joint_probabilities,
                    np.complex128_t [:] tmp_probabilities_fraction,
                    np.complex128_t [:, :] regime_transition,
                    np.complex128_t [:] predicted_joint_probabilities,
                    np.complex128_t [:] filtered_joint_probabilities,
                    np.complex128_t [:] prev_smoothed_joint_probabilities,
                    np.complex128_t [:] next_smoothed_joint_probabilities) nogil
