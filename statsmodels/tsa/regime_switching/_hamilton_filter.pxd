#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
Hamilton filter declarations

Author: Chad Fulton
License: Simplified-BSD
"""

cimport numpy as np

cdef void shamilton_filter_log_iteration(int t, int k_regimes, int order,
                                np.float32_t [:,:] transition,
                                np.float32_t [:] weighted_likelihoods,
                                np.float32_t [:] prev_filtered_marginalized_probabilities,
                                np.float32_t [:] conditional_likelihoods,
                                np.float32_t [:] joint_likelihoods,
                                np.float32_t [:] curr_predicted_joint_probabilities,
                                np.float32_t [:] prev_filtered_joint_probabilities,
                                np.float32_t [:] curr_filtered_joint_probabilities,
                                np.float32_t [:] tmp_predicted_joint_probabilities) nogil
cdef void dhamilton_filter_log_iteration(int t, int k_regimes, int order,
                                np.float64_t [:,:] transition,
                                np.float64_t [:] weighted_likelihoods,
                                np.float64_t [:] prev_filtered_marginalized_probabilities,
                                np.float64_t [:] conditional_likelihoods,
                                np.float64_t [:] joint_likelihoods,
                                np.float64_t [:] curr_predicted_joint_probabilities,
                                np.float64_t [:] prev_filtered_joint_probabilities,
                                np.float64_t [:] curr_filtered_joint_probabilities,
                                np.float64_t [:] tmp_predicted_joint_probabilities) nogil
cdef void chamilton_filter_log_iteration(int t, int k_regimes, int order,
                                np.complex64_t [:,:] transition,
                                np.complex64_t [:] weighted_likelihoods,
                                np.complex64_t [:] prev_filtered_marginalized_probabilities,
                                np.complex64_t [:] conditional_likelihoods,
                                np.complex64_t [:] joint_likelihoods,
                                np.complex64_t [:] curr_predicted_joint_probabilities,
                                np.complex64_t [:] prev_filtered_joint_probabilities,
                                np.complex64_t [:] curr_filtered_joint_probabilities,
                                np.complex64_t [:] tmp_predicted_joint_probabilities) nogil
cdef void zhamilton_filter_log_iteration(int t, int k_regimes, int order,
                                np.complex128_t [:,:] transition,
                                np.complex128_t [:] weighted_likelihoods,
                                np.complex128_t [:] prev_filtered_marginalized_probabilities,
                                np.complex128_t [:] conditional_likelihoods,
                                np.complex128_t [:] joint_likelihoods,
                                np.complex128_t [:] curr_predicted_joint_probabilities,
                                np.complex128_t [:] prev_filtered_joint_probabilities,
                                np.complex128_t [:] curr_filtered_joint_probabilities,
                                np.complex128_t [:] tmp_predicted_joint_probabilities) nogil
