#!python
#cython: wraparound=False, boundscheck=False, cdivision=True

from cpython cimport bool
import numpy as np
cimport numpy as np


np.import_array()

ctypedef fused numeric:
    np.float32_t
    np.float64_t
    np.complex64_t
    np.complex128_t

cpdef _initialize_ets_smooth(
    numeric [:] params,
    numeric[:,:] xhat,
    long [:] is_fixed,
    numeric [:] fixed_values,
    bool use_beta_star=False,
    bool use_gamma_star=False,
):
    """Extracts parameters and initializes states xhat"""
    cdef numeric [:] full_params
    cdef numeric alpha, beta, gamma, phi, beta_star, gamma_star
    cdef Py_ssize_t idx_all, idx_params, n_all, m, n

    n_all = len(is_fixed)
    full_params = np.empty_like(fixed_values)
    idx_params = 0
    for idx_all in range(n_all):
        if not is_fixed[idx_all]:
            full_params[idx_all] = params[idx_params]
            idx_params += 1
        else:
            full_params[idx_all] = fixed_values[idx_all]


    # get params
    alpha, beta, gamma, phi = full_params[0:4]
    m = len(full_params[6:])
    n = len(xhat)

    # calculate beta_star and gamma_star
    # if the use flags are true, the parameters are already the starred versions
    if use_beta_star:
        beta_star = beta
    else:
        beta_star = beta/alpha
    if use_gamma_star:
        gamma_star = gamma
    else:
        gamma_star = gamma / (1 - alpha)

    # initialize states
    # l = xhat[:,0], b = xhat[:,1], s = xhat[:,2:2+m]
    # seasons are sorted such that xhat[:,2+m-1] contains s[-m]
    xhat[n-1, :] = full_params[4:]

    return alpha, beta_star, gamma_star, phi, m, n

def _ets_smooth_add_add(numeric [:] params,
                        numeric [:] y,
                        numeric [:] yhat,
                        numeric [:,:] xhat,
                        long [:] is_fixed,
                        numeric [:] fixed_values,
                        bool use_beta_star=False,
                        bool use_gamma_star=False):
    """Smoothing with additive trend and additive season"""
    cdef numeric alpha, beta_star, gamma_star, phi
    cdef Py_ssize_t i, n, m, prev

    # get params
    alpha, beta_star, gamma_star, phi, m, n = _initialize_ets_smooth(
        params, xhat, is_fixed, fixed_values, use_beta_star, use_gamma_star
    )

    # smooth
    for i in range(n):
        prev = (n+i-1) % n

        # s[t-m] = xhat[prev, 2+m-1]
        yhat[i] = xhat[prev, 0] + phi * xhat[prev, 1] + xhat[prev, 2+m-1]
        # l_t = a * (y_t - s_t-m) + (1-a) * (l_t-1 + phi*b_t-1)
        xhat[i, 0] = (alpha * (y[i] - xhat[prev, 2+m-1])
                      + (1 - alpha) * (xhat[prev, 0] + phi * xhat[prev, 1]))
        # b_t = (b*) * (l_t - l_t-1) + (1 - (b*)) * phi * b_t-1
        xhat[i, 1] = (beta_star * (xhat[i, 0] - xhat[prev, 0])
                      + (1 - beta_star) * phi * xhat[prev, 1])
        # s_t = (g*) * (y_t - l_t) + (1 - (g*)) * s_t-m
        xhat[i, 2] = (gamma_star * (y[i] - xhat[i, 0])
                      + (1 - gamma_star) * xhat[prev, 2+m-1])
        xhat[i, 3:] = xhat[prev, 2:2+m-1]


def _ets_smooth_add_mul(numeric [:] params,
                        numeric [:] y,
                        numeric [:] yhat,
                        numeric [:,:] xhat,
                        long [:] is_fixed,
                        numeric [:] fixed_values,
                        bool use_beta_star=False,
                        bool use_gamma_star=False):
    """Smoothing with additive trend and multiplicative season"""
    cdef numeric alpha, beta_star, gamma_star, phi
    cdef Py_ssize_t i, n, m, prev

    # get params
    alpha, beta_star, gamma_star, phi, m, n = _initialize_ets_smooth(
        params, xhat, is_fixed, fixed_values, use_beta_star, use_gamma_star
    )

    # smooth
    for i in range(n):
        prev = (n+i-1) % n

        # s[t-m] = xhat[prev, 2+m-1]
        yhat[i] = (xhat[prev, 0] + phi * xhat[prev, 1]) * xhat[prev, 2+m-1]
        # l_t = a * (y_t / s_t-m) + (1-a) * (l_t-1 + phi*b_t-1)
        xhat[i, 0] = (alpha * (y[i] / xhat[prev, 2+m-1])
                      + (1 - alpha) * (xhat[prev, 0] + phi * xhat[prev, 1]))
        # b_t = (b*) * (l_t - l_t-1) + (1 - (b*)) * phi * b_t-1
        xhat[i, 1] = (beta_star * (xhat[i, 0] - xhat[prev, 0])
                      + (1 - beta_star) * phi * xhat[prev, 1])
        # s_t = g * (y_t / (l_t-1 - phi*b_t-1)) + (1 - g) * s_t-m
        xhat[i, 2] = (gamma_star * (y[i] / xhat[i, 0])
                      + (1 - gamma_star) * xhat[prev, 2+m-1])
        xhat[i, 3:] = xhat[prev, 2:2+m-1]


def _ets_smooth_mul_add(numeric [:] params,
                        numeric [:] y,
                        numeric [:] yhat,
                        numeric [:,:] xhat,
                        long [:] is_fixed,
                        numeric [:] fixed_values,
                        bool use_beta_star=False,
                        bool use_gamma_star=False):
    """Smoothing with multiplicative trend and additive season"""
    cdef numeric alpha, beta_star, gamma_star, phi
    cdef Py_ssize_t i, n, m, prev

    # get params
    alpha, beta_star, gamma_star, phi, m, n = _initialize_ets_smooth(
        params, xhat, is_fixed, fixed_values, use_beta_star, use_gamma_star
    )

    # smooth
    for i in range(n):
        prev = (n+i-1) % n

        # s[t-m] = xhat[prev, 2+m-1]
        yhat[i] = (xhat[prev, 0] * xhat[prev, 1]**phi) + xhat[prev, 2+m-1]
        # l_t = a * (y_t - s_t-m) + (1-a) * (l_t-1 * b_t-1**phi)
        xhat[i, 0] = (alpha * (y[i] - xhat[prev, 2+m-1])
                      + (1 - alpha) * (xhat[prev, 0] * xhat[prev, 1]**phi))
        # b_t = (b*) * (l_t / l_t-1) + (1 - (b*)) * b_t-1**phi
        xhat[i, 1] = (beta_star * (xhat[i, 0] / xhat[prev, 0])
                      + (1 - beta_star) * xhat[prev, 1]**phi)
        # s_t = g * (y_t - (l_t-1 * b_t-1**phi)) + (1 - g) * s_t-m
        xhat[i, 2] = (gamma_star * (y[i] - xhat[i, 0])
                      + (1 - gamma_star) * xhat[prev, 2+m-1])
        xhat[i, 3:] = xhat[prev, 2:2+m-1]


def _ets_smooth_mul_mul(numeric [:] params,
                        numeric [:] y,
                        numeric [:] yhat,
                        numeric [:,:] xhat,
                        long [:] is_fixed,
                        numeric [:] fixed_values,
                        bool use_beta_star=False,
                        bool use_gamma_star=False):
    """Smoothing with multiplicative trend and multiplicative season"""
    cdef numeric alpha, beta_star, gamma_star, phi
    cdef Py_ssize_t i, n, m, prev

    # get params
    alpha, beta_star, gamma_star, phi, m, n = _initialize_ets_smooth(
        params, xhat, is_fixed, fixed_values, use_beta_star, use_gamma_star
    )

    # smooth
    for i in range(n):
        prev = (n+i-1) % n

        # s[t-m] = xhat[prev, 2+m-1]
        yhat[i] = (xhat[prev, 0] * xhat[prev, 1]**phi) * xhat[prev, 2+m-1]
        # l_t = a * (y_t / s_t-m) + (1-a) * (l_t-1 * b_t-1**phi)
        xhat[i, 0] = (alpha * (y[i] / xhat[prev, 2+m-1])
                      + (1 - alpha) * (xhat[prev, 0] * xhat[prev, 1]**phi))
        # b_t = (b*) * (l_t / l_t-1) + (1 - (b*)) * b_t-1**phi
        xhat[i, 1] = (beta_star * (xhat[i, 0] / xhat[prev, 0])
                      + (1 - beta_star) * xhat[prev, 1]**phi)
        # s_t = g * (y_t / (l_t-1 * b_t-1**phi)) + (1 - g) * s_t-m
        xhat[i, 2] = (gamma_star * y[i] / xhat[i, 0]
                      + (1 - gamma_star) * xhat[prev, 2+m-1])
        xhat[i, 3:] = xhat[prev, 2:2+m-1]
