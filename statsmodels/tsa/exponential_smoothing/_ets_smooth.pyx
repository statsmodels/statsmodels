#!python
#cython: wraparound=False, boundscheck=False, cdivision=True, annotate=True

import numpy as np
cimport numpy as np

np.import_array()

cpdef _initialize_ets_smooth(np.float_t [:] params, Py_ssize_t n):
    """Extracts parameters and initializes states xhat"""
    cdef np.float_t alpha, beta_star, gamma, phi, l0, b0
    cdef np.float_t [:] s0
    cdef np.float_t [:,:] xhat
    cdef Py_ssize_t m

    # get params
    alpha, beta_star, gamma, phi = params[0:4]
    m = len(params[6:])

    # initialize states
    # l = xhat[:,0], b = xhat[:,1], s = xhat[:,2:2+m]
    # seasons are sorted such that xhat[:,2+m-1] contains s[-m]
    xhat = np.zeros((n, 2 + m))
    xhat[n-1, :] = params[4:]

    return xhat, alpha, beta_star, gamma, phi, m

def _ets_smooth_add_add(np.float_t [:] params, np.float_t [:] y):
    """Smoothing with additive trend and additive season"""
    cdef np.float_t alpha, beta_star, gamma, phi
    cdef np.float_t [:,:] xhat
    cdef np.float_t [:] yhat
    cdef Py_ssize_t i, n, m, prev

    # get params
    n = len(y)
    xhat, alpha, beta_star, gamma, phi, m = _initialize_ets_smooth(params, n)
    yhat = np.empty(n, dtype=np.float)

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
        # s_t = g * (y_t - l_t-1 - phi*b_t-1) + (1 - g) * s_t-m
        xhat[i, 2] = (gamma * (y[i] - xhat[prev, 0] - phi * xhat[prev, 1])
                      + (1 - gamma) * xhat[prev, 2+m-1])
        xhat[i, 3:] = xhat[prev, 2:2+m-1]

    return yhat, xhat


def _ets_smooth_add_mul(np.float_t [:] params, np.float_t [:] y):
    """Smoothing with additive trend and multiplicative season"""
    cdef np.float_t alpha, beta_star, gamma, phi
    cdef np.float_t [:,:] xhat
    cdef np.float_t [:] yhat
    cdef Py_ssize_t i, n, m, prev

    # get params
    n = len(y)
    xhat, alpha, beta_star, gamma, phi, m = _initialize_ets_smooth(params, n)
    yhat = np.empty(n, dtype=np.float)

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
        xhat[i, 2] = (gamma * (y[i] / (xhat[prev, 0] - phi * xhat[prev, 1]))
                      + (1 - gamma) * xhat[prev, 2+m-1])
        xhat[i, 3:] = xhat[prev, 2:2+m-1]

    return yhat, xhat


def _ets_smooth_mul_add(np.float_t [:] params, np.float_t [:] y):
    """Smoothing with multiplicative trend and additive season"""
    cdef np.float_t alpha, beta_star, gamma, phi
    cdef np.float_t [:,:] xhat
    cdef np.float_t [:] yhat
    cdef Py_ssize_t i, n, m, prev

    # get params
    n = len(y)
    xhat, alpha, beta_star, gamma, phi, m = _initialize_ets_smooth(params, n)
    yhat = np.empty(n, dtype=np.float)

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
        xhat[i, 2] = (gamma * (y[i] - (xhat[prev, 0] * xhat[prev, 1]**phi))
                      + (1 - gamma) * xhat[prev, 2+m-1])
        xhat[i, 3:] = xhat[prev, 2:2+m-1]

    return yhat, xhat


def _ets_smooth_mul_mul(np.float_t [:] params, np.float_t [:] y):
    """Smoothing with multiplicative trend and multiplicative season"""
    cdef np.float_t alpha, beta_star, gamma, phi
    cdef np.float_t [:,:] xhat
    cdef np.float_t [:] yhat
    cdef Py_ssize_t i, n, m, prev

    # get params
    n = len(y)
    xhat, alpha, beta_star, gamma, phi, m = _initialize_ets_smooth(params, n)
    yhat = np.empty(n, dtype=np.float)

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
        xhat[i, 2] = (gamma * (y[i] / (xhat[prev, 0] * xhat[prev, 1]**phi))
                      + (1 - gamma) * xhat[prev, 2+m-1])
        xhat[i, 3:] = xhat[prev, 2:2+m-1]

    return yhat, xhat
