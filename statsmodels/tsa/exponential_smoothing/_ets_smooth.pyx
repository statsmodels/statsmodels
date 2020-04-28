#!python
#cython: wraparound=False, boundscheck=False, cdivision=True, annotate=True

import numpy as np
cimport numpy as np

np.import_array()

cdef _initialize_hw_smooth(np.float_t [:] params, Py_ssize_t n):
    """Extracts parameters and initializes states xhat"""
    cdef np.float_t alpha, beta_star, gamma, phi, l0, b0
    cdef np.float_t [:] s0
    cdef np.float_t [:,:] xhat
    cdef Py_ssize_t i, m

    # get params
    alpha, beta_star, gamma, phi, l0, b0 = params[0:6]
    s0 = params[6:]

    # initialize states
    m = len(s0)
    # l = xhat[:,0], b = xhat[:,1], s = xhat[:,2]
    xhat = np.zeros((n, 3))
    xhat[n-1, 0] = l0
    xhat[n-1, 1] = b0
    xhat[n-m:n, 2] = s0

    return xhat, alpha, beta_star, gamma, phi, m

def _hw_smooth_add_add(np.float_t [:] params, np.float_t [:] y):
    """Smoothing with additive trend and additive season"""
    cdef np.float_t alpha, beta_star, gamma, phi
    cdef np.float_t [:,:] xhat
    cdef np.float_t [:] yhat
    cdef Py_ssize_t i, n, m, prev, previous_seas

    # get params
    n = len(y)
    xhat, alpha, beta_star, gamma, phi, m = _initialize_hw_smooth(params, n)
    yhat = np.empty(n, dtype=np.float)

    # smooth
    for i in range(n):
        prev = (n+i-1) % n
        prev_seas = (n+i-m) % n
        yhat[i] = xhat[prev, 0] + phi * xhat[prev, 1] + xhat[prev_seas, 2]
        # l_t = a * (y_t - s_t-m) + (1-a) * (l_t-1 + phi*b_t-1)
        xhat[i, 0] = (alpha * (y[i] - xhat[prev_seas, 2])
                      + (1 - alpha) * (xhat[prev, 0] + phi * xhat[prev, 1]))
        # b_t = (b*) * (l_t - l_t-1) + (1 - (b*)) * phi * b_t-1
        xhat[i, 1] = (beta_star * (xhat[i, 0] - xhat[prev, 0])
                      + (1 - beta_star) * phi * xhat[prev, 1])
        # s_t = g * (y_t - l_t-1 - phi*b_t-1) + (1 - g) * s_t-m
        xhat[i, 2] = (gamma * (y[i] - xhat[prev, 0] - phi * xhat[prev, 1])
                      + (1 - gamma) * xhat[prev_seas, 2])

    return yhat, xhat


def _hw_smooth_add_mul(np.float_t [:] params, np.float_t [:] y):
    """Smoothing with additive trend and multiplicative season"""
    cdef np.float_t alpha, beta_star, gamma, phi
    cdef np.float_t [:,:] xhat
    cdef np.float_t [:] yhat
    cdef Py_ssize_t i, n, m, prev, prev_seas

    # get params
    n = len(y)
    xhat, alpha, beta_star, gamma, phi, m = _initialize_hw_smooth(params, n)
    yhat = np.empty(n, dtype=np.float)

    # smooth
    for i in range(n):
        prev = (n+i-1) % n
        prev_seas = (n+i-m) % n
        yhat[i] = (xhat[prev, 0] + phi * xhat[prev, 1]) * xhat[prev_seas, 2]
        # l_t = a * (y_t / s_t-m) + (1-a) * (l_t-1 + phi*b_t-1)
        xhat[i, 0] = (alpha * (y[i] / xhat[prev_seas, 2])
                      + (1 - alpha) * (xhat[prev, 0] + phi * xhat[prev, 1]))
        # b_t = (b*) * (l_t - l_t-1) + (1 - (b*)) * phi * b_t-1
        xhat[i, 1] = (beta_star * (xhat[i, 0] - xhat[prev, 0])
                      + (1 - beta_star) * phi * xhat[prev, 1])
        # s_t = g * (y_t / (l_t-1 - phi*b_t-1)) + (1 - g) * s_t-m
        xhat[i, 2] = (gamma * (y[i] / (xhat[prev, 0] - phi * xhat[prev, 1]))
                      + (1 - gamma) * xhat[prev_seas, 2])

    return yhat, xhat


def _hw_smooth_mul_add(np.float_t [:] params, np.float_t [:] y):
    """Smoothing with multiplicative trend and additive season"""
    cdef np.float_t alpha, beta_star, gamma, phi
    cdef np.float_t [:,:] xhat
    cdef np.float_t [:] yhat
    cdef Py_ssize_t i, n, m, prev, prev_seas

    # get params
    n = len(y)
    xhat, alpha, beta_star, gamma, phi, m = _initialize_hw_smooth(params, n)
    yhat = np.empty(n, dtype=np.float)

    # smooth
    for i in range(n):
        prev = (i-1) % n
        prev_seas = (i-m) % n
        yhat[i] = (xhat[prev, 0] * xhat[prev, 1]**phi) + xhat[prev_seas, 2]
        # l_t = a * (y_t - s_t-m) + (1-a) * (l_t-1 * b_t-1**phi)
        xhat[i, 0] = (alpha * (y[i] - xhat[prev_seas, 2])
                      + (1 - alpha) * (xhat[prev, 0] * xhat[prev, 1]**phi))
        # b_t = (b*) * (l_t / l_t-1) + (1 - (b*)) * b_t-1**phi
        xhat[i, 1] = (beta_star * (xhat[i, 0] / xhat[prev, 0])
                      + (1 - beta_star) * xhat[prev, 1]**phi)
        # s_t = g * (y_t - (l_t-1 * b_t-1**phi)) + (1 - g) * s_t-m
        xhat[i, 2] = (gamma * (y[i] - (xhat[prev, 0] * xhat[prev, 1]**phi))
                      + (1 - gamma) * xhat[prev_seas, 2])

    return yhat, xhat


def _hw_smooth_mul_mul(np.float_t [:] params, np.float_t [:] y):
    """Smoothing with multiplicative trend and multiplicative season"""
    cdef np.float_t alpha, beta_star, gamma, phi
    cdef np.float_t [:,:] xhat
    cdef np.float_t [:] yhat
    cdef Py_ssize_t i, n, m, prev, prev_seas

    # get params
    n = len(y)
    xhat, alpha, beta_star, gamma, phi, m = _initialize_hw_smooth(params, n)
    yhat = np.empty(n, dtype=np.float)

    # smooth
    for i in range(n):
        prev = (n+i-1) % n
        prev_seas = (n+i-m) % n
        yhat[i] = (xhat[prev, 0] * xhat[prev, 1]**phi) * xhat[prev_seas, 2]
        # l_t = a * (y_t / s_t-m) + (1-a) * (l_t-1 * b_t-1**phi)
        xhat[i, 0] = (alpha * (y[i] / xhat[prev_seas, 2])
                      + (1 - alpha) * (xhat[prev, 0] * xhat[prev, 1]**phi))
        # b_t = (b*) * (l_t / l_t-1) + (1 - (b*)) * b_t-1**phi
        xhat[i, 1] = (beta_star * (xhat[i, 0] / xhat[prev, 0])
                      + (1 - beta_star) * xhat[prev, 1]**phi)
        # s_t = g * (y_t / (l_t-1 * b_t-1**phi)) + (1 - g) * s_t-m
        xhat[i, 2] = (gamma * (y[i] / (xhat[prev, 0] * xhat[prev, 1]**phi))
                      + (1 - gamma) * xhat[prev_seas, 2])

    return yhat, xhat
