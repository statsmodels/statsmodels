#!python
#cython: wraparound=False, boundscheck=False, cdivision=True

import numpy as np
cimport numpy as np

np.import_array()


cdef object _holt_init(double[::1] x, np.uint8_t[::1] xi, double[::1] p,
                       y, double[::1] l, double[::1] b):
    """Initialization for the Holt Models"""
    cdef double alpha, beta, phi, alphac, betac, l0, b0
    cdef Py_ssize_t i, n, idx = 0
    n = p.shape[0]
    for i in range(n):
        if xi[i]:
            p[i] = x[idx]
            idx += 1
    alpha = p[0]
    beta = p[1]
    l0 = p[3]
    b0 = p[4]
    phi = p[5]
    alphac = 1 - alpha
    betac = 1 - beta
    l[0] = l0
    b[0] = b0
    return alpha, beta, phi, alphac, betac


cdef double[::1] ensure_1d(object x):
    """
    This is a work aound function that ensures that X is a 1-d array.  It is needed since
    scipy.optimize.brute in version <= 1.0 calls squueze so that 1-d arrays are squeezed to
    scalars.

    Fixed in SciPy 1.1
    """
    if x.ndim == 0:
        # Due to bug in SciPy 1.0 that was fixed in 1.1 that squeezes
        x = np.array([x], dtype=np.double)
    return <np.ndarray>x


def _holt__(object x, np.uint8_t[::1] xi, double[::1] p, double[::1] y, double[::1] l,
            double[::1] b, double[::1] s, Py_ssize_t m, Py_ssize_t n, double max_seen):
    """
    Compute the sum of squared residuals for Simple Exponential Smoothing

    Returns
    -------
    sse : float
        Sum of squared errors
    """
    cdef double alpha, beta, phi, betac, alphac, err, sse
    cdef double[::1] x_arr
    cdef Py_ssize_t i

    x_arr = ensure_1d(x)
    alpha, beta, phi, alphac, betac = _holt_init(x_arr, xi, p, y, l, b)

    err = y[0] - l[0]
    sse = err * err
    for i in range(1, n):
        l[i] = (alpha * y[i - 1]) + (alphac * l[i - 1])
        err = y[i] - l[i]
        sse += err * err
    return sse


def _holt_mul_dam(object x, np.uint8_t[::1] xi, double[::1] p, double[::1] y, double[::1] l,
                  double[::1] b, double[::1] s, Py_ssize_t m, Py_ssize_t n, double max_seen):
    """
    Multiplicative and Multiplicative Damped
    Minimization Function
    (M,) & (Md,)
    """
    cdef double alpha, beta, phi, betac, alphac, err, sse
    cdef double[::1] x_arr
    cdef Py_ssize_t i

    x_arr = ensure_1d(x)
    alpha, beta, phi, alphac, betac = _holt_init(x_arr, xi, p, y, l, b)
    if alpha == 0.0:
        return max_seen
    if beta > alpha:
        return max_seen
    err = y[0] - (l[0] * b[0]**phi)
    sse = err * err
    for i in range(1, n):
        l[i] = (alpha * y[i - 1]) + (alphac * (l[i - 1] * b[i - 1]**phi))
        b[i] = (beta * (l[i] / l[i - 1])) + (betac * b[i - 1]**phi)
        err = y[i] - (l[i] * b[i]**phi)
        sse += err * err
    return sse


def _holt_add_dam(object x, np.uint8_t[::1] xi, double[::1] p, double[::1] y, double[::1] l,
                  double[::1] b, double[::1] s, Py_ssize_t m, Py_ssize_t n, double max_seen):
    """
    Additive and Additive Damped
    Minimization Function
    (A,) & (Ad,)
    """
    cdef double alpha, beta, phi, betac, alphac, err, sse
    cdef double[::1] x_arr
    cdef Py_ssize_t i

    x_arr = ensure_1d(x)
    alpha, beta, phi, alphac, betac = _holt_init(x_arr, xi, p, y, l, b)
    if alpha == 0.0:
        return max_seen
    if beta > alpha:
        return max_seen

    err = y[0] - (l[0] + phi * b[0])
    sse = err * err
    for i in range(1, n):
        l[i] = (alpha * y[i - 1]) + (alphac * (l[i - 1] + phi * b[i - 1]))
        b[i] = (beta * (l[i] - l[i - 1])) + (betac * phi * b[i - 1])
        err = y[i] - (l[i] + phi * b[i])
        sse += err * err
    return sse


cdef object _holt_win_init(double[::1] x, np.uint8_t[::1] xi, double[::1] p, y,
                           double[::1] l, double[::1] b, double[::1] s, Py_ssize_t m):
    """Initialization for the Holt Winters Seasonal Models"""
    cdef double alpha, beta, gamma, phi, alphac, betac, l0, b0
    cdef Py_ssize_t i, n, idx = 0

    n = p.shape[0]
    for i in range(n):
        if xi[i]:
            p[i] = x[idx]
            idx += 1
    alpha = p[0]
    beta = p[1]
    gamma = p[2]
    l0 = p[3]
    b0 = p[4]
    phi = p[5]

    alphac = 1 - alpha
    betac = 1 - beta
    gammac = 1 - gamma

    l[0] = l0
    b[0] = b0
    for i in range(m):
        s[i] = p[6+i]

    return alpha, beta, gamma, phi, alphac, betac, gammac


def _holt_win_add_add_dam(double[::1] x, np.uint8_t[::1] xi, double[::1] p, double[::1] y,
                          double[::1] l, double[::1] b, double[::1] s, Py_ssize_t m,
                          Py_ssize_t n, double max_seen):
    """
    Additive and Additive Damped with Additive Seasonal
    Minimization Function
    (A,A) & (Ad,A)
    """
    cdef double alpha, beta, gamma, phi, alphac, betac, gammac, err, sse
    cdef double[::1] x_arr
    cdef Py_ssize_t i

    x_arr = ensure_1d(x)
    alpha, beta, gamma, phi, alphac, betac, gammac = _holt_win_init(x_arr, xi, p, y, l, b, s, m)
    if alpha * beta == 0.0:
        return max_seen
    if beta > alpha or gamma > 1 - alpha:
        return max_seen

    err = y[0] - (l[0] + phi * b[0] + s[0])
    sse = err * err
    for i in range(1, n):
        l[i] = (alpha * y[i - 1]) - (alpha * s[i - 1]) + \
            (alphac * (l[i - 1] + phi * b[i - 1]))
        b[i] = (beta * (l[i] - l[i - 1])) + (betac * phi * b[i - 1])
        s[i + m - 1] = gamma * y[i - 1] - \
                       (gamma * (l[i - 1] + phi * b[i - 1])) + (gammac * s[i - 1])
        err = y[i] - (l[i] + phi * b[i] + s[i])
        sse += err * err
    return sse

def _holt_win__add(double[::1] x, np.uint8_t[::1] xi, double[::1] p, double[::1] y, double[::1] l,
                   double[::1] b, double[::1] s, Py_ssize_t m, Py_ssize_t n, double max_seen):
    """
    Additive Seasonal
    Minimization Function
    (,A)
    """
    cdef double alpha, beta, gamma, phi, alphac, betac, gammac, err, sse
    cdef double[::1] x_arr
    cdef Py_ssize_t i

    x_arr = ensure_1d(x)
    alpha, beta, gamma, phi, alphac, betac, gammac= _holt_win_init(x_arr, xi, p, y, l, b, s, m)
    if alpha == 0.0:
        return max_seen
    if gamma > 1 - alpha:
        return max_seen

    err = y[0] - (l[0] + s[0])
    sse = err * err
    for i in range(1, n):
        l[i] = (alpha * y[i - 1]) - (alpha * s[i - 1]) + (alphac * (l[i - 1]))
        s[i + m - 1] = gamma * y[i - 1] - \
            (gamma * (l[i - 1])) + (gammac * s[i - 1])
        err = y[i] - (l[i] + s[i])
        sse += err * err
    return sse


def _holt_win__mul(double[::1] x, np.uint8_t[::1] xi, double[::1] p, double[::1] y, double[::1] l,
                   double[::1] b, double[::1] s, Py_ssize_t m, Py_ssize_t n, double max_seen):
    """
    Multiplicative Seasonal
    Minimization Function
    (,M)
    """
    cdef double alpha, beta, gamma, phi, alphac, betac, gammac, err, sse
    cdef double[::1] x_arr
    cdef Py_ssize_t i

    x_arr = ensure_1d(x)
    alpha, beta, gamma, phi, alphac, betac, gammac= _holt_win_init(x_arr, xi, p, y, l, b, s, m)

    if alpha == 0.0:
        return max_seen
    if gamma > 1 - alpha:
        return max_seen

    err = y[0] - (l[0] * s[0])
    sse = err * err
    for i in range(1, n):
        l[i] = (alpha * y[i - 1] / s[i - 1]) + (alphac * (l[i - 1]))
        s[i + m - 1] = (gamma * y[i - 1] / (l[i - 1])) + (gammac * s[i - 1])
        err = y[i] - (l[i] * s[i])
        sse += err * err
    return sse


def _holt_win_mul_mul_dam(double[::1] x, np.uint8_t[::1] xi, double[::1] p, double[::1] y,
                          double[::1] l, double[::1] b, double[::1] s, Py_ssize_t m, Py_ssize_t n,
                          double max_seen):
    """
    Multiplicative and Multiplicative Damped with Multiplicative Seasonal
    Minimization Function
    (M,M) & (Md,M)
    """
    cdef double alpha, beta, gamma, phi, alphac, betac, gammac, err, sse
    cdef double[::1] x_arr
    cdef Py_ssize_t i

    x_arr = ensure_1d(x)
    alpha, beta, gamma, phi, alphac, betac, gammac= _holt_win_init(x_arr, xi, p, y, l, b, s, m)

    if alpha * beta == 0.0:
        return max_seen
    if beta > alpha or gamma > 1 - alpha:
        return max_seen

    err = y[0] - ((l[0] * b[0]**phi) * s[0])
    sse = err * err
    for i in range(1, n):
        l[i] = (alpha * y[i - 1] / s[i - 1]) + \
            (alphac * (l[i - 1] * b[i - 1]**phi))
        b[i] = (beta * (l[i] / l[i - 1])) + (betac * b[i - 1]**phi)
        s[i + m - 1] = (gamma * y[i - 1] / (l[i - 1] *
                                          b[i - 1]**phi)) + (gammac * s[i - 1])
        err = y[i] - ((l[i] * b[i]**phi) * s[i])
        sse += err * err
    return sse



def _holt_win_add_mul_dam(double[::1] x, np.uint8_t[::1] xi, double[::1] p, double[::1] y,
                          double[::1] l, double[::1] b, double[::1] s, Py_ssize_t m, Py_ssize_t n,
                          double max_seen):
    """
    Additive and Additive Damped with Multiplicative Seasonal
    Minimization Function
    (A,M) & (Ad,M)
    """
    cdef double alpha, beta, gamma, phi, alphac, betac, gammac, err, sse
    cdef double[::1] x_arr
    cdef Py_ssize_t i

    x_arr = ensure_1d(x)
    alpha, beta, gamma, phi, alphac, betac, gammac= _holt_win_init(x_arr, xi, p, y, l, b, s, m)

    if alpha * beta == 0.0:
        return max_seen
    if beta > alpha or gamma > 1 - alpha:
        return max_seen

    err = y[0] - ((l[0] + phi * b[0]) * s[0])
    sse = err * err
    for i in range(1, n):
        l[i] = (alpha * y[i - 1] / s[i - 1]) + \
            (alphac * (l[i - 1] + phi * b[i - 1]))
        b[i] = (beta * (l[i] - l[i - 1])) + (betac * phi * b[i - 1])
        s[i + m - 1] = (gamma * y[i - 1] / (l[i - 1] + phi *
                                          b[i - 1])) + (gammac * s[i - 1])
        err = y[i] - ((l[i] + phi * b[i]) * s[i])
        sse += err * err
    return sse



def _holt_win_mul_add_dam(double[::1] x, np.uint8_t[::1] xi, double[::1] p, double[::1] y,
                          double[::1] l, double[::1] b, double[::1] s, Py_ssize_t m, Py_ssize_t n,
                          double max_seen):
    """
    Multiplicative and Multiplicative Damped with Additive Seasonal
    Minimization Function
    (M,A) & (M,Ad)
    """
    cdef double alpha, beta, gamma, phi, alphac, betac, gammac, err, sse
    cdef double[::1] x_arr
    cdef Py_ssize_t i

    x_arr = ensure_1d(x)
    alpha, beta, gamma, phi, alphac, betac, gammac= _holt_win_init(x_arr, xi, p, y, l, b, s, m)

    if alpha * beta == 0.0:
        return max_seen
    if beta > alpha or gamma > 1 - alpha:
        return max_seen

    err = y[0] - ((l[0] * phi * b[0]) + s[0])
    sse = err * err
    for i in range(1, n):
        l[i] = (alpha * y[i - 1]) - (alpha * s[i - 1]) + \
            (alphac * (l[i - 1] * b[i - 1]**phi))
        b[i] = (beta * (l[i] / l[i - 1])) + (betac * b[i - 1]**phi)
        s[i + m - 1] = gamma * y[i - 1] - \
            (gamma * (l[i - 1] * b[i - 1]**phi)) + (gammac * s[i - 1])
        err = y[i] - ((l[i] * phi * b[i]) + s[i])
        sse += err * err
    return sse
