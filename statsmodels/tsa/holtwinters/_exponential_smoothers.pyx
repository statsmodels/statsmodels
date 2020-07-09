#!python
#cython: language_level=3, wraparound=False, boundscheck=False, cdivision=True

import numpy as np

cimport numpy as np

np.import_array()

cdef double LOWER_BOUND = np.sqrt(np.finfo(float).eps)


cdef class HoltWintersArgs:
    cdef long[::1] _xi
    cdef double[::1] _p, _y, _l, _b, _s
    cdef double[:, ::1] _bounds
    cdef Py_ssize_t _m, _n
    cdef bint _transform

    def __init__(
        self,
        long[::1] xi,
        double[::1] p,
        double[:, ::1] bounds,
        double[::1] y,
        Py_ssize_t m,
        Py_ssize_t n,
        bint transform=False,
    ):
        self._xi = xi
        self._p = p
        self._bounds = bounds
        self._y = y
        self._l = np.empty(n)
        self._b = np.empty(n)
        self._s = np.empty(n + m - 1)
        self._m = m
        self._n = n
        self._transform = transform

    @property
    def xi(self):
        return np.asarray(self._xi)

    @xi.setter
    def xi(self, value):
        self._xi = value

    @property
    def p(self):
        return np.asarray(self._p)

    @property
    def bounds(self):
        return np.asarray(self._bounds)

    @property
    def y(self):
        return np.asarray(self._y)

    @property
    def lvl(self):
        return np.asarray(self._l)

    @property
    def b(self):
        return np.asarray(self._b)

    @property
    def s(self):
        return np.asarray(self._s)

    @property
    def m(self):
        return self._m

    @property
    def n(self):
        return self._n

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        self._transform = value

cdef to_restricted(double[::1] p, long[::1] sel, double[:, ::1] bounds):
    """
    Transform parameters from the unrestricted [0,1] space

    Parameters
    ----------
    p : ndarray
        The parameters to transform
    sel : ndarray
        Array indicating whether a parameter is being estiamted
    bounds : ndarray
        2-d array of bounds where bound for element i is in row i
        and stored as [lb, ub]

    Returns
    -------

    """
    cdef double a, b, g, ub, lb
    a = p[0]
    b = p[1]
    g = p[2]

    if sel[0]:
        lb = max(LOWER_BOUND, bounds[0, 0])
        ub = min(1 - LOWER_BOUND, bounds[0, 1])
        a = lb + a * (ub - lb)
    if sel[1]:
        lb = bounds[1, 0]
        ub = min(a, bounds[1, 1])
        b = lb + b * (ub - lb)
    if sel[2]:
        lb = bounds[2, 0]
        ub = min(1. - a, bounds[2, 1])
        g = lb + g * (ub - lb)

    return a, b, g

def _test_to_restricted(p, sel, bounds):
    """Testing harness"""
    return to_restricted(p, sel, bounds)


# noinspection PyProtectedMember
cdef object holt_init(double[::1] x, HoltWintersArgs hw_args):
    """Initialization for the Holt Models"""
    cdef double alpha, beta, phi, alphac, betac, l0, b0, _
    cdef Py_ssize_t i, n, idx = 0
    cdef double[::1] p
    cdef long[::1] xi

    p = hw_args._p
    xi = hw_args._xi
    n = p.shape[0]
    for i in range(n):
        if xi[i]:
            p[i] = x[idx]
            idx += 1
    if hw_args._transform:
        alpha, beta, _ = to_restricted(p, xi, hw_args._bounds)
    else:
        alpha = p[0]
        beta = p[1]
    l0 = p[3]
    b0 = p[4]
    phi = p[5]
    alphac = 1 - alpha
    betac = 1 - beta
    hw_args._l[0] = l0
    hw_args._b[0] = b0
    return alpha, beta, phi, alphac, betac


# noinspection PyProtectedMember
def holt__(double[::1] x, HoltWintersArgs hw_args):
    """
    Compute the sum of squared residuals for Simple Exponential Smoothing

    Returns
    -------
    ndarray
        Array containing model errors
    """
    cdef double alpha, beta, phi, betac, alphac
    cdef double[::1] err, l, y
    cdef Py_ssize_t i

    l = hw_args._l
    y = hw_args._y
    err = np.empty(hw_args._n)
    alpha, beta, phi, alphac, betac = holt_init(x, hw_args)

    err[0] = y[0] - l[0]
    for i in range(1, hw_args._n):
        l[i] = (alpha * y[i - 1]) + (alphac * l[i - 1])
        err[i] = y[i] - l[i]
    return np.asarray(err)

# noinspection PyProtectedMember
def holt_mul_dam(double[::1] x, HoltWintersArgs hw_args):
    """
    Multiplicative and Multiplicative Damped
    Minimization Function
    (M,) & (Md,)
    """
    cdef double alpha, beta, phi, betac, alphac
    cdef double[::1] err, l, b, y
    cdef Py_ssize_t i

    err = np.empty(hw_args._n)
    alpha, beta, phi, alphac, betac = holt_init(x, hw_args)
    y = hw_args._y
    l = hw_args._l
    b = hw_args._b
    err[0] = y[0] - (l[0] * b[0]**phi)
    for i in range(1, hw_args._n):
        l[i] = (alpha * y[i - 1]) + (alphac * (l[i - 1] * b[i - 1]**phi))
        b[i] = (beta * (l[i] / l[i - 1])) + (betac * b[i - 1]**phi)
        err[i] = y[i] - (l[i] * b[i]**phi)
    return np.asarray(err)

# noinspection PyProtectedMember
def holt_add_dam(double[::1] x, HoltWintersArgs hw_args):
    """
    Additive and Additive Damped
    Minimization Function
    (A,) & (Ad,)
    """
    cdef double alpha, beta, phi, betac, alphac
    cdef double[::1] err, l, b, y
    cdef Py_ssize_t i

    err = np.empty(hw_args._n)
    alpha, beta, phi, alphac, betac = holt_init(x, hw_args)

    y = hw_args._y
    l = hw_args._l
    b = hw_args._b
    err[0] = y[0] - (l[0] + phi * b[0])
    for i in range(1, hw_args._n):
        l[i] = (alpha * y[i - 1]) + (alphac * (l[i - 1] + phi * b[i - 1]))
        b[i] = (beta * (l[i] - l[i - 1])) + (betac * phi * b[i - 1])
        err[i] = y[i] - (l[i] + phi * b[i])
    return np.asarray(err)

# noinspection PyProtectedMember
cdef object holt_win_init(double[::1] x, HoltWintersArgs hw_args):
    """Initialization for the Holt Winters Seasonal Models"""
    cdef double alpha, beta, gamma, phi, alphac, betac, l0, b0
    cdef Py_ssize_t i, n, idx = 0
    cdef double[::1] p, s
    cdef long[::1] xi
    
    p = hw_args._p
    xi = hw_args._xi
    n = p.shape[0]
    for i in range(n):
        if xi[i]:
            p[i] = x[idx]
            idx += 1
    if hw_args._transform:
        alpha, beta, gamma = to_restricted(p, xi, hw_args._bounds)
    else:
        alpha = p[0]
        beta = p[1]
        gamma = p[2]
    l0 = p[3]
    b0 = p[4]
    phi = p[5]

    alphac = 1 - alpha
    betac = 1 - beta
    gammac = 1 - gamma

    hw_args._l[0] = l0
    hw_args._b[0] = b0
    s = hw_args._s
    for i in range(hw_args._m):
        s[i] = p[6+i]

    return alpha, beta, gamma, phi, alphac, betac, gammac

# noinspection PyProtectedMember
def holt_win_add_add_dam(double[::1] x, HoltWintersArgs hw_args):
    """
    Additive and Additive Damped with Additive Seasonal
    Minimization Function
    (A,A) & (Ad,A)
    """
    cdef double alpha, beta, gamma, phi, alphac, betac, gammac
    cdef double[::1] err, l, s, b, y
    cdef Py_ssize_t i, m

    err = np.empty(hw_args._n)
    alpha, beta, gamma, phi, alphac, betac, gammac = holt_win_init(x, hw_args)

    y = hw_args._y
    l = hw_args._l
    b = hw_args._b
    s = hw_args._s
    m = hw_args._m
    err[0] = y[0] - (l[0] + phi * b[0] + s[0])
    for i in range(1, hw_args._n):
        l[i] = (alpha * y[i - 1]) - (alpha * s[i - 1]) + \
            (alphac * (l[i - 1] + phi * b[i - 1]))
        b[i] = (beta * (l[i] - l[i - 1])) + (betac * phi * b[i - 1])
        s[i + m - 1] = gamma * y[i - 1] - \
                       (gamma * (l[i - 1] + phi * b[i - 1])) + (gammac * s[i - 1])
        err[i] = y[i] - (l[i] + phi * b[i] + s[i])
    return np.asarray(err)

# noinspection PyProtectedMember
def holt_win__add(double[::1] x, HoltWintersArgs hw_args):
    """
    Additive Seasonal
    Minimization Function
    (,A)
    """
    cdef double alpha, beta, gamma, phi, alphac, betac, gammac
    cdef double[::1] err, y, l, s
    cdef Py_ssize_t i, m

    err = np.empty(hw_args._n)
    alpha, beta, gamma, phi, alphac, betac, gammac= holt_win_init(x, hw_args)

    y = hw_args._y
    l = hw_args._l
    s = hw_args._s
    m = hw_args._m
    err[0] = y[0] - (l[0] + s[0])
    for i in range(1, hw_args._n):
        l[i] = (alpha * y[i - 1]) - (alpha * s[i - 1]) + (alphac * (l[i - 1]))
        s[i + m - 1] = gamma * y[i - 1] - \
            (gamma * (l[i - 1])) + (gammac * s[i - 1])
        err[i] = y[i] - (l[i] + s[i])
    return np.asarray(err)

# noinspection PyProtectedMember
def holt_win__mul(double[::1] x, HoltWintersArgs hw_args):
    """
    Multiplicative Seasonal
    Minimization Function
    (,M)
    """
    cdef double alpha, beta, gamma, phi, alphac, betac, gammac
    cdef double[::1] err, y, l, s
    cdef Py_ssize_t i, m

    err = np.empty(hw_args._n)
    alpha, beta, gamma, phi, alphac, betac, gammac= holt_win_init(x, hw_args)

    y = hw_args._y
    l = hw_args._l
    s = hw_args._s
    m = hw_args._m
    err[0] = y[0] - (l[0] * s[0])
    for i in range(1, hw_args._n):
        l[i] = (alpha * y[i - 1] / s[i - 1]) + (alphac * (l[i - 1]))
        s[i + m - 1] = (gamma * y[i - 1] / (l[i - 1])) + (gammac * s[i - 1])
        err[i] = y[i] - (l[i] * s[i])
    return np.asarray(err)

# noinspection PyProtectedMember
def holt_win_mul_mul_dam(double[::1] x, HoltWintersArgs hw_args):
    """
    Multiplicative and Multiplicative Damped with Multiplicative Seasonal
    Minimization Function
    (M,M) & (Md,M)
    """
    cdef double alpha, beta, gamma, phi, alphac, betac, gammac
    cdef double[::1] err, y, l, b, s
    cdef Py_ssize_t i, m

    err = np.empty(hw_args._n)
    alpha, beta, gamma, phi, alphac, betac, gammac= holt_win_init(x, hw_args)

    y = hw_args._y
    l = hw_args._l
    b = hw_args._b
    s = hw_args._s
    m = hw_args._m
    err[0] = y[0] - ((l[0] * b[0]**phi) * s[0])
    for i in range(1, hw_args._n):
        l[i] = (alpha * y[i - 1] / s[i - 1]) + \
            (alphac * (l[i - 1] * b[i - 1]**phi))
        b[i] = (beta * (l[i] / l[i - 1])) + (betac * b[i - 1]**phi)
        s[i + m - 1] = (gamma * y[i - 1] / (l[i - 1] *
                                          b[i - 1]**phi)) + (gammac * s[i - 1])
        err[i] = y[i] - ((l[i] * b[i]**phi) * s[i])
    return np.asarray(err)



# noinspection PyProtectedMember
def holt_win_add_mul_dam(double[::1] x, HoltWintersArgs hw_args):
    """
    Additive and Additive Damped with Multiplicative Seasonal
    Minimization Function
    (A,M) & (Ad,M)
    """
    cdef double alpha, beta, gamma, phi, alphac, betac, gammac
    cdef double[::1] err, l, s, b, y
    cdef Py_ssize_t i, m

    err = np.empty(hw_args._n)
    alpha, beta, gamma, phi, alphac, betac, gammac= holt_win_init(x, hw_args)

    y = hw_args._y
    l = hw_args._l
    b = hw_args._b
    s = hw_args._s
    m = hw_args._m
    err[0] = y[0] - ((l[0] + phi * b[0]) * s[0])
    for i in range(1, hw_args._n):
        l[i] = (alpha * y[i - 1] / s[i - 1]) + \
            (alphac * (l[i - 1] + phi * b[i - 1]))
        b[i] = (beta * (l[i] - l[i - 1])) + (betac * phi * b[i - 1])
        s[i + m - 1] = (gamma * y[i - 1] / (l[i - 1] + phi *
                                          b[i - 1])) + (gammac * s[i - 1])
        err[i] = y[i] - ((l[i] + phi * b[i]) * s[i])
    return np.asarray(err)



# noinspection PyProtectedMember
def holt_win_mul_add_dam(double[::1] x, HoltWintersArgs hw_args):
    """
    Multiplicative and Multiplicative Damped with Additive Seasonal
    Minimization Function
    (M,A) & (M,Ad)
    """
    cdef double alpha, beta, gamma, phi, alphac, betac, gammac
    cdef double[::1] err, l, s, b, y
    cdef Py_ssize_t i, m

    err = np.empty(hw_args._n)
    alpha, beta, gamma, phi, alphac, betac, gammac= holt_win_init(x, hw_args)

    y = hw_args._y
    l = hw_args._l
    b = hw_args._b
    s = hw_args._s
    m = hw_args._m
    err[0] = y[0] - ((l[0] * phi * b[0]) + s[0])
    for i in range(1, hw_args._n):
        l[i] = (alpha * y[i - 1]) - (alpha * s[i - 1]) + \
            (alphac * (l[i - 1] * b[i - 1]**phi))
        b[i] = (beta * (l[i] / l[i - 1])) + (betac * b[i - 1]**phi)
        s[i + m - 1] = gamma * y[i - 1] - \
            (gamma * (l[i - 1] * b[i - 1]**phi)) + (gammac * s[i - 1])
        err[i] = y[i] - ((l[i] * phi * b[i]) + s[i])
    return np.asarray(err)
