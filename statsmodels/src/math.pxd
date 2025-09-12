# ## Math Functions
# Real and complex log and abs functions
from libc.math cimport log as dlog, abs as dabs, exp as dexp
cimport numpy as np
from libc.string cimport memcpy

cdef extern from "_complex_shim.h":
    ctypedef double double_complex
    double sm_cabs(double_complex z) nogil
    double_complex sm_clog(double_complex z) nogil
    double_complex sm_cexp(double_complex z) nogil

cdef extern from "numpy/npy_math.h":
    np.float64_t NPY_PI

cdef inline np.float64_t zabs(np.complex128_t z) noexcept nogil:
    cdef double_complex x
    memcpy(&x, &z, sizeof(z))
    return sm_cabs(x)

cdef inline np.complex128_t zlog(np.complex128_t z) noexcept nogil:
    cdef double_complex x
    cdef np.complex128_t out
    memcpy(&x, &z, sizeof(z))
    x = sm_clog(x)
    memcpy(&out, &x, sizeof(x))
    return out

cdef inline np.complex128_t zexp(np.complex128_t z) noexcept nogil:
    cdef double_complex x
    cdef np.complex128_t out
    memcpy(&x, &z, sizeof(z))
    x = sm_cexp(x)
    memcpy(&out, &x, sizeof(x))
    return out
