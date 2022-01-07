# ## Math Functions
# Real and complex log and abs functions
from libc.math cimport log as dlog, abs as dabs, exp as dexp
cimport numpy as np
from libc.string cimport memcpy

cdef extern from "numpy/npy_math.h":
    np.float64_t NPY_PI
    np.float64_t npy_cabs(np.npy_cdouble z) nogil
    np.npy_cdouble npy_clog(np.npy_cdouble z) nogil
    np.npy_cdouble npy_cexp(np.npy_cdouble z) nogil

cdef inline np.float64_t zabs(np.complex128_t z) nogil:
    cdef np.npy_cdouble x
    memcpy(&x, &z, sizeof(z))
    return npy_cabs(x)

cdef inline np.complex128_t zlog(np.complex128_t z) nogil:
    cdef np.npy_cdouble x
    cdef np.complex128_t out
    memcpy(&x, &z, sizeof(z))
    x = npy_clog(x)
    memcpy(&out, &x, sizeof(x))
    return out

cdef inline np.complex128_t zexp(np.complex128_t z) nogil:
    cdef np.npy_cdouble x
    cdef np.complex128_t out
    memcpy(&x, &z, sizeof(z))
    x = npy_cexp(x)
    memcpy(&out, &x, sizeof(x))
    return out
