#!python
#cython: wraparound=False, boundscheck=False, cdivision=True

from cpython cimport bool
import numpy as np
cimport numpy as np

ctypedef fused numeric:
    np.float32_t
    np.float64_t
    np.complex64_t
    np.complex128_t

cpdef _tbats_w_caculation(
    np.ndarray[numeric,  ndim=2] D,
    np.ndarray[numeric,  ndim=1] Z,
    np.ndarray[numeric,  ndim=2] out
):
    cdef Py_ssize_t i, nobs
    nobs = out.shape[0]

    out[0] = Z
    for i in range(nobs - 1):
         out[i+1] = out[i].dot(D)
    return out


cpdef _tbats_recursive_compute(
    np.ndarray[numeric, ndim=2] w,
    np.ndarray[numeric, ndim=2] F,
    np.ndarray[numeric, ndim=2] g,
    np.ndarray[numeric, ndim=2] y,
    np.ndarray[numeric, ndim=2] e,
    np.ndarray[numeric, ndim=2] y_hat,
    np.ndarray[numeric, ndim=2] x0
):
    cdef Py_ssize_t i, nobs, k_states
    cdef np.ndarray[numeric, ndim=2] x
    nobs = y.shape[0]
    k_states = w.shape[1]

    x = x0
    for i in range(nobs):
        y_hat[:, i] = w @ x
        e[:, i] = y[:, i] - y_hat[:, i]
        x = F @ x + g * e[:, i]