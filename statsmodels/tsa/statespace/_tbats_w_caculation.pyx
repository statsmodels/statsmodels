#!python
#cython: wraparound=False, boundscheck=False, cdivision=True

from cpython cimport bool
import numpy as np
cimport numpy as np

cpdef _tbats_w_caculation(
    numeric [:,:] D,
    numeric [:] Z,
    numeric [:,:] out
):
    cdef numeric nobs
    nobs = out.shape[0]

    out[0] = Z
    for i in range(nobs - 1):
        w[i + 1] = w[i].dot(D)
    return out