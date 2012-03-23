#cython profile=True
"""
cython -a fast_linbin.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include/ -o fast_linbin.so fast_linbin.c
"""

cimport cython
cimport numpy as np
import numpy as np

ctypedef np.float64_t DOUBLE
ctypedef np.int_t INT

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def linbin(np.ndarray[DOUBLE, ndim=1] X, double a, double b, int M,
           int trunc=1):
    """
    Linear Binning as described in Fan and Marron (1994)
    """
    cdef int i, li_i
    cdef int nobs = X.shape[0]
    cdef double delta = (b - a)/(M - 1)

    cdef np.ndarray[DOUBLE, ndim=1] gcnts = np.zeros(M, np.float)
    cdef np.ndarray[DOUBLE, ndim=1] lxi = (X - a)/delta
    cdef np.ndarray[INT, ndim=1] li = lxi.astype(int)
    cdef np.ndarray[DOUBLE, ndim=1] rem = lxi - li


    for i in range(nobs):
        li_i = li[i]
        if li_i > 1 and li_i < M:
            gcnts[li_i] = gcnts[li_i] + 1 - rem[i]
            gcnts[li_i+1] = gcnts[li_i+1] + rem[i]
        if li_i > M and trunc == 0:
            gcnts[M] = gcnts[M] + 1
    return gcnts
