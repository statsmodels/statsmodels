#!python
#cython: boundscheck=False, wraparound=False, cdivision=True
"""
cython -a fast_linbin.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include/ -o fast_linbin.so fast_linbin.c
"""

cimport cython
cimport numpy as np
import numpy as np

ctypedef np.float64_t DOUBLE
ctypedef np.int64_t INT

def fast_linbin(np.ndarray[DOUBLE] X, double a, double b, int M, int trunc=1):
    """
    Linear Binning as described in Fan and Marron (1994)

    Each observation in [a, b] is split linearly between the two grid points
    of the equally spaced grid ``linspace(a, b, M)`` that enclose it, so that
    the bin counts sum to the number of observations in [a, b]. Observations
    outside [a, b] are dropped if ``trunc`` is nonzero and are otherwise
    assigned to the nearest end point of the grid.
    """
    cdef:
        Py_ssize_t i, li_i
        int nobs = np.PyArray_DIMS(X)[0]
        double delta = (b - a)/(M - 1)
        np.ndarray[DOUBLE] gcnts = np.zeros(M, float)
        np.ndarray[DOUBLE] lxi = (X - a)/delta
        np.ndarray[INT] li = lxi.astype(np.int64)
        np.ndarray[DOUBLE] rem = lxi - li


    for i in range(nobs):
        if not (a <= X[i] <= b):
            # outside [a, b] (or nan): drop, or move to the nearest end point
            if trunc == 0:
                if X[i] < a:
                    gcnts[0] = gcnts[0] + 1
                elif X[i] > b:
                    gcnts[M-1] = gcnts[M-1] + 1
            continue
        li_i = li[i]
        if li_i >= M - 1:
            # X[i] == b, up to rounding in lxi; all of the weight belongs to
            # the last grid point and gcnts[M] is out of bounds
            gcnts[M-1] = gcnts[M-1] + 1
        else:
            gcnts[li_i] = gcnts[li_i] + 1 - rem[i]
            gcnts[li_i+1] = gcnts[li_i+1] + rem[i]
    return gcnts
