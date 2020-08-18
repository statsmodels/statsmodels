#!python
#cython: wraparound=False, boundscheck=False, cdivision=True

cimport cython
import numpy as np
cimport numpy as np

ctypedef np.float64_t DOUBLE
ctypedef np.int_t INT


def _high_weighted_median(np.ndarray[DOUBLE] a, np.ndarray[INT] weights):
    """
    Computes a weighted high median of a. This is defined as the
    smallest a[j] such that the sum over all a[i]<=a[j] is strictly
    greater than half the total sum of the weights
    """
    cdef:
        np.ndarray[INT] arg_sort
        np.ndarray[DOUBLE] sorted_a
        np.ndarray[INT] sorted_weights
        np.ndarray[INT] cs_weights
        int idx
        float midpoint
        float w_median
    arg_sort = np.argsort(a)
    sorted_a = a[arg_sort]
    sorted_weights = weights[arg_sort]
    midpoint = 0.5 * sum(weights)
    cs_weights = np.cumsum(sorted_weights)
    idx = np.where(cs_weights > midpoint)[0][0]
    w_median = sorted_a[idx]
    return w_median
