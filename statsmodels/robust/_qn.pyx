#!python
#cython: wraparound=False, boundscheck=False, cdivision=True

cimport cython
import numpy as np
cimport numpy as np

DTYPE_INT = np.int
DTYPE_DOUBLE = np.float64

ctypedef np.float64_t DTYPE_DOUBLE_t
ctypedef np.int_t DTYPE_INT_t


def _high_weighted_median(np.ndarray[DTYPE_DOUBLE_t] a, np.ndarray[DTYPE_INT_t] weights):
    """
    Computes a weighted high median of a. This is defined as the
    smallest a[j] such that the sum over all a[i]<=a[j] is strictly
    greater than half the total sum of the weights
    """
    cdef:
        DTYPE_INT_t n = a.shape[0]
        np.ndarray[DTYPE_DOUBLE_t] sorted_a = np.zeros((n,), dtype=DTYPE_DOUBLE)
        np.ndarray[DTYPE_DOUBLE_t] a_cand = np.zeros((n,), dtype=DTYPE_DOUBLE)
        np.ndarray[DTYPE_INT_t] weights_cand = np.zeros((n,), dtype=DTYPE_INT)
        Py_ssize_t i= 0
        DTYPE_INT_t kcand = 0
        DTYPE_INT_t wleft, wright, wmid, wtot, wrest = 0
        DTYPE_DOUBLE_t trial = 0
    wtot = np.sum(weights)
    while True:
        wleft = 0
        wmid = 0
        wright = 0
        for i in range(n):
            sorted_a[i] = a[i]
        sorted_a.partition(n//2)
        trial = sorted_a[n//2]
        for i in range(n):
            if a[i] < trial:
                wleft = wleft + weights[i]
            elif a[i] > trial:
                wright = wright + weights[i]
            else:
                wmid = wmid + weights[i]
        kcand = 0
        if 2 * (wrest + wleft) > wtot:
            for i in range(n):
                if a[i] < trial:
                    a_cand[kcand] = a[i]
                    weights_cand[kcand] = weights[i]
                    kcand = kcand + 1
        elif 2 * (wrest + wleft + wmid) <= wtot:
            for i in range(n):
                if a[i] > trial:
                    a_cand[kcand] = a[i]
                    weights_cand[kcand] = weights[i]
                    kcand = kcand + 1
            wrest = wrest + wleft + wmid
        else:
            return trial
        n = kcand
        for i in range(n):
            a[i] = a_cand[i]
            weights[i] = weights_cand[i]


def _qn(np.ndarray[DTYPE_DOUBLE_t] a, DTYPE_DOUBLE_t c):
    """
    Computes the Qn robust estimator of scale, a more efficient alternative
    to the MAD. The implementation follows the algorithm described in Croux
    and Rousseeuw (1992).

    Parameters
    ----------
    a : array_like
        Input array.
    c : float, optional
        The normalization constant, used to get consistent estimates of the
        standard deviation at the normal distribution.  Defined as
        1/(np.sqrt(2) * scipy.stats.norm.ppf(5/8)), which is 2.219144.

    Returns
    -------
    The Qn robust estimator of scale
    """
    cdef:
        DTYPE_INT_t n = a.shape[0]
        DTYPE_INT_t h = n/2 + 1
        DTYPE_INT_t k = h * (h - 1) / 2
        DTYPE_INT_t n_left = n * (n + 1) / 2
        DTYPE_INT_t n_right = n * n
        DTYPE_INT_t k_new = k + n_left
        Py_ssize_t i, j, jh, l = 0
        DTYPE_INT_t sump, sumq = 0
        DTYPE_DOUBLE_t trial, output = 0
        np.ndarray[DTYPE_DOUBLE_t] a_sorted = np.sort(a)
        np.ndarray[DTYPE_INT_t] left = np.array([n - i + 1 for i in range(0, n)], dtype=DTYPE_INT)
        np.ndarray[DTYPE_INT_t] right = np.array([n if i <= h else n - (i - h) for i in range(0, n)], dtype=DTYPE_INT)
        np.ndarray[DTYPE_INT_t] weights = np.zeros((n,), dtype=DTYPE_INT)
        np.ndarray[DTYPE_DOUBLE_t] work = np.zeros((n,), dtype=DTYPE_DOUBLE)
        np.ndarray[DTYPE_INT_t] p = np.zeros((n,), dtype=DTYPE_INT)
        np.ndarray[DTYPE_INT_t] q = np.zeros((n,), dtype=DTYPE_INT)
    while n_right - n_left > n:
        j = 0
        for i in range(1, n):
            if left[i] <= right[i]:
                weights[j] = right[i] - left[i] + 1
                jh = left[i] + weights[j] // 2
                work[j] = a_sorted[i] - a_sorted[n - jh]
                j = j + 1
        trial = _high_weighted_median(work[:j], weights[:j])
        j = 0
        for i in range(n - 1, -1, -1):
            while j < n and (a_sorted[i] - a_sorted[n - j - 1]) < trial:
                j = j + 1
            p[i] = j
        j = n + 1
        for i in range(n):
            while (a_sorted[i] - a_sorted[n - j + 1]) > trial:
                j = j - 1
            q[i] = j
        sump = np.sum(p)
        sumq = np.sum(q - 1)
        if k_new <= sump:
            right = np.copy(p)
            n_right = sump
        elif k_new > sumq:
            left = np.copy(q)
            n_left = sumq
        else:
            output = c * trial
            return output
    j = 0
    for i in range(1, n):
        for l in range(left[i], right[i] + 1):
            work[j] = a_sorted[i] - a_sorted[n - l]
            j = j + 1
    k_new = k_new - (n_left + 1)
    output = c * np.sort(work[:j])[k_new]
    return output
