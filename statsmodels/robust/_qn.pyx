#!python
#cython: wraparound=False, boundscheck=False, cdivision=True
"""
Qn robust estimator of scale, following Croux and Rousseeuw (1992).

``_high_weighted_median`` selects in place on preallocated buffers, as the
authors' Fortran and R's robustbase do, rather than partitioning a fresh copy
at every iteration. The counters that index the n(n + 1) / 2 candidate
differences are 64-bit, so n is not limited to 46 340.
"""

from libc.stdint cimport int64_t

import numpy as np


cdef void _psort(double* a, Py_ssize_t n, Py_ssize_t k) noexcept nogil:
    """Partial sort in place: a[k] becomes the k-th smallest (0-indexed),
    with a[:k] <= a[k] <= a[k+1:]. Quickselect, median-of-three pivot."""
    cdef Py_ssize_t lo = 0, hi = n - 1, i, j, mid
    cdef double v, t
    while hi > lo:
        mid = lo + (hi - lo) // 2
        if a[mid] < a[lo]:
            t = a[mid]; a[mid] = a[lo]; a[lo] = t
        if a[hi] < a[lo]:
            t = a[hi]; a[hi] = a[lo]; a[lo] = t
        if a[hi] < a[mid]:
            t = a[hi]; a[hi] = a[mid]; a[mid] = t
        v = a[mid]
        i = lo
        j = hi
        while i <= j:
            while a[i] < v:
                i += 1
            while a[j] > v:
                j -= 1
            if i <= j:
                t = a[i]; a[i] = a[j]; a[j] = t
                i += 1
                j -= 1
        if k <= j:
            hi = j
        elif k >= i:
            lo = i
        else:
            break


cdef double _whimed(double* a, int* w, Py_ssize_t n,
                    double* a_cand, double* a_srt, int* w_cand) noexcept nogil:
    """Weighted high median in O(n): the smallest a[j] such that the total
    weight of all a[i] <= a[j] is strictly greater than half the total
    weight. ``a`` and ``w`` are consumed (shrunk in place)."""
    cdef int64_t wleft, wmid, wright, w_tot = 0, wrest = 0
    cdef double trial
    cdef Py_ssize_t i, n2, kcand
    for i in range(n):
        w_tot += w[i]
    while True:
        n2 = n // 2
        for i in range(n):
            a_srt[i] = a[i]
        _psort(a_srt, n, n2)
        trial = a_srt[n2]
        wleft = 0; wmid = 0; wright = 0
        for i in range(n):
            if a[i] < trial:
                wleft += w[i]
            elif a[i] > trial:
                wright += w[i]
            else:
                wmid += w[i]
        kcand = 0
        if 2 * (wrest + wleft) > w_tot:
            for i in range(n):
                if a[i] < trial:
                    a_cand[kcand] = a[i]; w_cand[kcand] = w[i]; kcand += 1
        elif 2 * (wrest + wleft + wmid) <= w_tot:
            for i in range(n):
                if a[i] > trial:
                    a_cand[kcand] = a[i]; w_cand[kcand] = w[i]; kcand += 1
            wrest += wleft + wmid
        else:
            return trial
        n = kcand
        for i in range(n):
            a[i] = a_cand[i]; w[i] = w_cand[i]


def _high_weighted_median(double[::1] a, int[::1] weights):
    """
    Computes a weighted high median of a. This is defined as the
    smallest a[j] such that the sum over all a[i]<=a[j] is strictly
    greater than half the total sum of the weights
    """
    cdef Py_ssize_t n = a.shape[0]
    cdef double[::1] a_cp = np.array(a, dtype=np.float64, copy=True)
    cdef int[::1] w_cp = np.array(weights, dtype=np.intc, copy=True)
    cdef double[::1] a_cand = np.empty(n, dtype=np.float64)
    cdef double[::1] a_srt = np.empty(n, dtype=np.float64)
    cdef int[::1] w_cand = np.empty(n, dtype=np.intc)
    if n == 0:
        raise ValueError("weighted median of an empty array")
    return _whimed(&a_cp[0], &w_cp[0], n, &a_cand[0], &a_srt[0], &w_cand[0])


def _qn(double[:] a, double c):
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
    cdef Py_ssize_t n = a.shape[0]
    if n < 2:
        raise ValueError("Qn needs at least 2 observations, got %d" % n)
    cdef Py_ssize_t h = n // 2 + 1
    cdef int64_t k = (<int64_t>h) * (h - 1) // 2
    cdef int64_t n_left = (<int64_t>n) * (n + 1) // 2
    cdef int64_t n_right = (<int64_t>n) * n
    cdef int64_t k_new = k + n_left
    cdef int64_t sump, sumq
    cdef Py_ssize_t i, j, jh, l
    cdef double trial = 0.0
    cdef int found = 0, overrun = 0

    cdef double[::1] a_sorted = np.sort(a)
    cdef double[::1] work = np.empty(n, dtype=np.float64)
    cdef double[::1] a_cand = np.empty(n, dtype=np.float64)
    cdef double[::1] a_srt = np.empty(n, dtype=np.float64)
    cdef int[::1] left = np.empty(n, dtype=np.intc)
    cdef int[::1] right = np.empty(n, dtype=np.intc)
    cdef int[::1] weights = np.empty(n, dtype=np.intc)
    cdef int[::1] p = np.empty(n, dtype=np.intc)
    cdef int[::1] q = np.empty(n, dtype=np.intc)

    with nogil:
        for i in range(n):
            left[i] = <int>(n - i + 1)
            right[i] = <int>(n if i <= h else n - (i - h))

        while n_right - n_left > n:
            j = 0
            for i in range(1, n):
                if left[i] <= right[i]:
                    weights[j] = right[i] - left[i] + 1
                    jh = left[i] + weights[j] // 2
                    work[j] = a_sorted[i] - a_sorted[n - jh]
                    j += 1
            trial = _whimed(&work[0], &weights[0], j,
                            &a_cand[0], &a_srt[0], &p[0])
            j = 0
            for i in range(n - 1, -1, -1):
                while j < n and (a_sorted[i] - a_sorted[n - j - 1]) < trial:
                    j += 1
                p[i] = <int>j
            j = n + 1
            for i in range(n):
                while (a_sorted[i] - a_sorted[n - j + 1]) > trial:
                    j -= 1
                q[i] = <int>j
            sump = 0
            sumq = 0
            for i in range(n):
                sump += p[i]
                sumq += q[i]
            sumq -= n
            if k_new <= sump:
                for i in range(n):
                    right[i] = p[i]
                n_right = sump
            elif k_new > sumq:
                for i in range(n):
                    left[i] = q[i]
                n_left = sumq
            else:
                found = 1
                break

        if not found:
            j = 0
            for i in range(1, n):
                for l in range(left[i], right[i] + 1):
                    if j >= n:
                        overrun = 1
                        break
                    work[j] = a_sorted[i] - a_sorted[n - l]
                    j += 1
                if overrun:
                    break
            if not overrun:
                k_new = k_new - (n_left + 1)
                if k_new > j - 1:
                    k_new = j - 1
                elif k_new < 0:
                    k_new = 0
                _psort(&work[0], j, <Py_ssize_t>k_new)
                trial = work[k_new]

    if overrun:
        raise RuntimeError("_qn: candidate set exceeded n")
    return c * trial
