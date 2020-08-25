#!python
#cython: wraparound=False, boundscheck=False, cdivision=True

import numpy as np


def _high_weighted_median(double[::1] a, int[::1] weights):
    """
    Computes a weighted high median of a. This is defined as the
    smallest a[j] such that the sum over all a[i]<=a[j] is strictly
    greater than half the total sum of the weights
    """
    cdef:
        double[::1] a_cp = np.copy(a)
        int[::1] weights_cp = np.copy(weights)
        int n = a_cp.shape[0]
        double[::1] sorted_a = np.zeros((n,), dtype=np.double)
        double[::1] a_cand = np.zeros((n,), dtype=np.double)
        int[::1] weights_cand = np.zeros((n,), dtype=np.intc)
        int kcand = 0
        int wleft, wright, wmid, wtot, wrest = 0
        double trial = 0
    wtot = np.sum(weights_cp)
    while True:
        wleft = 0
        wmid = 0
        wright = 0
        for i in range(n):
            sorted_a[i] = a_cp[i]
        sorted_a = np.partition(sorted_a, kth=n//2)
        trial = sorted_a[n//2]
        for i in range(n):
            if a_cp[i] < trial:
                wleft = wleft + weights_cp[i]
            elif a_cp[i] > trial:
                wright = wright + weights_cp[i]
            else:
                wmid = wmid + weights_cp[i]
        kcand = 0
        if 2 * (wrest + wleft) > wtot:
            for i in range(n):
                if a_cp[i] < trial:
                    a_cand[kcand] = a_cp[i]
                    weights_cand[kcand] = weights_cp[i]
                    kcand = kcand + 1
        elif 2 * (wrest + wleft + wmid) <= wtot:
            for i in range(n):
                if a_cp[i] > trial:
                    a_cand[kcand] = a_cp[i]
                    weights_cand[kcand] = weights_cp[i]
                    kcand = kcand + 1
            wrest = wrest + wleft + wmid
        else:
            break
        n = kcand
        for i in range(n):
            a_cp[i] = a_cand[i]
            weights_cp[i] = weights_cand[i]
    return trial


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
    cdef:
        int n = a.shape[0]
        int h = n/2 + 1
        int k = h * (h - 1) / 2
        int n_left = n * (n + 1) / 2
        int n_right = n * n
        int k_new = k + n_left
        Py_ssize_t i, j, jh, l = 0
        int sump, sumq = 0
        double trial, output = 0
        double[::1] a_sorted = np.sort(a)
        int[::1] left = np.array([n - i + 1 for i in range(0, n)], dtype=np.intc)
        int[::1] right = np.array([n if i <= h else n - (i - h) for i in range(0, n)], dtype=np.intc)
        int[::1] weights = np.zeros((n,), dtype=np.intc)
        double[::1] work = np.zeros((n,), dtype=np.double)
        int[::1] p = np.zeros((n,), dtype=np.intc)
        int[::1] q = np.zeros((n,), dtype=np.intc)
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
        sumq = np.sum(q) - n
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
