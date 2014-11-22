"""
Critical value simulation for the Dickey-Fuller GLS model.  Similar in design to
MacKinnon (2010).  Makes use of parallel_fun in statsmodels which works best when
joblib is installed.
"""
from __future__ import division, print_function
from statsmodels.compat import range
import datetime

from numpy import ones, vstack, arange, diff, cumsum, sqrt, sum
from numpy.linalg import pinv
import numpy as np

from statsmodels.tools.parallel import parallel_func

# Controls memory use, in MiB
MAX_MEMORY_SIZE = 100
NUM_JOBS = 4
EX_NUM = 500
EX_SIZE = 200000


def wrapper(n, trend, b, seed=0):
    """
    Wraps and blocks the main simulation so that the maximum amount of memory 
    can be controlled on multi processor systems when executing in parallel
    """
    rng = np.random.RandomState()
    rng.seed(seed)
    remaining = b
    res = np.zeros(b)
    finished = 0
    block_size = int(2 ** 20.0 * MAX_MEMORY_SIZE / (8.0 * n))
    for j in range(0, b, block_size):
        if block_size < remaining:
            count = block_size
        else:
            count = remaining
        st = finished
        en = finished + count
        res[st:en] = dfgsl_simulation(n, trend, count, rng)
        finished += count
        remaining -= count

    return res


def dfgsl_simulation(n, trend, b, rng=None):
    """
    Simulates the empirical distribution of the DFGLS test statistic
    """
    if rng is None:
        np.random.seed(0)
        from numpy.random import standard_normal
    else:
        standard_normal = rng.standard_normal

    nobs = n
    if trend == 'c':
        c = -7.0
        z = ones((nobs, 1))
    else:
        c = -13.5
        z = vstack((ones(nobs), arange(1, nobs + 1))).T

    ct = c / nobs

    delta_z = np.copy(z)
    delta_z[1:, :] = delta_z[1:, :] - (1 + ct) * delta_z[:-1, :]
    delta_z_inv = pinv(delta_z)
    y = standard_normal((n + 50, b))
    y = cumsum(y, axis=0)
    y = y[50:, :]
    delta_y = y.copy()
    delta_y[1:, :] = delta_y[1:, :] - (1 + ct) * delta_y[:-1, :]
    detrend_coef = delta_z_inv.dot(delta_y)
    y_detrended = y - z.dot(detrend_coef)

    delta_y_detrended = diff(y_detrended, axis=0)
    rhs = y_detrended[:-1, :]
    lhs = delta_y_detrended

    xpy = sum(rhs * lhs, 0)
    xpx = sum(rhs ** 2.0, 0)
    gamma = xpy / xpx
    e = lhs - rhs * gamma
    sigma2 = sum(e ** 2.0, axis=0) / (n - 1)  # DOF correction?
    gamma_var = sigma2 / xpx

    stat = gamma / sqrt(gamma_var)
    return stat


if __name__ == '__main__':
    trends = ('c', 'ct')
    T = np.array(
        (20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160,
         180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900,
         1000, 1200, 1400, 2000))
    T = T[::-1]
    percentiles = list(np.arange(0.5, 100.0, 0.5))
    seeds = np.arange(0, 2 ** 32, step=2 ** 23)
    for tr in trends:
        results = np.zeros((len(percentiles), len(T), EX_NUM))

        for i in range(EX_NUM):
            print("Experiment Number {0} of {1} (trend {2})".format(i + 1,
                                                                    EX_NUM, tr))
            now = datetime.datetime.now()
            parallel, p_func, n_jobs = parallel_func(wrapper,
                                                     n_jobs=NUM_JOBS,
                                                     verbose=2)
            out = parallel(p_func(t, tr, EX_SIZE, seed=seeds[i]) for t in T)
            q = lambda x: np.percentile(x, percentiles)
            quantiles = map(q, out)
            results[:, :, i] = np.array(quantiles).T
            print('Elapsed time {0} seconds'.format(
                datetime.datetime.now() - now))

            if i % 50 == 0:
                np.savez('dfgls_' + tr + '.npz',
                         trend=tr,
                         results=results,
                         percentiles=percentiles,
                         T=T)

        np.savez('dfgls_' + tr + '.npz',
                 trend=tr,
                 results=results,
                 percentiles=percentiles,
                 T=T)