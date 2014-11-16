from __future__ import print_function
from statsmodels.compat import range
import numpy as np
from scipy.stats import norm

from statsmodels.regression.linear_model import OLS, WLS


trends = ('c', 'ct')
critical_values = (1.0, 5.0, 10.0)
dfgls_cv_approx = {}
for t in trends:
    print(t)
    data = np.load('dfgls_' + t + '.npz')
    percentiles = data['percentiles']
    trend = data['trend']
    results = data['results']
    # T = data['T']
    data.close()

    # Remove later
    T = np.array(
        (20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160,
         180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900,
         1000, 1200, 1400, 2000))
    T = T[::-1]

    # For percentiles 1, 5 and 10, regress on a constant, and powers of 1/T
    out = []
    for cv in critical_values:
        num_ex = results.shape[2]
        loc = np.where(percentiles == cv)[0][0]
        lhs = np.squeeze(results[loc, :, :])
        # Adjust for effective sample size, this is what lookup the code uses
        tau = np.ones((num_ex, 1)).dot(T[None, :]) - 1.0
        tau = tau.T
        lhs = lhs.ravel()
        tau = tau.ravel()
        tau = tau[:, None]
        n = lhs.shape[0]
        rhs = np.ones((n, 1))
        rhs = np.hstack((rhs, 1.0 / tau))
        rhs = np.hstack((rhs, (1.0 / tau) ** 2.0))
        rhs = np.hstack((rhs, (1.0 / tau) ** 3.0))
        res = OLS(lhs, rhs).fit()
        out.append(res.params)

    dfgls_cv_approx[t] = np.array(out)

trends = ('c', 'ct')
dfgls_large_p = {}
dfgls_small_p = {}
dfgls_tau_star = {}
dfgls_tau_max = {}
dfgls_tau_min = {}
for t in trends:
    data = np.load('dfgls_' + t + '.npz')
    percentiles = data['percentiles']
    results = data['results']  # Remove later
    # LHS is norm cdf inv of percentiles
    lhs = norm().ppf(percentiles / 100.0)
    lhs_large = lhs
    # RHS is made up of avg test stats for largest T, which is in pos 1
    avg_test_stats = results[:, 1, :].mean(axis=1)
    avg_test_std = results[:, 1, :].std(axis=1)
    avg_test_stats = avg_test_stats[:, None]
    m = lhs.shape[0]
    rhs = np.ones((m, 1))
    rhs = np.hstack((rhs, avg_test_stats))
    rhs = np.hstack((rhs, avg_test_stats ** 2.0))
    rhs = np.hstack((rhs, avg_test_stats ** 3.0))
    rhs_large = rhs
    res_large = WLS(lhs, rhs, weights=1.0 / avg_test_std).fit()
    dfgls_large_p[t] = res_large.params
    # Compute tau_max, by finding the func maximum
    p = res_large.params
    poly_roots = np.roots(np.array([3, 2, 1.0]) * p[:0:-1])
    dfgls_tau_max[t] = float(np.squeeze(np.real(np.max(poly_roots))))

    # Small p regression using only p<=15%
    cutoff = np.where(percentiles <= 15.0)[0]
    avg_test_stats = results[cutoff, 1, :].mean(axis=1)
    avg_test_std = results[cutoff, 1, :].std(axis=1)
    avg_test_stats = avg_test_stats[:, None]
    lhs = lhs[cutoff]
    m = lhs.shape[0]
    rhs = np.ones((m, 1))
    rhs = np.hstack((rhs, avg_test_stats))
    rhs = np.hstack((rhs, avg_test_stats ** 2.0))
    res_small = WLS(lhs, rhs, weights=1.0 / avg_test_std).fit()
    dfgls_small_p[t] = res_small.params

    # Compute tau star
    err_large = res_large.resid
    # Missing 1 parameter here, replace with 0
    params = np.append(res_small.params, 0.0)
    err_small = lhs_large - rhs_large.dot(params)
    # Find the location that minimizes the total absolute error
    m = lhs_large.shape[0]
    abs_err = np.zeros((m, 1))
    for i in range(m):
        abs_err[i] = np.abs(err_large[i:]).sum() + np.abs(err_small[:i]).sum()
    loc = np.argmin(abs_err)
    dfgls_tau_star[t] = rhs_large[loc, 1]
    # Compute tau min
    dfgls_tau_min[t] = -params[1] / (2 * params[2])

print('from numpy import array')
print('')
print('dfgls_cv_approx = ' + str(dfgls_cv_approx))
print('')
print('dfgls_tau_max = ' + str(dfgls_tau_max))
print('')
print('dfgls_tau_min = ' + str(dfgls_tau_min))
print('')
print('dfgls_tau_star = ' + str(dfgls_tau_star))
print('')
print('dfgls_large_p = ' + str(dfgls_large_p))
print('')
print('dfgls_small_p = ' + str(dfgls_small_p))
