from __future__ import print_function
import numpy as np
from statsmodels.regression.linear_model import OLS

trends = ('nc', 'c', 'ct', 'ctt')
critical_values = (1.0, 5.0, 10.0)
adf_z_cv_approx = {}
for t in trends:
    print(t)
    data = np.load('adf_z_' + t + '.npz')
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
        res.params[np.abs(res.tvalues) < 1.96] = 0.0
        out.append(res.params)

    adf_z_cv_approx[t] = np.array(out)

print('from numpy import array')
print('')
print('adf_z_cv_approx = ' + str(adf_z_cv_approx))
