'''Example to illustrate Quantile Regression

Author: Josef Perktold

polynomial regression with systematic deviations above

'''

import numpy as np
from scipy import stats
import statsmodels.api as sm

from statsmodels.sandbox.regression.quantile_regression import quantilereg

sige = 0.1
nobs, k_vars = 1500, 3
x = np.random.uniform(-1, 1, size=nobs)
x.sort()
exog = np.vander(x, k_vars+1)[:,::-1]
mix = 0.1 * stats.norm.pdf(x[:,None], loc=np.linspace(-0.5, 0.75, 4), scale=0.01).sum(1)
y = exog.sum(1) + mix + sige * (np.random.randn(nobs)/2 + 1)**3

p = 0.5
x0 = exog[:, 1:]    #quantilereg includes constant already!
res_qr = quantilereg(y, x0, p)

res_qr2 = quantilereg(y, x0, 0.1)
res_qr3 = quantilereg(y, x0, 0.75)
res_ols = sm.OLS(y, exog).fit()

params = [res_ols.params, res_qr2, res_qr, res_qr3]
labels = ['ols', 'qr 0.1', 'qr 0.5', 'qr 0.75']

import matplotlib.pyplot as plt

plt.figure()
plt.plot(x, y, '.', alpha=0.5)
for lab, beta in zip(['ols', 'qr 0.1', 'qr 0.5', 'qr 0.75'], params):
    print('%-8s'%lab, np.round(beta, 4))
    fitted = np.dot(exog, beta)
    lw = 2
    plt.plot(x, fitted, lw=lw, label=lab)
plt.legend()
plt.title('Quantile Regression')

plt.show()
