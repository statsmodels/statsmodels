'''Example to illustrate Quantile Regression

Author: Josef Perktold

polynomial regression with systematic deviations above

'''

import numpy as np
from statsmodels.compat.python import zip
from scipy import stats
import statsmodels.api as sm

from statsmodels.regression.quantile_regression import QuantReg

sige = 0.1
nobs, k_vars = 500, 3
x = np.random.uniform(-1, 1, size=nobs)
x.sort()
exog = np.vander(x, k_vars+1)[:,::-1]
mix = 0.1 * stats.norm.pdf(x[:,None], loc=np.linspace(-0.5, 0.75, 4), scale=0.01).sum(1)
y = exog.sum(1) + mix + sige * (np.random.randn(nobs)/2 + 1)**3

p = 0.5
res_qr = QuantReg(y, exog).fit(p)
res_qr2 = QuantReg(y, exog).fit(0.1)
res_qr3 = QuantReg(y, exog).fit(0.75)
res_ols = sm.OLS(y, exog).fit()

params = [res_ols.params, res_qr2.params, res_qr.params, res_qr3.params]
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
