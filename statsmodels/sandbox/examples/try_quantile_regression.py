'''Example to illustrate Quantile Regression

Author: Josef Perktold

'''

import numpy as np
from statsmodels.compat.python import zip
import statsmodels.api as sm

from statsmodels.regression.quantile_regression import QuantReg
sige = 5
nobs, k_vars = 500, 5
x = np.random.randn(nobs, k_vars)
#x[:,0] = 1
y = x.sum(1) + sige * (np.random.randn(nobs)/2 + 1)**3
p = 0.5
exog = np.column_stack((np.ones(nobs), x))
res_qr = QuantReg(y, exog).fit(p)

res_qr2 = QuantReg(y, exog).fit(0.25)
res_qr3 = QuantReg(y, exog).fit(0.75)
res_ols = sm.OLS(y, exog).fit()


##print 'ols ', res_ols.params
##print '0.25', res_qr2
##print '0.5 ', res_qr
##print '0.75', res_qr3

params = [res_ols.params, res_qr2.params, res_qr.params, res_qr3.params]
labels = ['ols', 'qr 0.25', 'qr 0.5', 'qr 0.75']

import matplotlib.pyplot as plt
#sortidx = np.argsort(y)
fitted_ols = np.dot(res_ols.model.exog, params[0])
sortidx = np.argsort(fitted_ols)
x_sorted = res_ols.model.exog[sortidx]
fitted_ols = np.dot(x_sorted, params[0])
plt.figure()
plt.plot(y[sortidx], 'o', alpha=0.75)
for lab, beta in zip(['ols', 'qr 0.25', 'qr 0.5', 'qr 0.75'], params):
    print('%-8s'%lab, np.round(beta, 4))
    fitted = np.dot(x_sorted, beta)
    lw = 2 if lab == 'ols' else 1
    plt.plot(fitted, lw=lw, label=lab)
plt.legend()

plt.show()
