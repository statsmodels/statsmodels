# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 11:17:06 2015

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from scipy import stats
import pandas

from statsmodels.miscmodels.ordinal_model import OrderedModel

nobs, k_vars = 1000, 3
x = np.random.randn(nobs, k_vars)
# x = np.column_stack((np.ones(nobs), x))
# #constant will be in integration limits
xb = x.dot(np.ones(k_vars))
y_latent = xb + np.random.randn(nobs)
y = np.round(np.clip(y_latent, -2.4, 2.4)).astype(int) + 2

print(np.unique(y))
print(np.bincount(y))

mod = OrderedModel(y, x)
# start_params = np.ones(k_vars + 4)
# start_params = np.concatenate((np.ones(k_vars), np.arange(4)))
start_ppf = stats.norm.ppf((np.bincount(y) / len(y)).cumsum())
start_threshold = np.concatenate((start_ppf[:1],
                                  np.log(np.diff(start_ppf[:-1]))))
start_params = np.concatenate((np.zeros(k_vars), start_threshold))
res = mod.fit(start_params=start_params, maxiter=5000, maxfun=5000)
print(res.params)
# res = mod.fit(start_params=res.params, method='bfgs')
res = mod.fit(start_params=start_params, method='bfgs')

print(res.params)
print(np.exp(res.params[-(mod.k_levels - 1):]).cumsum())
# print(res.summary())

predicted = res.model.predict(res.params)
pred_choice = predicted.argmax(1)
print('Fraction of correct choice predictions')
print((y == pred_choice).mean())

print('\ncomparing bincount')
print(np.bincount(res.model.predict(res.params).argmax(1)))
print(np.bincount(res.model.endog))

res_log = OrderedModel(y, x, distr='logit').fit(method='bfgs')
pred_choice_log = res_log.predict().argmax(1)
print((y == pred_choice_log).mean())
print(res_log.summary())

# example form UCLA Stats pages
# http://www.ats.ucla.edu/stat/stata/dae/ologit.htm
# requires downloaded dataset ologit.dta

dataf = pandas.read_stata(r"M:\josef_new\scripts\ologit_ucla.dta")

# this works but sorts category levels alphabetically
res_log2 = OrderedModel(np.asarray(dataf['apply']),
                        np.asarray(dataf[['pared', 'public', 'gpa']], float),
                        distr='logit').fit(method='bfgs')

# this replicates the UCLA example except
# for different parameterization of par2
res_log3 = OrderedModel(dataf['apply'].values.codes,
                        np.asarray(dataf[['pared', 'public', 'gpa']], float),
                        distr='logit').fit(method='bfgs')

print(res_log3.summary())

# with ordered probit - not on UCLA page
print(
    OrderedModel(dataf['apply'].values.codes,
                 np.asarray(dataf[['pared', 'public', 'gpa']], float),
                 distr='probit').fit(method='bfgs').summary())


# example with a custom distribution - not on UCLA page
# definition of the SciPy dist
class CLogLog(stats.rv_continuous):
    def _ppf(self, q):
        return np.log(-np.log(1 - q))

    def _cdf(self, x):
        return 1 - np.exp(-np.exp(x))


cloglog = CLogLog()

res_cloglog = OrderedModel(dataf['apply'],
                           dataf[['pared', 'public', 'gpa']],
                           distr=cloglog).fit(method='bfgs', disp=False)
print(res_cloglog.summary())
