# -*- coding: utf-8 -*-
"""

Created on Fri Mar 09 16:00:27 2012

Author: Josef Perktold
"""

import numpy as np
import statsmodels.api as sm

nobs = 10000
np.random.seed(987689)
x = np.random.randn(nobs, 3)
x = sm.add_constant(x, prepend=True)
y = x.sum(1) + np.random.randn(nobs)



xf = 0.25 * np.ones((2,4))

model = sm.OLS(y, x)
#y_count = np.random.poisson(np.exp(x.sum(1)-x.mean()))
#model = sm.Poisson(y_count, x)#, exposure=np.ones(nobs), offset=np.zeros(nobs)) #bug with default
results = model.fit()


#print results.predict(xf)
print results.model.predict(results.params, xf)

shrinkit = 1
if shrinkit:
    results.remove_data()

import pickle
fname = 'try_shrink%d_ols.pickle' % shrinkit
fh = open(fname, 'w')
pickle.dump(results._results, fh)  #pickling wrapper doesn't work
fh.close()
fh = open(fname, 'r')
results2 = pickle.load(fh)
fh.close()
print results2.predict(xf)
print results2.model.predict(results.params, xf)


y_count = np.random.poisson(np.exp(x.sum(1)-x.mean()))
model = sm.Poisson(y_count, x)#, exposure=np.ones(nobs), offset=np.zeros(nobs)) #bug with default
results = model.fit(method='bfgs')

print results.model.predict(results.params, xf, exposure=1, offset=0)

if shrinkit:
    results.remove_data()
else:
    #work around pickling bug
    results.mle_settings['callback'] = None

import pickle
fname = 'try_shrink%d_poisson.pickle' % shrinkit
fh = open(fname, 'w')
pickle.dump(results._results, fh)  #pickling wrapper doesn't work
fh.close()
fh = open(fname, 'r')
results2 = pickle.load(fh)
fh.close()
print results2.predict(xf, exposure=1, offset=0)
print results2.model.predict(results.params, xf, exposure=1, offset=0)
