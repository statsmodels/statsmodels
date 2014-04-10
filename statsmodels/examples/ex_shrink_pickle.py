# -*- coding: utf-8 -*-
"""

Created on Fri Mar 09 16:00:27 2012

Author: Josef Perktold
"""
from statsmodels.compat.python import StringIO
import numpy as np
import statsmodels.api as sm

nobs = 10000
np.random.seed(987689)
x = np.random.randn(nobs, 3)
x = sm.add_constant(x)
y = x.sum(1) + np.random.randn(nobs)



xf = 0.25 * np.ones((2,4))

model = sm.OLS(y, x)
#y_count = np.random.poisson(np.exp(x.sum(1)-x.mean()))
#model = sm.Poisson(y_count, x)#, exposure=np.ones(nobs), offset=np.zeros(nobs)) #bug with default
results = model.fit()


#print results.predict(xf)
print(results.model.predict(results.params, xf))
results.summary()

shrinkit = 1
if shrinkit:
    results.remove_data()

from statsmodels.compat.python import cPickle
fname = 'try_shrink%d_ols.pickle' % shrinkit
fh = open(fname, 'w')
cPickle.dump(results._results, fh)  #pickling wrapper doesn't work
fh.close()
fh = open(fname, 'r')
results2 = cPickle.load(fh)
fh.close()
print(results2.predict(xf))
print(results2.model.predict(results.params, xf))


y_count = np.random.poisson(np.exp(x.sum(1)-x.mean()))
model = sm.Poisson(y_count, x)#, exposure=np.ones(nobs), offset=np.zeros(nobs)) #bug with default
results = model.fit(method='bfgs')

results.summary()

print(results.model.predict(results.params, xf, exposure=1, offset=0))

if shrinkit:
    results.remove_data()
else:
    #work around pickling bug
    results.mle_settings['callback'] = None

import pickle
fname = 'try_shrink%d_poisson.pickle' % shrinkit
fh = open(fname, 'w')
cPickle.dump(results._results, fh)  #pickling wrapper doesn't work
fh.close()
fh = open(fname, 'r')
results3 = cPickle.load(fh)
fh.close()
print(results3.predict(xf, exposure=1, offset=0))
print(results3.model.predict(results.params, xf, exposure=1, offset=0))

def check_pickle(obj):
    fh =StringIO()
    cPickle.dump(obj, fh)
    plen = fh.pos
    fh.seek(0,0)
    res = cPickle.load(fh)
    fh.close()
    return res, plen

def test_remove_data_pickle(results, xf):
    res, l = check_pickle(results)
    #Note: 10000 is just a guess for the limit on the length of the pickle
    np.testing.assert_(l < 10000, msg='pickle length not %d < %d' % (l, 10000))
    pred1 = results.predict(xf, exposure=1, offset=0)
    pred2 = res.predict(xf, exposure=1, offset=0)
    np.testing.assert_equal(pred2, pred1)

test_remove_data_pickle(results._results, xf)
