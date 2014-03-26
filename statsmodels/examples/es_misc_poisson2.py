
from __future__ import print_function
import numpy as np
from numpy.testing import assert_almost_equal
import statsmodels.api as sm
from statsmodels.miscmodels.count import (PoissonGMLE, PoissonOffsetGMLE,
                                          PoissonZiGMLE)

DEC = 3

class Dummy(object):
    pass

self = Dummy()

# generate artificial data
np.random.seed(98765678)
nobs = 200
rvs = np.random.randn(nobs,6)
data_exog = rvs
data_exog = sm.add_constant(data_exog, prepend=False)
xbeta = 1 + 0.1*rvs.sum(1)
data_endog = np.random.poisson(np.exp(xbeta))

#estimate discretemod.Poisson as benchmark
from statsmodels.discrete.discrete_model import Poisson
res_discrete = Poisson(data_endog, data_exog).fit()

mod_glm = sm.GLM(data_endog, data_exog, family=sm.families.Poisson())
res_glm = mod_glm.fit()

#estimate generic MLE
self.mod = PoissonGMLE(data_endog, data_exog)
res = self.mod.fit()
offset = res.params[0] * data_exog[:,0]  #1d ???

mod1 = PoissonOffsetGMLE(data_endog, data_exog[:,1:], offset=offset)
start_params = np.ones(6)/2.
start_params = res.params[1:]
res1 = mod1.fit(start_params=start_params, method='nm', maxiter=1000, maxfun=1000)

print('mod2')
mod2 = PoissonZiGMLE(data_endog, data_exog[:,1:], offset=offset)
start_params = np.r_[np.ones(6)/2.,10]
start_params = np.r_[res.params[1:], 20.] #-100]
res2 = mod2.fit(start_params=start_params, method='bfgs', maxiter=1000, maxfun=2000)

print('mod3')
mod3 = PoissonZiGMLE(data_endog, data_exog, offset=None)
start_params = np.r_[np.ones(7)/2.,10]
start_params = np.r_[res.params, 20.]
res3 = mod3.fit(start_params=start_params, method='nm', maxiter=1000, maxfun=2000)

print('mod4')
data_endog2 = np.r_[data_endog, np.zeros(nobs)]
data_exog2 = np.r_[data_exog, data_exog]

mod4 = PoissonZiGMLE(data_endog2, data_exog2, offset=None)
start_params = np.r_[np.ones(7)/2.,10]
start_params = np.r_[res.params, 0.]
res4 = mod4.fit(start_params=start_params, method='nm', maxiter=1000, maxfun=1000)
print(res4.summary())
