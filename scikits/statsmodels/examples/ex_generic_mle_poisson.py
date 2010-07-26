# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 08:34:59 2010

Author: josef-pktd
"""

import numpy as np
from scipy import stats, factorial, special, optimize
import scikits.statsmodels as sm
from scikits.statsmodels.model import GenericLikelihoodModel


class NonlinearDeltaCov(object):
    '''Asymptotic covariance by Deltamethod

    the function is designed for 2d array, with rows equal to
    the number of equations and columns equal to the number
    of parameters. 1d params work by chance ?

    fun: R^{m*k) -> R^{m}  where m is number of equations and k is
    the number of parameters.

    equations follow Greene

    '''
    def __init__(self, fun, params, cov_params):
        self.fun = fun
        self.params = params
        self.cov_params = cov_params

    def grad(self, params=None, **kwds):
        if params is None:
            params = self.params
        kwds.setdefault('epsilon', 1e-4)
        from scikits.statsmodels.sandbox.regression.numdiff import approx_fprime1
        return approx_fprime1(params, self.fun, **kwds)

    def cov(self):
        g = self.grad()
        covar = np.dot(np.dot(g, self.cov_params), g.T)
        return covar

    def expected(self):
        return self.fun(self.params)

    def wald(self, value):
        m = self.expected()
        v = self.cov()
        df = np.size(m)
        diff = m - value
        lmstat = np.dot(np.dot(diff.T, np.linalg.inv(v)), diff)
        return lmstat, stats.chi2.sf(lmstat, df)




class MyPoisson(GenericLikelihoodModel):
    def nloglikeobs(self, params):
        """
        Loglikelihood of Poisson model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        The log likelihood of the model evaluated at `params`

        Notes
        --------
        .. math :: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]
        """
        XB = np.dot(self.exog, params)
        endog = self.endog
        return np.exp(XB) -  endog*XB + np.log(factorial(endog))


#Example:
np.random.seed(98765678)
nobs = 1000
rvs = np.random.randn(nobs,6)
data_exog = rvs
data_exog = sm.add_constant(data_exog)
xbeta = 1 + 0.1*rvs.sum(1)
data_endog = np.random.poisson(np.exp(xbeta))
#print data_endog

modp = MyPoisson(data_endog, data_exog)
resp = modp.fit()
print resp.params
print resp.bse

from scikits.statsmodels.discretemod import Poisson
resdp = Poisson(data_endog, data_exog).fit()
print 'compare params'
print resdp.params - resp.params
print 'compare bse'
print resdp.bse - resp.bse

lam = np.exp(np.dot(data_exog, resp.params))
predmean = stats.poisson.stats(lam,moments='m')
print np.max(np.abs(predmean - lam))

fun = lambda params: np.exp(np.dot(data_exog.mean(0), params))

lamcov = NonlinearDeltaCov(fun, resp.params, resdp.cov_params())
print lamcov.cov().shape
print lamcov.cov()

print 'analytical'
xm = data_exog.mean(0)
print np.dot(np.dot(xm, resdp.cov_params()), xm.T) * \
        np.exp(2*np.dot(data_exog.mean(0), resp.params))

print lamcov.wald(1.)
print lamcov.wald(2.)
print lamcov.wald(2.6)








'''
C:\Programs\Python25\lib\site-packages\matplotlib-0.99.1-py2.5-win32.egg\matplotlib\rcsetup.py:117: UserWarning: rcParams key "numerix" is obsolete and has no effect;
 please delete it from your matplotlibrc file
  warnings.warn('rcParams key "numerix" is obsolete and has no effect;\n'
Optimization terminated successfully.
         Current function value: 1850.464763
         Iterations 7
[ 0.11467485  0.11410121  0.09912345  0.09128155  0.11822654  0.12593207
  0.99811283]
[ 0.01889237  0.0192463   0.01893035  0.01899633  0.01887275  0.01958139
  0.01945616]
Optimization terminated successfully.
         Current function value: 1850.464743
         Iterations 7
compare params
[  4.86161605e-05   4.40232398e-05   4.48784529e-05   4.56971083e-05
   4.01606873e-05   3.91021040e-05   2.77126042e-05]
compare bse
[ -4.09437138e-07  -1.74232766e-07  -2.29642314e-07  -3.98918483e-07
  -3.62972858e-07  -3.45755876e-07   4.48484326e-07]
0.0
(1, 1)
>>> lamcov.cov()
array([[ 0.00275933]])
>>> m = np.dot(data_exog.mean(0), params)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'params' is not defined

>>> m = np.dot(data_exog.mean(0), resp.params)
>>> m
0.97947971307838566
>>> m = np.dot(data_exog.mean(0), resp.params)
>>> m
0.97947971307838566
>>> xm = data_exog.mean(0)
>>> np.dot(np.dot(xm, resdp.cov_params()), xm')
  File "<stdin>", line 1
    np.dot(np.dot(xm, resdp.cov_params()), xm')
                                               ^
SyntaxError: EOL while scanning single-quoted string

>>> np.dot(np.dot(xm, resdp.cov_params()), xm.T)
0.0003890413012758283
>>> np.dot(np.dot(xm, resdp.cov_params()), xm.T) * np.exp(np.dot(data_exog.mean(0), resp.params))
0.0010360443429787217
>>> np.dot(np.dot(xm, resdp.cov_params()), xm.T) * np.exp(2*np.dot(data_exog.mean(0), resp.params))
0.0027590589407811604
>>> fun(resp.params)
2.6630703207631199
>>> resp.params
array([ 0.11467485,  0.11410121,  0.09912345,  0.09128155,  0.11822654,
        0.12593207,  0.99811283])
>>> fun(resp.params).shape
()
>>> fun(resp.params).shape[0]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: tuple index out of range

>>> np.size(resp.params)
7
>>>
>>> np.size(fun(resp.params))
1
>>>
'''
