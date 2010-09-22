# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 08:34:59 2010

Author: josef-pktd
"""

import numpy as np
from scipy import stats, factorial
import scikits.statsmodels as sm
from scikits.statsmodels.model import GenericLikelihoodModel

def maxabs(arr1, arr2):
    return np.max(np.abs(arr1 - arr2))

def maxabsrel(arr1, arr2):
    return np.max(np.abs(arr2 / arr1 - 1))

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
        # rename: misnomer, this is the MLE of the fun
        return self.fun(self.params)

    def wald(self, value):
        m = self.expected()
        v = self.cov()
        df = np.size(m)
        diff = m - value
        lmstat = np.dot(np.dot(diff.T, np.linalg.inv(v)), diff)
        return lmstat, stats.chi2.sf(lmstat, df)




class MyPoisson(GenericLikelihoodModel):
    '''Maximum Likelihood Estimation of Poisson Model

    This is an example for generic MLE which has the same
    statistical model as discretemod.Poisson.

    Except for defining the negative log-likelihood method, all
    methods and results are generic. Gradients and Hessian
    and all resulting statistics are based on numerical
    differentiation.

    '''

    # copied from discretemod.Poisson
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
print '\ncompare with discretemod'
print 'compare params'
print resdp.params - resp.params
print 'compare bse'
print resdp.bse - resp.bse

gmlp = sm.GLM(data_endog, data_exog, family=sm.families.Poisson())
resgp = gmlp.fit()
''' this creates a warning, bug bse is double defined ???
c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\decorators.py:105: CacheWriteWarning: The attribute 'bse' cannot be overwritten
  warnings.warn(errmsg, CacheWriteWarning)
'''
print '\ncompare with GLM'
print 'compare params'
print resgp.params - resp.params
print 'compare bse'
print resgp.bse - resp.bse

lam = np.exp(np.dot(data_exog, resp.params))
'''mean of Poisson distribution'''
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

''' cov of linear transformation of params
>>> np.dot(np.dot(xm, resdp.cov_params()), xm.T)
0.00038904130127582825
>>> resp.cov_params(xm)
0.00038902428119179394
>>> np.dot(np.dot(xm, resp.cov_params()), xm.T)
0.00038902428119179394
'''

print lamcov.wald(1.)
print lamcov.wald(2.)
print lamcov.wald(2.6)

do_bootstrap = False
if do_bootstrap:
    m,s,r = resp.bootstrap(method='newton')
    print m
    print s
    print resp.bse


print '\ncomparison maxabs, masabsrel'
print 'discr params', maxabs(resdp.params, resp.params), maxabsrel(resdp.params, resp.params)
print 'discr bse   ', maxabs(resdp.bse, resp.bse), maxabsrel(resdp.bse, resp.bse)
print 'discr bsejac', maxabs(resdp.bse, resp.bsejac), maxabsrel(resdp.bse, resp.bsejac)
print 'discr bsejhj', maxabs(resdp.bse, resp.bsejhj), maxabsrel(resdp.bse, resp.bsejhj)
print
print 'glm params  ', maxabs(resdp.params, resp.params), maxabsrel(resdp.params, resp.params)
print 'glm bse     ', maxabs(resdp.bse, resp.bse), maxabsrel(resdp.bse, resp.bse)

from numpy.testing import assert_almost_equal
class TestPoissonMLE(object):
    def __init__(self):

        # generate artificial data
        np.random.seed(98765678)
        nobs = 1000
        rvs = np.random.randn(nobs,6)
        data_exog = rvs
        data_exog = sm.add_constant(data_exog)
        xbeta = 1 + 0.1*rvs.sum(1)
        data_endog = np.random.poisson(np.exp(xbeta))

        #estimate generic MLE
        self.modp = MyPoisson(data_endog, data_exog)
        self.resp = modp.fit()

        #estimate discretemod.Poisson as benchmark
        from scikits.statsmodels.discretemod import Poisson
        self.resdp = Poisson(data_endog, data_exog).fit()





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
[[ 0.00275933]]
analytical
0.00275905894078
(array([[ 1002.34583847]]), array([[  5.55093011e-220]]))
(array([[ 159.33659922]]), array([[  1.57977502e-36]]))
(array([[ 1.44160555]]), array([[ 0.2298797]]))
>>> resp.model.covjac()
array([[  3.31749436e-04,   1.96735481e-05,  -2.40937560e-06,
          5.63343253e-05,   2.28848835e-05,  -7.80558541e-06,
         -3.50911022e-05],
       [  1.96735481e-05,   3.85161759e-04,  -7.44716324e-06,
         -3.04404923e-05,  -2.18321747e-05,   3.32043698e-05,
         -4.32518477e-05],
       [ -2.40937560e-06,  -7.44716324e-06,   3.77540102e-04,
          3.56778598e-06,  -2.71579995e-05,   3.34932738e-05,
         -1.63277634e-05],
       [  5.63343253e-05,  -3.04404923e-05,   3.56778598e-06,
          3.48560346e-04,   1.18124215e-05,  -2.80244861e-05,
         -2.56710456e-05],
       [  2.28848835e-05,  -2.18321747e-05,  -2.71579995e-05,
          1.18124215e-05,   3.93179273e-04,  -1.64149536e-05,
         -3.70355922e-06],
       [ -7.80558541e-06,   3.32043698e-05,   3.34932738e-05,
         -2.80244861e-05,  -1.64149536e-05,   3.58024985e-04,
         -4.30556640e-05],
       [ -3.50911022e-05,  -4.32518477e-05,  -1.63277634e-05,
         -2.56710456e-05,  -3.70355922e-06,  -4.30556640e-05,
          3.85748107e-04]])
>>> resp.model.covjhj()
array([[  3.97002964e-04,  -3.92860475e-05,   1.50807118e-05,
         -3.63798223e-05,  -4.32334161e-05,   3.10448414e-05,
         -1.94014861e-05],
       [ -3.92860475e-05,   3.64248873e-04,   1.84273065e-05,
          2.44031895e-05,   1.53835642e-05,  -3.33299519e-05,
         -3.88629003e-05],
       [  1.50807118e-05,   1.84273065e-05,   3.45182208e-04,
         -1.58078497e-05,  -6.20594864e-06,  -3.62167500e-05,
         -4.19290042e-05],
       [ -3.63798223e-05,   2.44031895e-05,  -1.58078497e-05,
          3.83125993e-04,  -3.61749041e-06,  -4.15931876e-05,
          8.45997433e-06],
       [ -4.32334161e-05,   1.53835642e-05,  -6.20594864e-06,
         -3.61749041e-06,   3.28507483e-04,  -2.88790409e-05,
         -4.90737495e-05],
       [  3.10448414e-05,  -3.33299519e-05,  -3.62167500e-05,
         -4.15931876e-05,  -2.88790409e-05,   4.18662519e-04,
         -4.58404343e-05],
       [ -1.94014861e-05,  -3.88629003e-05,  -4.19290042e-05,
          8.45997433e-06,  -4.90737495e-05,  -4.58404343e-05,
          3.74414760e-04]])
>>> np.sqrt(np.diag(resp.model.covjhj()))
array([ 0.01992493,  0.01908531,  0.01857908,  0.0195736 ,  0.01812478,
        0.02046124,  0.0193498 ])
>>> resp.bse
array([ 0.01889237,  0.0192463 ,  0.01893035,  0.01899633,  0.01887275,
        0.01958139,  0.01945616])
>>> np.sqrt(np.diag(resp.model.covjac()))
array([ 0.01821399,  0.01962554,  0.01943039,  0.01866977,  0.01982875,
        0.01892155,  0.01964047])
>>> resdp.bse
array([ 0.01889196,  0.01924613,  0.01893012,  0.01899593,  0.01887239,
        0.01958105,  0.01945661])
>>> np.sqrt(np.diag(resp.model.covjhj())) - resdp.bse
array([ 0.00103297, -0.00016083, -0.00035104,  0.00057767, -0.00074762,
        0.0008802 , -0.00010681])
>>> np.sqrt(np.diag(resp.model.covjac())) - resdp.bse
array([-0.00067797,  0.00037941,  0.00050027, -0.00032616,  0.00095636,
       -0.0006595 ,  0.00018386])

>>> resp.bse - resdp.bse
array([  4.09437138e-07,   1.74232766e-07,   2.29642314e-07,
         3.98918483e-07,   3.62972858e-07,   3.45755876e-07,
        -4.48484326e-07])
>>> np.sqrt(np.diag(resp.model.covjhj())) / resdp.bse -1
array([ 0.05467782, -0.00835626, -0.01854421,  0.03041031, -0.03961428,
        0.04495154, -0.00548979])
>>> (np.sqrt(np.diag(resp.model.covjhj())) / resdp.bse -1)*100
array([ 5.46778165, -0.83562557, -1.85442053,  3.04103056, -3.96142751,
        4.49515422, -0.5489788 ])
>>>
'''

'''
>>> modnew = resp.model.__class__(data_endog[:100], data_exog[:100])
>>> resnew = modnew.fit()
Optimization terminated successfully.
         Current function value: 175.693241
         Iterations 8
>>> resnew.params
array([ 0.04242913,  0.24725651,  0.01703473,  0.14246281,  0.20778146,
        0.06525944,  0.88193324])
'''

# bootstrap results
'''
>>> lamboots = np.exp(np.dot(r, xm))
>>> lamboots.shape
(100,)
>>> lamboots.mean
<built-in method mean of numpy.ndarray object at 0x023C9658>
>>> lamboots.mean()
2.652769263132293
>>> lamboots.std()
0.048289762589747605
>>> lamcov.cov()
array([[ 0.00275944]])
>>> lamboots.var()
0.002331901170974187
>>> lamcov.mean
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'NonlinearDeltaCov' object has no attribute 'mean'

>>> lamcov.expected
<bound method NonlinearDeltaCov.expected of <__main__.NonlinearDeltaCov object at 0x020F2D70>>
>>> lamcov.expected()
2.6631247247748115
>>> lamboots_s = np.sort(lamboots)
>>> lamboots_s[floor(lamboots_s.shape[0]*0.05)]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'floor' is not defined

>>> lamboots_s[np.floor(lamboots_s.shape[0]*0.05)]
2.5751457398224877
>>> lamboots_s[np.ceil(lamboots_s.shape[0]*0.05)]
2.5751457398224877
>>> lamboots_s[-np.ceil(lamboots_s.shape[0]*0.05)]
2.7339727361167099
>>> lamboots_s[-np.ceil(lamboots_s.shape[0]*0.025)]
2.7544312022416912
>>> lamboots_s[-np.ceil(lamboots_s.shape[0]*0.025)]
2.7544312022416912
>>> r_s = np.sort(r,0)
>>> r_s[-np.ceil(lamboots_s.shape[0]*0.025)]
array([ 0.15634232,  0.14924568,  0.12959952,  0.12890382,  0.15470283,
        0.1619681 ,  1.0295304 ])
>>> r_s[np.floor(lamboots_s.shape[0]*0.05)]
array([ 0.08728524,  0.0887237 ,  0.06700413,  0.06556841,  0.09322076,
        0.08857776,  0.9657677 ])
>>>

>>> resp.model.df_resid = 1000
>>> resp.conf_int()
array([[ 0.07760053,  0.15174624],
       [ 0.07633444,  0.15186927],
       [ 0.06197996,  0.13627473],
       [ 0.05401091,  0.12856468],
       [ 0.08118493,  0.15525368],
       [ 0.0875024 ,  0.16435227],
       [ 0.95995415,  1.03631253]])
>>> np.column_stack([r_s[np.floor(lamboots_s.shape[0]*0.025)],
                     r_s[-np.ceil(lamboots_s.shape[0]*0.025)]])
array([[ 0.08180546,  0.15634232],
       [ 0.0812217 ,  0.14924568],
       [ 0.06374989,  0.12959952],
       [ 0.05860489,  0.12890382],
       [ 0.08772002,  0.15470283],
       [ 0.08340517,  0.1619681 ],
       [ 0.95598879,  1.0295304 ]])

'''

'''
C:\Programs\Python25\lib\site-packages\matplotlib-0.99.1-py2.5-win32.egg\matplotlib\rcsetup.py:117: UserWarning: rcParams key "numerix" is obsolete and has no effect;
 please delete it from your matplotlibrc file
  warnings.warn('rcParams key "numerix" is obsolete and has no effect;\n'
Warning: Desired error not necessarily achieveddue to precision loss
         Current function value: 1850.464762
         Iterations: 16
         Function evaluations: 42
         Gradient evaluations: 41
[ 0.11467339  0.11410185  0.09912734  0.09128779  0.11821931  0.12592733
  0.99813334]
[ 0.01889218  0.01924612  0.01893016  0.01899615  0.01887256  0.01958119
  0.01945596]
Optimization terminated successfully.
         Current function value: 1850.464743
         Iterations 7
compare params
[  5.00759798e-05   4.33838516e-05   4.09882887e-05   3.94532511e-05
   4.73972834e-05   4.38332826e-05   7.20893712e-06]
compare bse
[ -2.14180090e-07   1.01278767e-08  -3.15431400e-08  -2.16062253e-07
  -1.73459318e-07  -1.42927012e-07   6.53938779e-07]
0.0
(1, 1)
[[ 0.00275944]]
analytical
0.00275917167187
(array([[ 1002.37046364]]), array([[  5.48293565e-220]]))
(array([[ 159.35623593]]), array([[  1.56424491e-36]]))
(array([[ 1.44403466]]), array([[ 0.22948756]]))
<class '__main__.MyPoisson'>
Optimization terminated successfully.
         Current function value: 1838.716751
         Iterations 7
Optimization terminated successfully.
         Current function value: 1824.417107
         Iterations 7
Optimization terminated successfully.
         Current function value: 1872.612180
         Iterations 8
Optimization terminated successfully.
         Current function value: 1879.691734
         Iterations 7
Optimization terminated successfully.
         Current function value: 1855.399226
         Iterations 7
Optimization terminated successfully.
         Current function value: 1832.896494
         Iterations 7
Optimization terminated successfully.
         Current function value: 1854.982956
         Iterations 7
Optimization terminated successfully.
         Current function value: 1843.245409
         Iterations 7
Optimization terminated successfully.
         Current function value: 1842.070292
         Iterations 7
Optimization terminated successfully.
         Current function value: 1838.889974
         Iterations 7
Optimization terminated successfully.
         Current function value: 1797.894229
         Iterations 7
Optimization terminated successfully.
         Current function value: 1828.937198
         Iterations 7
Optimization terminated successfully.
         Current function value: 1831.862188
         Iterations 7
Optimization terminated successfully.
         Current function value: 1840.532065
         Iterations 7
Optimization terminated successfully.
         Current function value: 1840.317910
         Iterations 7
Optimization terminated successfully.
         Current function value: 1820.839432
         Iterations 8
Optimization terminated successfully.
         Current function value: 1887.109943
         Iterations 7
Optimization terminated successfully.
         Current function value: 1808.451292
         Iterations 7
Optimization terminated successfully.
         Current function value: 1796.839422
         Iterations 7
Optimization terminated successfully.
         Current function value: 1825.613056
         Iterations 7
Optimization terminated successfully.
         Current function value: 1834.838255
         Iterations 8
Optimization terminated successfully.
         Current function value: 1856.107145
         Iterations 8
Optimization terminated successfully.
         Current function value: 1864.356783
         Iterations 7
Optimization terminated successfully.
         Current function value: 1842.534425
         Iterations 7
Optimization terminated successfully.
         Current function value: 1872.919482
         Iterations 7
Optimization terminated successfully.
         Current function value: 1851.339345
         Iterations 7
Optimization terminated successfully.
         Current function value: 1892.392002
         Iterations 7
Optimization terminated successfully.
         Current function value: 1829.130336
         Iterations 7
Optimization terminated successfully.
         Current function value: 1836.077728
         Iterations 7
Optimization terminated successfully.
         Current function value: 1891.269760
         Iterations 7
Optimization terminated successfully.
         Current function value: 1802.134950
         Iterations 7
Optimization terminated successfully.
         Current function value: 1843.037757
         Iterations 7
Optimization terminated successfully.
         Current function value: 1797.600274
         Iterations 7
Optimization terminated successfully.
         Current function value: 1849.706459
         Iterations 7
Optimization terminated successfully.
         Current function value: 1861.218885
         Iterations 7
Optimization terminated successfully.
         Current function value: 1874.765195
         Iterations 8
Optimization terminated successfully.
         Current function value: 1805.010614
         Iterations 7
Optimization terminated successfully.
         Current function value: 1870.395327
         Iterations 7
Optimization terminated successfully.
         Current function value: 1841.912377
         Iterations 7
Optimization terminated successfully.
         Current function value: 1834.809537
         Iterations 7
Optimization terminated successfully.
         Current function value: 1787.088124
         Iterations 7
Optimization terminated successfully.
         Current function value: 1884.035193
         Iterations 7
Optimization terminated successfully.
         Current function value: 1829.665610
         Iterations 7
Optimization terminated successfully.
         Current function value: 1867.819272
         Iterations 7
Optimization terminated successfully.
         Current function value: 1831.384560
         Iterations 7
Optimization terminated successfully.
         Current function value: 1820.252540
         Iterations 7
Optimization terminated successfully.
         Current function value: 1840.781255
         Iterations 7
Optimization terminated successfully.
         Current function value: 1843.580701
         Iterations 7
Optimization terminated successfully.
         Current function value: 1852.075478
         Iterations 8
Optimization terminated successfully.
         Current function value: 1880.542815
         Iterations 7
Optimization terminated successfully.
         Current function value: 1864.418209
         Iterations 7
Optimization terminated successfully.
         Current function value: 1854.920292
         Iterations 8
Optimization terminated successfully.
         Current function value: 1862.379349
         Iterations 7
Optimization terminated successfully.
         Current function value: 1863.562129
         Iterations 7
Optimization terminated successfully.
         Current function value: 1822.813313
         Iterations 7
Optimization terminated successfully.
         Current function value: 1862.169448
         Iterations 7
Optimization terminated successfully.
         Current function value: 1797.168018
         Iterations 7
Optimization terminated successfully.
         Current function value: 1870.198841
         Iterations 7
Optimization terminated successfully.
         Current function value: 1842.179575
         Iterations 7
Optimization terminated successfully.
         Current function value: 1813.479475
         Iterations 7
Optimization terminated successfully.
         Current function value: 1836.700351
         Iterations 7
Optimization terminated successfully.
         Current function value: 1874.926819
         Iterations 7
Optimization terminated successfully.
         Current function value: 1885.473190
         Iterations 7
Optimization terminated successfully.
         Current function value: 1848.766660
         Iterations 7
Optimization terminated successfully.
         Current function value: 1818.960873
         Iterations 7
Optimization terminated successfully.
         Current function value: 1873.391803
         Iterations 7
Optimization terminated successfully.
         Current function value: 1824.689211
         Iterations 7
Optimization terminated successfully.
         Current function value: 1872.214561
         Iterations 8
Optimization terminated successfully.
         Current function value: 1858.395332
         Iterations 7
Optimization terminated successfully.
         Current function value: 1816.359558
         Iterations 7
Optimization terminated successfully.
         Current function value: 1812.359580
         Iterations 7
Optimization terminated successfully.
         Current function value: 1804.720621
         Iterations 8
Optimization terminated successfully.
         Current function value: 1836.573387
         Iterations 7
Optimization terminated successfully.
         Current function value: 1844.324791
         Iterations 7
Optimization terminated successfully.
         Current function value: 1855.622736
         Iterations 7
Optimization terminated successfully.
         Current function value: 1835.077022
         Iterations 7
Optimization terminated successfully.
         Current function value: 1861.741831
         Iterations 7
Optimization terminated successfully.
         Current function value: 1824.166800
         Iterations 7
Optimization terminated successfully.
         Current function value: 1858.920023
         Iterations 7
Optimization terminated successfully.
         Current function value: 1866.544502
         Iterations 7
Optimization terminated successfully.
         Current function value: 1866.994736
         Iterations 7
Optimization terminated successfully.
         Current function value: 1829.652056
         Iterations 7
Optimization terminated successfully.
         Current function value: 1845.241818
         Iterations 7
Optimization terminated successfully.
         Current function value: 1850.666330
         Iterations 7
Optimization terminated successfully.
         Current function value: 1852.045608
         Iterations 7
Optimization terminated successfully.
         Current function value: 1846.506391
         Iterations 7
Optimization terminated successfully.
         Current function value: 1826.910981
         Iterations 7
Optimization terminated successfully.
         Current function value: 1841.312615
         Iterations 7
Optimization terminated successfully.
         Current function value: 1817.899736
         Iterations 7
Optimization terminated successfully.
         Current function value: 1829.310500
         Iterations 8
Optimization terminated successfully.
         Current function value: 1857.793096
         Iterations 8
Optimization terminated successfully.
         Current function value: 1888.251117
         Iterations 7
Optimization terminated successfully.
         Current function value: 1858.137520
         Iterations 7
Optimization terminated successfully.
         Current function value: 1893.934929
         Iterations 7
Optimization terminated successfully.
         Current function value: 1858.501623
         Iterations 7
Optimization terminated successfully.
         Current function value: 1833.890184
         Iterations 7
Optimization terminated successfully.
         Current function value: 1847.655637
         Iterations 7
Optimization terminated successfully.
         Current function value: 1840.947925
         Iterations 7
Optimization terminated successfully.
         Current function value: 1805.539102
         Iterations 7
Optimization terminated successfully.
         Current function value: 1829.855631
         Iterations 7
[ 0.11706039  0.11295848  0.09752456  0.09286044  0.11712545  0.12434599
  0.99416779]
[ 0.01889407  0.01819808  0.0173015   0.01877438  0.01701709  0.01996202
  0.01774115]
[ 0.01889218  0.01924612  0.01893016  0.01899615  0.01887256  0.01958119
  0.01945596]
>>> xm
array([-0.03791887, -0.00486621, -0.01934583, -0.06387206, -0.04375079,
       -0.00642465,  1.        ])
>>> lamboots = np.exp(np.dot(r, xm))
>>> lamboots.shape
(100,)
>>> lamboots.mean
<built-in method mean of numpy.ndarray object at 0x023C9658>
>>> lamboots.mean()
2.652769263132293
>>> lamboots.std()
0.048289762589747605
>>> lamcov.cov()
array([[ 0.00275944]])
>>> lamboots.var()
0.002331901170974187
>>> lamcov.mean
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'NonlinearDeltaCov' object has no attribute 'mean'

>>> lamcov.expected
<bound method NonlinearDeltaCov.expected of <__main__.NonlinearDeltaCov object at 0x020F2D70>>
>>> lamcov.expected()
2.6631247247748115
>>> lamboots_s = np.sort(lamboots)
>>> lamboots_s[floor(lamboots_s.shape[0]*0.05)]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'floor' is not defined

>>> lamboots_s[np.floor(lamboots_s.shape[0]*0.05)]
2.5751457398224877
>>> lamboots_s[np.ceil(lamboots_s.shape[0]*0.05)]
2.5751457398224877
>>> lamboots_s[-np.ceil(lamboots_s.shape[0]*0.05)]
2.7339727361167099
>>> lamboots_s[-np.ceil(lamboots_s.shape[0]*0.025)]
2.7544312022416912
>>> lamboots_s[-np.ceil(lamboots_s.shape[0]*0.025)]
2.7544312022416912
>>> r_s = np.sort(r,0)
>>> r_s[-np.ceil(lamboots_s.shape[0]*0.025)]
array([ 0.15634232,  0.14924568,  0.12959952,  0.12890382,  0.15470283,
        0.1619681 ,  1.0295304 ])
>>> r_s[np.floor(lamboots_s.shape[0]*0.05)]
array([ 0.08728524,  0.0887237 ,  0.06700413,  0.06556841,  0.09322076,
        0.08857776,  0.9657677 ])
>>> resp.conf_int()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\model.py", line 1075, in conf_int
    lower = self.params - dist.ppf(1-alpha/2,self.model.df_resid) *\
AttributeError: 'MyPoisson' object has no attribute 'df_resid'

>>> resp.model.df_resid = 1000
>>> resp.conf_int()
array([[ 0.07760053,  0.15174624],
       [ 0.07633444,  0.15186927],
       [ 0.06197996,  0.13627473],
       [ 0.05401091,  0.12856468],
       [ 0.08118493,  0.15525368],
       [ 0.0875024 ,  0.16435227],
       [ 0.95995415,  1.03631253]])
>>> np.column_stack([r_s[np.floor(lamboots_s.shape[0]*0.05)], r_s[-np.ceil(lamboots_s.shape[0]*0.025)]])
array([[ 0.08728524,  0.15634232],
       [ 0.0887237 ,  0.14924568],
       [ 0.06700413,  0.12959952],
       [ 0.06556841,  0.12890382],
       [ 0.09322076,  0.15470283],
       [ 0.08857776,  0.1619681 ],
       [ 0.9657677 ,  1.0295304 ]])
>>> np.column_stack([r_s[np.floor(lamboots_s.shape[0]*0.025)], r_s[-np.ceil(lamboots_s.shape[0]*0.025)]])
array([[ 0.08180546,  0.15634232],
       [ 0.0812217 ,  0.14924568],
       [ 0.06374989,  0.12959952],
       [ 0.05860489,  0.12890382],
       [ 0.08772002,  0.15470283],
       [ 0.08340517,  0.1619681 ],
       [ 0.95598879,  1.0295304 ]])
>>> stats.t._logpdf([5,0.1],5)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Josef\_progs\Subversion\scipy-trunk_after\trunk\dist\scipy-0.9.0.dev6579.win32\Programs\Python25\Lib\site-packages\scipy\stats\distributions.py", line 3740, in _logpdf
    lPx -= 0.5*log(r*pi) + (r+1)/2*log(1+(x**2)/r)
TypeError: unsupported operand type(s) for ** or pow(): 'list' and 'int'

>>> stats.t._logpdf(np.array([5,0.1]),5)
array([-6.343898 , -0.9746136])
>>> np.log(stats.t._pdf(np.array([5,0.1]),5))
array([-6.343898 , -0.9746136])
>>> np.gumbel
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'module' object has no attribute 'gumbel'

>>>
'''

'''
C:\Programs\Python25\lib\site-packages\matplotlib-0.99.1-py2.5-win32.egg\matplotlib\rcsetup.py:117: UserWarning: rcParams key "numerix" is obsolete and has no effect;
 please delete it from your matplotlibrc file
  warnings.warn('rcParams key "numerix" is obsolete and has no effect;\n'
Warning: Desired error not necessarily achieveddue to precision loss
         Current function value: 1850.464762
         Iterations: 16
         Function evaluations: 42
         Gradient evaluations: 41
[ 0.11467339  0.11410185  0.09912734  0.09128779  0.11821931  0.12592733
  0.99813334]
[ 0.01889218  0.01924612  0.01893016  0.01899615  0.01887256  0.01958119
  0.01945596]
Optimization terminated successfully.
         Current function value: 1850.464743
         Iterations 7
compare params
[  5.00759874e-05   4.33838499e-05   4.09882926e-05   3.94532382e-05
   4.73972823e-05   4.38332934e-05   7.20893378e-06]
compare bse
[ -2.16317918e-07   8.64674746e-09  -3.44553395e-08  -2.13453529e-07
  -1.72783642e-07  -1.44169333e-07   6.53689146e-07]
0.0
(1, 1)
[[ 0.00275944]]
analytical
0.00275917167189
(array([[ 1002.37046364]]), array([[  5.48293564e-220]]))
(array([[ 159.35623594]]), array([[  1.56424491e-36]]))
(array([[ 1.44403466]]), array([[ 0.22948756]]))
<class '__main__.MyPoisson'>
[ 0.11706039  0.11295848  0.09752456  0.09286044  0.11712545  0.12434599
  0.99416779]
[ 0.01889407  0.01819808  0.0173015   0.01877438  0.01701709  0.01996202
  0.01774115]
[ 0.01889218  0.01924612  0.01893016  0.01899615  0.01887256  0.01958119
  0.01945596]
>>> resp.bsejhj
array([ 0.01992451,  0.01908492,  0.01857873,  0.01957318,  0.0181244 ,
        0.02046083,  0.01934938])
>>> resp.bsejac
array([ 0.01821398,  0.01962556,  0.01943036,  0.0186698 ,  0.01982873,
        0.01892153,  0.01964047])
>>> s
array([ 0.01889407,  0.01819808,  0.0173015 ,  0.01877438,  0.01701709,
        0.01996202,  0.01774115])
>>> m5,s5,r5 = resp.bootstrap(nrep=500,method='newton')
<class '__main__.MyPoisson'>
>>> m5
array([ 0.11466849,  0.11365848,  0.09733133,  0.09020553,  0.11727044,
        0.12676167,  0.9985733 ])
>>> s5
array([ 0.01960173,  0.01919744,  0.01883891,  0.01854194,  0.01909822,
        0.02017819,  0.01971702])
>>> resp.bse
array([ 0.01889218,  0.01924612,  0.01893016,  0.01899615,  0.01887256,
        0.01958119,  0.01945596])
>>> m5/s5
array([  5.84991759,   5.92050046,   5.16650497,   4.86494442,
         6.14038532,   6.28211424,  50.64523658])
>>> res2.bse
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'res2' is not defined

>>> resdp.bse
array([ 0.01889196,  0.01924613,  0.01893012,  0.01899593,  0.01887239,
        0.01958105,  0.01945661])
>>> resdp.bse - resp.bse
array([ -2.16317918e-07,   8.64674746e-09,  -3.44553395e-08,
        -2.13453529e-07,  -1.72783642e-07,  -1.44169333e-07,
         6.53689146e-07])
>>> resdp.bse - s5
array([ -7.09766269e-04,   4.86864027e-05,   9.12120361e-05,
         4.53988542e-04,  -2.25830315e-04,  -5.97139582e-04,
        -2.60410331e-04])
>>> resdp.t()
array([  6.07260721,   5.93081484,   5.23865204,   4.80772633,
         6.26665166,   6.43332162,  51.3008386 ])
>>> resp.t()
array([  6.06988706,   5.92856334,   5.23647726,   4.8055954 ,
         6.26408285,   6.43103571,  51.3021917 ])
>>> lam = np.exp(np.dot(data_exog, resp.params))
>>> predmean = stats.poisson.stats(lam,moments='m')
>>> print np.max(np.abs(predmean - lam))
0.0
>>> 'mean of Poisson distribution'
'mean of Poisson distribution'
>>> predmean = stats.poisson.stats(lam,moments='m')
>>>
>>> resdp.bse - resp.bsejac
array([ 0.00067798, -0.00037943, -0.00050023,  0.00032613, -0.00095634,
        0.00065952, -0.00018386])
>>> from numpy.testing import assert_almost_equal
>>> assert_almost_equal(resdp.bse, resp.bsejac, 5)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Programs\Python25\lib\site-packages\numpy\testing\utils.py", line 441, in assert_almost_equal
    return assert_array_almost_equal(actual, desired, decimal, err_msg)
  File "C:\Programs\Python25\lib\site-packages\numpy\testing\utils.py", line 765, in assert_array_almost_equal
    header='Arrays are not almost equal')
  File "C:\Programs\Python25\lib\site-packages\numpy\testing\utils.py", line 609, in assert_array_compare
    raise AssertionError(msg)
AssertionError:
Arrays are not almost equal

(mismatch 100.0%)
 x: array([ 0.01889196,  0.01924613,  0.01893012,  0.01899593,  0.01887239,
        0.01958105,  0.01945661])
 y: array([ 0.01821398,  0.01962556,  0.01943036,  0.0186698 ,  0.01982873,
        0.01892153,  0.01964047])

>>> assert_almost_equal(resdp.bse, resp.bsejac, 4)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Programs\Python25\lib\site-packages\numpy\testing\utils.py", line 441, in assert_almost_equal
    return assert_array_almost_equal(actual, desired, decimal, err_msg)
  File "C:\Programs\Python25\lib\site-packages\numpy\testing\utils.py", line 765, in assert_array_almost_equal
    header='Arrays are not almost equal')
  File "C:\Programs\Python25\lib\site-packages\numpy\testing\utils.py", line 609, in assert_array_compare
    raise AssertionError(msg)
AssertionError:
Arrays are not almost equal

(mismatch 100.0%)
 x: array([ 0.01889196,  0.01924613,  0.01893012,  0.01899593,  0.01887239,
        0.01958105,  0.01945661])
 y: array([ 0.01821398,  0.01962556,  0.01943036,  0.0186698 ,  0.01982873,
        0.01892153,  0.01964047])

>>> assert_almost_equal(resdp.bse, resp.bsejac, 3)
>>> def maxabs(arr1, arr2):
...     return np.max(np.abs(arr1, arr2))
...
>>> maxabs(resdp.bse, resp.bsejac)
0.019581046101296185
>>> def maxabs(arr1, arr2):
...     return np.max(np.abs(arr1 - arr2))
...
>>> maxabs(resdp.bse, resp.bsejac)
0.0
>>> np.max(np.abs(resdp.bse - resp.bsejac)
...
... )
0.0
>>> resdp.bse - resp.bsejac
array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.])
>>> resdp.bse - resp.bsejhj
array([-0.00103255,  0.00016121,  0.0003514 , -0.00057725,  0.00074799,
       -0.00087979,  0.00010723])
>>> np.max(np.abs(resdp.bse - resp.bsejhj)
... )
0.0010325508056756791
>>> def maxabsrel(arr1, arr2):
...     return np.max(np.abs(arr1 / arr2 - 1))
...
>>> maxabsrel(resdp.bse, resp.bsejhj)
0.051823139477062408
>>>
>>> resdp.model.predict()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: predict() takes exactly 2 arguments (1 given)

>>> resdp.model.fitted_values[:10]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'Poisson' object has no attribute 'fitted_values'

>>> resdp.fitted_values[:10]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'DiscreteResults' object has no attribute 'fitted_values'

>>> dir(resdp)
['__class__', '__delattr__', '__dict__', '__doc__', '__getattribute__', '__hash__', '__init__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__str__', '__weakref__', '_cache', 'aic', 'bic', 'bse', 'conf_int', 'cov_params', 'df_model', 'df_resid', 'f_test', 'fittedvalues', 'initialize', 'llf', 'llnull', 'llr', 'llr_pvalue', 'margeff', 'mle_retvals', 'mle_settings', 'model', 'nobs', 'normalized_cov_params', 'params', 'prsquared', 'scale', 't', 't_test', 'tval']
>>> resdp.fittedvalues[:10]
array([ 1.22064817,  0.68525511,  0.67017905,  0.85995342,  1.09986713,
        1.71281003,  0.78698615,  0.49580334,  0.65307384,  0.70446772])
>>> dir(resp)
['__class__', '__delattr__', '__dict__', '__doc__', '__getattribute__', '__hash__', '__init__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__str__', '__weakref__', '_cache', 'bootstrap', 'bse', 'bsejac', 'bsejhj', 'conf_int', 'cov_params', 'covjac', 'covjhj', 'endog', 'exog', 'f_test', 'hessv', 'initialize', 'jacv', 'llf', 'mle_retvals', 'mle_settings', 'model', 'nobs', 'normalized_cov_params', 'params', 'scale', 't', 't_test', 'tval']
>>> lam[:10]
array([ 3.38907965,  1.98451501,  1.95485303,  2.36314107,  3.00361542,
        5.54291517,  2.19693142,  1.64213302,  1.92168367,  2.02300577])
>>> np.log(lam[:10])
array([ 1.2205584 ,  0.68537456,  0.67031501,  0.85999169,  1.0998167 ,
        1.71252057,  0.78706158,  0.49599602,  0.65320171,  0.70458441])
>>> resdp.prsquared.shape
()
>>> resdp.prsquared
0.054975512934849591
>>> sm.GLM(data.endog, data.exog,        \
...                              family=sm.families.Gamma())
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'data' is not defined

>>> gmlp = sm.GLM(data_endog, data_exog, family=sm.families.Poisson())
>>> resgp = gmlp.fit()
c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\decorators.py:105: CacheWriteWarning: The attribute 'bse' cannot be overwritten
  warnings.warn(errmsg, CacheWriteWarning)
>>> resgp.bse
array([ 0.01889196,  0.01924613,  0.01893012,  0.01899593,  0.01887239,
        0.01958105,  0.01945661])
>>> resgp.params
array([ 0.11472346,  0.11414524,  0.09916833,  0.09132724,  0.1182667 ,
        0.12597117,  0.99814055])
>>> resdp.params
array([ 0.11472346,  0.11414524,  0.09916833,  0.09132724,  0.1182667 ,
        0.12597117,  0.99814055])
>>> resp.params
array([ 0.11467339,  0.11410185,  0.09912734,  0.09128779,  0.11821931,
        0.12592733,  0.99813334])
>>> resdp.params - resp.params
array([  5.00759874e-05,   4.33838499e-05,   4.09882926e-05,
         3.94532382e-05,   4.73972823e-05,   4.38332934e-05,
         7.20893378e-06])
>>> resdp.bse - resp.bse
array([ -2.16317918e-07,   8.64674746e-09,  -3.44553395e-08,
        -2.13453529e-07,  -1.72783642e-07,  -1.44169333e-07,
         6.53689146e-07])
>>>
'''
