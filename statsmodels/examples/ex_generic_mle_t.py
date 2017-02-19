# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 08:28:04 2010

Author: josef-pktd
"""


from __future__ import print_function
import numpy as np

from scipy import stats, special
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel

#redefine some shortcuts
np_log = np.log
np_pi = np.pi
sps_gamln = special.gammaln


def maxabs(arr1, arr2):
    return np.max(np.abs(arr1 - arr2))

def maxabsrel(arr1, arr2):
    return np.max(np.abs(arr2 / arr1 - 1))



class MyT(GenericLikelihoodModel):
    '''Maximum Likelihood Estimation of Poisson Model

    This is an example for generic MLE which has the same
    statistical model as discretemod.Poisson.

    Except for defining the negative log-likelihood method, all
    methods and results are generic. Gradients and Hessian
    and all resulting statistics are based on numerical
    differentiation.

    '''

    def loglike(self, params):
        return -self.nloglikeobs(params).sum(0)

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
        .. math:: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]
        """
        #print len(params),
        beta = params[:-2]
        df = params[-2]
        scale = params[-1]
        loc = np.dot(self.exog, beta)
        endog = self.endog
        x = (endog - loc)/scale
        #next part is stats.t._logpdf
        lPx = sps_gamln((df+1)/2) - sps_gamln(df/2.)
        lPx -= 0.5*np_log(df*np_pi) + (df+1)/2.*np_log(1+(x**2)/df)
        lPx -= np_log(scale)  # correction for scale
        return -lPx


#Example:
np.random.seed(98765678)
nobs = 1000
rvs = np.random.randn(nobs,5)
data_exog = sm.add_constant(rvs, prepend=False)
xbeta = 0.9 + 0.1*rvs.sum(1)
data_endog = xbeta + 0.1*np.random.standard_t(5, size=nobs)
#print data_endog

modp = MyT(data_endog, data_exog)
modp.start_value = np.ones(data_exog.shape[1]+2)
modp.start_value[-2] = 10
modp.start_params = modp.start_value
resp = modp.fit(start_params = modp.start_value)
print(resp.params)
print(resp.bse)

from statsmodels.tools.numdiff import approx_fprime, approx_hess

hb=-approx_hess(modp.start_value, modp.loglike, epsilon=-1e-4)
tmp = modp.loglike(modp.start_value)
print(tmp.shape)


'''
>>> tmp = modp.loglike(modp.start_value)
8
>>> tmp.shape
(100,)
>>> tmp.sum(0)
-24220.877108016182
>>> tmp = modp.nloglikeobs(modp.start_value)
8
>>> tmp.shape
(100, 100)
>>> np.dot(modp.exog, beta).shape
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'beta' is not defined

>>> params = modp.start_value
>>> beta = params[:-2]
>>> beta.shape
(6,)
>>> np.dot(modp.exog, beta).shape
(100,)
>>> modp.endog.shape
(100, 100)
>>> xbeta.shape
(100,)
>>>
'''

'''
C:\Programs\Python25\lib\site-packages\matplotlib-0.99.1-py2.5-win32.egg\matplotlib\rcsetup.py:117: UserWarning: rcParams key "numerix" is obsolete and has no effect;
 please delete it from your matplotlibrc file
  warnings.warn('rcParams key "numerix" is obsolete and has no effect;\n'
repr(start_params) array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
Optimization terminated successfully.
         Current function value: 91.897859
         Iterations: 108
         Function evaluations: 173
         Gradient evaluations: 173
[  1.58253308e-01   1.73188603e-01   1.77357447e-01   2.06707494e-02
  -1.31174789e-01   8.79915580e-01   6.47663840e+03   6.73457641e+02]
[         NaN          NaN          NaN          NaN          NaN
  28.26906182          NaN          NaN]
()
>>> resp.params
array([  1.58253308e-01,   1.73188603e-01,   1.77357447e-01,
         2.06707494e-02,  -1.31174789e-01,   8.79915580e-01,
         6.47663840e+03,   6.73457641e+02])
>>> resp.bse
array([         NaN,          NaN,          NaN,          NaN,
                NaN,  28.26906182,          NaN,          NaN])
>>> resp.jac
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'GenericLikelihoodModelResults' object has no attribute 'jac'

>>> resp.bsejac
array([    45243.35919908,     51997.80776897,     41418.33021984,
           42763.46575168,     50101.91631612,     42804.92083525,
         3005625.35649203,  13826948.68708931])
>>> resp.bsejhj
array([ 1.51643931,  0.80229636,  0.27720185,  0.4711138 ,  0.9028682 ,
        0.31673747,  0.00524426,  0.69729368])
>>> resp.covjac
array([[  2.04696155e+09,   1.46643494e+08,   7.59932781e+06,
         -2.39993397e+08,   5.62644255e+08,   2.34300598e+08,
         -3.07824799e+09,  -1.93425470e+10],
       [  1.46643494e+08,   2.70377201e+09,   1.06005712e+08,
          3.76824011e+08,  -1.21778986e+08,   5.38612723e+08,
         -2.12575784e+10,  -1.69503271e+11],
       [  7.59932781e+06,   1.06005712e+08,   1.71547808e+09,
         -5.94451158e+07,  -1.44586401e+08,  -5.41830441e+06,
          1.25899515e+10,   1.06372065e+11],
       [ -2.39993397e+08,   3.76824011e+08,  -5.94451158e+07,
          1.82871400e+09,  -5.66930891e+08,   3.75061111e+08,
         -6.84681772e+09,  -7.29993789e+10],
       [  5.62644255e+08,  -1.21778986e+08,  -1.44586401e+08,
         -5.66930891e+08,   2.51020202e+09,  -4.67886982e+08,
          1.78890380e+10,   1.75428694e+11],
       [  2.34300598e+08,   5.38612723e+08,  -5.41830441e+06,
          3.75061111e+08,  -4.67886982e+08,   1.83226125e+09,
         -1.27484996e+10,  -1.12550321e+11],
       [ -3.07824799e+09,  -2.12575784e+10,   1.25899515e+10,
         -6.84681772e+09,   1.78890380e+10,  -1.27484996e+10,
          9.03378378e+12,   2.15188047e+13],
       [ -1.93425470e+10,  -1.69503271e+11,   1.06372065e+11,
         -7.29993789e+10,   1.75428694e+11,  -1.12550321e+11,
          2.15188047e+13,   1.91184510e+14]])
>>> hb
array([[  33.68732564,   -2.33209221,  -13.51255321,   -1.60840159,
         -13.03920385,   -9.3506543 ,    4.86239173,   -9.30409101],
       [  -2.33209221,    3.12512611,   -6.08530968,   -6.79232244,
           3.66804898,    1.26497071,    5.10113409,   -2.53482995],
       [ -13.51255321,   -6.08530968,   31.14883498,   -5.01514705,
         -10.48819911,   -2.62533035,    3.82241581,  -12.51046342],
       [  -1.60840159,   -6.79232244,   -5.01514705,   28.40141917,
          -8.72489636,   -8.82449456,    5.47584023,  -18.20500017],
       [ -13.03920385,    3.66804898,  -10.48819911,   -8.72489636,
           9.03650914,    3.65206176,    6.55926726,   -1.8233635 ],
       [  -9.3506543 ,    1.26497071,   -2.62533035,   -8.82449456,
           3.65206176,   21.41825348,   -1.28610793,    4.28101146],
       [   4.86239173,    5.10113409,    3.82241581,    5.47584023,
           6.55926726,   -1.28610793,   46.52354448,  -32.23861427],
       [  -9.30409101,   -2.53482995,  -12.51046342,  -18.20500017,
          -1.8233635 ,    4.28101146,  -32.23861427,  178.61978279]])
>>> np.linalg.eigh(hb)
(array([ -10.50373649,    0.7460258 ,   14.73131793,   29.72453087,
         36.24103832,   41.98042979,   48.99815223,  190.04303734]), array([[-0.40303259,  0.10181305,  0.18164206,  0.48201456,  0.03916688,
         0.00903695,  0.74620692,  0.05853619],
       [-0.3201713 , -0.88444855, -0.19867642,  0.02828812,  0.16733946,
        -0.21440765, -0.02927317,  0.01176904],
       [-0.41847094,  0.00170161,  0.04973298,  0.43276118, -0.55894304,
         0.26454728, -0.49745582,  0.07251685],
       [-0.3508729 , -0.08302723,  0.25004884, -0.73495077, -0.38936448,
         0.20677082,  0.24464779,  0.11448238],
       [-0.62065653,  0.44662675, -0.37388565, -0.19453047,  0.29084735,
        -0.34151809, -0.19088978,  0.00342713],
       [-0.15119802, -0.01099165,  0.84377273,  0.00554863,  0.37332324,
        -0.17917015, -0.30371283, -0.03635211],
       [ 0.15813581,  0.0293601 ,  0.09882271,  0.03515962, -0.48768565,
        -0.81960996,  0.05248464,  0.22533642],
       [-0.06118044, -0.00549223,  0.03205047, -0.01782649, -0.21128588,
        -0.14391393,  0.05973658, -0.96226835]]))
>>> np.linalg.eigh(np.linalg.inv(hb))
(array([-0.09520422,  0.00526197,  0.02040893,  0.02382062,  0.02759303,
        0.03364225,  0.06788259,  1.34043621]), array([[-0.40303259,  0.05853619,  0.74620692, -0.00903695, -0.03916688,
         0.48201456,  0.18164206,  0.10181305],
       [-0.3201713 ,  0.01176904, -0.02927317,  0.21440765, -0.16733946,
         0.02828812, -0.19867642, -0.88444855],
       [-0.41847094,  0.07251685, -0.49745582, -0.26454728,  0.55894304,
         0.43276118,  0.04973298,  0.00170161],
       [-0.3508729 ,  0.11448238,  0.24464779, -0.20677082,  0.38936448,
        -0.73495077,  0.25004884, -0.08302723],
       [-0.62065653,  0.00342713, -0.19088978,  0.34151809, -0.29084735,
        -0.19453047, -0.37388565,  0.44662675],
       [-0.15119802, -0.03635211, -0.30371283,  0.17917015, -0.37332324,
         0.00554863,  0.84377273, -0.01099165],
       [ 0.15813581,  0.22533642,  0.05248464,  0.81960996,  0.48768565,
         0.03515962,  0.09882271,  0.0293601 ],
       [-0.06118044, -0.96226835,  0.05973658,  0.14391393,  0.21128588,
        -0.01782649,  0.03205047, -0.00549223]]))
>>> np.diag(np.linalg.inv(hb))
array([ 0.01991288,  1.0433882 ,  0.00516616,  0.02642799,  0.24732871,
        0.05281555,  0.02236704,  0.00643486])
>>> np.sqrt(np.diag(np.linalg.inv(hb)))
array([ 0.14111302,  1.02146375,  0.07187597,  0.16256686,  0.49732154,
        0.22981633,  0.14955616,  0.08021756])
>>> hess = modp.hessian(resp.params)
>>> np.sqrt(np.diag(np.linalg.inv(hess)))
array([ 231.3823423 ,  117.79508218,   31.46595143,   53.44753106,
        132.4855704 ,           NaN,    5.47881705,   90.75332693])
>>> hb=-approx_hess(resp.params, modp.loglike, epsilon=-1e-4)
>>> np.sqrt(np.diag(np.linalg.inv(hb)))
array([ 31.93524822,  22.0333515 ,          NaN,  29.90198792,
        38.82615785,          NaN,          NaN,          NaN])
>>> hb=-approx_hess(resp.params, modp.loglike, epsilon=-1e-8)
>>> np.sqrt(np.diag(np.linalg.inv(hb)))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Programs\Python25\lib\site-packages\numpy\linalg\linalg.py", line 423, in inv
    return wrap(solve(a, identity(a.shape[0], dtype=a.dtype)))
  File "C:\Programs\Python25\lib\site-packages\numpy\linalg\linalg.py", line 306, in solve
    raise LinAlgError, 'Singular matrix'
numpy.linalg.linalg.LinAlgError: Singular matrix
>>> resp.params
array([  1.58253308e-01,   1.73188603e-01,   1.77357447e-01,
         2.06707494e-02,  -1.31174789e-01,   8.79915580e-01,
         6.47663840e+03,   6.73457641e+02])
>>>
'''
