
from __future__ import print_function
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel


data = sm.datasets.spector.load()
data.exog = sm.add_constant(data.exog, prepend=False)
# in this dir

probit_mod = sm.Probit(data.endog, data.exog)
probit_res = probit_mod.fit()
loglike = probit_mod.loglike
score = probit_mod.score
mod = GenericLikelihoodModel(data.endog, data.exog*2, loglike, score)
res = mod.fit(method="nm", maxiter = 500)

def probitloglike(params, endog, exog):
      """
      Log likelihood for the probit
      """
      q = 2*endog - 1
      X = exog
      return np.add.reduce(stats.norm.logcdf(q*np.dot(X,params)))

mod = GenericLikelihoodModel(data.endog, data.exog, loglike=probitloglike)
res = mod.fit(method="nm", fargs=(data.endog,data.exog), maxiter=500)
print(res)


#np.allclose(res.params, probit_res.params)

print(res.params, probit_res.params)

#datal = sm.datasets.longley.load()
datal = sm.datasets.ccard.load()
datal.exog = sm.add_constant(datal.exog, prepend=False)
# Instance of GenericLikelihood model doesn't work directly, because loglike
# cannot get access to data in self.endog, self.exog

nobs = 5000
rvs = np.random.randn(nobs,6)
datal.exog = rvs[:,:-1]
datal.exog = sm.add_constant(datal.exog, prepend=False)
datal.endog = 1 + rvs.sum(1)

show_error = False
show_error2 = 1#False
if show_error:
    def loglike_norm_xb(self, params):
        beta = params[:-1]
        sigma = params[-1]
        xb = np.dot(self.exog, beta)
        return stats.norm.logpdf(self.endog, loc=xb, scale=sigma)

    mod_norm = GenericLikelihoodModel(datal.endog, datal.exog, loglike_norm_xb)
    res_norm = mod_norm.fit(method="nm", maxiter = 500)

    print(res_norm.params)

if show_error2:
    def loglike_norm_xb(params, endog, exog):
        beta = params[:-1]
        sigma = params[-1]
        #print exog.shape, beta.shape
        xb = np.dot(exog, beta)
        #print xb.shape, stats.norm.logpdf(endog, loc=xb, scale=sigma).shape
        return stats.norm.logpdf(endog, loc=xb, scale=sigma).sum()

    mod_norm = GenericLikelihoodModel(datal.endog, datal.exog, loglike_norm_xb)
    res_norm = mod_norm.fit(start_params=np.ones(datal.exog.shape[1]+1),
                            method="nm", maxiter = 5000,
                            fargs=(datal.endog, datal.exog))

    print(res_norm.params)

class MygMLE(GenericLikelihoodModel):
    # just for testing
    def loglike(self, params):
        beta = params[:-1]
        sigma = params[-1]
        xb = np.dot(self.exog, beta)
        return stats.norm.logpdf(self.endog, loc=xb, scale=sigma).sum()

    def loglikeobs(self, params):
        beta = params[:-1]
        sigma = params[-1]
        xb = np.dot(self.exog, beta)
        return stats.norm.logpdf(self.endog, loc=xb, scale=sigma)

mod_norm2 = MygMLE(datal.endog, datal.exog)
#res_norm = mod_norm.fit(start_params=np.ones(datal.exog.shape[1]+1), method="nm", maxiter = 500)
res_norm2 = mod_norm2.fit(start_params=[1.]*datal.exog.shape[1]+[1], method="nm", maxiter = 500)
print(res_norm2.params)

res2 = sm.OLS(datal.endog, datal.exog).fit()
start_params = np.hstack((res2.params, np.sqrt(res2.mse_resid)))
res_norm3 = mod_norm2.fit(start_params=start_params, method="nm", maxiter = 500,
                          retall=0)
print(start_params)
print(res_norm3.params)
print(res2.bse)
#print res_norm3.bse   # not available
print('llf', res2.llf, res_norm3.llf)

bse = np.sqrt(np.diag(np.linalg.inv(res_norm3.model.hessian(res_norm3.params))))
res_norm3.model.score(res_norm3.params)

#fprime in fit option cannot be overwritten, set to None, when score is defined
# exception is fixed, but I don't think score was supposed to be called
'''
>>> mod_norm2.fit(start_params=start_params, method="bfgs", fprime=None, maxiter
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\s
tatsmodels\model.py", line 316, in fit
    disp=disp, retall=retall, callback=callback)
  File "C:\Josef\_progs\Subversion\scipy-trunk_after\trunk\dist\scipy-0.9.0.dev6
579.win32\Programs\Python25\Lib\site-packages\scipy\optimize\optimize.py", line
710, in fmin_bfgs
    gfk = myfprime(x0)
  File "C:\Josef\_progs\Subversion\scipy-trunk_after\trunk\dist\scipy-0.9.0.dev6
579.win32\Programs\Python25\Lib\site-packages\scipy\optimize\optimize.py", line
103, in function_wrapper
    return function(x, *args)
  File "c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\s
tatsmodels\model.py", line 240, in <lambda>
    score = lambda params: -self.score(params)
  File "c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\s
tatsmodels\model.py", line 480, in score
    return approx_fprime1(params, self.nloglike)
  File "c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\s
tatsmodels\sandbox\regression\numdiff.py", line 81, in approx_fprime1
    nobs = np.size(f0) #len(f0)
TypeError: object of type 'numpy.float64' has no len()

'''

res_bfgs = mod_norm2.fit(start_params=start_params, method="bfgs", fprime=None,
maxiter = 500, retall=0)

from statsmodels.tools.numdiff import approx_fprime, approx_hess
hb=-approx_hess(res_norm3.params, mod_norm2.loglike, epsilon=-1e-4)
hf=-approx_hess(res_norm3.params, mod_norm2.loglike, epsilon=1e-4)
hh = (hf+hb)/2.
print(np.linalg.eigh(hh))

grad = -approx_fprime(res_norm3.params, mod_norm2.loglike, epsilon=-1e-4)
print(grad)
gradb = -approx_fprime(res_norm3.params, mod_norm2.loglike, epsilon=-1e-4)
gradf = -approx_fprime(res_norm3.params, mod_norm2.loglike, epsilon=1e-4)
print((gradb+gradf)/2.)

print(res_norm3.model.score(res_norm3.params))
print(res_norm3.model.score(start_params))
mod_norm2.loglike(start_params/2.)
print(np.linalg.inv(-1*mod_norm2.hessian(res_norm3.params)))
print(np.sqrt(np.diag(res_bfgs.cov_params())))
print(res_norm3.bse)

print("MLE - OLS parameter estimates")
print(res_norm3.params[:-1] - res2.params)
print("bse diff in percent")
print((res_norm3.bse[:-1] / res2.bse)*100. - 100)

'''
C:\Programs\Python25\lib\site-packages\matplotlib-0.99.1-py2.5-win32.egg\matplotlib\rcsetup.py:117: UserWarning: rcParams key "numerix" is obsolete and has no effect;
 please delete it from your matplotlibrc file
  warnings.warn('rcParams key "numerix" is obsolete and has no effect;\n'
Optimization terminated successfully.
         Current function value: 12.818804
         Iterations 6
Optimization terminated successfully.
         Current function value: 12.818804
         Iterations: 439
         Function evaluations: 735
Optimization terminated successfully.
         Current function value: 12.818804
         Iterations: 439
         Function evaluations: 735
<statsmodels.model.LikelihoodModelResults object at 0x02131290>
[ 1.6258006   0.05172931  1.42632252 -7.45229732] [ 1.62581004  0.05172895  1.42633234 -7.45231965]
Warning: Maximum number of function evaluations has been exceeded.
[  -1.18109149  246.94438535  -16.21235536   24.05282629 -324.80867176
  274.07378453]
Warning: Maximum number of iterations has been exceeded
[  17.57107    -149.87528787   19.89079376  -72.49810777  -50.06067953
  306.14170418]
Optimization terminated successfully.
         Current function value: 506.488765
         Iterations: 339
         Function evaluations: 550
[  -3.08181404  234.34702702  -14.99684418   27.94090839 -237.1465136
  284.75079529]
[  -3.08181304  234.34701361  -14.99684381   27.94088692 -237.14649571
  274.6857294 ]
[   5.51471653   80.36595035    7.46933695   82.92232357  199.35166485]
llf -506.488764864 -506.488764864
Optimization terminated successfully.
         Current function value: 506.488765
         Iterations: 9
         Function evaluations: 13
         Gradient evaluations: 13
(array([  2.41772580e-05,   1.62492628e-04,   2.79438138e-04,
         1.90996240e-03,   2.07117946e-01,   1.28747174e+00]), array([[  1.52225754e-02,   2.01838216e-02,   6.90127235e-02,
         -2.57002471e-04,  -5.25941060e-01,  -8.47339404e-01],
       [  2.39797491e-01,  -2.32325602e-01,  -9.36235262e-01,
          3.02434938e-03,   3.95614029e-02,  -1.02035585e-01],
       [ -2.11381471e-02,   3.01074776e-02,   7.97208277e-02,
         -2.94955832e-04,   8.49402362e-01,  -5.20391053e-01],
       [ -1.55821981e-01,  -9.66926643e-01,   2.01517298e-01,
          1.52397702e-03,   4.13805882e-03,  -1.19878714e-02],
       [ -9.57881586e-01,   9.87911166e-02,  -2.67819451e-01,
          1.55192932e-03,  -1.78717579e-02,  -2.55757014e-02],
       [ -9.96486655e-04,  -2.03697290e-03,  -2.98130314e-03,
         -9.99992985e-01,  -1.71500426e-05,   4.70854949e-06]]))
[[ -4.91007768e-05  -7.28732630e-07  -2.51941401e-05  -2.50111043e-08
   -4.77484718e-08  -9.72022463e-08]]
[[ -1.64845915e-08  -2.87059265e-08  -2.88764568e-07  -6.82121026e-09
    2.84217094e-10  -1.70530257e-09]]
[ -4.90678076e-05  -6.71320777e-07  -2.46166110e-05  -1.13686838e-08
  -4.83169060e-08  -9.37916411e-08]
[ -4.56753924e-05  -6.50857146e-07  -2.31756303e-05  -1.70530257e-08
  -4.43378667e-08  -1.75592936e-02]
[[  2.99386348e+01  -1.24442928e+02   9.67254672e+00  -1.58968536e+02
   -5.91960010e+02  -2.48738183e+00]
 [ -1.24442928e+02   5.62972166e+03  -5.00079203e+02  -7.13057475e+02
   -7.82440674e+03  -1.05126925e+01]
 [  9.67254672e+00  -5.00079203e+02   4.87472259e+01   3.37373299e+00
    6.96960872e+02   7.69866589e-01]
 [ -1.58968536e+02  -7.13057475e+02   3.37373299e+00   6.82417837e+03
    4.84485862e+03   3.21440021e+01]
 [ -5.91960010e+02  -7.82440674e+03   6.96960872e+02   4.84485862e+03
    3.43753691e+04   9.37524459e+01]
 [ -2.48738183e+00  -1.05126925e+01   7.69866589e-01   3.21440021e+01
    9.37524459e+01   5.23915258e+02]]
>>> res_norm3.bse
array([   5.47162086,   75.03147114,    6.98192136,   82.60858536,
        185.40595756,   22.88919522])
>>> print res_norm3.model.score(res_norm3.params)
[ -4.90678076e-05  -6.71320777e-07  -2.46166110e-05  -1.13686838e-08
  -4.83169060e-08  -9.37916411e-08]
>>> print res_norm3.model.score(start_params)
[ -4.56753924e-05  -6.50857146e-07  -2.31756303e-05  -1.70530257e-08
  -4.43378667e-08  -1.75592936e-02]
>>> mod_norm2.loglike(start_params/2.)
-598.56178102781314
>>> print np.linalg.inv(-1*mod_norm2.hessian(res_norm3.params))
[[  2.99386348e+01  -1.24442928e+02   9.67254672e+00  -1.58968536e+02
   -5.91960010e+02  -2.48738183e+00]
 [ -1.24442928e+02   5.62972166e+03  -5.00079203e+02  -7.13057475e+02
   -7.82440674e+03  -1.05126925e+01]
 [  9.67254672e+00  -5.00079203e+02   4.87472259e+01   3.37373299e+00
    6.96960872e+02   7.69866589e-01]
 [ -1.58968536e+02  -7.13057475e+02   3.37373299e+00   6.82417837e+03
    4.84485862e+03   3.21440021e+01]
 [ -5.91960010e+02  -7.82440674e+03   6.96960872e+02   4.84485862e+03
    3.43753691e+04   9.37524459e+01]
 [ -2.48738183e+00  -1.05126925e+01   7.69866589e-01   3.21440021e+01
    9.37524459e+01   5.23915258e+02]]
>>> print np.sqrt(np.diag(res_bfgs.cov_params()))
[   5.10032831   74.34988912    6.96522122   76.7091604   169.8117832
   22.91695494]
>>> print res_norm3.bse
[   5.47162086   75.03147114    6.98192136   82.60858536  185.40595756
   22.88919522]
>>> res_norm3.conf_int
<bound method LikelihoodModelResults.conf_int of <statsmodels.model.LikelihoodModelResults object at 0x021317F0>>
>>> res_norm3.conf_int()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\model.py", line 993, in conf_int
    lower = self.params - dist.ppf(1-alpha/2,self.model.df_resid) *\
AttributeError: 'MygMLE' object has no attribute 'df_resid'

>>> res_norm3.params
array([  -3.08181304,  234.34701361,  -14.99684381,   27.94088692,
       -237.14649571,  274.6857294 ])
>>> res2.params
array([  -3.08181404,  234.34702702,  -14.99684418,   27.94090839,
       -237.1465136 ])
>>>
>>> res_norm3.params - res2.params
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: shape mismatch: objects cannot be broadcast to a single shape

>>> res_norm3.params[:-1] - res2.params
array([  9.96859735e-07,  -1.34122981e-05,   3.72278400e-07,
        -2.14645839e-05,   1.78919019e-05])
>>>
>>> res_norm3.bse[:-1] - res2.bse
array([ -0.04309567,  -5.33447922,  -0.48741559,  -0.31373822, -13.94570729])
>>> (res_norm3.bse[:-1] / res2.bse) - 1
array([-0.00781467, -0.06637735, -0.06525554, -0.00378352, -0.06995531])
>>> (res_norm3.bse[:-1] / res2.bse)*100. - 100
array([-0.7814667 , -6.6377355 , -6.52555369, -0.37835193, -6.99553089])
>>> np.sqrt(np.diag(np.linalg.inv(res_norm3.model.hessian(res_bfgs.params))))
array([ NaN,  NaN,  NaN,  NaN,  NaN,  NaN])
>>> np.sqrt(np.diag(np.linalg.inv(-res_norm3.model.hessian(res_bfgs.params))))
array([   5.10032831,   74.34988912,    6.96522122,   76.7091604 ,
        169.8117832 ,   22.91695494])
>>> res_norm3.bse
array([   5.47162086,   75.03147114,    6.98192136,   82.60858536,
        185.40595756,   22.88919522])
>>> res2.bse
array([   5.51471653,   80.36595035,    7.46933695,   82.92232357,
        199.35166485])
>>>
>>> bse_bfgs = np.sqrt(np.diag(np.linalg.inv(-res_norm3.model.hessian(res_bfgs.params))))
>>> (bse_bfgs[:-1] / res2.bse)*100. - 100
array([ -7.51422527,  -7.4858335 ,  -6.74913633,  -7.49275094, -14.8179759 ])
>>> hb=-approx_hess(res_bfgs.params, mod_norm2.loglike, epsilon=-1e-4)
>>> hf=-approx_hess(res_bfgs.params, mod_norm2.loglike, epsilon=1e-4)
>>> hh = (hf+hb)/2.
>>> bse_bfgs = np.sqrt(np.diag(np.linalg.inv(-hh)))
>>> bse_bfgs
array([ NaN,  NaN,  NaN,  NaN,  NaN,  NaN])
>>> bse_bfgs = np.sqrt(np.diag(np.linalg.inv(hh)))
>>> np.diag(hh)
array([  9.81680159e-01,   1.39920076e-02,   4.98101826e-01,
         3.60955710e-04,   9.57811608e-04,   1.90709670e-03])
>>> np.diag(np.inv(hh))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'module' object has no attribute 'inv'

>>> np.diag(np.linalg.inv(hh))
array([  2.64875153e+01,   5.91578496e+03,   5.13279911e+01,
         6.11533345e+03,   3.33775960e+04,   5.24357391e+02])
>>> res2.bse**2
array([  3.04120984e+01,   6.45868598e+03,   5.57909945e+01,
         6.87611175e+03,   3.97410863e+04])
>>> bse_bfgs
array([   5.14660231,   76.91414015,    7.1643556 ,   78.20059751,
        182.69536402,   22.89885131])
>>> bse_bfgs - res_norm3.bse
array([-0.32501855,  1.88266901,  0.18243424, -4.40798785, -2.71059354,
        0.00965609])
>>> (bse_bfgs[:-1] / res2.bse)*100. - 100
array([-6.67512508, -4.29511526, -4.0831115 , -5.69415552, -8.35523538])
>>> (res_norm3.bse[:-1] / res2.bse)*100. - 100
array([-0.7814667 , -6.6377355 , -6.52555369, -0.37835193, -6.99553089])
>>> (bse_bfgs / res_norm3.bse)*100. - 100
array([-5.94007812,  2.50917247,  2.61295176, -5.33599242, -1.46197759,
        0.04218624])
>>> bse_bfgs
array([   5.14660231,   76.91414015,    7.1643556 ,   78.20059751,
        182.69536402,   22.89885131])
>>> res_norm3.bse
array([   5.47162086,   75.03147114,    6.98192136,   82.60858536,
        185.40595756,   22.88919522])
>>> res2.bse
array([   5.51471653,   80.36595035,    7.46933695,   82.92232357,
        199.35166485])
>>> dir(res_bfgs)
['__class__', '__delattr__', '__dict__', '__doc__', '__getattribute__', '__hash__', '__init__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__str__', '__weakref__', 'bse', 'conf_int', 'cov_params', 'f_test', 'initialize', 'llf', 'mle_retvals', 'mle_settings', 'model', 'normalized_cov_params', 'params', 'scale', 't', 't_test']
>>> res_bfgs.scale
1.0
>>> res2.scale
81083.015420213851
>>> res2.mse_resid
81083.015420213851
>>> print np.sqrt(np.diag(np.linalg.inv(-1*mod_norm2.hessian(res_bfgs.params))))
[   5.10032831   74.34988912    6.96522122   76.7091604   169.8117832
   22.91695494]
>>> print np.sqrt(np.diag(np.linalg.inv(-1*res_bfgs.model.hessian(res_bfgs.params))))
[   5.10032831   74.34988912    6.96522122   76.7091604   169.8117832
   22.91695494]

Is scale a misnomer, actually scale squared, i.e. variance of error term ?
'''

print(res_norm3.model.score_obs(res_norm3.params).shape)

jac = res_norm3.model.score_obs(res_norm3.params)
print(np.sqrt(np.diag(np.dot(jac.T, jac)))/start_params)
jac2 = res_norm3.model.score_obs(res_norm3.params, centered=True)

print(np.sqrt(np.diag(np.linalg.inv(np.dot(jac.T, jac)))))
print(res_norm3.bse)
print(res2.bse)
