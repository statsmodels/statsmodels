
import numpy as np
from scipy import stats
import scikits.statsmodels as sm
from scikits.statsmodels.model import GenericLikelihoodModel


data = sm.datasets.spector.load()
data.exog = sm.add_constant(data.exog)
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
print res


#np.allclose(res.params, probit_res.params)

print res.params, probit_res.params

#datal = sm.datasets.longley.load()
datal = sm.datasets.ccard.load()
datal.exog = sm.add_constant(datal.exog)
# Instance of GenericLikelihood model doesn't work directly, because loglike
# cannot get access to data in self.endog, self.exog

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

    print res_norm.params

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

    print res_norm.params

class MygMLE(GenericLikelihoodModel):
    # just for testing
    def loglike(self, params):
        beta = params[:-1]
        sigma = params[-1]
        xb = np.dot(self.exog, beta)
        return stats.norm.logpdf(self.endog, loc=xb, scale=sigma).sum()

mod_norm2 = MygMLE(datal.endog, datal.exog)
#res_norm = mod_norm.fit(start_params=np.ones(datal.exog.shape[1]+1), method="nm", maxiter = 500)
res_norm2 = mod_norm2.fit(start_params=[1.]*datal.exog.shape[1]+[1], method="nm", maxiter = 500)
print res_norm2.params

res2 = sm.OLS(datal.endog, datal.exog).fit()
start_params = np.hstack((res2.params, np.sqrt(res2.mse_resid)))
res_norm3 = mod_norm2.fit(start_params=start_params, method="nm", maxiter = 500,
                          retall=0)
print start_params
print res_norm3.params
print res2.bse
#print res_norm3.bse   # not available
print 'llf', res2.llf, res_norm3.llf

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

from scikits.statsmodels.sandbox.regression.numdiff import approx_fprime1, approx_hess
hb=-approx_hess(res_norm3.params, mod_norm2.loglike, epsilon=-1e-4)[0]
hf=-approx_hess(res_norm3.params, mod_norm2.loglike, epsilon=1e-4)[0]
hh = (hf+hb)/2.
print np.linalg.eigh(hh)

grad = -approx_fprime1(res_norm3.params, mod_norm2.loglike, epsilon=-1e-4)
print grad
gradb = -approx_fprime1(res_norm3.params, mod_norm2.loglike, epsilon=-1e-4)
gradf = -approx_fprime1(res_norm3.params, mod_norm2.loglike, epsilon=1e-4)
print (gradb+gradf)/2.

print res_norm3.model.score(res_norm3.params)
print res_norm3.model.score(start_params)
mod_norm2.loglike(start_params/2.)
print np.linalg.inv(-1*mod_norm2.hessian(res_norm3.params))
print np.sqrt(np.diag(res_bfgs.cov_params()))
print res_norm3.bse

print "MLE - OLS parameter estimates"
print res_norm3.params[:-1] - res2.params
