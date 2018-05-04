# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 08:28:04 2010

Author: josef-pktd
"""


from __future__ import print_function
from statsmodels.compat.python import zip
import numpy as np

from scipy import stats, special, optimize
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

#global
store_params = []

class MyT(GenericLikelihoodModel):
    '''Maximum Likelihood Estimation of Linear Model with t-distributed errors

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
        store_params.append(params)
        if not self.fixed_params is None:
            #print 'using fixed'
            params = self.expandparams(params)

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
nvars = 6
df = 5
rvs = np.random.randn(nobs, nvars-1)
data_exog = sm.add_constant(rvs, prepend=False)
xbeta = 0.9 + 0.1*rvs.sum(1)
data_endog = xbeta + 0.1*np.random.standard_t(df, size=nobs)
print(data_endog.var())

res_ols = sm.OLS(data_endog, data_exog).fit()
print(res_ols.scale)
print(np.sqrt(res_ols.scale))
print(res_ols.params)
kurt = stats.kurtosis(res_ols.resid)
df_fromkurt = 6./kurt + 4
print(stats.t.stats(df_fromkurt, moments='mvsk'))
print(stats.t.stats(df, moments='mvsk'))

modp = MyT(data_endog, data_exog)
start_value = 0.1*np.ones(data_exog.shape[1]+2)
#start_value = np.zeros(data_exog.shape[1]+2)
#start_value[:nvars] = sm.OLS(data_endog, data_exog).fit().params
start_value[:nvars] = res_ols.params
start_value[-2] = df_fromkurt #10
start_value[-1] = np.sqrt(res_ols.scale) #0.5
modp.start_params = start_value

#adding fixed parameters

fixdf = np.nan * np.zeros(modp.start_params.shape)
fixdf[-2] = 100

fixone = 0
if fixone:
    modp.fixed_params = fixdf
    modp.fixed_paramsmask = np.isnan(fixdf)
    modp.start_params = modp.start_params[modp.fixed_paramsmask]
else:
    modp.fixed_params = None
    modp.fixed_paramsmask = None


resp = modp.fit(start_params = modp.start_params, disp=1, method='nm')#'newton')
#resp = modp.fit(start_params = modp.start_params, disp=1, method='newton')
print('\nestimation results t-dist')
print(resp.params)
print(resp.bse)
resp2 = modp.fit(start_params = resp.params, method='Newton')
print('using Newton')
print(resp2.params)
print(resp2.bse)

from statsmodels.tools.numdiff import approx_fprime, approx_hess

hb=-approx_hess(modp.start_params, modp.loglike, epsilon=-1e-4)
tmp = modp.loglike(modp.start_params)
print(tmp.shape)
#np.linalg.eigh(np.linalg.inv(hb))[0]

pp=np.array(store_params)
print(pp.min(0))
print(pp.max(0))




##################### Example: Pareto
# estimating scale doesn't work yet, a bug somewhere ?
# fit_ks works well, but no bse or other result statistics yet





#import for kstest based estimation
#should be replace
import statsmodels.sandbox.distributions.sppatch

class MyPareto(GenericLikelihoodModel):
    '''Maximum Likelihood Estimation pareto distribution

    first version: iid case, with constant parameters
    '''

    #copied from stats.distribution
    def pdf(self, x, b):
        return b * x**(-b-1)

    def loglike(self, params):
        return -self.nloglikeobs(params).sum(0)

    def nloglikeobs(self, params):
        #print params.shape
        if not self.fixed_params is None:
            #print 'using fixed'
            params = self.expandparams(params)
        b = params[0]
        loc = params[1]
        scale = params[2]
        #loc = np.dot(self.exog, beta)
        endog = self.endog
        x = (endog - loc)/scale
        logpdf = np_log(b) - (b+1.)*np_log(x)  #use np_log(1 + x) for Pareto II
        logpdf -= np.log(scale)
        #lb = loc + scale
        #logpdf[endog<lb] = -inf
        #import pdb; pdb.set_trace()
        logpdf[x<1] = -10000 #-np.inf
        return -logpdf

    def fit_ks(self):
        '''fit Pareto with nested optimization

        originally published on stackoverflow
        this doesn't trim lower values during ks optimization

        '''
        rvs = self.endog
        rvsmin = rvs.min()
        fixdf = np.nan * np.ones(3)
        self.fixed_params = fixdf
        self.fixed_paramsmask = np.isnan(fixdf)

        def pareto_ks(loc, rvs):
            #start_scale = rvs.min() - loc # not used yet
            #est = self.fit_fr(rvs, 1., frozen=[np.nan, loc, np.nan])
            self.fixed_params[1] = loc
            est = self.fit(start_params=self.start_params[self.fixed_paramsmask]).params
            #est = self.fit(start_params=self.start_params, method='nm').params
            args = (est[0], loc, est[1])
            return stats.kstest(rvs,'pareto',args)[0]

        locest = optimize.fmin(pareto_ks, rvsmin - 1.5, (rvs,))
        est = stats.pareto.fit_fr(rvs, 0., frozen=[np.nan, locest, np.nan])
        args = (est[0], locest[0], est[1])
        return args


    def fit_ks1_trim(self):
        '''fit Pareto with nested optimization

        originally published on stackoverflow

        '''
        self.nobs = self.endog.shape[0]
        rvs = np.sort(self.endog)
        rvsmin = rvs.min()

        def pareto_ks(loc, rvs):
            #start_scale = rvs.min() - loc # not used yet
            est = stats.pareto.fit_fr(rvs, frozen=[np.nan, loc, np.nan])
            args = (est[0], loc, est[1])
            return stats.kstest(rvs,'pareto',args)[0]

        #locest = optimize.fmin(pareto_ks, rvsmin*0.7, (rvs,))
        maxind = min(np.floor(self.nobs*0.95).astype(int), self.nobs-10)
        res = []
        for trimidx in range(self.nobs//2, maxind):
            xmin = loc = rvs[trimidx]
            res.append([trimidx, pareto_ks(loc-1e-10, rvs[trimidx:])])
        res = np.array(res)
        bestidx = res[np.argmin(res[:,1]),0].astype(int)
        print(bestidx)
        locest = rvs[bestidx]

        est = stats.pareto.fit_fr(rvs[bestidx:], 1., frozen=[np.nan, locest, np.nan])
        args = (est[0], locest, est[1])
        return args

    def fit_ks1(self):
        '''fit Pareto with nested optimization

        originally published on stackoverflow

        '''
        rvs = self.endog
        rvsmin = rvs.min()

        def pareto_ks(loc, rvs):
            #start_scale = rvs.min() - loc # not used yet
            est = stats.pareto.fit_fr(rvs, 1., frozen=[np.nan, loc, np.nan])
            args = (est[0], loc, est[1])
            return stats.kstest(rvs,'pareto',args)[0]

        #locest = optimize.fmin(pareto_ks, rvsmin*0.7, (rvs,))
        locest = optimize.fmin(pareto_ks, rvsmin - 1.5, (rvs,))
        est = stats.pareto.fit_fr(rvs, 1., frozen=[np.nan, locest, np.nan])
        args = (est[0], locest[0], est[1])
        return args

#y = stats.pareto.rvs(1, loc=10, scale=2, size=nobs)
y = stats.pareto.rvs(1, loc=0, scale=2, size=nobs)


par_start_params = np.array([1., 9., 2.])

mod_par = MyPareto(y)
mod_par.start_params = np.array([1., 10., 2.])
mod_par.start_params = np.array([1., -9., 2.])
mod_par.fixed_params = None

fixdf = np.nan * np.ones(mod_par.start_params.shape)
fixdf[1] = 9.9
#fixdf[2] = 2.
fixone = 0
if fixone:
    mod_par.fixed_params = fixdf
    mod_par.fixed_paramsmask = np.isnan(fixdf)
    mod_par.start_params = mod_par.start_params[mod_par.fixed_paramsmask]
    mod_par.df_model = 2
    mod_par.df_resid = mod_par.endog.shape[0] - mod_par.df_model
    mod_par.data.xnames = ['shape', 'scale']
else:
    mod_par.fixed_params = None
    mod_par.fixed_paramsmask = None
    mod_par.df_model = 3
    mod_par.df_resid = mod_par.endog.shape[0] - mod_par.df_model
    mod_par.data.xnames = ['shape', 'loc', 'scale']

res_par = mod_par.fit(start_params=mod_par.start_params, method='nm', maxfun=10000, maxiter=5000)
#res_par2 = mod_par.fit(start_params=res_par.params, method='newton', maxfun=10000, maxiter=5000)

res_parks = mod_par.fit_ks1()

print(res_par.params)
#print res_par2.params
print(res_parks)

print(res_par.params[1:].sum(), sum(res_parks[1:]), mod_par.endog.min())

#start new model, so we don't get two result instances with the same model instance
mod_par = MyPareto(y)
mod_par.fixed_params = fixdf
mod_par.fixed_paramsmask = np.isnan(fixdf)
mod_par.df_model = mod_par.fixed_paramsmask.sum()
mod_par.df_resid = mod_par.endog.shape[0] - mod_par.df_model
#mod_par.data.xnames = np.array(['shape', 'loc', 'scale'])[mod_par.fixed_paramsmask].tolist() # works also
mod_par.data.xnames = [name for (name, incl) in zip(['shape', 'loc', 'scale'], mod_par.fixed_paramsmask) if incl]

res_par3 = mod_par.start_params = par_start_params[mod_par.fixed_paramsmask]
res5 = mod_par.fit(start_params=mod_par.start_params)
##res_parks2 = mod_par.fit_ks()
##
##res_parkst = mod_par.fit_ks1_trim()
##print res_parkst

print(res5.summary())
print(res5.t_test([[1,0]]))

'''
C:\Programs\Python25\lib\site-packages\matplotlib-0.99.1-py2.5-win32.egg\matplotlib\rcsetup.py:117: UserWarning: rcParams key "numerix" is obsolete and has no effect;
 please delete it from your matplotlibrc file
  warnings.warn('rcParams key "numerix" is obsolete and has no effect;\n'
0.0686702747648
0.0164150896481
0.128121386381
[ 0.10370428  0.09921315  0.09676723  0.10457413  0.10201618  0.89964496]
(array(0.0), array(1.4552599885729831), array(0.0), array(2.5072143354058238))
(array(0.0), array(1.6666666666666667), array(0.0), array(6.0))
repr(start_params) array([ 0.10370428,  0.09921315,  0.09676723,  0.10457413,  0.10201618,
        0.89964496,  6.39309417,  0.12812139])
Optimization terminated successfully.
         Current function value: -679.951339
         Iterations: 398
         Function evaluations: 609

estimation results t-dist
[ 0.10400826  0.10111893  0.09725133  0.10507788  0.10086163  0.8996041
  4.72131318  0.09825355]
[ 0.00365493  0.00356149  0.00349329  0.00362333  0.003732    0.00362716
  0.7232824   0.00388829]
repr(start_params) array([ 0.10400826,  0.10111893,  0.09725133,  0.10507788,  0.10086163,
        0.8996041 ,  4.72131318,  0.09825355])
Optimization terminated successfully.
         Current function value: -679.950443
         Iterations 3
using Newton
[ 0.10395383  0.10106762  0.09720665  0.10503384  0.10080599  0.89954546
  4.70918964  0.09815885]
[ 0.00365299  0.00355968  0.00349147  0.00362166  0.00373015  0.00362533
  0.72014031  0.00388434]
()
[ 0.09992709  0.09786601  0.09387356  0.10229919  0.09756623  0.85466272
  4.60459182  0.09661986]
[ 0.11308292  0.10828401  0.1028508   0.11268895  0.10934726  0.94462721
  7.15412655  0.13452746]
repr(start_params) array([ 1.,  2.])
Warning: Maximum number of function evaluations has been exceeded.
>>> res_par.params
array([  7.42705803e+152,   2.17339053e+153])
>>> mod_par.loglike(mod_p.start_params)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'mod_p' is not defined

>>> mod_par.loglike(mod_par.start_params)
-1085.1993430947232
>>> np.log(mod_par.pdf(mod_par.start_params))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: pdf() takes exactly 3 arguments (2 given)

>>> np.log(mod_par.pdf(*mod_par.start_params))
0.69314718055994529
>>> mod_par.loglike(*mod_par.start_params)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: loglike() takes exactly 2 arguments (3 given)

>>> mod_par.loglike(mod_par.start_params)
-1085.1993430947232
>>> np.log(stats.pareto.pdf(y[0],*mod_par.start_params))
-4.6414308627431353
>>> mod_par.loglike(mod_par.start_params)
-1085.1993430947232
>>> mod_par.nloglikeobs(mod_par.start_params)[0]
0.29377232943845044
>>> mod_par.start_params
array([ 1.,  2.])
>>> np.log(stats.pareto.pdf(y[0],1,9.5,2))
-1.2806918394368461
>>> mod_par.fixed_params= None
>>> mod_par.nloglikeobs(np.array([1., 10., 2.]))[0]
0.087533156771285828
>>> y[0]
12.182956907488885
>>> mod_para.endog[0]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'mod_para' is not defined

>>> mod_par.endog[0]
12.182956907488885
>>> np.log(stats.pareto.pdf(y[0],1,10,2))
-0.86821349410251702
>>> np.log(stats.pareto.pdf(y[0],1.,10.,2.))
-0.86821349410251702
>>> stats.pareto.pdf(y[0],1.,10.,2.)
0.41970067762301644
>>> mod_par.loglikeobs(np.array([1., 10., 2.]))[0]
-0.087533156771285828
>>>
'''

'''
>>> mod_par.nloglikeobs(np.array([1., 10., 2.]))[0]
0.86821349410251691
>>> np.log(stats.pareto.pdf(y,1.,10.,2.)).sum()
-2627.9403758026938
'''


#'''
#C:\Programs\Python25\lib\site-packages\matplotlib-0.99.1-py2.5-win32.egg\matplotlib\rcsetup.py:117: UserWarning: rcParams key "numerix" is obsolete and has no effect;
# please delete it from your matplotlibrc file
#  warnings.warn('rcParams key "numerix" is obsolete and has no effect;\n'
#0.0686702747648
#0.0164150896481
#0.128121386381
#[ 0.10370428  0.09921315  0.09676723  0.10457413  0.10201618  0.89964496]
#(array(0.0), array(1.4552599885729827), array(0.0), array(2.5072143354058203))
#(array(0.0), array(1.6666666666666667), array(0.0), array(6.0))
#repr(start_params) array([ 0.10370428,  0.09921315,  0.09676723,  0.10457413,  0.10201618,
#        0.89964496,  6.39309417,  0.12812139])
#Optimization terminated successfully.
#         Current function value: -679.951339
#         Iterations: 398
#         Function evaluations: 609
#
#estimation results t-dist
#[ 0.10400826  0.10111893  0.09725133  0.10507788  0.10086163  0.8996041
#  4.72131318  0.09825355]
#[ 0.00365493  0.00356149  0.00349329  0.00362333  0.003732    0.00362716
#  0.72325227  0.00388822]
#repr(start_params) array([ 0.10400826,  0.10111893,  0.09725133,  0.10507788,  0.10086163,
#        0.8996041 ,  4.72131318,  0.09825355])
#Optimization terminated successfully.
#         Current function value: -679.950443
#         Iterations 3
#using Newton
#[ 0.10395383  0.10106762  0.09720665  0.10503384  0.10080599  0.89954546
#  4.70918964  0.09815885]
#[ 0.00365299  0.00355968  0.00349147  0.00362166  0.00373015  0.00362533
#  0.72014669  0.00388436]
#()
#[ 0.09992709  0.09786601  0.09387356  0.10229919  0.09756623  0.85466272
#  4.60459182  0.09661986]
#[ 0.11308292  0.10828401  0.1028508   0.11268895  0.10934726  0.94462721
#  7.15412655  0.13452746]
#repr(start_params) array([ 1.,  2.])
#Warning: Maximum number of function evaluations has been exceeded.
#repr(start_params) array([  3.06504406e+302,   3.29325579e+303])
#Traceback (most recent call last):
#  File "C:\Josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\examples\ex_generic_mle_tdist.py", line 222, in <module>
#    res_par2 = mod_par.fit(start_params=res_par.params, method='newton', maxfun=10000, maxiter=5000)
#  File "c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\model.py", line 547, in fit
#    disp=disp, callback=callback, **kwargs)
#  File "c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\model.py", line 262, in fit
#    newparams = oldparams - np.dot(np.linalg.inv(H),
#  File "C:\Programs\Python25\lib\site-packages\numpy\linalg\linalg.py", line 423, in inv
#    return wrap(solve(a, identity(a.shape[0], dtype=a.dtype)))
#  File "C:\Programs\Python25\lib\site-packages\numpy\linalg\linalg.py", line 306, in solve
#    raise LinAlgError, 'Singular matrix'
#numpy.linalg.linalg.LinAlgError: Singular matrix
#
#>>> mod_par.fixed_params
#array([ NaN,  10.,  NaN])
#>>> mod_par.start_params
#array([ 1.,  2.])
#>>> np.source(stats.pareto.fit_fr)
#In file: c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\sandbox\stats\distributions_patch.py
#
#def fit_fr(self, data, *args, **kwds):
#    '''estimate distribution parameters by MLE taking some parameters as fixed
#
#    Parameters
#    ----------
#    data : array, 1d
#        data for which the distribution parameters are estimated,
#    args : list ? check
#        starting values for optimization
#    kwds :
#
#      - 'frozen' : array_like
#           values for frozen distribution parameters and, for elements with
#           np.nan, the corresponding parameter will be estimated
#
#    Returns
#    -------
#    argest : array
#        estimated parameters
#
#
#    Examples
#    --------
#    generate random sample
#    >>> np.random.seed(12345)
#    >>> x = stats.gamma.rvs(2.5, loc=0, scale=1.2, size=200)
#
#    estimate all parameters
#    >>> stats.gamma.fit(x)
#    array([ 2.0243194 ,  0.20395655,  1.44411371])
#    >>> stats.gamma.fit_fr(x, frozen=[np.nan, np.nan, np.nan])
#    array([ 2.0243194 ,  0.20395655,  1.44411371])
#
#    keep loc fixed, estimate shape and scale parameters
#    >>> stats.gamma.fit_fr(x, frozen=[np.nan, 0.0, np.nan])
#    array([ 2.45603985,  1.27333105])
#
#    keep loc and scale fixed, estimate shape parameter
#    >>> stats.gamma.fit_fr(x, frozen=[np.nan, 0.0, 1.0])
#    array([ 3.00048828])
#    >>> stats.gamma.fit_fr(x, frozen=[np.nan, 0.0, 1.2])
#    array([ 2.57792969])
#
#    estimate only scale parameter for fixed shape and loc
#    >>> stats.gamma.fit_fr(x, frozen=[2.5, 0.0, np.nan])
#    array([ 1.25087891])
#
#    Notes
#    -----
#    self is an instance of a distribution class. This can be attached to
#    scipy.stats.distributions.rv_continuous
#
#    *Todo*
#
#    * check if docstring is correct
#    * more input checking, args is list ? might also apply to current fit method
#
#    '''
#    loc0, scale0 = map(kwds.get, ['loc', 'scale'],[0.0, 1.0])
#    Narg = len(args)
#
#    if Narg == 0 and hasattr(self, '_fitstart'):
#        x0 = self._fitstart(data)
#    elif Narg > self.numargs:
#        raise ValueError("Too many input arguments.")
#    else:
#        args += (1.0,)*(self.numargs-Narg)
#        # location and scale are at the end
#        x0 = args + (loc0, scale0)
#
#    if 'frozen' in kwds:
#        frmask = np.array(kwds['frozen'])
#        if len(frmask) != self.numargs+2:
#            raise ValueError("Incorrect number of frozen arguments.")
#        else:
#            # keep starting values for not frozen parameters
#            x0  = np.array(x0)[np.isnan(frmask)]
#    else:
#        frmask = None
#
#    #print x0
#    #print frmask
#    return optimize.fmin(self.nnlf_fr, x0,
#                args=(np.ravel(data), frmask), disp=0)
#
#>>> stats.pareto.fit_fr(y, 1., frozen=[np.nan, loc, np.nan])
#Traceback (most recent call last):
#  File "<stdin>", line 1, in <module>
#NameError: name 'loc' is not defined
#
#>>> stats.pareto.fit_fr(y, 1., frozen=[np.nan, 10., np.nan])
#array([ 1.0346268 ,  2.00184808])
#>>> stats.pareto.fit_fr(y, (1.,2), frozen=[np.nan, 10., np.nan])
#Traceback (most recent call last):
#  File "<stdin>", line 1, in <module>
#  File "c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\sandbox\stats\distributions_patch.py", line 273, in fit_fr
#    x0  = np.array(x0)[np.isnan(frmask)]
#ValueError: setting an array element with a sequence.
#
#>>> stats.pareto.fit_fr(y, [1.,2], frozen=[np.nan, 10., np.nan])
#Traceback (most recent call last):
#  File "<stdin>", line 1, in <module>
#  File "c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\sandbox\stats\distributions_patch.py", line 273, in fit_fr
#    x0  = np.array(x0)[np.isnan(frmask)]
#ValueError: setting an array element with a sequence.
#
#>>> stats.pareto.fit_fr(y, frozen=[np.nan, 10., np.nan])
#array([ 1.03463526,  2.00184809])
#>>> stats.pareto.pdf(y, 1.03463526, 10, 2.00184809).sum()
#173.33947284555239
#>>> mod_par(1.03463526, 10, 2.00184809)
#Traceback (most recent call last):
#  File "<stdin>", line 1, in <module>
#TypeError: 'MyPareto' object is not callable
#
#>>> mod_par.loglike(1.03463526, 10, 2.00184809)
#Traceback (most recent call last):
#  File "<stdin>", line 1, in <module>
#TypeError: loglike() takes exactly 2 arguments (4 given)
#
#>>> mod_par.loglike((1.03463526, 10, 2.00184809))
#-962.21623668859741
#>>> np.log(stats.pareto.pdf(y, 1.03463526, 10, 2.00184809)).sum()
#-inf
#>>> np.log(stats.pareto.pdf(y, 1.03463526, 9, 2.00184809)).sum()
#-3074.5947476137271
#>>> np.log(stats.pareto.pdf(y, 1.03463526, 10., 2.00184809)).sum()
#-inf
#>>> np.log(stats.pareto.pdf(y, 1.03463526, 9.9, 2.00184809)).sum()
#-2677.3867091635661
#>>> y.min()
#12.001848089426717
#>>> np.log(stats.pareto.pdf(y, 1.03463526, loc=9.9, scale=2.00184809)).sum()
#-2677.3867091635661
#>>> np.log(stats.pareto.pdf(y, 1.03463526, loc=10., scale=2.00184809)).sum()
#-inf
#>>> stats.pareto.logpdf(y, 1.03463526, loc=10., scale=2.00184809).sum()
#-inf
#>>> stats.pareto.logpdf(y, 1.03463526, loc=9.99, scale=2.00184809).sum()
#-2631.6120098202355
#>>> mod_par.loglike((1.03463526, 9.99, 2.00184809))
#-963.2513896113644
#>>> maxabs(y, mod_par.endog)
#0.0
#>>> np.source(stats.pareto.logpdf)
#In file: C:\Josef\_progs\Subversion\scipy-trunk_after\trunk\dist\scipy-0.9.0.dev6579.win32\Programs\Python25\Lib\site-packages\scipy\stats\distributions.py
#
#    def logpdf(self, x, *args, **kwds):
#        """
#        Log of the probability density function at x of the given RV.
#
#        This uses more numerically accurate calculation if available.
#
#        Parameters
#        ----------
#        x : array-like
#            quantiles
#        arg1, arg2, arg3,... : array-like
#            The shape parameter(s) for the distribution (see docstring of the
#            instance object for more information)
#        loc : array-like, optional
#            location parameter (default=0)
#        scale : array-like, optional
#            scale parameter (default=1)
#
#        Returns
#        -------
#        logpdf : array-like
#            Log of the probability density function evaluated at x
#
#        """
#        loc,scale=map(kwds.get,['loc','scale'])
#        args, loc, scale = self._fix_loc_scale(args, loc, scale)
#        x,loc,scale = map(arr,(x,loc,scale))
#        args = tuple(lmap(arr,args))
#        x = arr((x-loc)*1.0/scale)
#        cond0 = self._argcheck(*args) & (scale > 0)
#        cond1 = (scale > 0) & (x >= self.a) & (x <= self.b)
#        cond = cond0 & cond1
#        output = empty(shape(cond),'d')
#        output.fill(NINF)
#        putmask(output,(1-cond0)*array(cond1,bool),self.badvalue)
#        goodargs = argsreduce(cond, *((x,)+args+(scale,)))
#        scale, goodargs = goodargs[-1], goodargs[:-1]
#        place(output,cond,self._logpdf(*goodargs) - log(scale))
#        if output.ndim == 0:
#            return output[()]
#        return output
#
#>>> np.source(stats.pareto._logpdf)
#In file: C:\Josef\_progs\Subversion\scipy-trunk_after\trunk\dist\scipy-0.9.0.dev6579.win32\Programs\Python25\Lib\site-packages\scipy\stats\distributions.py
#
#    def _logpdf(self, x, *args):
#        return log(self._pdf(x, *args))
#
#>>> np.source(stats.pareto._pdf)
#In file: C:\Josef\_progs\Subversion\scipy-trunk_after\trunk\dist\scipy-0.9.0.dev6579.win32\Programs\Python25\Lib\site-packages\scipy\stats\distributions.py
#
#    def _pdf(self, x, b):
#        return b * x**(-b-1)
#
#>>> stats.pareto.a
#1.0
#>>> (1-loc)/scale
#Traceback (most recent call last):
#  File "<stdin>", line 1, in <module>
#NameError: name 'loc' is not defined
#
#>>> b, loc, scale = (1.03463526, 9.99, 2.00184809)
#>>> (1-loc)/scale
#-4.4908502522786327
#>>> (x-loc)/scale == 1
#Traceback (most recent call last):
#  File "<stdin>", line 1, in <module>
#NameError: name 'x' is not defined
#
#>>> (lb-loc)/scale == 1
#Traceback (most recent call last):
#  File "<stdin>", line 1, in <module>
#NameError: name 'lb' is not defined
#
#>>> lb = scale + loc
#>>> lb
#11.991848090000001
#>>> (lb-loc)/scale == 1
#False
#>>> (lb-loc)/scale
#1.0000000000000004
#>>>
#'''

'''
repr(start_params) array([  1.,  10.,   2.])
Optimization terminated successfully.
         Current function value: 2626.436870
         Iterations: 102
         Function evaluations: 210
Optimization terminated successfully.
         Current function value: 0.016555
         Iterations: 16
         Function evaluations: 35
[  1.03482659  10.00737039   1.9944777 ]
(1.0596088578825995, 9.9043376069230007, 2.0975104813987118)
>>> 9.9043376069230007 + 2.0975104813987118
12.001848088321712
>>> y.min()
12.001848089426717

'''

'''
C:\Programs\Python25\lib\site-packages\matplotlib-0.99.1-py2.5-win32.egg\matplotlib\rcsetup.py:117: UserWarning: rcParams key "numerix" is obsolete and has no effect;
 please delete it from your matplotlibrc file
  warnings.warn('rcParams key "numerix" is obsolete and has no effect;\n'
0.0686702747648
0.0164150896481
0.128121386381
[ 0.10370428  0.09921315  0.09676723  0.10457413  0.10201618  0.89964496]
(array(0.0), array(1.4552599885729829), array(0.0), array(2.5072143354058221))
(array(0.0), array(1.6666666666666667), array(0.0), array(6.0))
repr(start_params) array([ 0.10370428,  0.09921315,  0.09676723,  0.10457413,  0.10201618,
        0.89964496,  6.39309417,  0.12812139])
Optimization terminated successfully.
         Current function value: -679.951339
         Iterations: 398
         Function evaluations: 609

estimation results t-dist
[ 0.10400826  0.10111893  0.09725133  0.10507788  0.10086163  0.8996041
  4.72131318  0.09825355]
[ 0.00365493  0.00356149  0.00349329  0.00362333  0.003732    0.00362716
  0.72329352  0.00388832]
repr(start_params) array([ 0.10400826,  0.10111893,  0.09725133,  0.10507788,  0.10086163,
        0.8996041 ,  4.72131318,  0.09825355])
Optimization terminated successfully.
         Current function value: -679.950443
         Iterations 3
using Newton
[ 0.10395383  0.10106762  0.09720665  0.10503384  0.10080599  0.89954546
  4.70918964  0.09815885]
[ 0.00365299  0.00355968  0.00349147  0.00362166  0.00373015  0.00362533
  0.7201488   0.00388437]
()
[ 0.09992709  0.09786601  0.09387356  0.10229919  0.09756623  0.85466272
  4.60459182  0.09661986]
[ 0.11308292  0.10828401  0.1028508   0.11268895  0.10934726  0.94462721
  7.15412655  0.13452746]
repr(start_params) array([ 1.,  9.,  2.])
Optimization terminated successfully.
         Current function value: 2636.129089
         Iterations: 147
         Function evaluations: 279
Optimization terminated successfully.
         Current function value: 0.016555
         Iterations: 16
         Function evaluations: 35
[  0.84856418  10.2197801    1.78206799]
(1.0596088578825995, 9.9043376069230007, 2.0975104813987118)
12.0018480891 12.0018480883 12.0018480894
repr(start_params) array([ 1.,  2.])
Warning: Desired error not necessarily achieveddue to precision loss
         Current function value: 2643.549907
         Iterations: 2
         Function evaluations: 13
         Gradient evaluations: 12
>>> res_parks2 = mod_par.fit_ks()
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2642.465273
         Iterations: 92
         Function evaluations: 172
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2636.639863
         Iterations: 73
         Function evaluations: 136
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2631.568778
         Iterations: 75
         Function evaluations: 133
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2627.821044
         Iterations: 75
         Function evaluations: 135
repr(start_params) array([ 1.,  2.])
Warning: Maximum number of function evaluations has been exceeded.
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2631.568778
         Iterations: 75
         Function evaluations: 133
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.431596
         Iterations: 58
         Function evaluations: 109
repr(start_params) array([ 1.,  2.])
Warning: Maximum number of function evaluations has been exceeded.
repr(start_params) array([ 1.,  2.])
Warning: Maximum number of function evaluations has been exceeded.
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.737426
         Iterations: 60
         Function evaluations: 109
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2627.821044
         Iterations: 75
         Function evaluations: 135
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.471666
         Iterations: 48
         Function evaluations: 94
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2627.196314
         Iterations: 66
         Function evaluations: 119
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.578538
         Iterations: 56
         Function evaluations: 103
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.471666
         Iterations: 48
         Function evaluations: 94
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.651702
         Iterations: 67
         Function evaluations: 122
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.737426
         Iterations: 60
         Function evaluations: 109
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.613505
         Iterations: 73
         Function evaluations: 141
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.578538
         Iterations: 56
         Function evaluations: 103
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.632218
         Iterations: 64
         Function evaluations: 119
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.651702
         Iterations: 67
         Function evaluations: 122
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.622789
         Iterations: 63
         Function evaluations: 114
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.613505
         Iterations: 73
         Function evaluations: 141
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.627465
         Iterations: 59
         Function evaluations: 109
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.632218
         Iterations: 64
         Function evaluations: 119
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.625104
         Iterations: 59
         Function evaluations: 108
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.629829
         Iterations: 66
         Function evaluations: 118
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.632218
         Iterations: 64
         Function evaluations: 119
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.632218
         Iterations: 64
         Function evaluations: 119
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.628642
         Iterations: 67
         Function evaluations: 122
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.631023
         Iterations: 68
         Function evaluations: 129
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.630430
         Iterations: 57
         Function evaluations: 108
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.629598
         Iterations: 60
         Function evaluations: 112
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.630430
         Iterations: 57
         Function evaluations: 108
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.630130
         Iterations: 65
         Function evaluations: 122
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.629536
         Iterations: 62
         Function evaluations: 111
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.630130
         Iterations: 65
         Function evaluations: 122
repr(start_params) array([ 1.,  2.])
Optimization terminated successfully.
         Current function value: 2626.629984
         Iterations: 67
         Function evaluations: 123
Optimization terminated successfully.
         Current function value: 0.016560
         Iterations: 18
         Function evaluations: 38
>>> res_parks2
(1.0592352626264809, 9.9051580457572399, 2.0966900385041591)
>>> res_parks
(1.0596088578825995, 9.9043376069230007, 2.0975104813987118)
>>> res_par.params
array([  0.84856418,  10.2197801 ,   1.78206799])
>>> np.sqrt(np.diag(mod_par.hessian(res_par.params)))
array([ NaN,  NaN,  NaN])
>>> mod_par.hessian(res_par.params
... )
array([[ NaN,  NaN,  NaN],
       [ NaN,  NaN,  NaN],
       [ NaN,  NaN,  NaN]])
>>> mod_par.hessian(res_parks)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\model.py", line 533, in hessian
    return approx_hess(params, self.loglike)[0]  #need options for hess (epsilon)
  File "c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\sandbox\regression\numdiff.py", line 118, in approx_hess
    xh = x + h
TypeError: can only concatenate tuple (not "float") to tuple

>>> mod_par.hessian(np.array(res_parks))
array([[ NaN,  NaN,  NaN],
       [ NaN,  NaN,  NaN],
       [ NaN,  NaN,  NaN]])
>>> mod_par.fixed_params
array([        NaN,  9.90510677,         NaN])
>>> mod_par.fixed_params=None
>>> mod_par.hessian(np.array(res_parks))
array([[-890.48553491,           NaN,           NaN],
       [          NaN,           NaN,           NaN],
       [          NaN,           NaN,           NaN]])
>>> mod_par.loglike(np.array(res_parks))
-2626.6322080820569
>>> mod_par.bsejac
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\decorators.py", line 85, in __get__
    _cachedval = self.fget(obj)
  File "c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\model.py", line 592, in bsejac
    return np.sqrt(np.diag(self.covjac))
  File "c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\decorators.py", line 85, in __get__
    _cachedval = self.fget(obj)
  File "c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\model.py", line 574, in covjac
    jacv = self.jacv
  File "c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\decorators.py", line 85, in __get__
    _cachedval = self.fget(obj)
  File "c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\model.py", line 557, in jacv
    return self.jac(self._results.params)
  File "c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\model.py", line 530, in jac
    return approx_fprime1(params, self.loglikeobs, **kwds)
  File "c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\sandbox\regression\numdiff.py", line 80, in approx_fprime1
    f0 = f(*((xk,)+args))
  File "c:\josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\model.py", line 522, in loglikeobs
    return -self.nloglikeobs(params)
  File "C:\Josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\examples\ex_generic_mle_tdist.py", line 184, in nloglikeobs
    scale = params[2]
IndexError: index out of bounds
>>> hasattr(self, 'start_params')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'self' is not defined

>>> hasattr(mod_par, 'start_params')
True
>>> mod_par.start_params
array([ 1.,  2.])
>>> stats.pareto.stats(1., 9., 2., moments='mvsk')
(array(1.#INF), array(1.#INF), array(1.#QNAN), array(1.#QNAN))
>>> stats.pareto.stats(1., 8., 2., moments='mvsk')
(array(1.#INF), array(1.#INF), array(1.#QNAN), array(1.#QNAN))
>>> stats.pareto.stats(1., 8., 1., moments='mvsk')
(array(1.#INF), array(1.#INF), array(1.#QNAN), array(1.#QNAN))
>>> stats.pareto.stats(1., moments='mvsk')
(array(1.#INF), array(1.#INF), array(1.#QNAN), array(1.#QNAN))
>>> stats.pareto.stats(0.5., moments='mvsk')
  File "<stdin>", line 1
    stats.pareto.stats(0.5., moments='mvsk')
                           ^
SyntaxError: invalid syntax

>>> stats.pareto.stats(0.5, moments='mvsk')
(array(1.#INF), array(1.#INF), array(1.#QNAN), array(1.#QNAN))
>>> stats.pareto.stats(2, moments='mvsk')
(array(2.0), array(1.#INF), array(1.#QNAN), array(1.#QNAN))
>>> stats.pareto.stats(10, moments='mvsk')
(array(1.1111111111111112), array(0.015432098765432098), array(2.8110568859997356), array(14.828571428571429))
>>> stats.pareto.rvs(10, size=10)
array([ 1.07716265,  1.18977526,  1.07093   ,  1.05157081,  1.15991232,
        1.31015589,  1.06675107,  1.08082475,  1.19501243,  1.34967158])
>>> r = stats.pareto.rvs(10, size=1000)
>>> plt
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'plt' is not defined

>>> import matplotlib.pyplot as plt
>>> plt.hist(r)
(array([962,  32,   3,   2,   0,   0,   0,   0,   0,   1]), array([ 1.00013046,  1.3968991 ,  1.79366773,  2.19043637,  2.587205  ,
        2.98397364,  3.38074227,  3.77751091,  4.17427955,  4.57104818,
        4.96781682]), <a list of 10 Patch objects>)
>>> plt.show()

'''
