

import numpy as np
from numpy.testing import assert_almost_equal
import scikits.statsmodels as sm
import numdiff
from numdiff import approx_fprime, approx_fprime_cs, approx_hess_cs

def maxabs(x,y):
    return np.abs(x-y).max()

def fun(beta, x):
    return np.dot(x, beta).sum(0)

def fun1(beta, y, x):
    #print beta.shape, x.shape
    xb = np.dot(x, beta)
    return (y-xb)**2 #(xb-xb.mean(0))**2

def fun2(beta, y, x):
    #print beta.shape, x.shape
    return fun1(beta, y, x).sum(0)


class CheckGradLoglike(object):
    def test_score(self):
        pass
        #assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)



class TestGradMNLogit(CheckGradLoglike):
    def __init__(self):
        #from results.results_discrete import Anes
        data = sm.datasets.anes96.load()
        exog = data.exog
        exog[:,0] = np.log(exog[:,0] + .1)
        exog = np.column_stack((exog[:,0],exog[:,2],
            exog[:,5:8]))
        exog = sm.add_constant(exog)
        self.mod = sm.MNLogit(data.endog, exog)
        self.params = np.ones(6)



if __name__ == '__main__':

    epsilon = 1e-6
    nobs = 200
    x = np.arange(nobs*3).reshape(nobs,-1)
    x = np.random.randn(nobs,3)

    xk = np.array([1,2,3])
    xk = np.array([1.,1.,1.])
    #xk = np.zeros(3)
    beta = xk
    y = np.dot(x, beta) + 0.1*np.random.randn(nobs)
    xkols = np.dot(np.linalg.pinv(x),y)

    print approx_fprime((1,2,3),fun,epsilon,x)
    gradtrue = x.sum(0)
    print x.sum(0)
    gradcs = approx_fprime_cs((1,2,3), fun, (x,), h=1.0e-20)
    print gradcs, maxabs(gradcs, gradtrue)
    print approx_hess_cs((1,2,3), fun, (x,), h=1.0e-20)  #this is correctly zero

    print approx_hess_cs((1,2,3), fun2, (y,x), h=1.0e-20)-2*np.dot(x.T, x)
    print numdiff.approx_hess(xk,fun2,1e-3, y,x)[0] - 2*np.dot(x.T, x)

    gt = (-x*2*(y-np.dot(x, [1,2,3]))[:,None])
    g = approx_fprime_cs((1,2,3), fun1, (y,x), h=1.0e-20).T   #this shouldn't be transposed
    gd = numdiff.approx_fprime1((1,2,3),fun1,epsilon,(y,x))
    print maxabs(g, gt)
    print maxabs(gd, gt)


    import scikits.statsmodels as sm

    data = sm.datasets.spector.load()
    data.exog = sm.add_constant(data.exog)
    #mod = sm.Probit(data.endog, data.exog)
    mod = sm.Logit(data.endog, data.exog)
    #res = mod.fit(method="newton")
    test_params = [1,0.25,1.4,-7]
    loglike = mod.loglike
    score = mod.score
    hess = mod.hessian

    #cs doesn't work for Probit because special.ndtr doesn't support complex
    #maybe calculating ndtr for real and imag parts separately, if we need it
    #and if it still works in this case
    print 'sm', score(test_params)
    print 'fd', numdiff.approx_fprime1(test_params,loglike,epsilon)
    print 'cs', numdiff.approx_fprime_cs(test_params,loglike)
    print 'sm', hess(test_params)
    print 'fd', numdiff.approx_fprime1(test_params,score,epsilon)
    print 'cs', numdiff.approx_fprime_cs(test_params, score)

    #print 'fd', numdiff.approx_hess(test_params, loglike, epsilon) #TODO: bug
    '''
    Traceback (most recent call last):
      File "C:\Josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\sandbox\regression\test_numdiff.py", line 74, in <module>
        print 'fd', numdiff.approx_hess(test_params, loglike, epsilon)
      File "C:\Josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\sandbox\regression\numdiff.py", line 118, in approx_hess
        xh = x + h
    TypeError: can only concatenate list (not "float") to list
    '''
    hesscs = numdiff.approx_hess_cs(test_params, loglike)
    print 'cs', hesscs
    print maxabs(hess(test_params), hesscs)
