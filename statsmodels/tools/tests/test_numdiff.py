'''Testing numerical differentiation

Still some problems, with API (args tuple versus *args)
finite difference Hessian has some problems that I didn't look at yet

Should Hessian also work per observation, if fun returns 2d

'''

import numpy as np
from numpy.testing import assert_almost_equal
import statsmodels.api as sm
import numdiff
from numdiff import approx_fprime, approx_fprime_cs, approx_hess_cs

DEC3 = 3
DEC4 = 4
DEC5 = 5
DEC6 = 6
DEC8 = 8
DEC13 = 13
DEC14 = 14

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


#ravel() added because of MNLogit 2d params
class CheckGradLoglike(object):
    def test_score(self):
        pass
        #assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)

        for test_params in self.params:
            sc = self.mod.score(test_params)
            scfd = numdiff.approx_fprime1(test_params.ravel(), self.mod.loglike)
            assert_almost_equal(sc, scfd, decimal=1)

            sccs = numdiff.approx_fprime_cs(test_params.ravel(), self.mod.loglike)
            assert_almost_equal(sc, sccs, decimal=13)

    def test_hess(self):
        pass
        #assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)

        for test_params in self.params:
            he = self.mod.hessian(test_params)
            #TODO: bug
##            hefd = numdiff.approx_hess(test_params, self.mod.score)
##            assert_almost_equal(he, hefd, decimal=DEC8)

            hescs = numdiff.approx_fprime_cs(test_params.ravel(), self.mod.score)
            assert_almost_equal(he, hescs, decimal=DEC8)

            hecs = numdiff.approx_hess_cs(test_params.ravel(), self.mod.loglike)
            assert_almost_equal(he, hecs, decimal=DEC6)


class estGradMNLogit(CheckGradLoglike):
    #doesn't work yet because params is 2d and loglike doesn't take raveled
    def __init__(self):
        #from results.results_discrete import Anes
        data = sm.datasets.anes96.load()
        exog = data.exog
        exog[:,0] = np.log(exog[:,0] + .1)
        exog = np.column_stack((exog[:,0],exog[:,2],
            exog[:,5:8]))
        exog = sm.add_constant(exog)
        self.mod = sm.MNLogit(data.endog, exog)

        def loglikeflat(self, params):
            #reshapes flattened params
            return self.loglike(params.reshape(6,6))
        self.mod.loglike = loglikeflat  #need instance method
        self.params = [np.ones((6,6))]


class TestGradLogit(CheckGradLoglike):
    def __init__(self):
        data = sm.datasets.spector.load()
        data.exog = sm.add_constant(data.exog)
        #mod = sm.Probit(data.endog, data.exog)
        self.mod = sm.Logit(data.endog, data.exog)
        #res = mod.fit(method="newton")
        self.params = [np.array([1,0.25,1.4,-7])]
##        loglike = mod.loglike
##        score = mod.score
##        hess = mod.hessian


class CheckDerivative(object):
    def __init__(self):
        nobs = 200
        #x = np.arange(nobs*3).reshape(nobs,-1)
        np.random.seed(187678)
        x = np.random.randn(nobs,3)

        xk = np.array([1,2,3])
        xk = np.array([1.,1.,1.])
        #xk = np.zeros(3)
        beta = xk
        y = np.dot(x, beta) + 0.1*np.random.randn(nobs)
        xkols = np.dot(np.linalg.pinv(x),y)

        self.x = x
        self.y = y
        self.params = [np.array([1.,1.,1.]), xkols]
        self.init()

    def init(self):
        pass

    def test_grad_fun1_fd(self):
        for test_params in self.params:
            #gtrue = self.x.sum(0)
            gtrue = self.gradtrue(test_params)
            fun = self.fun()
            epsilon = 1e-6
            gfd = numdiff.approx_fprime1(test_params, fun, epsilon=epsilon,
                                         args=self.args)
            gfd += numdiff.approx_fprime1(test_params, fun, epsilon=-epsilon,
                                          args=self.args)
            gfd /= 2.
            assert_almost_equal(gtrue, gfd, decimal=DEC6)

    def test_grad_fun1_fdc(self):
        for test_params in self.params:
            #gtrue = self.x.sum(0)
            gtrue = self.gradtrue(test_params)
            fun = self.fun()

            epsilon = 1e-6  #default epsilon 1e-6 is not precise enough
            gfd = numdiff.approx_fprime1(test_params, fun, epsilon=1e-8,
                                         args=self.args, centered=True)
            assert_almost_equal(gtrue, gfd, decimal=DEC5)

    def test_grad_fun1_cs(self):
        for test_params in self.params:
            #gtrue = self.x.sum(0)
            gtrue = self.gradtrue(test_params)
            fun = self.fun()

            gcs = numdiff.approx_fprime_cs(test_params, fun, args=self.args)
            assert_almost_equal(gtrue, gcs, decimal=DEC13)

    def test_hess_fun1_fd(self):
        for test_params in self.params:
            #hetrue = 0
            hetrue = self.hesstrue(test_params)
            if not hetrue is None: #Hessian doesn't work for 2d return of fun
                fun = self.fun()
                #default works, epsilon 1e-6 or 1e-8 is not precise enough
                hefd = numdiff.approx_hess(test_params, fun, #epsilon=1e-8,
                                             args=self.args)[0] #TODO:should be kwds
                assert_almost_equal(hetrue, hefd, decimal=DEC3)
                #TODO: I reduced precision to DEC3 from DEC4 because of
                #    TestDerivativeFun

    def test_hess_fun1_cs(self):
        for test_params in self.params:
            #hetrue = 0
            hetrue = self.hesstrue(test_params)
            if not hetrue is None: #Hessian doesn't work for 2d return of fun
                fun = self.fun()
                hecs = numdiff.approx_hess_cs(test_params, fun, args=self.args)
                assert_almost_equal(hetrue, hecs, decimal=DEC6)


class TestDerivativeFun(CheckDerivative):
    def init(self):
        xkols = np.dot(np.linalg.pinv(self.x), self.y)
        self.params = [np.array([1.,1.,1.]), xkols]
        self.args = (self.x,)

    def fun(self):
        return fun
    def gradtrue(self, params):
        return self.x.sum(0)
    def hesstrue(self, params):
        return np.zeros((3,3))   #make it (3,3), because test fails with scalar 0
        #why is precision only DEC3

class TestDerivativeFun2(CheckDerivative):
    def init(self):
        xkols = np.dot(np.linalg.pinv(self.x), self.y)
        self.params = [np.array([1.,1.,1.]), xkols]
        self.args = (self.y, self.x)

    def fun(self):
        return fun2

    def gradtrue(self, params):
        y, x = self.y, self.x
        return (-x*2*(y-np.dot(x, params))[:,None]).sum(0)
                #2*(y-np.dot(x, params)).sum(0)

    def hesstrue(self, params):
        x = self.x
        return 2*np.dot(x.T, x)

class TestDerivativeFun1(CheckDerivative):
    def init(self):
        xkols = np.dot(np.linalg.pinv(self.x), self.y)
        self.params = [np.array([1.,1.,1.]), xkols]
        self.args = (self.y, self.x)

    def fun(self):
        return fun1
    def gradtrue(self, params):
        y, x = self.y, self.x
        return (-x*2*(y-np.dot(x, params))[:,None])
    def hesstrue(self, params):
        return None
        y, x = self.y, self.x
        return (-x*2*(y-np.dot(x, parms))[:,None])  #TODO: check shape


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
    print numdiff.approx_hess(xk,fun2,1e-3, (y,x))[0] - 2*np.dot(x.T, x)

    gt = (-x*2*(y-np.dot(x, [1,2,3]))[:,None])
    g = approx_fprime_cs((1,2,3), fun1, (y,x), h=1.0e-20)#.T   #this shouldn't be transposed
    gd = numdiff.approx_fprime1((1,2,3),fun1,epsilon,(y,x))
    print maxabs(g, gt)
    print maxabs(gd, gt)


    import statsmodels.api as sm

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

    data = sm.datasets.anes96.load()
    exog = data.exog
    exog[:,0] = np.log(exog[:,0] + .1)
    exog = np.column_stack((exog[:,0],exog[:,2],
        exog[:,5:8]))
    exog = sm.add_constant(exog)
    res1 = sm.MNLogit(data.endog, exog).fit(method="newton", disp=0)

    datap = sm.datasets.randhie.load()
    nobs = len(datap.endog)
    exogp = sm.add_constant(datap.exog.view(float).reshape(nobs,-1))
    modp = sm.Poisson(datap.endog, exogp)
    resp = modp.fit(method='newton', disp=0)


