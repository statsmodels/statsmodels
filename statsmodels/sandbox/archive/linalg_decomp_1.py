'''Recipes for more efficient work with linalg using classes


intended for use for multivariate normal and linear regression
calculations

x  is the data (nobs, nvars)
m  is the moment matrix (x'x) or a covariance matrix Sigma

examples:
x'sigma^{-1}x
z = Px  where P=Sigma^{-1/2}  or P=Sigma^{1/2}

Initially assume positive definite, then add spectral cutoff and
regularization of moment matrix, and extend to PCA

maybe extend to sparse if some examples work out
(transformation matrix P for random effect and for toeplitz)


Author: josef-pktd
Created on 2010-10-20
'''

from __future__ import print_function
from statsmodels.compat.python import get_function_name
import numpy as np
from scipy import linalg


#this has been copied from nitime a long time ago
#TODO: ceck whether class has changed in nitime
class OneTimeProperty(object):


    """A descriptor to make special properties that become normal attributes.

    This is meant to be used mostly by the auto_attr decorator in this module.
    Author: Fernando Perez, copied from nitime
    """
    def __init__(self,func):

        """Create a OneTimeProperty instance.

         Parameters
         ----------
           func : method

             The method that will be called the first time to compute a value.
             Afterwards, the method's name will be a standard attribute holding
             the value of this computation.
             """
        self.getter = func
        self.name = get_function_name(func)

    def __get__(self,obj,type=None):
        """This will be called on attribute access on the class or instance. """

        if obj is None:
            # Being called on the class, return the original function. This way,
            # introspection works on the class.
            #return func
            print('class access')
            return self.getter

        val = self.getter(obj)
        #print("** auto_attr - loading '%s'" % self.name  # dbg)
        setattr(obj, self.name, val)
        return val


class PlainMatrixArray(object):
    '''Class that defines linalg operation on an array

    simplest version as benchmark

    linear algebra recipes for multivariate normal and linear
    regression calculations

    '''
    def __init__(self, data=None, sym=None):
        if not data is None:
            if sym is None:
                self.x = np.asarray(data)
                self.m = np.dot(self.x.T, self.x)
            else:
                raise ValueError('data and sym cannot be both given')
        elif not sym is None:
            self.m = np.asarray(sym)
            self.x = np.eye(*self.m.shape) #default

        else:
            raise ValueError('either data or sym need to be given')

    @OneTimeProperty
    def minv(self):
        return np.linalg.inv(self.m)

    @OneTimeProperty
    def m_y(self, y):
        return np.dot(self.m, y)

    def minv_y(self, y):
        return np.dot(self.minv, y)

    @OneTimeProperty
    def mpinv(self):
        return linalg.pinv(self.m)

    @OneTimeProperty
    def xpinv(self):
        return linalg.pinv(self.x)

    def yt_m_y(self, y):
        return np.dot(y.T, np.dot(self.m, y))

    def yt_minv_y(self, y):
        return np.dot(y.T, np.dot(self.minv, y))

    #next two are redundant
    def y_m_yt(self, y):
        return np.dot(y, np.dot(self.m, y.T))

    def y_minv_yt(self, y):
        return np.dot(y, np.dot(self.minv, y.T))

    @OneTimeProperty
    def mdet(self):
        return linalg.det(self.m)

    @OneTimeProperty
    def mlogdet(self):
        return np.log(linalg.det(self.m))

    @OneTimeProperty
    def meigh(self):
        evals, evecs = linalg.eigh(self.m)
        sortind = np.argsort(evals)[::-1]
        return evals[sortind], evecs[:,sortind]

    @OneTimeProperty
    def mhalf(self):
        evals, evecs = self.meigh
        return np.dot(np.diag(evals**0.5), evecs.T)
        #return np.dot(evecs, np.dot(np.diag(evals**0.5), evecs.T))
        #return np.dot(evecs, 1./np.sqrt(evals) * evecs.T))

    @OneTimeProperty
    def minvhalf(self):
        evals, evecs = self.meigh
        return np.dot(evecs, 1./np.sqrt(evals) * evecs.T)



class SvdArray(PlainMatrixArray):
    '''Class that defines linalg operation on an array

    svd version, where svd is taken on original data array, if
    or when it matters

    no spectral cutoff in first version
    '''

    def __init__(self, data=None, sym=None):
        super(SvdArray, self).__init__(data=data, sym=sym)

        u, s, v = np.linalg.svd(self.x, full_matrices=1)
        self.u, self.s, self.v = u, s, v
        self.sdiag = linalg.diagsvd(s, *x.shape)
        self.sinvdiag = linalg.diagsvd(1./s, *x.shape)

    def _sdiagpow(self, p):
        return linalg.diagsvd(np.power(self.s, p), *x.shape)

    @OneTimeProperty
    def minv(self):
        sinvv = np.dot(self.sinvdiag, self.v)
        return np.dot(sinvv.T, sinvv)


    @OneTimeProperty
    def meigh(self):
        evecs = self.v.T
        evals = self.s**2
        return evals, evecs

    @OneTimeProperty
    def mdet(self):
        return self.meigh[0].prod()

    @OneTimeProperty
    def mlogdet(self):
        return np.log(self.meigh[0]).sum()

    @OneTimeProperty
    def mhalf(self):
        return np.dot(np.diag(self.s), self.v)

    @OneTimeProperty
    def xxthalf(self):
        return np.dot(self.u, self.sdiag)

    @OneTimeProperty
    def xxtinvhalf(self):
        return np.dot(self.u, self.sinvdiag)


class CholArray(PlainMatrixArray):
    '''Class that defines linalg operation on an array

    cholesky version, where svd is taken on original data array, if
    or when it matters

    plan: use cholesky factor and cholesky solve
    nothing implemented yet
    '''

    def __init__(self, data=None, sym=None):
        super(SvdArray, self).__init__(data=data, sym=sym)


    def yt_minv_y(self, y):
        '''xSigmainvx
        doesn't use stored cholesky yet
        '''
        return np.dot(x,linalg.cho_solve(linalg.cho_factor(self.m),x))
        #same as
        #lower = False   #if cholesky(sigma) is used, default is upper
        #np.dot(x,linalg.cho_solve((self.cholsigma, lower),x))



def testcompare(m1, m2):
    from numpy.testing import assert_almost_equal, assert_approx_equal
    decimal = 12

    #inv
    assert_almost_equal(m1.minv, m2.minv, decimal=decimal)

    #matrix half and invhalf
    #fix sign in test, should this be standardized
    s1 = np.sign(m1.mhalf.sum(1))[:,None]
    s2 = np.sign(m2.mhalf.sum(1))[:,None]
    scorr = s1/s2
    assert_almost_equal(m1.mhalf, m2.mhalf * scorr, decimal=decimal)
    assert_almost_equal(m1.minvhalf, m2.minvhalf, decimal=decimal)

    #eigenvalues, eigenvectors
    evals1, evecs1 = m1.meigh
    evals2, evecs2 = m2.meigh
    assert_almost_equal(evals1, evals2, decimal=decimal)
    #normalization can be different: evecs in columns
    s1 = np.sign(evecs1.sum(0))
    s2 = np.sign(evecs2.sum(0))
    scorr = s1/s2
    assert_almost_equal(evecs1, evecs2 * scorr, decimal=decimal)

    #determinant
    assert_approx_equal(m1.mdet, m2.mdet, significant=13)
    assert_approx_equal(m1.mlogdet, m2.mlogdet, significant=13)

####### helper function for interactive work
def tiny2zero(x, eps = 1e-15):
    '''replace abs values smaller than eps by zero, makes copy
    '''
    mask = np.abs(x.copy()) <  eps
    x[mask] = 0
    return x

def maxabs(x):
    return np.max(np.abs(x))


if __name__ == '__main__':


    n = 5
    y = np.arange(n)
    x = np.random.randn(100,n)
    autocov = 2*0.8**np.arange(n) +0.01 * np.random.randn(n)
    sigma = linalg.toeplitz(autocov)

    mat = PlainMatrixArray(sym=sigma)
    print(tiny2zero(mat.mhalf))
    mih = mat.minvhalf
    print(tiny2zero(mih)) #for nicer printing

    mat2 = PlainMatrixArray(data=x)
    print(maxabs(mat2.yt_minv_y(np.dot(x.T, x)) - mat2.m))
    print(tiny2zero(mat2.minv_y(mat2.m)))

    mat3 = SvdArray(data=x)
    print(mat3.meigh[0])
    print(mat2.meigh[0])

    testcompare(mat2, mat3)

    '''
    m = np.dot(x.T, x)

    u,s,v = np.linalg.svd(x, full_matrices=1)
    Sig = linalg.diagsvd(s,*x.shape)

    >>> np.max(np.abs(np.dot(u, np.dot(Sig, v)) - x))
    3.1086244689504383e-015
    >>> np.max(np.abs(np.dot(u.T, u) - np.eye(100)))
    3.3306690738754696e-016
    >>> np.max(np.abs(np.dot(v.T, v) - np.eye(5)))
    6.6613381477509392e-016
    >>> np.max(np.abs(np.dot(Sig.T, Sig) - np.diag(s**2)))
    5.6843418860808015e-014

    >>> evals,evecs = linalg.eigh(np.dot(x.T, x))
    >>> evals[::-1]
    array([ 123.36404464,  112.17036442,  102.04198468,   76.60832278,
             74.70484487])

    >>> s**2
    array([ 123.36404464,  112.17036442,  102.04198468,   76.60832278,
             74.70484487])

    >>> np.max(np.abs(np.dot(v.T, np.dot(np.diag(s**2), v)) - m))
    1.1368683772161603e-013

    >>> us = np.dot(u, Sig)
    >>> np.max(np.abs(np.dot(us, us.T) - np.dot(x, x.T)))
    1.0658141036401503e-014

    >>> sv = np.dot(Sig, v)
    >>> np.max(np.abs(np.dot(sv.T, sv) - np.dot(x.T, x)))
    1.1368683772161603e-013


    '''
