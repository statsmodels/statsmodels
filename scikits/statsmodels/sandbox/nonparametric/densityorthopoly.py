# -*- coding: cp1252 -*-
# some cut and paste characters are not ASCII
'''density estimation based on orthogonal polynomials


Author: Josef Perktold
Created: 2011-05017
License: BSD

2 versions work: based on Fourier, FPoly, and chebychev T, ChebyTPoly
other versions need normalization


TODO:

* check fourier case again:  base is orthonormal,
  but needs offsetfact = 0 and doesn't integrate to 1, rescaled looks good
* not implemented methods:
  - add bonafide density correction
  - add transformation to domain of polynomial base DONE
    possible problem what is the behavior at the boundary,
    offsetfact requires more work, check different cases, add as option
* convert examples to test cases
* organize poly classes in separate module, check new numpy.polynomials,
  polyvander
* MISE measures, order selection, ...

enhancements:
  * other polynomial bases: especially for open and half open support
  * wavelets
  * local or piecewise approximations


'''

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scikits.statsmodels.sandbox.distributions.mixture_rvs import mixture_rvs
##from scikits.statsmodels.sandbox.nonparametric.kde import (kdensity,
##                                                           kdensityfft)
##from scikits.statsmodels.sandbox.nonparametric import bandwidths

sqr2 = np.sqrt(2.)

class FPoly(object):
    '''

    orthonormal polynomial but density needs corfactor that I don't see what
    it is analytically

    parameterization on [0,1] from

    Sam Efromovich: Orthogonal series density estimation,
    2010 John Wiley & Sons, Inc. WIREs Comp Stat 2010 2 467–476


    '''

    def __init__(self, order):
        self.order = order
        self.domain = (0, 1)
        self.intdomain = self.domain

    def __call__(self, x):
        if self.order == 0:
            return np.ones_like(x)
        else:
            return sqr2 * np.cos(np.pi * self.order * x)

class F2Poly(object):
    '''

    is orthogonal but first component doesn't square-integrate to 1
    final result seems to need a correction factor of sqrt(pi)
    _corfactor = sqrt(pi) from integrating the density

    Parameterization on [0, pi] from

    Peter Hall, Cross-Validation and the Smoothing of Orthogonal Series Density
    Estimators, JOURNAL OF MULTIVARIATE ANALYSIS 21, 189-206 (1987)

    '''

    def __init__(self, order):
        self.order = order
        self.domain = (0, np.pi)
        self.intdomain = self.domain

    def __call__(self, x):
        if self.order == 0:
            return np.ones_like(x) / np.sqrt(np.pi)
        else:
            return sqr2 * np.cos(self.order * x) / np.sqrt(np.pi)

class ChebyTPoly(object):

    def __init__(self, order):
        self.order = order
        from scipy.special import legendre, hermitenorm, chebyt
        self.poly = chebyt(order)
        self.domain = (-1, 1)
        self.intdomain = (-1+1e-6, 1-1e-6)

    def __call__(self, x):
        if self.order == 0:
            return np.ones_like(x) / (1-x**2)**(1/4.) /np.sqrt(np.pi)

        else:
            return self.poly(x) / (1-x**2)**(1/4.) /np.sqrt(np.pi) *np.sqrt(2)


##        elif self.order == 1:
##            return self.poly(x)/ (1-x**2)**(1/4.) /np.sqrt(np.pi) *np.sqrt(2) #*4


from scipy.misc import factorial
from scipy import special

logpi2 = np.log(np.pi)/2

class HPoly(object):

    def __init__(self, order):
        self.order = order
        from scipy.special import legendre, hermitenorm, chebyt, hermite
        self.poly = hermite(order)
        self.domain = (-6, +6)

    def __call__(self, x):
        k = self.order
        #lnfact = -(k/2.)*np.log(2.) - 0.5*special.gammaln(k+1) - logpi4 - x*x/2.
        #following Hall:
        #lnfact = -(k/2.)*(k*np.log(2.) + special.gammaln(k+1) + logpi2) - x*x/2
        lnfact = -(1./2)*(k*np.log(2.) + special.gammaln(k+1) + logpi2) - x*x/2
        #lnfact = -x*x/4.

        fact = np.exp(lnfact) #*(-1)**k
        #fact = np.sqrt(fact)
        if self.order < 0:
            return np.ones_like(x) * fact
        else:
            return self.poly(x) * fact

def polyvander(x, polybase, order=5):
    polyarr = np.column_stack([polybase(i)(x) for i in range(order)])
    return polyarr

def inner_cont(polys, lower, upper):
    '''inner product of continuous function

    '''
    n_polys = len(polys)
    innerprod = np.empty((n_polys, n_polys))
    innerprod.fill(np.nan)
    interr = np.zeros((n_polys, n_polys))

    for i in range(n_polys):
        for j in range(i+1):
            p1 = polys[i]
            p2 = polys[j]
            innp, err = integrate.quad(lambda x: p1(x)*p2(x), lower, upper)
            innerprod[i,j] = innp
            interr[i,j] = err
            if not i == j:
                innerprod[j,i] = innp
                interr[j,i] = err

    return innerprod, interr


def is_orthonormal_cont(polys, lower, upper, rtol=0, atol=1e-14):
    '''check whether functions are orthonormal

    Parameters
    ----------
    polys : list of polynomials or function

    Returns
    -------
    is_orthonormal : bool
        is False if the innerproducts are not close to 0 or 1

    '''
    for i in range(len(polys)):
        for j in range(i+1):
            p1 = polys[i]
            p2 = polys[j]
            innerprod = integrate.quad(lambda x: p1(x)*p2(x), lower, upper)[0]
            #print i,j, innerprod
            if not np.allclose(innerprod, i==j, rtol=rtol, atol=atol):
                return False
    return True

#new versions


class DensityOrthoPoly(object):

    def __init__(self, polybase=None, order=5):
        if not polybase is None:
            self.polybase = polybase
            self.polys = polys = [polybase(i) for i in range(order)]
        self.offsetfac = 0.05
        self._corfactor = 1
        self._corshift = 0


    def fit(self, x, polybase=None, order=5, limits=None):
        '''estimate the orthogonal polynomial approximation to the density

        '''
        if polybase is None:
            polys = self.polys[:order]
        else:
            self.polybase = polybase
            self.polys = polys = [polybase(i) for i in range(order)]


        xmin, xmax = x.min(), x.max()
        if limits is None:
            self.offset = offset = (xmax - xmin) * self.offsetfac
            limits = self.limits = (xmin - offset, xmax + offset)

        interval_length = limits[1] - limits[0]
        xinterval = xmax - xmin
        # need to cover (half-)open intervalls
        self.shrink = 1. / interval_length #xinterval/interval_length
        offset = (interval_length - xinterval ) / 2.
        self.shift = xmin - offset

        self.x = x = self._transform(x)

        coeffs = [(p(x)).mean() for p in polys]
        self.coeffs = coeffs
        self.polys = polys
        self._verify()  #verify that it is a proper density

        return self #coeffs, polys

    def evaluate(self, xeval, order=None):
        xeval = self._transform(xeval)
        if order is None:
            order = len(self.polys)
        res = sum(c*p(xeval) for c, p in zip(self.coeffs, self.polys)[:order])
        res = self._correction(res)
        return res

    def __call__(self, xeval):
        '''alias for evaluate, except no order argument'''
        return self.evaluate(xeval)

    def _verify(self):
        '''check for bona fide density correction NotImplementedYet'''
        #watch out for circular/recursive usage
        #integrate.quad(lambda x: p(x)**2, -1,1)
        intdomain = self.limits #self.polys[0].intdomain
        self._corfactor = 1./integrate.quad(self.evaluate, *intdomain)[0]
        #self._corshift = 0
        #self._corfactor
        return self._corfactor



    def _correction(self, x):
        '''bona fide density correction NotImplementedYet'''
        if self._corfactor != 1:
            x *= self._corfactor

        if self._corshift != 0:
            x += self._corfactor

        return x

    def _transform(self, x): # limits=None):
        '''transform to domain of density, NotImplementedYet'''
        #domain = np.array([0.02, -0.02]) + self.polys[0].domain
        domain = self.polys[0].domain
        #class doesn't have domain  self.polybase.domain[0] AttributeError
        ilen = (domain[1] - domain[0])#*(1-self.offsetfac * 2)
        shift = self.shift - domain[0]/self.shrink/ilen #*(1-self.offsetfac) #check
        shrink = self.shrink * ilen
        #return x
        return (x - shift) * shrink


def density_orthopoly(x, polybase, order=5, xeval=None):
    from scipy.special import legendre, hermitenorm, chebyt, chebyu, hermite
    #polybase = legendre  #chebyt #hermitenorm#
    #polybase = chebyt
    #polybase = FPoly
    #polybase = ChtPoly
    #polybase = hermite
    #polybase = HPoly

    if xeval is None:
        xeval = np.linspace(x.min(),x.max(),50)

    #polys = [legendre(i) for i in range(order)]
    polys = [polybase(i) for i in range(order)]
    #coeffs = [(p(x)*(1-x**2)**(-1/2.)).mean() for p in polys]
    #coeffs = [(p(x)*np.exp(-x*x)).mean() for p in polys]
    coeffs = [(p(x)).mean() for p in polys]
    res = sum(c*p(xeval) for c, p in zip(coeffs, polys))
    #res *= (1-xeval**2)**(-1/2.)
    #res *= np.exp(-xeval**2./2)
    return res, xeval, coeffs, polys



if __name__ == '__main__':

    examples = ['fourier', 'hermite'][1]

    nobs = 10000

    #np.random.seed(12345)
    obs_dist = mixture_rvs([1/3.,2/3.], size=nobs, dist=[stats.norm, stats.norm],
                   kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.75)))
    obs_dist = mixture_rvs([1/3.,2/3.], size=nobs, dist=[stats.norm, stats.norm],
                   kwargs = (dict(loc=-0.5,scale=.5),dict(loc=1,scale=.2)))

    #obs_dist = np.random.randn(nobs)/4. #np.sqrt(2)

    #obs_dist = np.clip(obs_dist, -2, 2)/2.01
    #chebyt [0,1]
    obs_dist = obs_dist[(obs_dist>-2) & (obs_dist<2)]/2.0 #/4. + 2/4.0
    #fourier [0,1]
    #obs_dist = obs_dist[(obs_dist>-2) & (obs_dist<2)]/4. + 2/4.0
    f_hat, grid, coeffs, polys = density_orthopoly(obs_dist, ChebyTPoly, order=20, xeval=None)
    #f_hat /= f_hat.sum() * (grid.max() - grid.min())/len(grid)
    f_hat0 = f_hat
    from scipy import integrate
    fint = integrate.trapz(f_hat, grid)# dx=(grid.max() - grid.min())/len(grid))
    #f_hat -= fint/2.
    print 'f_hat.min()', f_hat.min()
    f_hat = (f_hat - f_hat.min()) #/ f_hat.max() - f_hat.min
    fint2 = integrate.trapz(f_hat, grid)# dx=(grid.max() - grid.min())/len(grid))
    print 'fint2', fint, fint2
    f_hat /= fint2

    # note that this uses a *huge* grid by default
    #f_hat, grid = kdensityfft(emp_dist, kernel="gauss", bw="scott")

    # check the plot

    doplot = 0
    if doplot:
        plt.hist(obs_dist, bins=50, normed=True, color='red')
        plt.plot(grid, f_hat, lw=2, color='black')
        plt.plot(grid, f_hat0, lw=2, color='g')
        plt.show()
    # See attached

    for i,p in enumerate(polys[:5]):
        for j,p2 in enumerate(polys[:5]):
            print i,j,integrate.quad(lambda x: p(x)*p2(x), -1,1)[0]

    for p in polys:
        print integrate.quad(lambda x: p(x)**2, -1,1)

    dop = DensityOrthoPoly().fit(obs_dist, ChebyTPoly, order=20)
    grid = np.linspace(obs_dist.min(), obs_dist.max())
    xf = dop(grid)
    print 'np.max(np.abs(xf - f_hat0))', np.max(np.abs(xf - f_hat0))
    dopint = integrate.quad(dop, *dop.limits)[0]
    print 'dop F integral', dopint
    doplot = 0
    if doplot:
        plt.figure()
        plt.hist(obs_dist, bins=50, normed=True, color='red')
        plt.plot(grid, xf, lw=2, color='black')
        #plt.show()

    if "fourier" in examples:
        dop = DensityOrthoPoly()
        dop.offsetfac = 0.5
        dop = dop.fit(obs_dist, F2Poly, order=20)
        grid = np.linspace(obs_dist.min(), obs_dist.max())
        xf = dop(grid)
        print np.max(np.abs(xf - f_hat0))
        dopint = integrate.quad(dop, *dop.limits)[0]
        print 'dop F integral', dopint
        doplot = 0
        if doplot:
            plt.figure()
            plt.hist(obs_dist, bins=50, normed=True, color='red')
            plt.plot(grid, xf, lw=2, color='black')
            plt.show()

        #check orthonormality:
        print np.max(np.abs(inner_cont(dop.polys[:5], 0, 1)[0] -np.eye(5)))

    if "hermite" in examples:
        dop = DensityOrthoPoly()
        dop.offsetfac = 0
        dop = dop.fit(obs_dist, HPoly, order=20)
        grid = np.linspace(obs_dist.min(), obs_dist.max())
        xf = dop(grid)
        print np.max(np.abs(xf - f_hat0))
        dopint = integrate.quad(dop, *dop.limits)[0]
        print 'dop F integral', dopint
        doplot = 1
        if doplot:
            plt.figure()
            plt.hist(obs_dist, bins=50, normed=True, color='red')
            plt.plot(grid, xf, lw=2, color='black')
            plt.show()

        #check orthonormality:
        print np.max(np.abs(inner_cont(dop.polys[:5], 0, 1)[0] -np.eye(5)))

    hpolys = [HPoly(i) for i in range(5)]
    inn = inner_cont(hpolys, -6, 6)[0]
    print np.max(np.abs(inn - np.eye(5)))
    print (inn*100000).astype(int)

    htpolys = [hermite(i) for i in range(5)]
    innt = inner_cont(htpolys, -10, 10)[0]
    print (innt*100000).astype(int)
