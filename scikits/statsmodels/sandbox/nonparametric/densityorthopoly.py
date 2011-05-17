'''density estimation based on orthogonal polynomials


Author: SJosef Perktold
Created: 2011-05017
License: BSD

2 versions work: based on Fourier, FPoly, and chebychev T, ChebyTPoly
other versions need normalization


TODO:

* check fourier case again
* not implemented methods:
  - add bonafide density correction
  - add transformation to domain of polynomial base
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

    def __init__(self, order):
        self.order = order

    def __call__(self, x):
        if self.order == 0:
            return np.ones_like(x)
        else:
            return sqr2 * np.cos(np.pi * self.order * x)


class ChebyTPoly(object):

    def __init__(self, order):
        self.order = order
        from scipy.special import legendre, hermitenorm, chebyt
        self.poly = chebyt(order)

    def __call__(self, x):
        if self.order == 0:
            return np.ones_like(x) / (1-x**2)**(1/4.) /np.sqrt(np.pi)

        else:
            return self.poly(x) / (1-x**2)**(1/4.) /np.sqrt(np.pi) *np.sqrt(2)


##        elif self.order == 1:
##            return self.poly(x)/ (1-x**2)**(1/4.) /np.sqrt(np.pi) *np.sqrt(2) #*4


from scipy.misc import factorial
from scipy import special

logpi4 = np.log(np.pi)/4

class HPoly(object):

    def __init__(self, order):
        self.order = order
        from scipy.special import legendre, hermitenorm, chebyt, hermitenorm
        self.poly = hermitenorm(order)

    def __call__(self, x):
        k = self.order
        lnfact = -(k/2.)*np.log(2.) - 0.5*special.gammaln(k+1) - logpi4# - x*x/2.
        fact = np.exp(lnfact) #*(-1)**k
        if self.order == 0:
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


    def fit(self, x, polybase=None, order=5, limits=None):
        '''estimate the orthogonal polynomial approximation to the density

        '''
        if polybase is None:
            polys = self.polys[:order]
        else:
            self.polys = polys = [polybase(i) for i in range(order)]

        x = self._transform(x, limits=None)

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
        pass

    def _correction(self, x):
        '''bona fide density correction NotImplementedYet'''
        return x

    def _transform(self, x, limits=None):
        '''transform to domain of density, NotImplementedYet'''
        return x


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
    nobs = 10000

    #np.random.seed(12345)
    obs_dist = mixture_rvs([1/3.,2/3.], size=nobs, dist=[stats.norm, stats.norm],
                   kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.75)))
    obs_dist = mixture_rvs([1/3.,2/3.], size=nobs, dist=[stats.norm, stats.norm],
                   kwargs = (dict(loc=-0.5,scale=.5),dict(loc=1,scale=.2)))

    #obs_dist = np.random.randn(nobs)/2 #np.sqrt(2)

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
    xf = dop(grid)
    print np.max(np.abs(xf - f_hat0))

