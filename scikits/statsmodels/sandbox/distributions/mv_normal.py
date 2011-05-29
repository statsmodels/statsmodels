# -*- coding: utf-8 -*-
"""
Created on Sat May 28 15:38:23 2011

@author: josef
"""

import numpy as np

def expect_mc(dist, func=lambda x: 1):
    def fun(x):
        return func(x) * dist.pdf(x)
    rvs = dist.rvs(size=size)
    return fun(rvs).mean()


def bivariate_normal(x, mu, cov):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.

    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    """
    X, Y = np.transpose(x)
    mux, muy = mu
    sigmax, sigmaxy, tmp, sigmay = np.ravel(cov)
    sigmax, sigmay = np.sqrt(sigmax), np.sqrt(sigmay)
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp( -z/(2*(1-rho**2))) / denom



class BivariateNormal(object):


    #TODO: make integration limits more flexible
    #      or normalize before integration

    def __init__(self, mean, cov):
        self.mean = mu
        self.cov = cov
        self.sigmax, self.sigmaxy, tmp, self.sigmay = np.ravel(cov)
        self.nvars = 2

    def rvs(self, size=1):
        return np.random.multivariate_normal(self.mean, self.cov, size=size)

    def pdf(self, x):
        return bivariate_normal(x, self.mean, self.cov)

    def logpdf(self, x):
        #TODO: replace this
        return np.log(self.pdf(x))

    def cdf(self, x):
        return self.expect(upper=x)

    def expect(self, func=lambda x: 1, lower=(-10,-10), upper=(10,10)):
        def fun(x, y):
            x = np.column_stack((x,y))
            return func(x) * self.pdf(x)
        from scipy.integrate import dblquad
        return dblquad(fun, lower[0], upper[0], lambda y: lower[1],
                       lambda y: upper[1])

    def kl(self, other):
        '''Kullback-Leibler divergence between this and another distribution

        int f(x) (log f(x) - log g(x)) dx

        where f is the pdf of self, and g is the pdf of other

        uses double integration with scipy.integrate.dblquad

        limits currently hardcoded

        '''
        fun = lambda x : self.logpdf(x) - other.logpdf(x)
        return self.expect(fun)

    def kl_mc(self, other, size=500000):
        fun = lambda x : self.logpdf(x) - other.logpdf(x)
        rvs = self.rvs(size=size)
        return fun(rvs).mean()

#parts taken from linear_model, but heavy adjustments
class MVNormal(object):


    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov = np.asarray(cov)
        cov = np.squeeze(cov)
        self.nvars = nvars = len(mean)
        #self.covchol = np.linalg.cholesky(cov)


        #in the following cov is original, self.cov is full matrix
        if cov.shape == ():
            #iid
            self.cov = np.eye(nvars) * cov
            self.covinv = np.eye(nvars) / cov
            self.cholcovinv = np.eye(nvars) / np.sqrt(cov)
        elif (cov.ndim == 1) and (len(cov) == nvars):
            #independent heteroscedastic
            self.cov = np.diag(cov)
            self.covinv = np.diag(1. / cov)
            self.cholcovinv = np.diag( 1. / np.sqrt(cov))
        elif cov.shape == (nvars, nvars): #python tuple comparison
            #general
            self.covinv = np.linalg.pinv(cov)
            self.cholcovinv = np.linalg.cholesky(self.covinv).T
        else:
            raise ValueError('cov has invalid shape')

        #store logdetcov for logpdf
        self.logdetcov = np.log(np.linalg.det(self.cov))

    def whiten(self, x):
        """
        GLS whiten method.

        Parameters
        -----------
        X : array-like
            Data to be whitened.

        Returns
        -------
        np.dot(cholsigmainv,X)

        See Also
        --------
        regression.GLS
        """
        x = np.asarray(x)
        if np.any(self.cov):
            #return np.dot(self.cholcovinv, x)
            return np.dot(x, self.cholcovinv.T)
        else:
            return x

    def rvs(self, size=1):
        return np.random.multivariate_normal(self.mean, self.cov, size=size)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def logpdf(self, x):
        '''log of pdf

        this should be made to work with 2d x,
        with multivariate normal vector in each row and iid across rows
        doesn't work now because of dot in whiten

        '''
        x = np.asarray(x)
        x_whitened = self.whiten(x - self.mean)
        SSR = np.sum(x_whitened**2, -1)
        llf = -SSR
        llf -= self.nvars * np.log(2. * np.pi)
        llf -= self.logdetcov
        llf *= 0.5
        return llf

    expect_mc = expect_mc


def quad2d(func=lambda x: 1, lower=(-10,-10), upper=(10,10)):
    def fun(x, y):
        x = np.column_stack((x,y))
        return func(x)
    from scipy.integrate import dblquad
    return dblquad(fun, lower[0], upper[0], lambda y: lower[1],
                   lambda y: upper[1])

if __name__ == '__main__':
    mu = (0,0)
    covx = np.array([[1.0, 0.5], [0.5, 1.0]])
    bvn = BivariateNormal(mu, covx)
    rvs = bvn.rvs(size=1000)
    print rvs.mean(0)
    print np.cov(rvs, rowvar=0)
    print bvn.expect()
    print bvn.cdf([0,0])
    bvn1 = BivariateNormal(mu, np.eye(2))
    bvn2 = BivariateNormal(mu, 4*np.eye(2))
    fun = lambda(x) : np.log(bvn1.pdf(x)) - np.log(bvn.pdf(x))
    print bvn1.expect(fun)
    print bvn1.kl(bvn2), bvn1.kl_mc(bvn2)
    print bvn2.kl(bvn1), bvn2.kl_mc(bvn1)
    print bvn1.kl(bvn), bvn1.kl_mc(bvn)
    mvn = MVNormal(mu, covx)
    mvn.pdf([0,0])
    mvn.pdf(np.zeros((2,2)))
    #np.dot(mvn.cholcovinv.T, mvn.cholcovinv) - mvn.covinv

    cov3 = np.array([[ 1.  ,  0.5 ,  0.75],
                     [ 0.5 ,  1.5 ,  0.6 ],
                     [ 0.75,  0.6 ,  2.  ]])
    mu3 = [-1, 0., 2.]
    mvn3 = MVNormal(mu3, cov3)
    mvn3.pdf((0., 2., 3.))
    mvn3.logpdf((0., 2., 3.))


'''
>>> bvn1.expect(lambda x: np.transpose(x)[0])
(0.0, 0.0)
>>> bvn1.expect(lambda x: 0.1+np.transpose(x)[0])
(0.09999999999999995, 8.6710319810497246e-11)
>>> bvn1.expect(lambda x: 0.1+np.transpose(x)[1])
(0.10000000000000014, 1.0691905740743197e-08)
>>> fun = lambda(x) : np.log(bvn1.pdf(x) / bvn2.pdf(x))
>>> bvn1.expect(fun)
(1.8350887222397816, 2.6290829608852095e-09)
>>> stats.norm.pdf(0, scale=2)*stats.norm.pdf(0, scale=2)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'stats' is not defined
>>> from scipy import stats
>>> stats.norm.pdf(0, scale=2)*stats.norm.pdf(0, scale=2)
0.039788735772973836
>>> bvn1.pdf((0,0))
0.15915494309189535
>>> bvn2.pdf((0,0))
0.0099471839432434591
>>>
>>> stats.norm.pdf(0, scale=2)
0.19947114020071635
>>> stats.norm.pdf(0, scale=1)
0.3989422804014327
>>> stats.norm.pdf(0, scale=1)**2
0.15915494309189535
>>>
>>> stats.norm.pdf(0, scale=2.)*stats.norm.pdf(0, scale=2.)
0.039788735772973836
>>> stats.norm.pdf(0, scale=4.)
0.099735570100358176
>>> stats.norm.pdf(0, scale=4.)**2
0.0099471839432434591
>>> bvn2.pdf((0,0))
0.0099471839432434591
>>>
'''
