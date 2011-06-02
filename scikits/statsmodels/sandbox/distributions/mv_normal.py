# -*- coding: utf-8 -*-
"""Multivariate Normal and t distributions



Created on Sat May 28 15:38:23 2011

@author: Josef Perktold

TODO: rename again,
    after adding t distribution, cov doesn't make sense for Sigma    DONE
    I kept the original MVNormal0 class as reference, can be deleted

"""

import numpy as np

from scikits.statsmodels.sandbox.distributions.multivariate import (
                mvstdtprob, mvstdnormcdf)

def expect_mc(dist, func=lambda x: 1, size=50000):
    def fun(x):
        return func(x) * dist.pdf(x)
    rvs = dist.rvs(size=size)
    return fun(rvs).mean(0)

def expect_mc_bounds(dist, func=lambda x: 1, size=50000, lower=None, upper=None,
                     overfact=1.2):
    '''calculate expected value of function by Monte Carlo integration

    Notes
    -----
    this doesn't batch

    not checked yet

    '''
    def fun(x):
        return func(x) * dist.pdf(x)

    rvsli = []
    remain = size
    while True:
        rvs = dist.rvs(size=int(size * overfact))
        rvsok = rvs[(rvs >= lower) & (rvs <= upper)]
        #if rvsok.ndim == 1: #possible shape problems if only 1 random vector
        rvsok = np.atleast_2d(rvsok)
        remain -= rvsok.shape[0]
        rvsli.append(rvsok[:remain])
        if remain <= 0: break
    rvs = np.vstack(rvsli)
    return fun(rvs).mean(0)


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

class MVElliptical(object):
    '''Base Class for multivariate elliptical distributions, normal and t

    contains common initialization, and some common methods
    subclass needs to implement at least rvs and logpdf methods

    '''
    #getting common things between normal and t distribution


    def __init__(self, mean, sigma, *args, **kwds):
        self.mean = mean
        self.sigma = sigma = np.asarray(sigma)
        sigma = np.squeeze(sigma)
        self.nvars = nvars = len(mean)
        #self.covchol = np.linalg.cholesky(sigma)


        #in the following sigma is original, self.sigma is full matrix
        if sigma.shape == ():
            #iid
            self.sigma = np.eye(nvars) * sigma
            self.sigmainv = np.eye(nvars) / sigma
            self.cholsigmainv = np.eye(nvars) / np.sqrt(sigma)
        elif (sigma.ndim == 1) and (len(sigma) == nvars):
            #independent heteroscedastic
            self.sigma = np.diag(sigma)
            self.sigmainv = np.diag(1. / sigma)
            self.cholsigmainv = np.diag( 1. / np.sqrt(sigma))
        elif sigma.shape == (nvars, nvars): #python tuple comparison
            #general
            self.sigmainv = np.linalg.pinv(sigma)
            self.cholsigmainv = np.linalg.cholesky(self.sigmainv).T
        else:
            raise ValueError('sigma has invalid shape')

        #store logdetsigma for logpdf
        self.logdetsigma = np.log(np.linalg.det(self.sigma))

    def rvs(self, size=1):
        '''random variable

        Parameters
        ----------
        size : int or tuple
            the number and shape of random variables to draw.

        Returns
        -------
        rvs : ndarray
            the returned random variables with shape given by size and the
            dimension of the multivariate random vector as additional last
            dimension

        Notes
        -----
        uses numpy.random.multivariate_normal directly

        '''
        raise NotImplementedError

    def logpdf(self, x):
        '''logarithm of probability density function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector

        Returns
        -------
        logpdf : float or array
            probability density value of each random vector


        this should be made to work with 2d x,
        with multivariate normal vector in each row and iid across rows
        doesn't work now because of dot in whiten

        '''
        raise NotImplementedError

    def whiten(self, x):
        """
        whiten the data by linear transformation

        Parameters
        -----------
        x : array-like, 1d or 2d
            Data to be whitened, if 2d then each row contains an independent
            sample of the multivariate random vector

        Returns
        -------
        np.dot(x, self.cholsigmainv.T)

        Notes
        -----
        This only does rescaling, it doesn't subtract the mean, use standardize
        for this instead

        See Also
        --------
        standardize : subtract mean and rescale to standardized random variable.

        """
        x = np.asarray(x)
        return np.dot(x, self.cholsigmainv.T)

    def pdf(self, x):
        '''probability density function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector

        Returns
        -------
        pdf : float or array
            probability density value of each random vector

        '''
        return np.exp(self.logpdf(x))

    def standardize(self, x):
        '''standardize the random variable, i.e. subtract mean and whiten

        Parameters
        -----------
        x : array-like, 1d or 2d
            Data to be whitened, if 2d then each row contains an independent
            sample of the multivariate random vector

        Returns
        -------
        np.dot(x - self.mean, self.cholsigmainv.T)

        Notes
        -----


        See Also
        --------
        whiten : rescale random variable, standardize without subtracting mean.


        '''
        return self.whiten(x - self.mean)

    @property
    def std(self):
        '''standard deviation, square root of diagonal elements of cov
        '''
        return np.sqrt(np.diag(self.cov))


    @property
    def corr(self):
        '''correlation matrix'''
        return self.cov / np.outer(self.std, self.std)

    expect_mc = expect_mc


#parts taken from linear_model, but heavy adjustments
class MVNormal0(object):
    '''Class for Multivariate Normal Distribution

    original full version, kept for testing, new version inherits from
    MVElliptical

    uses Cholesky decomposition of covariance matrix for the transformation
    of the data

    '''


    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov = np.asarray(cov)
        cov = np.squeeze(cov)
        self.nvars = nvars = len(mean)


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
        whiten the data by linear transformation

        Parameters
        -----------
        X : array-like, 1d or 2d
            Data to be whitened, if 2d then each row contains an independent
            sample of the multivariate random vector

        Returns
        -------
        np.dot(x, self.cholcovinv.T)

        Notes
        -----
        This only does rescaling, it doesn't subtract the mean, use standardize
        for this instead

        See Also
        --------
        standardize : subtract mean and rescale to standardized random variable.

        """
        x = np.asarray(x)
        if np.any(self.cov):
            #return np.dot(self.cholcovinv, x)
            return np.dot(x, self.cholcovinv.T)
        else:
            return x

    def rvs(self, size=1):
        '''random variable

        Parameters
        ----------
        size : int or tuple
            the number and shape of random variables to draw.

        Returns
        -------
        rvs : ndarray
            the returned random variables with shape given by size and the
            dimension of the multivariate random vector as additional last
            dimension

        Notes
        -----
        uses numpy.random.multivariate_normal directly

        '''
        return np.random.multivariate_normal(self.mean, self.cov, size=size)

    def pdf(self, x):
        '''probability density function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector

        Returns
        -------
        pdf : float or array
            probability density value of each random vector

        '''

        return np.exp(self.logpdf(x))

    def logpdf(self, x):
        '''logarithm of probability density function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector

        Returns
        -------
        logpdf : float or array
            probability density value of each random vector


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


class MVNormal(MVElliptical):
    '''Class for Multivariate Normal Distribution

    uses Cholesky decomposition of covariance matrix for the transformation
    of the data

    '''

    def rvs(self, size=1):
        '''random variable

        Parameters
        ----------
        size : int or tuple
            the number and shape of random variables to draw.

        Returns
        -------
        rvs : ndarray
            the returned random variables with shape given by size and the
            dimension of the multivariate random vector as additional last
            dimension

        Notes
        -----
        uses numpy.random.multivariate_normal directly

        '''
        return np.random.multivariate_normal(self.mean, self.sigma, size=size)

    def logpdf(self, x):
        '''logarithm of probability density function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector

        Returns
        -------
        logpdf : float or array
            probability density value of each random vector


        this should be made to work with 2d x,
        with multivariate normal vector in each row and iid across rows
        doesn't work now because of dot in whiten

        '''
        x = np.asarray(x)
        x_whitened = self.whiten(x - self.mean)
        SSR = np.sum(x_whitened**2, -1)
        llf = -SSR
        llf -= self.nvars * np.log(2. * np.pi)
        llf -= self.logdetsigma
        llf *= 0.5
        return llf

    def cdf(self, x, **kwds):
        '''TODO: test the normalization in here or use non-standardized cdf'''
        lower = -np.inf * np.ones_like(x)
        return mvstdnormcdf(lower, self.standardize(upper), self.corr, **kwds)

    @property
    def cov(self):
        '''covariance matrix'''
        return self.sigma



from scipy import special
#redefine some shortcuts
np_log = np.log
np_pi = np.pi
sps_gamln = special.gammaln

class MVT(MVElliptical):

    def __init__(self, mean, sigma, df):
        self.df = df
        super(MVT, self).__init__(mean, sigma)

    def rvs(self, size=1):
        from multivariate import multivariate_t_rvs
        return multivariate_t_rvs(self.mean, self.sigma, df=self.df, n=size)


    def logpdf(self, x):

        x = np.asarray(x)

        df = self.df
        nvars = self.nvars

        x_whitened = self.whiten(x - self.mean) #should be float

        llf = - nvars * np_log(df * np_pi)
        llf -= self.logdetsigma
        llf -= (df + nvars) * np_log(1 + np.sum(x_whitened**2,-1) / df)
        llf *= 0.5
        llf += sps_gamln((df + nvars) / 2.) - sps_gamln(df / 2.)

        return llf

    def cdf(self, x, **kwds):
        '''TODO: test the normalization in here'''
        lower = -np.inf * np.ones_like(x)
        return mvstdtcdf(lower, self.standardize(upper), self.corr, df, **kwds)

    @property
    def cov(self):
        if self.df <= 2:
            return np.nan * np.ones_like(self.sigma)
        else:
            return self.df / (self.df - 2.) * self.sigma


def quad2d(func=lambda x: 1, lower=(-10,-10), upper=(10,10)):
    def fun(x, y):
        x = np.column_stack((x,y))
        return func(x)
    from scipy.integrate import dblquad
    return dblquad(fun, lower[0], upper[0], lambda y: lower[1],
                   lambda y: upper[1])

if __name__ == '__main__':

    from numpy.testing import assert_almost_equal, assert_array_almost_equal

    examples = ['mvn']

    mu = (0,0)
    covx = np.array([[1.0, 0.5], [0.5, 1.0]])
    mu3 = [-1, 0., 2.]
    cov3 = np.array([[ 1.  ,  0.5 ,  0.75],
                     [ 0.5 ,  1.5 ,  0.6 ],
                     [ 0.75,  0.6 ,  2.  ]])


    if 'mvn' in examples:
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
        #comparisons with R mvtnorm::dmvnorm
        #decimal=14
#        mvn3.logpdf(cov3) - [-7.667977543898155, -6.917977543898155, -5.167977543898155]
#        #decimal 18
#        mvn3.pdf(cov3) - [0.000467562492721686, 0.000989829804859273, 0.005696077243833402]
#        #cheating new mean, same cov
#        mvn3.mean = np.array([0,0,0])
#        #decimal= 16
#        mvn3.pdf(cov3) - [0.02914269740502042, 0.02269635555984291, 0.01767593948287269]

        #as asserts
        r_val = [-7.667977543898155, -6.917977543898155, -5.167977543898155]
        assert_array_almost_equal( mvn3.logpdf(cov3), r_val, decimal = 14)
        #decimal 18
        r_val = [0.000467562492721686, 0.000989829804859273, 0.005696077243833402]
        assert_array_almost_equal( mvn3.pdf(cov3), r_val, decimal = 17)
        #cheating new mean, same cov, too dangerous, got wrong instance in tests
        #mvn3.mean = np.array([0,0,0])
        mvn3c = MVNormal(np.array([0,0,0]), cov3)
        r_val = [0.02914269740502042, 0.02269635555984291, 0.01767593948287269]
        assert_array_almost_equal( mvn3c.pdf(cov3), r_val, decimal = 16)

        mvn3b = MVNormal((0,0,0), 1)
        fun = lambda(x) : np.log(mvn3.pdf(x)) - np.log(mvn3b.pdf(x))
        print mvn3.expect_mc(fun)
        print mvn3.expect_mc(fun, size=200000)


    mvt = MVT((0,0), 1, 5)
    assert_almost_equal(mvt.logpdf(np.array([0.,0.])), -1.837877066409345,
                        decimal=15)
    assert_almost_equal(mvt.pdf(np.array([0.,0.])), 0.1591549430918953,
                        decimal=15)

    mvt.logpdf(np.array([1.,1.]))-(-3.01552989458359)

    mvt1 = MVT((0,0), 1, 1)
    mvt1.logpdf(np.array([1.,1.]))-(-3.48579549941151) #decimal=16

    rvs = mvt.rvs(100000)
    assert_almost_equal(np.cov(rvs, rowvar=0), mvt.cov, decimal=1)

    mvt31 = MVT(mu3, cov3, 1)
    assert_almost_equal(mvt31.pdf(cov3),
        [0.0007276818698165781, 0.0009980625182293658, 0.0027661422056214652],
        decimal=18)

    mvt = MVT(mu3, cov3, 3)
    assert_almost_equal(mvt.pdf(cov3),
        [0.000863777424247410, 0.001277510788307594, 0.004156314279452241],
        decimal=17)

