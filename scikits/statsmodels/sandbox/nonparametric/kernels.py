# -*- coding: utf-8 -*-
#
"""
This models contains the Kernels for Kernel smoothing.

Hopefully in the future they may be reused/extended for other kernel based
method

References:
----------

Pointwise Kernel Confidence Bounds
(smoothconf)
http://fedc.wiwi.hu-berlin.de/xplore/ebooks/html/anr/anrhtmlframe62.html
"""

# pylint: disable-msg=C0103
# pylint: disable-msg=W0142
# pylint: disable-msg=E1101
# pylint: disable-msg=E0611

import math
import numpy as np
import scipy.integrate
from numpy import exp, multiply, square, divide, subtract, inf

class NdKernel(object):
    """Generic N-dimensial kernel

    Parameters
    ----------
    n : int
        The number of series for kernel estimates
    kernels : list
        kernels

    Can be constructed from either
    a) a list of n kernels which will be treated as
    indepent marginals on a gaussian copula (specified by H)
    or b) a single univariate kernel which will be applied radially to the
    mahalanobis distance defined by H.

    In the case of the Gaussian these are both equivalent, and the second constructiong
    is prefered.
    """
    def __init__(self, n, kernels = None, H = None):
        if kernels is None:
            kernels = Gaussian()

        self._kernels = kernels

        if H is None:
            H = np.matrix( np.identity(n))

        self._H = H
        self._Hrootinv = np.linalg.cholesky( H.I )

    def getH(self):
        """Getter for kernel bandwidth, H"""
        return self._H

    def setH(self, value):
        """Setter for kernel bandwidth, H"""
        self._H = value

    H = property(getH, setH, doc="Kernel bandwidth matrix")

    def density(self, xs, x):
        n = len(xs)
        #xs = self.inDomain( xs, xs, x )[0]

        if len(xs)>0:  ## Need to do product of marginal distributions
            w = np.sum([self(self._Hrootinv * (xx-x) ) for xx in xs])/n
            return w
        else:
            return np.nan

    def _kernweight(self, x ):
        """returns the kernel weight for the independent multivariate kernel"""
        if isinstance( self._kernels, CustomKernel ):
            ## Radial case
            d = math.sqrt( x.T * x )
            return self._kernels( d )


class CustomKernel(object):
    """
    Generic 1D Kernel object.
    Can be constructed by selecting a standard named Kernel,
    or providing a lambda expression and domain.
    The domain allows some algorithms to run faster for finite domain kernels.
    """
    # MC: Not sure how this will look in the end - or even still exist.
    # Main purpose of this is to allow custom kernels and to allow speed up
    # from finite support.

    def __init__(self, shape, h = 1.0, domain = None, norm = None):
        """
        shape should be a lambda taking and returning numeric type.

        For sanity it should always return positive or zero but this isn't
        enforced incase you want to do weird things.  Bear in mind that the
        statistical tests etc. may not be valid for non-positive kernels.

        The bandwidth of the kernel is supplied as h.

        You may specify a domain as a list of 2 values [min,max], in which case
        kernel will be treated as zero outside these values.  This will speed up
        calculation.

        You may also specify the normalisation constant for the supplied Kernel.
        If you do this number will be stored and used as the normalisation
        without calculation.  It is recommended you do this if you know the
        constant, to speed up calculation.  In particular if the shape function
        provided is already normalised you should provide
        norm = 1.0
        or
        norm = True
        """
        if norm is True:
            norm = 1.0
        self._normconst = norm
        self.domain = domain
        if callable(shape):
            self._shape = shape
        else:
            raise TypeError("shape must be a callable object/function")
        self._h = h
        self._L2Norm = None

    def geth(self):
        """Getter for kernel bandwidth, h"""
        return self._h
    def seth(self, value):
        """Setter for kernel bandwidth, h"""
        self._h = value
    h = property(geth, seth, doc="Kernel Bandwidth")

    def inDomain(self, xs, ys, x):
        """
        Returns the filtered (xs, ys) based on the Kernel domain centred on x
        """
        # Disable black-list functions: filter used for speed instead of
        # list-comprehension
        # pylint: disable-msg=W0141
        def isInDomain(xy):
            """Used for filter to check if point is in the domain"""
            u = (xy[0]-x)/self.h
            return u >= self.domain[0] and u <= self.domain[1]

        if self.domain is None:
            return (xs, ys)
        else:
            filtered = filter(isInDomain, zip(xs, ys))
            if len(filtered) > 0:
                xs, ys = zip(*filtered)
                return (xs, ys)
            else:
                return ([], [])

    def density(self, xs, x):
        """Returns the kernel density estimate for point x based on x-values
        xs
        """
        xs = np.asarray(xs)
        n = len(xs) # before inDomain?
        xs = self.inDomain( xs, xs, x )[0]
        if xs.ndim == 1:
            xs = xs[:,None]
        if len(xs)>0:
            h = self.h
            w = 1/h * np.mean(self((xs-x)/h), axis=0)
            return w
        else:
            return np.nan

    def smooth(self, xs, ys, x):
        """Returns the kernel smoothing estimate for point x based on x-values
        xs and y-values ys.
        Not expected to be called by the user.
        """
        xs, ys = self.inDomain(xs, ys, x)

        if len(xs)>0:
            w = np.sum(self((xs-x)/self.h))
            #TODO: change the below to broadcasting when shape is sorted
            v = np.sum([yy*self((xx-x)/self.h) for xx, yy in zip(xs, ys)])
            return v / w
        else:
            return np.nan

    def smoothvar(self, xs, ys, x):
        """Returns the kernel smoothing estimate of the variance at point x.
        """
        xs, ys = self.inDomain(xs, ys, x)

        if len(xs) > 0:
            fittedvals = np.array([self.smooth(xs, ys, xx) for xx in xs])
            sqresid = square( subtract(ys, fittedvals) )
            w = np.sum(self((xs-x)/self.h))
            v = np.sum([rr*self((xx-x)/self.h) for xx, rr in zip(xs, sqresid)])
            return v / w
        else:
            return np.nan

    def smoothconf(self, xs, ys, x):
        """Returns the kernel smoothing estimate with confidence 1sigma bounds
        """
        xs, ys = self.inDomain(xs, ys, x)

        if len(xs) > 0:
            fittedvals = np.array([self.smooth(xs, ys, xx) for xx in xs])
            sqresid = square(
                subtract(ys, fittedvals)
            )
            w = np.sum(self((xs-x)/self.h))
            v = np.sum([rr*self((xx-x)/self.h) for xx, rr in zip(xs, sqresid)])
            var = v / w
            sd = np.sqrt(var)
            K = self.L2Norm
            yhat = self.smooth(xs, ys, x)
            err = sd * K / np.sqrt(w * self.h * self.norm_const)
            return (yhat - err, yhat, yhat + err)
        else:
            return (np.nan, np.nan, np.nan)

    @property
    def L2Norm(self):
        """Returns the integral of the square of the kernal from -inf to inf"""
        if self._L2Norm is None:
            L2Func = lambda x: (self.norm_const*self._shape(x))**2
            if self.domain is None:
                self._L2Norm = scipy.integrate.quad(L2Func, -inf, inf)[0]
            else:
                self._L2Norm = scipy.integrate.quad(L2Func, self.domain[0],
                                               self.domain[1])[0]
        return self._L2Norm

    @property
    def norm_const(self):
        """
        Normalising constant for kernel (integral from -inf to inf)
        """
        if self._normconst is None:
            if self.domain is None:
                quadres = scipy.integrate.quad(self._shape, -inf, inf)
            else:
                quadres = scipy.integrate.quad(self._shape, self.domain[0],
                                               self.domain[1])
            self._normconst = 1.0/(quadres[0])
        return self._normconst

    def weight(self, x):
        """This returns the normalised weight at distance x"""
        return self.norm_const*self._shape(x)

    def __call__(self, x):
        """
        This simply returns the value of the kernel function at x

        Does the same as weight if the function is normalised
        """
        return self._shape(x)

class Uniform(CustomKernel):
    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 0.5, h=h,
                              domain=[-1.0, 1.0], norm = 1.0)
        self._L2Norm = 0.5

class Triangular(CustomKernel):
    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 1 - abs(x), h=h,
                              domain=[-1.0, 1.0], norm = 1.0)
        self._L2Norm = 2.0/3.0

class Epanechnikov(CustomKernel):
    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 0.75*(1 - x*x), h=h,
                              domain=[-1.0, 1.0], norm = 1.0)
        self._L2Norm = 0.6

class Biweight(CustomKernel):
    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 0.9375*(1 - x*x)**2, h=h,
                              domain=[-1.0, 1.0], norm = 1.0)
        self._L2Norm = 5.0/7.0

    def smooth(self, xs, ys, x):
        """Returns the kernel smoothing estimate for point x based on x-values
        xs and y-values ys.
        Not expected to be called by the user.

        Special implementation optimised for Biweight.
        """
        xs, ys = self.inDomain(xs, ys, x)

        if len(xs) > 0:
            w = np.sum(square(subtract(1, square(divide(subtract(xs, x),
                                                        self.h)))))
            v = np.sum(multiply(ys, square(subtract(1, square(divide(
                                                subtract(xs, x), self.h))))))
            return v / w
        else:
            return np.nan

    def smoothvar(self, xs, ys, x):
        """
        Returns the kernel smoothing estimate of the variance at point x.
        """
        xs, ys = self.inDomain(xs, ys, x)

        if len(xs) > 0:
            fittedvals = np.array([self.smooth(xs, ys, xx) for xx in xs])
            rs = square(subtract(ys, fittedvals))
            w = np.sum(square(subtract(1.0, square(divide(subtract(xs, x),
                                                        self.h)))))
            v = np.sum(multiply(rs, square(subtract(1, square(divide(
                                                subtract(xs, x), self.h))))))
            return v / w
        else:
            return np.nan

    def smoothconf(self, xs, ys, x):
        """Returns the kernel smoothing estimate with confidence 1sigma bounds
        """
        xs, ys = self.inDomain(xs, ys, x)

        if len(xs) > 0:
            fittedvals = np.array([self.smooth(xs, ys, xx) for xx in xs])
            rs = square(subtract(ys, fittedvals))
            w = np.sum(square(subtract(1.0, square(divide(subtract(xs, x),
                                                        self.h)))))
            v = np.sum(multiply(rs, square(subtract(1, square(divide(
                                                subtract(xs, x), self.h))))))
            var = v / w
            sd = np.sqrt(var)
            K = self.L2Norm
            yhat = self.smooth(xs, ys, x)
            err = sd * K / np.sqrt(0.9375 * w * self.h)
            return (yhat - err, yhat, yhat + err)
        else:
            return (np.nan, np.nan, np.nan)

class Triweight(CustomKernel):
    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 1.09375*(1 - x*x)**3, h=h,
                              domain=[-1.0, 1.0], norm = 1.0)
        self._L2Norm = 350.0/429.0

class Gaussian(CustomKernel):
    """
    Gaussian (Normal) Kernel

    K(u) = 1 / (sqrt(2*pi)) exp(-0.5 u**2)
    """
    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape = lambda x: 0.3989422804014327 *
                        np.exp(-x**2/2.0), h = h, domain = None, norm = 1.0)
        self._L2Norm = 1.0/(2.0*np.sqrt(np.pi))

    def smooth(self, xs, ys, x):
        """Returns the kernel smoothing estimate for point x based on x-values
        xs and y-values ys.
        Not expected to be called by the user.

        Special implementation optimised for Gaussian.
        """
        w = np.sum(exp(multiply(square(divide(subtract(xs, x),
                                              self.h)),-0.5)))
        v = np.sum(multiply(ys, exp(multiply(square(divide(subtract(xs, x),
                                                          self.h)), -0.5))))
        return v/w

class Cosine(CustomKernel):
    """
    Cosine Kernel

    K(u) = pi/4 cos(0.5 * pi * u) between -1.0 and 1.0
    """
    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 0.78539816339744828 *
                np.cos(np.pi/2.0 * x), h=h, domain=[-1.0, 1.0], norm = 1.0)
        self._L2Norm = np.pi**2/16.0
