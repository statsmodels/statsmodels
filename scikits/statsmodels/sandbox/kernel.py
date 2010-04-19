# -*- coding: utf-8 -*-
"""
This models contains the Kernels for Kernel smoothing.

Hopefully in the future they may be reused/extended for other kernel based method

References:
----------

Pointwise Kernel Confidence Bounds
(smoothconf)
http://fedc.wiwi.hu-berlin.de/xplore/ebooks/html/anr/anrhtmlframe62.html
"""

import numpy as np
from numpy import exp, multiply, square, divide, subtract

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

    def __init__(self, shape, h = 1.0, domain = None):
        """
        shape should be a lambda taking and returning numeric type.

        For sanity it should always return positive or zero but this isn't
        enforces.
        """
        self.domain = domain
        # TODO: Add checking code that shape is valid
        self._shape = shape
        self.h = h

    def smooth(self, xs, ys, x):
        """Returns the kernel smoothing estimate for point x based on x-values
        xs and y-values ys.
        Not expected to be called by the user.
        """
        # TODO: make filtering more efficient
        # TODO: Implement WARPing algorithm when possible
        if self.domain is None:
            filtered = zip(xs, ys)
        else:
            filtered = [
                (xx,yy)
                for xx,yy in zip(xs,ys)
                if (xx-x)/self.h >= self.domain[0]
                and (xx-x)/self.h <= self.domain[1]
            ]

        if len(filtered) > 0:
            xs,ys = zip(*filtered)
            w = np.sum([self((xx-x)/self.h) for xx in xs])
            v = np.sum([yy*self((xx-x)/self.h) for xx, yy in zip(xs,ys)])
            return v/w
        else:
            return np.nan

    def smoothvar(self, xs, ys, x):
        """Returns the kernel smoothing estimate of the variance at point x.
        """
        # TODO: linear interpolation of the fit can speed this up for large
        # datasets without noticable error.
        # TODO: this is thoroughly inefficient way of doing this at the moment
        # but it works for small dataset - remove repeated w calc.
        if self.domain is None:
            filtered = zip(xs, ys)
        else:
            filtered = [
                (xx,yy)
                for xx,yy in zip(xs, ys)
                if (xx-x)/self.h >= self.domain[0]
                and (xx-x)/self.h <= self.domain[1]
            ]

        if len(filtered) > 0:
            xs,ys = zip(*filtered)
            fittedvals = np.array([self.smooth(xs, ys, xx) for xx in xs])
            sqresid = square(
                subtract(ys, fittedvals)
            )
            w = np.sum([self((xx-x)/self.h) for xx in xs])
            v = np.sum([rr*self((xx-x)/self.h) for xx, rr in zip(xs, rr)])
            var = v/w
        else:
            return np.nan

    def smoothconf(self, xs, ys, x):
        """Returns the kernel smoothing estimate with confidence 1sigma bounds
        """
        # TODO:  This is a HORRIBLE implementation - needs unhorribleising
        # Also without the correct L2Norm and norm_const this will be out by a
        # factor.
        # Also I think there is a bug in this.
        if self.domain is None:
            filtered = zip(xs, ys)
        else:
            filtered = [
                (xx,yy)
                for xx,yy in zip(xs, ys)
                if (xx-x)/self.h >= self.domain[0]
                and (xx-x)/self.h <= self.domain[1]
            ]

        if len(filtered) > 0:
            xs,ys = zip(*filtered)
            fittedvals = np.array([self.smooth(xs, ys, xx) for xx in xs])
            sqresid = square(
                subtract(ys, fittedvals)
            )
            w = np.sum([self((xx-x)/self.h) for xx in xs])
            v = np.sum([rr*self((xx-x)/self.h) for xx, rr in zip(xs, sqresid)])
            var = v/w
            sd = np.sqrt(var)
            K = self.L2Norm()
            yhat = self.smooth(xs, ys, x)
            err = sd * K/np.sqrt(w)
            return (yhat-err, yhat, yhat+err)
        else:
            return (np.nan, np.nan, np.nan)

    # TODO: make this a property
    def L2Norm(self):
        """Returns the integral of the square of the kernal from -inf to inf"""
        #TODO: yeah right.  For now just stick a number here
        # easy enough to sort this for the specific kernels
        # will use scipy or similar for custom kernels
        return 1

    # TODO: make this a property
    def norm_const(self, x):
        """Normalising constant for kernel (integral from -inf to inf)"""
        # TODO: again, this needs sorting
        # will use scipy or similar for custom kernels
        return 1

    def __call__(self, x):
        return self._shape(x)



class DefaultKernel(object):
    """Represents the default kernel - should not be used directly
    This contains no functionality and acts as a placeholder for the consuming
    function.
    """
    pass

class Uniform(CustomKernel):
    def __init__(self, h=1.0):
        self.h = h
        self._shape = lambda x: 1.0
        self.domain = [-1.0,1.0]

class Triangular(CustomKernel):
    def __init__(self, h=1.0):
        self.h = h
        self._shape = lambda x: 1-abs(x)
        self.domain = [-1.0,1.0]

class Epanechnikov(CustomKernel):
    def __init__(self, h=1.0):
        self.h = h
        self._shape = lambda x: (1-x*x)
        self.domain = [-1.0,1.0]

class Biweight(CustomKernel):
    def __init__(self, h=1.0):
        self.h = h
        self._shape = lambda x: (1-x*x)**2
        self.domain = [-1.0,1.0]

class Triweight(CustomKernel):
    def __init__(self, h=1.0):
        self.h = h
        self._shape = lambda x: (1-x*x)**3
        self.domain = [-1.0,1.0]

class Gaussian(CustomKernel):
    def __init__(self, h=1.0):
        self.h = h
        self._shape = lambda x: np.exp(-x**2/2.0)
        self.domain = None

    def smooth(self, xs, ys, x):
        w = np.sum(exp(multiply(square(divide(subtract(xs, x),
                                              self.h)),-0.5)))

        v = np.sum(multiply(ys,exp(multiply(square(divide(subtract( xs, x),
                                                          self.h)),-0.5))))

        return v/w

    def norm_const(self):
        return 0.3989422804014327  # 1/sqrt(2 pi)

    def L2Norm(self):
        return 0.88622692545275794   # sqrt(pi)/2

class Cosine(CustomKernel):
    def __init__(self, h=1.0):
        self.h = h
        self._shape = lambda x: np.cos(np.pi/2.0 * x)
        self.domain = [-1.0,1.0]
