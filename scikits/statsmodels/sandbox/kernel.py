# -*- coding: utf-8 -*-
"""
This models contains the Kernels for Kernel smoothing.

Hopefully in the future they may be reused/extended for other kernel based method
"""

import numpy as np
from numpy import exp, multiply, square, divide, subtract

class Custom(object):
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
        # TODO: make filtering more efficient
        if self.domain is None:
            filtered = zip(xs, ys)
        else:
            filtered = [(xx,yy) for xx,yy in zip(xs,ys) if (xx-x)/self.h >= self.domain[0] and (xx-x)/self.h <= self.domain[1]]
        if len(filtered) > 0:
            xs,ys = zip(*filtered)
            w = np.sum([self((xx-x)/self.h) for xx in xs])
            v = np.sum([yy*self((xx-x)/self.h) for xx, yy in zip(xs,ys)])
            return v/w
        else:
            return np.nan

    def __call__(self, x):
        return self._shape(x)

    def norm_const(self, x):
        """Don't need normalisation for Kernel Regression"""
        raise NotImplementedError

class Default(object):
    """Represents the default kernel - should not be used directly
    This contains no functionality and acts as a placeholder for the consuming
    function.
    """
    pass

class Uniform(Custom):
    def __init__(self, h=1.0):
        self.h = h
        self._shape = lambda x: 1.0
        self.domain = [-1.0,1.0]

class Triangular(Custom):
    def __init__(self, h=1.0):
        self.h = h
        self._shape = lambda x: 1-abs(x)
        self.domain = [-1.0,1.0]

class Epanechnikov(Custom):
    def __init__(self, h=1.0):
        self.h = h
        self._shape = lambda x: (1-x*x)
        self.domain = [-1.0,1.0]

class Biweight(Custom):
    def __init__(self, h=1.0):
        self.h = h
        self._shape = lambda x: (1-x*x)**2
        self.domain = [-1.0,1.0]

class Triweight(Custom):
    def __init__(self, h=1.0):
        self.h = h
        self._shape = lambda x: (1-x*x)**3
        self.domain = [-1.0,1.0]

class Gaussian(Custom):
    def __init__(self, h=1.0):
        self.h = h
        self._shape = lambda x: np.exp(-x**2/2.0)
        self.domain = None

    def smooth(self, xs, ys, x):
        w = np.sum( exp( multiply( square( divide( subtract( xs, x), self.h)), -0.5)))
        v = np.sum( multiply( ys, exp( multiply( square( divide( subtract( xs, x), self.h)), -0.5))))
        return v/w

class Cosine(Custom):
    def __init__(self, h=1.0):
        self.h = h
        self._shape = lambda x: np.cos(np.pi/2.0 * x)
        self.domain = [-1.0,1.0]
