# -*- coding: utf-8 -*-
"""
This models contains the Kernels for Kernel smoothing.

Hopefully in the future they may be reused/extended for other kernel based method
"""
class Kernel(object):
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

        For sanity it should always return positive or zero.
        """
        self.domain = domain
        # TODO: Add checking code that shape is valid
        self._shape = shape
        self.h = h

    def evaluate(self, xs, ys, x):
        # TODO: make filtering more efficient
        filtered = [(xx,yy) for xx,yy in zip(xs,ys) if (xx-x)/self.h >= self.domain[0] and (xx-x)/self.h <= self.domain[1]]
        if len(filtered) > 0:
            xs,ys = zip(*filtered)
            w = np.sum([self((xx-x)/self.h) for xx in xs])
            v = np.sum([yy*self((xx-x)/self.h) for xx, yy in zip(xs,ys)])
            return v/w
        else:
            return 0

    def __call__(self, x):
        return self._shape(x)

class Gaussian(Kernel):
    def __init__(self, h=1.0):
            self.h = h
            self._shape = lambda x: np.exp(-x**2/2.0)

