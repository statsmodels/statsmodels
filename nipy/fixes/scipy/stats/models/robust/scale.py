import numpy as np
from scipy.stats import norm
from numpy import median

def unsqueeze(data, axis, oldshape):
    """
    unsqueeze a collapsed array

    >>> from numpy import mean
    >>> from numpy.random import standard_normal
    >>> x = standard_normal((3,4,5))
    >>> m = mean(x, axis=1)
    >>> m.shape
    (3, 5)
    >>> m = unsqueeze(m, 1, x.shape)
    >>> m.shape
    (3, 1, 5)
    >>>
    """

    newshape = list(oldshape)
    newshape[axis] = 1
    return data.reshape(newshape)


def MAD(a, c=0.6745, axis=0):
    """
    Median Absolute Deviation along given axis of an array:

    median(abs(a - median(a))) / c

    """

    a = np.asarray(a, np.float64)
    d = median(a, axis=axis)
    d = unsqueeze(d, axis, a.shape)

    return median(np.fabs(a - d) / c, axis=axis)

class Huber:
    """
    Huber's proposal 2 for estimating scale.

    R Venables, B Ripley. \'Modern Applied Statistics in S\'
    Springer, New York, 2002.
    """

    c = 1.5
    tol = 1.0e-06

    tmp = 2 * norm.cdf(c) - 1
    gamma = tmp + c**2 * (1 - tmp) - 2 * c * norm.pdf(c)
    del tmp

    niter = 30

    def __call__(self, a, mu=None, scale=None, axis=0):
        """
        Compute Huber\'s proposal 2 estimate of scale, using an optional
        initial value of scale and an optional estimate of mu. If mu
        is supplied, it is not reestimated.
        """

        self.axis = axis
        self.a = np.asarray(a, np.float64)
        if mu is None:
            self.n = self.a.shape[0] - 1
            self.mu = median(self.a, axis=axis)
            self.est_mu = True
        else:
            self.n = self.a.shape[0]
            self.mu = mu
            self.est_mu = False

        if scale is None:
            self.scale = MAD(self.a, axis=self.axis)**2
        else:
            self.scale = scale

        self.scale = unsqueeze(self.scale, self.axis, self.a.shape)
        self.mu = unsqueeze(self.mu, self.axis, self.a.shape)

        for donothing in self:
            pass

        self.s = np.squeeze(np.sqrt(self.scale))
        del(self.scale); del(self.mu); del(self.a)
        return self.s

    def __iter__(self):
        self.iter = 0
        return self

    def next(self):
        a = self.a
        subset = self.subset(a)
        if self.est_mu:
            mu = np.sum(subset * a + (1 - Huber.c) * subset, axis=self.axis) / a.shape[self.axis]
        else:
            mu = self.mu
        self.axis = unsqueeze(mu, self.axis, self.a.shape)

        scale = np.sum(subset * (a - mu)**2, axis=self.axis) / (self.n * Huber.gamma - np.sum(1. - subset, axis=self.axis) * Huber.c**2)

        self.iter += 1

        if np.alltrue(np.less_equal(np.fabs(np.sqrt(scale) - np.sqrt(self.scale)), np.sqrt(self.scale) * Huber.tol)) and np.alltrue(np.less_equal(np.fabs(mu - self.mu), np.sqrt(self.scale) * Huber.tol)):
            self.scale = scale
            self.mu = mu
            raise StopIteration
        else:
            self.scale = scale
            self.mu = mu

        self.scale = unsqueeze(self.scale, self.axis, self.a.shape)

        if self.iter >= self.niter:
            raise StopIteration

    def subset(self, a):
        tmp = (a - self.mu) / np.sqrt(self.scale)
        return np.greater(tmp, -Huber.c) * np.less(tmp, Huber.c)

huber = Huber()
