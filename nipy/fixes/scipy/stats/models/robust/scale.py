import numpy as np
from scipy.stats import norm as Gaussian
import norms


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
# what would this be used for?

def MAD(a, c=Gaussian.ppf(3/4.), axis=0):  # c \approx .6745
    """
    The Median Absolute Deviation along given axis of an array:

    median(abs(a)) / c

    Reference
    ---------
    Venables and Ripley
    """
    a = np.asarray(a)
    return np.median((np.fabs(a))/c, axis=axis)

def stand_MAD(a, c=Gaussian.ppf(3/4.), axis=0):
    """
    The standardized Median Absolute Deviation along given axis of an array:

    MAD = median(abs(a - median(a))) / c
    """
    a = np.asarray(a)
    d = np.median(a, axis = axis)
    d = unsqueeze(d, axis, a.shape)
    return np.median(np.fabs(a - d)/c, axis = axis)

class Huber(object):
    """
    Huber's proposal 2 for estimating scale.

    R Venables, B Ripley. \'Modern Applied Statistics in S\'
    Springer, New York, 2002.
    """

    def __init__(self, c=1.5, tol=1.0e-08, maxiter=30, norm=None):
        """
        Instance of Huber's proposal 2 for estimating
        (location, scale) jointly.

        Inputs:
        -------
        c : float
            Threshold used in threshold for chi=psi**2
        tol : float
            Tolerance for convergence
        maxiter : int
            Maximum number of iterations
        norm : ``norms.RobustNorm``
            A robust norm used in M estimator of location. If None,
            the location estimator defaults to a one-step
            fixed point version of the M-estimator using norms.HuberT
        """
        self.c = c
        self.maxiter = maxiter
        self.tol = tol
        self.norm = norm
        tmp = 2 * Gaussian.cdf(c) - 1
        self.gamma = tmp + c**2 * (1 - tmp) - 2 * c * Gaussian.pdf(c)


    def __call__(self, a, mu=None, scale=None, axis=0):
        """
        Compute Huber\'s proposal 2 estimate of scale, using an optional
        initial value of scale and an optional estimate of mu. If mu
        is supplied, it is not reestimated.

        Given a one-dimensional array a,
        this function minimises the quantity

        sum(psi((a[i]-mu)/scale)**2)

        as a function of (mu, scale), where

        psi(x) = np.clip(x, -self.c, self.c)

        """
        a = np.asarray(a)
        if mu is None:
            n = a.shape[0] - 1
            mu = np.median(a, axis=axis)
            est_mu = True
        else:
            n = a.shape[0]
            mu = mu
            est_mu = False

        if scale is None:
            scale = stand_MAD(a, axis=axis)   # note that MAD got changed
        else:
            scale = scale
        scale = unsqueeze(scale, axis, a.shape) # why??
        mu = unsqueeze(mu, axis, a.shape)
        return self._estimate_both(a, scale, mu, axis, est_mu, n)

    def _estimate_both(self, a, scale, mu, axis, est_mu, n):
        """
        Estimate scale and location simultaneously with the following
        pseudo_loop:

        while not_converged:
            mu, scale = estimate_location(a, scale, mu), estimate_scale(a, scale, mu)

        where estimate_location is an M-estimator and estimate_scale implements
        the check used in Section 5.5 of Venables & Ripley

        """
        for _ in range(self.maxiter):
            # Estimate the mean along a given axis
            if est_mu:
                if self.norm is None:   # it will always be None as written
                                        # allowing to specify norm in IRLS
                                        # resulted in nonconvergence for
                                        # HuberT()
                    # This is a one-step fixed-point estimator
                    # if self.norm == norms.HuberT
                    # It should be faster than using norms.HuberT
                    nmu = np.clip(a, mu-self.c*scale,
                        mu+self.c*scale).sum(axis) / a.shape[axis]
                else:
                    nmu = norms.estimate_location(a, scale, self.norm, axis, mu,                        self.maxiter, self.tol)
            else:
                # Effectively, do nothing
                nmu = mu.squeeze()
            nmu = unsqueeze(nmu, axis, a.shape)

            subset = np.less_equal(np.fabs((a - mu)/scale), self.c)
            card = subset.sum(axis)

            nscale = np.sqrt(np.sum(subset * (a - nmu)**2, axis) \
                    / (n * self.gamma - (a.shape[axis] - card) * self.c**2))

            nscale = unsqueeze(nscale, axis, a.shape)

            test1 = np.alltrue(np.less_equal(np.fabs(scale - nscale),
                        nscale * self.tol))
#            print np.fabs(scale/nscale - 1.).max()
            test2 = np.alltrue(np.less_equal(np.fabs(mu - nmu), nscale*self.tol))
            if not (test1 and test2):
                mu = nmu; scale = nscale
            else:
                return nmu.squeeze(), nscale.squeeze()
        raise ValueError('joint estimation of location and scale failed to converge in %d iterations' % self.maxiter)

huber = Huber()
