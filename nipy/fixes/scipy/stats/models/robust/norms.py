import numpy as np

class RobustNorm(object):

    def rho(self, z):
        """
        Abstract method:

        -2 log L used in M-estimator
        """
        raise NotImplementedError

    def psi(self, z):
        """
        Abstract method:

        psi = rho'
        """
        raise NotImplementedError

    def weights(self, z):
        """
        Abstract method:
        psi(z) / z

        TODO: is the above correct... need to look at V&R
        """
        raise NotImplementedError

    def __call__(self, z):
        return self.rho(z)

class LeastSquares(RobustNorm):

    """
    Least squares rho for M estimation.

    DC Montgomery, EA Peck. \'Introduction to Linear Regression Analysis\',
    John Wiley and Sons, Inc., New York, 2001.

    """

    def rho(self, z):
        return z**2 * 0.5

    def psi(self, z):
        return np.asarray(z)

    def weights(self, z):
        return np.ones(z.shape, np.float64)

class HuberT(RobustNorm):

    """
    Huber\'s T for M estimation.

    DC Montgomery, EA Peck. \'Introduction to Linear Regression Analysis\',
    John Wiley and Sons, Inc., New York, 2001.

    R Venables, B Ripley. \'Modern Applied Statistics in S\'
    Springer, New York, 2002.

    """

    def __init__(self, t=1.345):
        self.t = t

    def subset(self, z):
        z = np.asarray(z)
        return np.less_equal(np.fabs(z), self.t)

    def rho(self, z):
        z = np.asarray(z)
        test = self.subset(z)
        return (test * 0.5 * z**2 +
                (1 - test) * (np.fabs(z) * self.t - 0.5 * self.t**2))

    def psi(self, z):
        z = np.asarray(z)
        test = self.subset(z)
        return test * z + (1 - test) * self.t * np.sign(z)

    def weights(self, z):
        z = np.asarray(z)
        test = self.subset(z)
        return test + (1 - test) * self.t / np.fabs(z)

class RamsayE(RobustNorm):

    """
    Ramsay\'s Ea for M estimation.

    DC Montgomery, EA Peck. \'Introduction to Linear Regression Analysis\',
    John Wiley and Sons, Inc., New York, 2001.

    """
    a = 0.3

    def rho(self, z):
        z = np.asarray(z)
        return (1 - np.exp(-self.a * np.fabs(z)) *
                (1 + self.a * np.fabs(z))) / self.a**2

    def psi(self, z):
        z = np.asarray(z)
        return z * np.exp(-self.a * np.fabs(z))

    def weights(self, z):
        z = np.asarray(z)
        return np.exp(-self.a * np.fabs(z))

class AndrewWave(RobustNorm):

    """
    Andrew\'s wave for M estimation.

    DC Montgomery, EA Peck. \'Introduction to Linear Regression Analysis\',
    John Wiley and Sons, Inc., New York, 2001.

    """
    a = 1.339

    def subset(self, z):
        z = np.asarray(z)
        return np.less_equal(np.fabs(z), self.a * np.pi)

    def rho(self, z):
        a = self.a
        z = np.asarray(z)
        test = self.subset(z)
        return (test * a * (1 - np.cos(z / a)) +
                (1 - test) * 2 * a)

    def psi(self, z):
        a = self.a
        z = np.asarray(z)
        test = self.subset(z)
        return test * np.sin(z / a)

    def weights(self, z):
        a = self.a
        z = np.asarray(z)
        test = self.subset(z)
        return test * np.sin(z / a) / (z / a)

class TrimmedMean(RobustNorm):
    """

    Trimmed mean function for M-estimation.

    R Venables, B Ripley. \'Modern Applied Statistics in S\'
    Springer, New York, 2002.
    """

    c = 2

    def subset(self, z):
        z = np.asarray(z)
        return np.less_equal(np.fabs(z), self.c)

    def rho(self, z):
        z = np.asarray(z)
        test = self.subset(z)
        return test * np.power(z, 2) * 0.5

    def psi(self, z):
        z = np.asarray(z)
        test = self.subset(z)
        return test * z

    def weights(self, z):
        z = np.asarray(z)
        test = self.subset(z)
        return test

class Hampel(RobustNorm):
    """

    Hampel function for M-estimation.

    R Venables, B Ripley. \'Modern Applied Statistics in S\'
    Springer, New York, 2002.
    """

#   Default values from Montgomery and Peck
#    a = 1.7
#    b = 3.4
#    c = 8.5
    def __init__(self, a = 2., b = 4., c = 8.):
        self.a = a
        self.b = b
        self.c = c

    def subset(self, z):
        z = np.fabs(np.asarray(z))
        t1 = np.less_equal(z, self.a)
        t2 = np.less_equal(z, self.b) * np.greater(z, self.a)
        t3 = np.less_equal(z, self.c) * np.greater(z, self.b)
        return t1, t2, t3

    def rho(self, z):
        z = np.fabs(z)
        a = self.a; b = self.b; c = self.c
        t1, t2, t3 = self.subset(z)
        v = (t1 * z**2 * 0.5 +
             t2 * (a * z - a**2 * 0.5) +
             t3 * (a * (c * z - z**2 * 0.5) / (c - b) - 7 * a**2 / 6.) +     #(7/6) not (7/2) from M&P
             (1 - t1 + t2 + t3) * a * (b + c - a))
        return v

    def psi(self, z):
        z = np.asarray(z)
        a = self.a; b = self.b; c = self.c
        t1, t2, t3 = self.subset(z)
        s = np.sign(z)
        z = np.fabs(z)
        v = s * (t1 * z +
                 t2 * a +
                 t3 * a * (c - z) / (c - b))
        return v

    def weights(self, z):
        z = np.asarray(z)
#        test = np.not_equal(z, 0)
#        return self.psi(z) * test / z + (1 - test)     # check Venables, this is different than M&P
# don't think the above handles the signs correctly, need to check
        a = self.a; b = self.b; c = self.c
        t1, t2, t3 = self.subset(z)
        v = (t1 +
            t2 * a/np.fabs(z) +
            t3 * a*(c-np.fabs(z))/(np.fabs(z)*(c-b)))
        return v



class TukeyBiweight(RobustNorm):
    """

    Tukey\'s biweight function for M-estimation.

    R Venables, B Ripley. \'Modern Applied Statistics in S\'
    Springer, New York, 2002.
    """


    R = 4.685

    def subset(self, z):
        z = np.fabs(np.asarray(z))
        return np.less_equal(z, self.R)

    def psi(self, z):
        z = np.asarray(z)
        subset = self.subset(z)
        return z * (1 - (z / self.R)**2)**2 * subset

    def rho(self, z):
        subset = self.subset(z)
        return -(1 - (z / self.R)**2)**3 * subset * self.R**2 / 6

    def weights(self, z):
        subset = self.subset(z)
        return (1 - (z / self.R)**2)**2 * subset

def estimate_location(a, scale, norm=HuberT(), axis=0, initial=None,
                      niter=30, tol=1.0e-06):
    """
    M-estimator of location using self.norm and a current
    estimator of scale.

    This iteratively finds a solution to

    norm.psi((a-mu)/scale).sum() == 0

    Inputs:
    -------
    a : ndarray
        Array over which the location parameter is to be estimated
    scale : ndarray
        Scale parameter to be used in M-estimator
    norm : RobustNorm
        Robust norm used in the M-estimator.
    axis : int
        Axis along which to estimate the location parameter.
    initial : ndarray
        Optional initial condition for the location parameter
    niter : int
        Maximum number of iterations
    tol : float
        Toleration for convergence
    Outputs:
    --------
    mu : ndarray
        Estimate of location
    """

    if initial is None:
        mu = np.median(a, axis)
    else:
        mu = initial

    for iter in range(niter):
        W = norm.weights((a-mu)/scale)
        nmu = np.sum(W*a, axis) / np.sum(W, axis)
        if np.alltrue(np.less(np.fabs(mu - nmu), scale * tol)):
            return nmu
        else:
            mu = nmu
    raise ValueError("location estimator failed to converge in %d iterations" % niter)

