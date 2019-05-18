from statsmodels.compat.python import range
import numpy as np

#TODO: add plots to weighting functions for online docs.


class RobustNorm(object):
    """
    The parent class for the norms used for robust regression.

    Lays out the methods expected of the robust norms to be used
    by statsmodels.RLM.

    Parameters
    ----------
    None :
        Some subclasses have optional tuning constants.

    References
    ----------
    PJ Huber.  'Robust Statistics' John Wiley and Sons, Inc., New York, 1981.

    DC Montgomery, EA Peck. 'Introduction to Linear Regression Analysis',
        John Wiley and Sons, Inc., New York, 2001.

    R Venables, B Ripley. 'Modern Applied Statistics in S'
        Springer, New York, 2002.

    See Also
    --------
    statsmodels.rlm

    Notes
    -----
    Currently only M-estimators are available.
    """

    def rho(self, z):
        """
        The robust criterion estimator function.

        Abstract method:

        -2 loglike used in M-estimator
        """
        raise NotImplementedError

    def psi(self, z):
        """
        Derivative of rho.  Sometimes referred to as the influence function.

        Abstract method:

        psi = rho'
        """
        raise NotImplementedError

    def weights(self, z):
        """
        Returns the value of psi(z) / z

        Abstract method:

        psi(z) / z
        """
        raise NotImplementedError

    def psi_deriv(self, z):
        """
        Deriative of psi.  Used to obtain robust covariance matrix.

        See statsmodels.rlm for more information.

        Abstract method:

        psi_derive = psi'
        """
        raise NotImplementedError

    def __call__(self, z):
        """
        Returns the value of estimator rho applied to an input
        """
        return self.rho(z)


class LeastSquares(RobustNorm):

    """
    Least squares rho for M-estimation and its derived functions.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """

    def rho(self, z):
        """
        The least squares estimator rho function

        Parameters
        ----------
        z : array
            1d array

        Returns
        -------
        rho : array
            rho(z) = (1/2.)*z**2
        """

        return z**2 * 0.5

    def psi(self, z):
        """
        The psi function for the least squares estimator

        The analytic derivative of rho

        Parameters
        ----------
        z : array-like
            1d array

        Returns
        -------
        psi : array
            psi(z) = z
        """

        return np.asarray(z)

    def weights(self, z):
        """
        The least squares estimator weighting function for the IRLS algorithm.

        The psi function scaled by the input z

        Parameters
        ----------
        z : array-like
            1d array

        Returns
        -------
        weights : array
            weights(z) = np.ones(z.shape)
        """

        z = np.asarray(z)
        return np.ones(z.shape, np.float64)

    def psi_deriv(self, z):
        """
        The derivative of the least squares psi function.

        Returns
        -------
        psi_deriv : array
            ones(z.shape)

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        return np.ones(z.shape, np.float64)


class HuberT(RobustNorm):
    """
    Huber's T for M estimation.

    Parameters
    ----------
    t : float, optional
        The tuning constant for Huber's t function. The default value is
        1.345.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """

    def __init__(self, t=1.345):
        self.t = t

    def _subset(self, z):
        """
        Huber's T is defined piecewise over the range for z
        """
        z = np.asarray(z)
        return np.less_equal(np.fabs(z), self.t)

    def rho(self, z):
        r"""
        The robust criterion function for Huber's t.

        Parameters
        ----------
        z : array-like
            1d array

        Returns
        -------
        rho : array
            rho(z) = .5*z**2            for \|z\| <= t

            rho(z) = \|z\|*t - .5*t**2    for \|z\| > t
        """
        z = np.asarray(z)
        test = self._subset(z)
        return (test * 0.5 * z**2 +
                (1 - test) * (np.fabs(z) * self.t - 0.5 * self.t**2))

    def psi(self, z):
        r"""
        The psi function for Huber's t estimator

        The analytic derivative of rho

        Parameters
        ----------
        z : array-like
            1d array

        Returns
        -------
        psi : array
            psi(z) = z      for \|z\| <= t

            psi(z) = sign(z)*t for \|z\| > t
        """
        z = np.asarray(z)
        test = self._subset(z)
        return test * z + (1 - test) * self.t * np.sign(z)

    def weights(self, z):
        r"""
        Huber's t weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array-like
            1d array

        Returns
        -------
        weights : array
            weights(z) = 1          for \|z\| <= t

            weights(z) = t/\|z\|      for \|z\| > t
        """
        z = np.asarray(z)
        test = self._subset(z)
        absz = np.fabs(z)
        absz[test] = 1.0
        return test + (1 - test) * self.t / absz

    def psi_deriv(self, z):
        """
        The derivative of Huber's t psi function

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        return np.less_equal(np.fabs(z), self.t)


#TODO: untested, but looks right.  RamsayE not available in R or SAS?
class RamsayE(RobustNorm):
    """
    Ramsay's Ea for M estimation.

    Parameters
    ----------
    a : float, optional
        The tuning constant for Ramsay's Ea function.  The default value is
        0.3.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """

    def __init__(self, a = .3):
        self.a = a

    def rho(self, z):
        r"""
        The robust criterion function for Ramsay's Ea.

        Parameters
        ----------
        z : array-like
            1d array

        Returns
        -------
        rho : array
            rho(z) = a**-2 * (1 - exp(-a*\|z\|)*(1 + a*\|z\|))
        """
        z = np.asarray(z)
        return (1 - np.exp(-self.a * np.fabs(z)) *
                (1 + self.a * np.fabs(z))) / self.a**2

    def psi(self, z):
        r"""
        The psi function for Ramsay's Ea estimator

        The analytic derivative of rho

        Parameters
        ----------
        z : array-like
            1d array

        Returns
        -------
        psi : array
            psi(z) = z*exp(-a*\|z\|)
        """
        z = np.asarray(z)
        return z * np.exp(-self.a * np.fabs(z))

    def weights(self, z):
        r"""
        Ramsay's Ea weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array-like
            1d array

        Returns
        -------
        weights : array
            weights(z) = exp(-a*\|z\|)
        """

        z = np.asarray(z)
        return np.exp(-self.a * np.fabs(z))

    def psi_deriv(self, z):
        """
        The derivative of Ramsay's Ea psi function.

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """

        return np.exp(-self.a * np.fabs(z)) + z**2*\
                np.exp(-self.a*np.fabs(z))*-self.a/np.fabs(z)


class AndrewWave(RobustNorm):

    """
    Andrew's wave for M estimation.

    Parameters
    ----------
    a : float, optional
        The tuning constant for Andrew's Wave function.  The default value is
        1.339.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """
    def __init__(self, a=1.339):
        self.a = a

    def _subset(self, z):
        """
        Andrew's wave is defined piecewise over the range of z.
        """
        z = np.asarray(z)
        return np.less_equal(np.fabs(z), self.a * np.pi)

    def rho(self, z):
        r"""
        The robust criterion function for Andrew's wave.

        Parameters
        ----------
        z : array-like
            1d array

        Returns
        -------
        rho : array
            rho(z) = a*(1-cos(z/a))     for \|z\| <= a*pi

            rho(z) = 2*a                for \|z\| > a*pi
        """

        a = self.a
        z = np.asarray(z)
        test = self._subset(z)
        return (test * a * (1 - np.cos(z / a)) +
                (1 - test) * 2 * a)

    def psi(self, z):
        r"""
        The psi function for Andrew's wave

        The analytic derivative of rho

        Parameters
        ----------
        z : array-like
            1d array

        Returns
        -------
        psi : array
            psi(z) = sin(z/a)       for \|z\| <= a*pi

            psi(z) = 0              for \|z\| > a*pi
        """

        a = self.a
        z = np.asarray(z)
        test = self._subset(z)
        return test * np.sin(z / a)

    def weights(self, z):
        r"""
        Andrew's wave weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array-like
            1d array

        Returns
        -------
        weights : array
            weights(z) = sin(z/a)/(z/a)     for \|z\| <= a*pi

            weights(z) = 0                  for \|z\| > a*pi
        """
        a = self.a
        z = np.asarray(z)
        test = self._subset(z)
        return test * np.sin(z / a) / (z / a)

    def psi_deriv(self, z):
        """
        The derivative of Andrew's wave psi function

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """

        test = self._subset(z)
        return test*np.cos(z / self.a)/self.a


# TODO: this is untested
class TrimmedMean(RobustNorm):
    """
    Trimmed mean function for M-estimation.

    Parameters
    ----------
    c : float, optional
        The tuning constant for Ramsay's Ea function.  The default value is
        2.0.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """

    def __init__(self, c=2.):
        self.c = c

    def _subset(self, z):
        """
        Least trimmed mean is defined piecewise over the range of z.
        """

        z = np.asarray(z)
        return np.less_equal(np.fabs(z), self.c)

    def rho(self, z):
        r"""
        The robust criterion function for least trimmed mean.

        Parameters
        ----------
        z : array-like
            1d array

        Returns
        -------
        rho : array
            rho(z) = (1/2.)*z**2    for \|z\| <= c

            rho(z) = 0              for \|z\| > c
        """

        z = np.asarray(z)
        test = self._subset(z)
        return test * z**2 * 0.5

    def psi(self, z):
        r"""
        The psi function for least trimmed mean

        The analytic derivative of rho

        Parameters
        ----------
        z : array-like
            1d array

        Returns
        -------
        psi : array
            psi(z) = z              for \|z\| <= c

            psi(z) = 0              for \|z\| > c

        """
        z = np.asarray(z)
        test = self._subset(z)
        return test * z

    def weights(self, z):
        r"""
        Least trimmed mean weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array-like
            1d array

        Returns
        -------
        weights : array
            weights(z) = 1             for \|z\| <= c

            weights(z) = 0             for \|z\| > c

        """
        z = np.asarray(z)
        test = self._subset(z)
        return test

    def psi_deriv(self, z):
        """
        The derivative of least trimmed mean psi function

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        test = self._subset(z)
        return test


class Hampel(RobustNorm):
    """

    Hampel function for M-estimation.

    Parameters
    ----------
    a : float, optional
    b : float, optional
    c : float, optional
        The tuning constants for Hampel's function.  The default values are
        a,b,c = 2, 4, 8.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """

    def __init__(self, a=2., b=4., c=8.):
        self.a = a
        self.b = b
        self.c = c

    def _subset(self, z):
        """
        Hampel's function is defined piecewise over the range of z
        """
        z = np.fabs(np.asarray(z))
        t1 = np.less_equal(z, self.a)
        t2 = np.less_equal(z, self.b) * np.greater(z, self.a)
        t3 = np.less_equal(z, self.c) * np.greater(z, self.b)
        return t1, t2, t3

    def rho(self, z):
        r"""
        The robust criterion function for Hampel's estimator

        Parameters
        ----------
        z : array-like
            1d array

        Returns
        -------
        rho : array
            rho(z) = (1/2.)*z**2                    for \|z\| <= a

            rho(z) = a*\|z\| - 1/2.*a**2              for a < \|z\| <= b

            rho(z) = a*(c*\|z\|-(1/2.)*z**2)/(c-b)    for b < \|z\| <= c

            rho(z) = a*(b + c - a)                  for \|z\| > c
        """

        z = np.fabs(z)
        a = self.a
        b = self.b
        c = self.c
        t1, t2, t3 = self._subset(z)
        v = (t1 * z**2 * 0.5 +
             t2 * (a * z - a**2 * 0.5) +
             t3 * (a * (c * z - z**2 * 0.5) / (c - b) - 7 * a**2 / 6.) +
             (1 - t1 + t2 + t3) * a * (b + c - a))
        return v

    def psi(self, z):
        r"""
        The psi function for Hampel's estimator

        The analytic derivative of rho

        Parameters
        ----------
        z : array-like
            1d array

        Returns
        -------
        psi : array
            psi(z) = z                            for \|z\| <= a

            psi(z) = a*sign(z)                    for a < \|z\| <= b

            psi(z) = a*sign(z)*(c - \|z\|)/(c-b)    for b < \|z\| <= c

            psi(z) = 0                            for \|z\| > c
        """
        z = np.asarray(z)
        a = self.a
        b = self.b
        c = self.c
        t1, t2, t3 = self._subset(z)
        s = np.sign(z)
        z = np.fabs(z)
        v = s * (t1 * z +
                 t2 * a*s +
                 t3 * a*s * (c - z) / (c - b))
        return v

    def weights(self, z):
        r"""
        Hampel weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array-like
            1d array

        Returns
        -------
        weights : array
            weights(z) = 1                            for \|z\| <= a

            weights(z) = a/\|z\|                        for a < \|z\| <= b

            weights(z) = a*(c - \|z\|)/(\|z\|*(c-b))      for b < \|z\| <= c

            weights(z) = 0                            for \|z\| > c

        """
        z = np.asarray(z)
        a = self.a
        b = self.b
        c = self.c
        t1, t2, t3 = self._subset(z)
        v = (t1 +
            t2 * a/np.fabs(z) +
            t3 * a*(c-np.fabs(z))/(np.fabs(z)*(c-b)))
        v[np.where(np.isnan(v))]=1. # for some reason 0 returns a nan?
        return v

    def psi_deriv(self, z):
        t1, t2, t3 = self._subset(z)
        return t1 + t3 * (self.a*np.sign(z)*z)/(np.fabs(z)*(self.c-self.b))


class TukeyBiweight(RobustNorm):
    """

    Tukey's biweight function for M-estimation.

    Parameters
    ----------
    c : float, optional
        The tuning constant for Tukey's Biweight.  The default value is
        c = 4.685.

    Notes
    -----
    Tukey's biweight is sometime's called bisquare.
    """

    def __init__(self, c = 4.685):
        self.c = c

    def _subset(self, z):
        """
        Tukey's biweight is defined piecewise over the range of z
        """
        z = np.fabs(np.asarray(z))
        return np.less_equal(z, self.c)

    def rho(self, z):
        r"""
        The robust criterion function for Tukey's biweight estimator

        Parameters
        ----------
        z : array-like
            1d array

        Returns
        -------
        rho : array
            rho(z) = -(1 - (z/c)**2)**3 * c**2/6.   for \|z\| <= R

            rho(z) = 0                              for \|z\| > R
        """
        subset = self._subset(z)
        return -(1 - (z / self.c)**2)**3 * subset * self.c**2 / 6.

    def psi(self, z):
        r"""
        The psi function for Tukey's biweight estimator

        The analytic derivative of rho

        Parameters
        ----------
        z : array-like
            1d array

        Returns
        -------
        psi : array
            psi(z) = z*(1 - (z/c)**2)**2        for \|z\| <= R

            psi(z) = 0                           for \|z\| > R
        """

        z = np.asarray(z)
        subset = self._subset(z)
        return z * (1 - (z / self.c)**2)**2 * subset

    def weights(self, z):
        r"""
        Tukey's biweight weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array-like
            1d array

        Returns
        -------
        weights : array
            psi(z) = (1 - (z/c)**2)**2          for \|z\| <= R

            psi(z) = 0                          for \|z\| > R
        """

        subset = self._subset(z)
        return (1 - (z / self.c)**2)**2 * subset

    def psi_deriv(self, z):
        """
        The derivative of Tukey's biweight psi function

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        subset = self._subset(z)
        return subset*((1 - (z/self.c)**2)**2 - (4*z**2/self.c**2) *\
                    (1-(z/self.c)**2))

def estimate_location(a, scale, norm=None, axis=0, initial=None,
                      maxiter=30, tol=1.0e-06):
    """
    M-estimator of location using self.norm and a current
    estimator of scale.

    This iteratively finds a solution to

    norm.psi((a-mu)/scale).sum() == 0

    Parameters
    ----------
    a : array
        Array over which the location parameter is to be estimated
    scale : array
        Scale parameter to be used in M-estimator
    norm : RobustNorm, optional
        Robust norm used in the M-estimator.  The default is HuberT().
    axis : int, optional
        Axis along which to estimate the location parameter.  The default is 0.
    initial : array, optional
        Initial condition for the location parameter.  Default is None, which
        uses the median of a.
    niter : int, optional
        Maximum number of iterations.  The default is 30.
    tol : float, optional
        Toleration for convergence.  The default is 1e-06.

    Returns
    -------
    mu : array
        Estimate of location
    """
    if norm is None:
        norm = HuberT()

    if initial is None:
        mu = np.median(a, axis)
    else:
        mu = initial

    for iter in range(maxiter):
        W = norm.weights((a-mu)/scale)
        nmu = np.sum(W*a, axis) / np.sum(W, axis)
        if np.alltrue(np.less(np.fabs(mu - nmu), scale * tol)):
            return nmu
        else:
            mu = nmu
    raise ValueError("location estimator failed to converge in %d iterations"
                     % maxiter)
