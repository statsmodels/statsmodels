"""
Module of kernels that are able to handle continuous as well as categorical
variables (both ordered and unordered).

Having kernel functions rather than classes makes extension to a multivariate
kernel density estimation much easier.

NOTE: As it is, this module does not interact with the existing API

References:
----------
Pointwise Kernel Confidence Bounds
(smoothconf)
http://fedc.wiwi.hu-berlin.de/xplore/ebooks/html/anr/anrhtmlframe62.html
"""

from __future__ import division

import numpy as np

from scipy.special import erf
from scipy import stats, integrate

from statsmodels.compat import lzip, lfilter
from statsmodels.compat.scipy import factorial

#TODO:
# - make sure we only receive int input for wang-ryzin and aitchison-aitken
# - Check for the scalar Xi case everywhere


def aitchison_aitken(h, Xi, x, num_levels=None):
    r"""
    The Aitchison-Aitken kernel, used for unordered discrete random variables.

    Parameters
    ----------
    h : 1-D ndarray, shape (K,)
        The bandwidths used to estimate the value of the kernel function.
    Xi : 2-D ndarray of ints, shape (nobs, K)
        The value of the training set.
    x: 1-D ndarray, shape (K,)
        The value at which the kernel density is being estimated.
    num_levels: bool, optional
        Gives the user the option to specify the number of levels for the
        random variable.  If False, the number of levels is calculated from
        the data.

    Returns
    -------
    kernel_value : ndarray, shape (nobs, K)
        The value of the kernel function at each training point for each var.

    Notes
    -----
    See p.18 of [2]_ for details.  The value of the kernel L if :math:`X_{i}=x`
    is :math:`1-\lambda`, otherwise it is :math:`\frac{\lambda}{c-1}`.
    Here :math:`c` is the number of levels plus one of the RV.

    References
    ----------
    .. [*] J. Aitchison and C.G.G. Aitken, "Multivariate binary discrimination
           by the kernel method", Biometrika, vol. 63, pp. 413-420, 1976.
    .. [*] Racine, Jeff. "Nonparametric Econometrics: A Primer," Foundation
           and Trends in Econometrics: Vol 3: No 1, pp1-88., 2008.
    """
    Xi = Xi.reshape(Xi.size)  # seems needed in case Xi is scalar
    if num_levels is None:
        num_levels = np.asarray(np.unique(Xi).size)

    kernel_value = np.ones(Xi.size) * h / (num_levels - 1)
    idx = Xi == x
    kernel_value[idx] = (idx * (1 - h))[idx]
    return kernel_value


def wang_ryzin(h, Xi, x):
    r"""
    The Wang-Ryzin kernel, used for ordered discrete random variables.

    Parameters
    ----------
    h : scalar or 1-D ndarray, shape (K,)
        The bandwidths used to estimate the value of the kernel function.
    Xi : ndarray of ints, shape (nobs, K)
        The value of the training set.
    x : scalar or 1-D ndarray of shape (K,)
        The value at which the kernel density is being estimated.

    Returns
    -------
    kernel_value : ndarray, shape (nobs, K)
        The value of the kernel function at each training point for each var.

    Notes
    -----
    See p. 19 in [1]_ for details.  The value of the kernel L if
    :math:`X_{i}=x` is :math:`1-\lambda`, otherwise it is
    :math:`\frac{1-\lambda}{2}\lambda^{|X_{i}-x|}`, where :math:`\lambda` is
    the bandwidth.

    References
    ----------
    .. [*] Racine, Jeff. "Nonparametric Econometrics: A Primer," Foundation
           and Trends in Econometrics: Vol 3: No 1, pp1-88., 2008.
           http://dx.doi.org/10.1561/0800000009
    .. [*] M.-C. Wang and J. van Ryzin, "A class of smooth estimators for
           discrete distributions", Biometrika, vol. 68, pp. 301-309, 1981.
    """
    Xi = Xi.reshape(Xi.size)  # seems needed in case Xi is scalar
    kernel_value = 0.5 * (1 - h) * (h ** abs(Xi - x))
    idx = Xi == x
    kernel_value[idx] = (idx * (1 - h))[idx]
    return kernel_value


def gaussian(h, Xi, x):
    """
    Gaussian Kernel for continuous variables
    Parameters
    ----------
    h : 1-D ndarray, shape (K,)
        The bandwidths used to estimate the value of the kernel function.
    Xi : 1-D ndarray, shape (K,)
        The value of the training set.
    x : 1-D ndarray, shape (K,)
        The value at which the kernel density is being estimated.

    Returns
    -------
    kernel_value : ndarray, shape (nobs, K)
        The value of the kernel function at each training point for each var.

    """
    return (1. / np.sqrt(2 * np.pi)) * np.exp(-(Xi - x)**2 / (h**2 * 2.))


def gaussian_convolution(h, Xi, x):
    """ Calculates the Gaussian Convolution Kernel """
    return (1. / np.sqrt(4 * np.pi)) * np.exp(- (Xi - x)**2 / (h**2 * 4.))


def wang_ryzin_convolution(h, Xi, Xj):
    # This is the equivalent of the convolution case with the Gaussian Kernel
    # However it is not exactly convolution. Think of a better name
    # References
    ordered = np.zeros(Xi.size)
    for x in np.unique(Xi):
        ordered += wang_ryzin(h, Xi, x) * wang_ryzin(h, Xj, x)

    return ordered


def aitchison_aitken_convolution(h, Xi, Xj):
    Xi_vals = np.unique(Xi)
    ordered = np.zeros(Xi.size)
    num_levels = Xi_vals.size
    for x in Xi_vals:
        ordered += aitchison_aitken(h, Xi, x, num_levels=num_levels) * \
                   aitchison_aitken(h, Xj, x, num_levels=num_levels)

    return ordered


def gaussian_cdf(h, Xi, x):
    return 0.5 * h * (1 + erf((x - Xi) / (h * np.sqrt(2))))


def aitchison_aitken_cdf(h, Xi, x_u):
    x_u = int(x_u)
    Xi_vals = np.unique(Xi)
    ordered = np.zeros(Xi.size)
    num_levels = Xi_vals.size
    for x in Xi_vals:
        if x <= x_u:  #FIXME: why a comparison for unordered variables?
            ordered += aitchison_aitken(h, Xi, x, num_levels=num_levels)

    return ordered


def wang_ryzin_cdf(h, Xi, x_u):
    ordered = np.zeros(Xi.size)
    for x in np.unique(Xi):
        if x <= x_u:
            ordered += wang_ryzin(h, Xi, x)

    return ordered


def d_gaussian(h, Xi, x):
    # The derivative of the Gaussian Kernel
    return 2 * (Xi - x) * gaussian(h, Xi, x) / h**2


def aitchison_aitken_reg(h, Xi, x):
    """
    A version for the Aitchison-Aitken kernel for nonparametric regression.

    Suggested by Li and Racine.
    """
    kernel_value = np.ones(Xi.size)
    ix = Xi != x
    inDom = ix * h
    kernel_value[ix] = inDom[ix]
    return kernel_value


def wang_ryzin_reg(h, Xi, x):
    """
    A version for the Wang-Ryzin kernel for nonparametric regression.

    Suggested by Li and Racine in [1] ch.4
    """
    return h ** abs(Xi - x)


# -----------------------------------------------------------------
# Class-Based kernel implementations

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
        self.weights = None

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
        #xs = self.in_domain( xs, xs, x )[0]

        if len(xs)>0:  ## Need to do product of marginal distributions
            #w = np.sum([self(self._Hrootinv * (xx-x).T ) for xx in xs])/n
            #vectorized doesn't work:
            if self.weights is not None:
                w = np.mean(self((xs-x) * self._Hrootinv).T * self.weights)/sum(self.weights)
            else:
                w = np.mean(self((xs-x) * self._Hrootinv )) #transposed
            #w = np.mean([self(xd) for xd in ((xs-x) * self._Hrootinv)] ) #transposed
            return w
        else:
            return np.nan

    def _kernweight(self, x ):
        """returns the kernel weight for the independent multivariate kernel"""
        if isinstance( self._kernels, CustomKernel ):
            ## Radial case
            #d = x.T * x
            #x is matrix, 2d, element wise sqrt looks wrong
            #d = np.sqrt( x.T * x )
            x = np.asarray(x)
            #d = np.sqrt( (x * x).sum(-1) )
            d = (x * x).sum(-1)
            return self._kernels( np.asarray(d) )

    def __call__(self, x):
        """
        This simply returns the value of the kernel function at x

        Does the same as weight if the function is normalised
        """
        return self._kernweight(x)


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
        shape should be a function taking and returning numeric type.

        For sanity it should always return positive or zero but this isn't
        enforced in case you want to do weird things. Bear in mind that the
        statistical tests etc. may not be valid for non-positive kernels.

        The bandwidth of the kernel is supplied as h.

        You may specify a domain as a list of 2 values [min, max], in which case
        kernel will be treated as zero outside these values. This will speed up
        calculation.

        You may also specify the normalisation constant for the supplied Kernel.
        If you do this number will be stored and used as the normalisation
        without calculation.  It is recommended you do this if you know the
        constant, to speed up calculation.  In particular if the shape function
        provided is already normalised you should provide norm = 1.0.

        Warning: I think several calculations assume that the kernel is
        normalized. No tests for non-normalized kernel.
        """
        self._normconst = norm   # a value or None, if None, then calculate
        self.domain = domain
        self.weights = None
        if callable(shape):
            self._shape = shape
        else:
            raise TypeError("shape must be a callable object/function")
        self._h = h
        self._L2Norm = None
        self._kernel_var = None
        self._normal_reference_constant = None
        self._order = None

    def geth(self):
        """Getter for kernel bandwidth, h"""
        return self._h
    def seth(self, value):
        """Setter for kernel bandwidth, h"""
        self._h = value
    h = property(geth, seth, doc="Kernel Bandwidth")

    def in_domain(self, xs, ys, x):
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
            filtered = lfilter(isInDomain, lzip(xs, ys))
            if len(filtered) > 0:
                xs, ys = lzip(*filtered)
                return (xs, ys)
            else:
                return ([], [])

    def density(self, xs, x):
        """Returns the kernel density estimate for point x based on x-values
        xs
        """
        xs = np.asarray(xs)
        n = len(xs) # before in_domain?
        if self.weights is not None:
            xs, weights = self.in_domain( xs, self.weights, x )
        else:
            xs = self.in_domain( xs, xs, x )[0]
        xs = np.asarray(xs)
        #print 'len(xs)', len(xs), x
        if xs.ndim == 1:
            xs = xs[:,None]
        if len(xs)>0:
            h = self.h
            if self.weights is not None:
                w = 1 / h * np.sum(self((xs-x)/h).T * weights, axis=1)
            else:
                w = 1. / (h * n) * np.sum(self((xs-x)/h), axis=0)
            return w
        else:
            return np.nan

    def density_var(self, density, nobs):
        """approximate pointwise variance for kernel density

        not verified

        Parameters
        ----------
        density : array_lie
            pdf of the kernel density
        nobs : int
            number of observations used in the KDE estimation

        Returns
        -------
        kde_var : ndarray
            estimated variance of the density estimate

        Notes
        -----
        This uses the asymptotic normal approximation to the distribution of
        the density estimate.
        """
        return np.asarray(density) * self.L2Norm / self.h / nobs

    def density_confint(self, density, nobs, alpha=0.05):
        """approximate pointwise confidence interval for kernel density

        The confidence interval is centered at the estimated density and
        ignores the bias of the density estimate.

        not verified

        Parameters
        ----------
        density : array_lie
            pdf of the kernel density
        nobs : int
            number of observations used in the KDE estimation

        Returns
        -------
        conf_int : ndarray
            estimated confidence interval of the density estimate, lower bound
            in first column and upper bound in second column

        Notes
        -----
        This uses the asymptotic normal approximation to the distribution of
        the density estimate. The lower bound can be negative for density
        values close to zero.

        """
        
        crit = stats.norm.isf(alpha / 2.)
        density = np.asarray(density)
        half_width = crit * np.sqrt(self.density_var(density, nobs))
        conf_int = np.column_stack((density - half_width, density + half_width))
        return conf_int

    def smooth(self, xs, ys, x):
        """Returns the kernel smoothing estimate for point x based on x-values
        xs and y-values ys.
        Not expected to be called by the user.
        """
        xs, ys = self.in_domain(xs, ys, x)

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
        xs, ys = self.in_domain(xs, ys, x)

        if len(xs) > 0:
            fittedvals = np.array([self.smooth(xs, ys, xx) for xx in xs])
            sqresid = np.square(np.subtract(ys, fittedvals))
            w = np.sum(self((xs-x)/self.h))
            v = np.sum([rr*self((xx-x)/self.h) for xx, rr in zip(xs, sqresid)])
            return v / w
        else:
            return np.nan

    def smoothconf(self, xs, ys, x, alpha=0.05):
        """Returns the kernel smoothing estimate with confidence 1sigma bounds
        """
        xs, ys = self.in_domain(xs, ys, x)

        if len(xs) > 0:
            fittedvals = np.array([self.smooth(xs, ys, xx) for xx in xs])
            #fittedvals = self.smooth(xs, ys, x) # x or xs in Haerdle
            sqresid = np.square(
                np.subtract(ys, fittedvals)
            )
            w = np.sum(self((xs-x)/self.h))
            #var = sqresid.sum() / (len(sqresid) - 0)  # nonlocal var ? JP just trying
            v = np.sum([rr*self((xx-x)/self.h) for xx, rr in zip(xs, sqresid)])
            var = v / w
            sd = np.sqrt(var)
            K = self.L2Norm
            yhat = self.smooth(xs, ys, x)
            crit = stats.norm.isf(alpha / 2)
            err = crit * sd * np.sqrt(K) / np.sqrt(w * self.h * self.norm_const)
            return (yhat - err, yhat, yhat + err)
        else:
            return (np.nan, np.nan, np.nan)

    @property
    def L2Norm(self):
        """Returns the integral of the square of the kernal from -inf to inf"""
        if self._L2Norm is None:
            L2Func = lambda x: (self.norm_const*self._shape(x))**2
            if self.domain is None:
                self._L2Norm = integrate.quad(L2Func, -np.inf, np.inf)[0]
            else:
                self._L2Norm = integrate.quad(L2Func, self.domain[0],
                                               self.domain[1])[0]
        return self._L2Norm

    @property
    def norm_const(self):
        """
        Normalising constant for kernel (integral from -inf to inf)
        """
        if self._normconst is None:
            if self.domain is None:
                quadres = integrate.quad(self._shape, -np.inf, np.inf)
            else:
                quadres = integrate.quad(self._shape, self.domain[0],
                                               self.domain[1])
            self._normconst = 1.0/(quadres[0])
        return self._normconst

    @property
    def kernel_var(self):
        """Returns the second moment of the kernel"""
        if self._kernel_var is None:
            func = lambda x: x**2 * self.norm_const * self._shape(x)
            if self.domain is None:
                self._kernel_var = integrate.quad(func, -np.inf, np.inf)[0]
            else:
                self._kernel_var = integrate.quad(func, self.domain[0],
                                               self.domain[1])[0]
        return self._kernel_var

    def moments(self, n):

        if n > 2:
            msg = "Only first and second moment currently implemented"
            raise NotImplementedError(msg)

        if n == 1:
            return 0

        if n == 2:
            return self.kernel_var

    @property
    def normal_reference_constant(self):
        """
        Constant used for silverman normal reference asymtotic bandwidth
        calculation.

        C  = 2((pi^(1/2)*(nu!)^3 R(k))/(2nu(2nu)!kap_nu(k)^2))^(1/(2nu+1))
        nu = kernel order
        kap_nu = nu'th moment of kernel
        R = kernel roughness (square of L^2 norm)

        Note: L2Norm property returns square of norm.
        """
        nu = self._order

        if not nu == 2:
            msg = "Only implemented for second order kernels"
            raise NotImplementedError(msg)

        if self._normal_reference_constant is None:
            C = np.pi**(.5) * factorial(nu)**3 * self.L2Norm
            C /= (2 * nu * factorial(2 * nu) * self.moments(nu)**2)
            C = 2*C**(1.0/(2*nu+1))
            self._normal_reference_constant = C

        return self._normal_reference_constant

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
        CustomKernel.__init__(self, shape=lambda x: 0.5 * np.ones(x.shape), h=h,
                              domain=[-1.0, 1.0], norm = 1.0)
        self._L2Norm = 0.5
        self._kernel_var = 1. / 3
        self._order = 2


class Triangular(CustomKernel):
    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 1 - abs(x), h=h,
                              domain=[-1.0, 1.0], norm = 1.0)
        self._L2Norm = 2.0/3.0
        self._kernel_var = 1. / 6
        self._order = 2


class Epanechnikov(CustomKernel):
    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 0.75*(1 - x*x), h=h,
                              domain=[-1.0, 1.0], norm = 1.0)
        self._L2Norm = 0.6
        self._kernel_var = 0.2
        self._order = 2


class Biweight(CustomKernel):
    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 0.9375*(1 - x*x)**2, h=h,
                              domain=[-1.0, 1.0], norm = 1.0)
        self._L2Norm = 5.0/7.0
        self._kernel_var = 1. / 7
        self._order = 2

    def smooth(self, xs, ys, x):
        """Returns the kernel smoothing estimate for point x based on x-values
        xs and y-values ys.
        Not expected to be called by the user.

        Special implementation optimised for Biweight.
        """
        xs, ys = self.in_domain(xs, ys, x)

        if len(xs) > 0:
            w = np.sum(np.square(np.subtract(1, np.square(np.divide(np.subtract(xs, x),
                                                        self.h)))))
            v = np.sum(np.multiply(ys, np.square(np.subtract(1, np.square(np.divide(
                                                np.subtract(xs, x), self.h))))))
            return v / w
        else:
            return np.nan

    def smoothvar(self, xs, ys, x):
        """
        Returns the kernel smoothing estimate of the variance at point x.
        """
        xs, ys = self.in_domain(xs, ys, x)

        if len(xs) > 0:
            fittedvals = np.array([self.smooth(xs, ys, xx) for xx in xs])
            rs = np.square(np.subtract(ys, fittedvals))
            w = np.sum(np.square(np.subtract(1.0, np.square(np.divide(np.subtract(xs, x),
                                                        self.h)))))
            v = np.sum(np.multiply(rs, np.square(np.subtract(1, np.square(np.divide(
                                                np.subtract(xs, x), self.h))))))
            return v / w
        else:
            return np.nan

    def smoothconf_(self, xs, ys, x):
        """Returns the kernel smoothing estimate with confidence 1sigma bounds
        """
        xs, ys = self.in_domain(xs, ys, x)

        if len(xs) > 0:
            fittedvals = np.array([self.smooth(xs, ys, xx) for xx in xs])
            rs = np.square(np.subtract(ys, fittedvals))
            w = np.sum(np.square(np.subtract(1.0, np.square(np.divide(np.subtract(xs, x),
                                                        self.h)))))
            v = np.sum(np.multiply(rs, np.square(np.subtract(1, np.square(np.divide(
                                                np.subtract(xs, x), self.h))))))
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
        self._kernel_var = 1. / 9
        self._order = 2


class Gaussian(CustomKernel):
    """
    Gaussian (Normal) Kernel

    K(u) = 1 / (sqrt(2*pi)) exp(-0.5 u**2)
    """
    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape = lambda x: 0.3989422804014327 *
                        np.exp(-x**2/2.0), h = h, domain = None, norm = 1.0)
        self._L2Norm = 1.0/(2.0*np.sqrt(np.pi))
        self._kernel_var = 1.0
        self._order = 2

    def smooth(self, xs, ys, x):
        """Returns the kernel smoothing estimate for point x based on x-values
        xs and y-values ys.
        Not expected to be called by the user.

        Special implementation optimised for Gaussian.
        """
        w = np.sum(np.exp(np.multiply(np.square(np.divide(np.subtract(xs, x),
                                              self.h)),-0.5)))
        v = np.sum(np.multiply(ys, np.exp(np.multiply(np.square(np.divide(np.subtract(xs, x),
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
        self._kernel_var = 0.1894305308612978  # = 1 - 8 / np.pi**2
        self._order = 2


class Cosine2(CustomKernel):
    """
    Cosine2 Kernel

    K(u) = 1 + cos(2 * pi * u) between -0.5 and 0.5

    Note: this  is the same Cosine kernel that Stata uses
    """
    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 1 + np.cos(2.0 * np.pi * x)
                , h=h, domain=[-0.5, 0.5], norm = 1.0)
        self._L2Norm = 1.5
        self._kernel_var = 0.03267274151216444  # = 1/12. - 0.5 / np.pi**2
        self._order = 2
