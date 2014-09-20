"""
Univariate Kernel Density Estimators

References
----------
Racine, Jeff. (2008) "Nonparametric Econometrics: A Primer," Foundation and
    Trends in Econometrics: Vol 3: No 1, pp1-88.
    http://dx.doi.org/10.1561/0800000009

http://en.wikipedia.org/wiki/Kernel_%28statistics%29

Silverman, B.W.  Density Estimation for Statistics and Data Analysis.
"""
from __future__ import absolute_import, print_function
from statsmodels.compat.python import range
# for 2to3 with extensions
import warnings

import numpy as np
import scipy as sp
from scipy import integrate, stats
from statsmodels.sandbox.nonparametric import kernels
from statsmodels.tools.decorators import (cache_readonly,
                                                    resettable_cache)
from . import bandwidths
from .kdetools import (forrt, revrt, silverman_transform, counts)
from .linbin import fast_linbin, fast_linbin_weights
from statsmodels.compat.scipy import _next_regular

#### Kernels Switch for estimators ####

kernel_switch = dict(gau=kernels.Gaussian, epa=kernels.Epanechnikov,
                    uni=kernels.Uniform, tri=kernels.Triangular,
                    biw=kernels.Biweight, triw=kernels.Triweight,
                    cos=kernels.Cosine, cos2=kernels.Cosine2)

def _checkisfit(self):
    try:
        self.density
    except:
        raise ValueError("Call fit to fit the density first")


#### Kernel Density Estimator Class ###


class KDEUnivariate(object):
    """
    Univariate Kernel Density Estimator.

    Parameters
    ----------
    endog : array-like
        The variable for which the density estimate is desired.

    Notes
    -----
    If cdf, sf, cumhazard, or entropy are computed, they are computed based on
    the definition of the kernel rather than the FFT approximation, even if
    the density is fit with FFT = True.

    `KDEUnivariate` is much faster than `KDEMultivariate`, due to its FFT-based
    implementation.  It should be preferred for univariate, continuous data.
    `KDEMultivariate` also supports mixed data.

    See Also
    --------
    KDEMultivariate
    kdensity, kdensityfft

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import matplotlib.pyplot as plt

    >>> nobs = 300
    >>> np.random.seed(1234)  # Seed random generator
    >>> dens = sm.nonparametric.KDEUnivariate(np.random.normal(size=nobs))
    >>> dens.fit()
    >>> plt.plot(dens.cdf)
    >>> plt.show()

    """

    def __init__(self, endog):
        self.endog = np.asarray(endog)

    def fit(self, kernel="gau", bw="normal_reference", fft=True, weights=None,
            gridsize=None, adjust=1, cut=3, clip=(-np.inf, np.inf)):
        """
        Attach the density estimate to the KDEUnivariate class.

        Parameters
        ----------
        kernel : str
            The Kernel to be used. Choices are:

            - "biw" for biweight
            - "cos" for cosine
            - "epa" for Epanechnikov
            - "gau" for Gaussian.
            - "tri" for triangular
            - "triw" for triweight
            - "uni" for uniform

        bw : str, float
            The bandwidth to use. Choices are:

            - "scott" - 1.059 * A * nobs ** (-1/5.), where A is
              `min(std(X),IQR/1.34)`
            - "silverman" - .9 * A * nobs ** (-1/5.), where A is
              `min(std(X),IQR/1.34)`
            - "normal_reference" - C * A * nobs ** (-1/5.), where C is
              calculated from the kernel. Equivalent (up to 2 dp) to the
              "scott" bandwidth for gaussian kernels. See bandwidths.py
            - If a float is given, it is the bandwidth.

        fft : bool
            Whether or not to use FFT. FFT implementation is more
            computationally efficient. However, only the Gaussian kernel
            is implemented. If FFT is False, then a 'nobs' x 'gridsize'
            intermediate array is created.
        gridsize : int
            If gridsize is None, max(len(X), 50) is used.
        cut : float
            Defines the length of the grid past the lowest and highest values
            of X so that the kernel goes to zero. The end points are
            -/+ cut*bw*{min(X) or max(X)}
        adjust : float
            An adjustment factor for the bw. Bandwidth becomes bw * adjust.
        """
        try:
            bw = float(bw)
            self.bw_method = "user-given"
        except:
            self.bw_method = bw
        endog = self.endog

        if fft:
            # if kernel != "gau":
            #     msg = "Only gaussian kernel is available for fft"
            #     raise NotImplementedError(msg)
            # if weights is not None:
            #     msg = "Weights are not implemented for fft"
            #     raise NotImplementedError(msg)
            density, grid, bw = kdensityfft(endog, kernel=kernel, bw=bw,
                    adjust=adjust, weights=weights, gridsize=gridsize,
                    clip=clip, cut=cut)
        else:
            density, grid, bw = kdensity(endog, kernel=kernel, bw=bw,
                    adjust=adjust, weights=weights, gridsize=gridsize,
                    clip=clip, cut=cut)
        self.density = density
        self.support = grid
        self.bw = bw
        self.kernel = kernel_switch[kernel](h=bw) # we instantiate twice,
                                                # should this passed to funcs?
        # put here to ensure empty cache after re-fit with new options
        self.kernel.weights = weights
        if weights is not None:
            self.kernel.weights /= weights.sum()
        self._cache = resettable_cache()

    @cache_readonly
    def cdf(self):
        """
        Returns the cumulative distribution function evaluated at the support.

        Notes
        -----
        Will not work if fit has not been called.
        """
        _checkisfit(self)
        density = self.density
        kern = self.kernel
        if kern.domain is None: # TODO: test for grid point at domain bound
            a,b = -np.inf,np.inf
        else:
            a,b = kern.domain
        func = lambda x,s: kern.density(s,x)

        support = self.support
        support = np.r_[a,support]
        gridsize = len(support)
        endog = self.endog
        probs = [integrate.quad(func, support[i-1], support[i],
                    args=endog)[0] for i in range(1,gridsize)]
        return np.cumsum(probs)

    @cache_readonly
    def cumhazard(self):
        """
        Returns the hazard function evaluated at the support.

        Notes
        -----
        Will not work if fit has not been called.

        """
        _checkisfit(self)
        return -np.log(self.sf)

    @cache_readonly
    def sf(self):
        """
        Returns the survival function evaluated at the support.

        Notes
        -----
        Will not work if fit has not been called.
        """
        _checkisfit(self)
        return 1 - self.cdf

    @cache_readonly
    def entropy(self):
        """
        Returns the differential entropy evaluated at the support

        Notes
        -----
        Will not work if fit has not been called. 1e-12 is added to each
        probability to ensure that log(0) is not called.
        """
        _checkisfit(self)

        def entr(x,s):
            pdf = kern.density(s,x)
            return pdf*np.log(pdf+1e-12)

        pdf = self.density
        kern = self.kernel

        if kern.domain is not None:
            a,b = self.domain
        else:
            a,b = -np.inf,np.inf
        endog = self.endog
        #TODO: below could run into integr problems, cf. stats.dist._entropy
        return -integrate.quad(entr, a,b, args=(endog,))[0]

    @cache_readonly
    def icdf(self):
        """
        Inverse Cumulative Distribution (Quantile) Function

        Notes
        -----
        Will not work if fit has not been called. Uses
        `scipy.stats.mstats.mquantiles`.
        """
        _checkisfit(self)
        gridsize = len(self.density)
        return stats.mstats.mquantiles(self.endog, np.linspace(0,1,
                    gridsize))

    def evaluate(self, point):
        """
        Evaluate density at a single point.

        Parameters
        ----------
        point : float
            Point at which to evaluate the density.
        """
        _checkisfit(self)
        return self.kernel.density(self.endog, point)


class KDE(KDEUnivariate):
    def __init__(self, endog):
        self.endog = np.asarray(endog)
        warnings.warn("KDE is deprecated and will be removed in 0.6, "
                      "use KDEUnivariate instead", FutureWarning)


#### Kernel Density Estimator Functions ####

def kdensity(X, kernel="gau", bw="normal_reference", weights=None, gridsize=None,
             adjust=1, clip=(-np.inf,np.inf), cut=3, retgrid=True):
    """
    Rosenblatt-Parzen univariate kernel density estimator.

    Parameters
    ----------
    X : array-like
        The variable for which the density estimate is desired.
    kernel : str
        The Kernel to be used. Choices are
        - "biw" for biweight
        - "cos" for cosine
        - "epa" for Epanechnikov
        - "gau" for Gaussian.
        - "tri" for triangular
        - "triw" for triweight
        - "uni" for uniform
    bw : str, float
        "scott" - 1.059 * A * nobs ** (-1/5.), where A is min(std(X),IQR/1.34)
        "silverman" - .9 * A * nobs ** (-1/5.), where A is min(std(X),IQR/1.34)
        If a float is given, it is the bandwidth.
    weights : array or None
        Optional  weights. If the X value is clipped, then this weight is
        also dropped.
    gridsize : int
        If gridsize is None, max(len(X), 50) is used.
    adjust : float
        An adjustment factor for the bw. Bandwidth becomes bw * adjust.
    clip : tuple
        Observations in X that are outside of the range given by clip are
        dropped. The number of observations in X is then shortened.
    cut : float
        Defines the length of the grid past the lowest and highest values of X
        so that the kernel goes to zero. The end points are
        -/+ cut*bw*{min(X) or max(X)}
    retgrid : bool
        Whether or not to return the grid over which the density is estimated.

    Returns
    -------
    density : array
        The densities estimated at the grid points.
    grid : array, optional
        The grid points at which the density is estimated.

    Notes
    -----
    Creates an intermediate (`gridsize` x `nobs`) array. Use FFT for a more
    computationally efficient version.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:,None]
    clip_x = np.logical_and(X>clip[0], X<clip[1])
    X = X[clip_x]

    nobs = float(len(X)) # after trim

    if gridsize == None:
        gridsize = max(nobs,50) # don't need to resize if no FFT

        # handle weights
    if weights is None:
        weights = np.ones(nobs)
        q = nobs
    else:
        # ensure weights is a numpy array
        weights = np.asarray(weights)
        if len(weights) != len(clip_x):
            msg = "The length of the weights must be the same as the given X."
            raise ValueError(msg)
        weights = weights[clip_x.squeeze()]
        q = weights.sum()

    # Get kernel object corresponding to selection
    kern = kernel_switch[kernel]()

    # if bw is None, select optimal bandwidth for kernel
    try:
        bw = float(bw)
    except:
        bw = bandwidths.select_bandwidth(X, bw, kern)
    bw *= adjust

    a = np.min(X,axis=0) - cut*bw
    b = np.max(X,axis=0) + cut*bw
    grid = np.linspace(a, b, gridsize)

    k = (X.T - grid[:,None])/bw  # uses broadcasting to make a gridsize x nobs

    # set kernel bandwidth
    kern.seth(bw)

    # truncate to domain
    if kern.domain is not None: # won't work for piecewise kernels like parzen
        z_lo, z_high = kern.domain
        domain_mask = (k < z_lo) | (k > z_high)
        k = kern(k) # estimate density
        k[domain_mask] = 0
    else:
        k = kern(k) # estimate density

    k[k<0] = 0 # get rid of any negative values, do we need this?

    dens = np.dot(k,weights)/(q*bw)

    if retgrid:
        return dens, grid, bw
    else:
        return dens, bw

def kdensityfft(X, kernel="gau", bw="normal_reference", weights=None, gridsize=None,
                adjust=1, clip=(-np.inf, np.inf), cut=3, retgrid=True):
    """
    Rosenblatt-Parzen univariate kernel density estimator

    Parameters
    ----------
    X : array-like
        The variable for which the density estimate is desired.
    kernel : str
        ONLY GAUSSIAN IS CURRENTLY IMPLEMENTED.
        "bi" for biweight
        "cos" for cosine
        "epa" for Epanechnikov, default
        "epa2" for alternative Epanechnikov
        "gau" for Gaussian.
        "par" for Parzen
        "rect" for rectangular
        "tri" for triangular
    bw : str, float
        "scott" - 1.059 * A * nobs ** (-1/5.), where A is min(std(X),IQR/1.34)
        "silverman" - .9 * A * nobs ** (-1/5.), where A is min(std(X),IQR/1.34)
        If a float is given, it is the bandwidth.
    weights : array or None
        WEIGHTS ARE NOT CURRENTLY IMPLEMENTED.
        Optional  weights. If the X value is clipped, then this weight is
        also dropped.
    gridsize : int
        If gridsize is None, min(len(X), 512) is used. Note that the provided
        number is rounded up to the next highest power of 2.
    adjust : float
        An adjustment factor for the bw. Bandwidth becomes bw * adjust.
        clip : tuple
        Observations in X that are outside of the range given by clip are
        dropped. The number of observations in X is then shortened.
    clip : 2-tuple
        Limits for the range of data. Data outside range is thrown away.
    cut : float
        Defines the length of the grid past the lowest and highest values of X
        so that the kernel goes to zero. The end points are
        -/+ cut*bw*{X.min() or X.max()}
    retgrid : bool
        Whether or not to return the grid over which the density is estimated.

    Returns
    -------
    density : array
        The densities estimated at the grid points.
    grid : array, optional
        The grid points at which the density is estimated.

    Notes
    -----
    Generic kernel is now supported as long as it has finite support or
    defines a cut off for effective support. This is based on the algorithm
    outline in Wand and Jones (1995)

    References
    ---------- ::
    Wand, M. P. and Jones, M. C. (1995). Kernel Smoothing. Chapman and Hall,
     London.
    """
    # Not convinced this is neccessary
    X = np.asarray(X)
    keep_mask = np.logical_and(X > clip[0], X < clip[1])
    X = X[keep_mask]
    
    # Get kernel object corresponding to selection
    kern = kernel_switch[kernel]()

    # This kernel selection should be moved outside of this function.
    # bw should be required as as float to this function.
    try:
        bw = float(bw)
    except:
        # will cross-val fit this pattern?
        bw = bandwidths.select_bandwidth(X, bw, kern)
    bw *= adjust

    nobs = float(len(X))  # after trim

    # step 1 Make grid and discretize the data
    if gridsize is None:
        # not convinced this is correct
        gridsize = np.max((nobs, 512.))
    # round to next power of 2
    gridsize = 2 ** np.ceil(np.log2(gridsize))

    a = np.min(X) - cut * bw
    b = np.max(X) + cut * bw
    grid, delta = np.linspace(a, b, gridsize, retstep=True)
    RANGE = b - a

    # Calculate the scaled bin counts with linear-binning
    
    if weights is None:
        binned = fast_linbin(X, a, b, gridsize)
    # handle weighted observations
    else:
        # ensure weights is a numpy array
        weights = np.asarray(weights)
        weights = weights[keep_mask]
        if len(weights) != len(X):
            msg = "The length of the weights must be the same as the given X."
            raise ValueError(msg)
        q = weights.mean()
        weights = weights / q
        #weights = np.ones(nobs)
        binned = fast_linbin_weights(X, weights, a, b, gridsize)

    # step 2 compute weights
    M = gridsize
    if kern.domain is None:
        L = M/2
        tau = np.inf
    else:
        tau = kern.domain[1]  # assumes support is symmetric.
        L =  M/2
    l = np.arange(0, L + 1)
    gridx = (delta * l) / bw
    kappa = kern(gridx)
    kappa = 1.0 / (nobs * bw) * kappa
    # throw away points evaluated outside support
    kappa[np.abs(gridx) > tau] = 0
    
    c = np.fft.rfft(binned)
    k = np.fft.rfft(np.r_[kappa, np.zeros(M % 2), kappa[::-1][1:-1]])

    # step 4 convolve using fourier transform
    f = np.fft.irfft(c * k, len(binned))

    if retgrid:
        return f, grid, bw
    else:
        return f, bw

if __name__ == "__main__":
    import numpy as np
    np.random.seed(12345)
    xi = np.random.randn(100)
    f,grid, bw1 = kdensity(xi, kernel="gau", bw=.372735, retgrid=True)
    f2, bw2 = kdensityfft(xi, kernel="gau", bw="silverman",retgrid=False)

# do some checking vs. silverman algo.
# you need denes.f, http://lib.stat.cmu.edu/apstat/176
#NOTE: I (SS) made some changes to the Fortran
# and the FFT stuff from Munro http://lib.stat.cmu.edu/apstat/97o
# then compile everything and link to denest with f2py
#Make pyf file as usual, then compile shared object
#f2py denest.f -m denest2 -h denest.pyf
#edit pyf
#-c flag makes it available to other programs, fPIC builds a shared library
#/usr/bin/gfortran -Wall -c -fPIC fft.f
#f2py -c denest.pyf ./fft.o denest.f

    try:
        from denest2 import denest # @UnresolvedImport
        a = -3.4884382032045504
        b = 4.3671504686785605
        RANGE = b - a
        bw = bandwidths.bw_silverman(xi)

        ft,smooth,ifault,weights,smooth1 = denest(xi,a,b,bw,np.zeros(512),np.zeros(512),0,
                np.zeros(512), np.zeros(512))
# We use a different binning algo, so only accurate up to 3 decimal places
        np.testing.assert_almost_equal(f2, smooth, 3)
#NOTE: for debugging
#        y2 = forrt(weights)
#        RJ = np.arange(512/2+1)
#        FAC1 = 2*(np.pi*bw/RANGE)**2
#        RJFAC = RJ**2*FAC1
#        BC = 1 - RJFAC/(6*(bw/((b-a)/M))**2)
#        FAC = np.exp(-RJFAC)/BC
#        SMOOTH = np.r_[FAC,FAC[1:-1]] * y2

#        dens = revrt(SMOOTH)

    except:
#        ft = np.loadtxt('./ft_silver.csv')
#        smooth = np.loadtxt('./smooth_silver.csv')
        print("Didn't get the estimates from the Silverman algorithm")
