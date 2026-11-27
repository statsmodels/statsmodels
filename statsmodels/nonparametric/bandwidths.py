from statsmodels.compat.pandas import Substitution

import numpy as np
from scipy import fft, optimize
from scipy.stats import norm, scoreatpercentile

from statsmodels.sandbox.nonparametric import kernels
from statsmodels.tools.validation import array_like, float_like


def _select_sigma(x, percentile=25):
    """
    Returns the smaller of std(X, ddof=1) or normalized IQR(X) over axis 0

    Parameters
    ----------
    x : array_like
        Array for which to get the dispersion estimate.
    percentile : float, optional
        The lower percentile of the inter-percentile range, in percent. The
        upper percentile is ``100 - percentile``. The range is normalized by
        the same range of the standard normal distribution, so that the
        result estimates the standard deviation when the data are normal.
        The default, 25, gives the interquartile range and uses Silverman's
        rounded normalizing constant of 1.349.  percentile must be strictly
        between 0 and 50.

    Returns
    -------
    float
        The smaller of the sample standard deviation and the normalized
        inter-percentile range.

    References
    ----------
    Silverman (1986) p.47
    """
    percentile = float_like(percentile, "percentile")
    if not 0 < percentile < 50:
        raise ValueError("percentile must be between 0 and 50")
    if percentile == 25:
        # Silverman (1986) p.47 rounds the normal interquartile range to
        # four digits. The rounded value is retained for the default so
        # that the bandwidths it produces are unchanged.
        normalize = 1.349
    else:
        normalize = norm.ppf(1 - percentile / 100.0) - norm.ppf(percentile / 100.0)
    iqr = scoreatpercentile(x, 100 - percentile) - scoreatpercentile(x, percentile)
    IQR = iqr / normalize
    std_dev = np.std(x, axis=0, ddof=1)
    if IQR > 0:
        return np.minimum(std_dev, IQR)
    else:
        return std_dev


# Univariate Rule of Thumb Bandwidths
def bw_scott(x, kernel=None):
    """
    Scott's Rule of Thumb

    Parameters
    ----------
    x : array_like
        Array for which to get the bandwidth
    kernel : CustomKernel instance, optional
        Unused

    Returns
    -------
    bw : float
        The estimate of the bandwidth

    Notes
    -----
    Returns 1.059 * A * n ** (-1/5.) where ::

       A = min(std(x, ddof=1), IQR/1.349)
       IQR = np.subtract.reduce(np.percentile(x, [75,25]))

    References
    ----------

    Scott, D.W. (1992) Multivariate Density Estimation: Theory, Practice, and
        Visualization.
    """
    A = _select_sigma(x)
    n = len(x)
    return 1.059 * A * n ** (-0.2)


def bw_silverman(x, kernel=None):
    """
    Silverman's Rule of Thumb

    Parameters
    ----------
    x : array_like
        Array for which to get the bandwidth
    kernel : CustomKernel instance, optional
        Unused

    Returns
    -------
    bw : float
        The estimate of the bandwidth

    Notes
    -----
    Returns .9 * A * n ** (-1/5.) where ::

       A = min(std(x, ddof=1), IQR/1.349)
       IQR = np.subtract.reduce(np.percentile(x, [75,25]))

    References
    ----------

    Silverman, B.W. (1986) `Density Estimation.`
    """
    A = _select_sigma(x)
    n = len(x)
    return 0.9 * A * n ** (-0.2)


def bw_normal_reference(x, kernel=None):
    """
    Plug-in bandwidth with kernel specific constant based on normal reference

    This bandwidth minimizes the mean integrated square error if the true
    distribution is the normal. This choice is an appropriate bandwidth for
    single peaked distributions that are similar to the normal distribution.

    Parameters
    ----------
    x : array_like
        Array for which to get the bandwidth
    kernel : CustomKernel instance, optional
        Used to calculate the constant for the plug-in bandwidth.
        The default is a Gaussian kernel.

    Returns
    -------
    bw : float
        The estimate of the bandwidth

    Notes
    -----
    Returns C * A * n ** (-1/5.) where ::

       A = min(std(x, ddof=1), IQR/1.349)
       IQR = np.subtract.reduce(np.percentile(x, [75,25]))
       C = constant from Hansen (2009)

    When using a Gaussian kernel this is equivalent to the 'scott' bandwidth up
    to two decimal places. This is the accuracy to which the 'scott' constant is
    specified.

    References
    ----------

    Silverman, B.W. (1986) `Density Estimation.`
    Hansen, B.E. (2009) `Lecture Notes on Nonparametrics.`
    """
    if kernel is None:
        kernel = kernels.Gaussian()
    C = kernel.normal_reference_constant
    A = _select_sigma(x)
    n = len(x)
    return C * A * n ** (-0.2)


# Plug-In Methods
def _isj_fixed_point(t, nobs, k2, a2):
    """
    Fixed point equation of the improved Sheather-Jones bandwidth

    Returns ``t - xi * gamma^[l](t)`` of Botev et al. (2010) with ``l = 7``
    for the data rescaled to the unit interval. ``k2`` are the squared
    frequencies and ``a2`` the squared coefficients of the cosine series of
    the binned data.
    """
    ell = 7
    k2pi2 = k2 * np.pi**2

    def deriv_norm(s, t_s):
        # squared L2 norm of the derivative of order s of the density
        # estimate with squared bandwidth t_s
        return 2 * np.pi ** (2 * s) * np.sum(k2**s * a2 * np.exp(-k2pi2 * t_s))

    f = deriv_norm(ell, t)
    for s in range(ell - 1, 1, -1):
        odd_prod = np.prod(np.arange(1, 2 * s, 2))
        const = (1 + 0.5 ** (s + 0.5)) / 3
        t_s = (2 * const * odd_prod / np.sqrt(2 * np.pi) / (nobs * f)) ** (
            2 / (3 + 2 * s)
        )
        f = deriv_norm(s, t_s)
    return t - (2 * nobs * np.sqrt(np.pi) * f) ** (-2 / 5)


def _bw_isj(x, gridsize=2**14):
    """
    Improved Sheather-Jones bandwidth using a histogram with gridsize bins

    See ``bw_isj``, which uses the default gridsize.
    """
    x = array_like(x, "x", dtype=np.float64, ndim=1)
    x_min = x.min()
    x_range = x.max() - x_min
    if x_range == 0:
        return 0.0
    # the grid extends the range of the data by one tenth on both sides
    lower = x_min - x_range / 10
    grid_range = 1.2 * x_range
    counts = np.histogram(x, bins=gridsize, range=(lower, lower + grid_range))[0]
    # coefficients of the cosine series of the binned data, which are
    # rescaled to the unit interval
    a2 = (fft.dct(counts / counts.sum(), type=2)[1:] / 2) ** 2
    k2 = np.arange(1, gridsize, dtype=np.float64) ** 2
    nobs = len(np.unique(x))
    args = (nobs, k2, a2)
    t_upp = 0.1
    with np.errstate(divide="ignore", invalid="ignore"):
        has_root = _isj_fixed_point(t_upp, *args) > 0
    if not has_root:
        raise RuntimeError(
            "The improved Sheather-Jones bandwidth could not be computed, the "
            "fixed point equation has no solution. This is common in small "
            "samples and if the data have few distinct values. Either provide "
            "the bandwidth or use an alternative method."
        )
    t_star = optimize.brentq(
        _isj_fixed_point, 0, t_upp, args=args, xtol=np.finfo(np.float64).tiny
    )
    return np.sqrt(t_star) * grid_range


def bw_isj(x, kernel=None):
    """
    Improved Sheather-Jones plug-in bandwidth for the Gaussian kernel.

    The bandwidth is the solution of the fixed point equation of Botev,
    Grotowski and Kroese (2010). It does not use a normal reference
    distribution, in contrast to the rule of thumb bandwidths, and is also
    appropriate for densities that are far from the normal distribution, for
    example multimodal densities.

    .. versionadded:: 0.15.1

    Parameters
    ----------
    x : array_like
        Array for which to get the bandwidth.
    kernel : CustomKernel instance, optional
        The bandwidth is only available for the Gaussian kernel, which is
        the default. A ValueError is raised for any other kernel.

    Returns
    -------
    float
        The estimate of the bandwidth.

    Raises
    ------
    RuntimeError
        If the fixed point equation has no solution. This is common in small
        samples and for data with few distinct values, see the Notes.

    See Also
    --------
    bw_normal_reference : Plug-in bandwidth based on a normal reference.
    bw_scott : Scott's rule of thumb.
    bw_silverman : Silverman's rule of thumb.

    Notes
    -----
    The plug-in bandwidth that minimizes the asymptotic mean integrated
    squared error depends on the squared L2 norm of the second derivative of
    the density. The norm of the derivative of order ``s`` is estimated with
    a bandwidth that depends on the norm of the derivative of order
    ``s + 1``. The method starts this recursion at ``s = 7`` with the
    bandwidth itself and solves for the bandwidth that is a fixed point of
    the recursion.

    The implementation follows the algorithm of Botev et al. (2010). The
    data are counted in a histogram with ``2**14`` bins on the range of the
    data extended by one tenth on both sides, the norms of the derivatives
    are computed from the discrete cosine transform of the bin counts, and
    the fixed point equation is solved for a squared bandwidth of at most
    one tenth of the squared length of the grid. The number of distinct
    values in ``x`` is used as number of observations.

    The fixed point equation often has no solution in small samples. In
    simulated samples from the normal distribution it has no solution in
    around 40 percent of the samples with 20 observations and in fewer than
    1 percent of the samples with 50 observations. In samples from the
    uniform distribution it has no solution in around 30 percent of the
    samples with 100 observations and in around 3 percent of the samples
    with 200 observations.

    The bandwidth cannot be resolved if it is small compared to the width of
    a bin, which can be the case if the range of the data is dominated by
    outliers.

    References
    ----------
    Botev, Z.I., Grotowski, J.F. and Kroese, D.P. (2010) Kernel density
        estimation via diffusion. The Annals of Statistics 38(5), 2916-2957.

    Examples
    --------
    The distribution of the yearly sunspot numbers is skewed to the right.

    >>> import statsmodels.api as sm
    >>> from statsmodels.nonparametric import bandwidths
    >>> x = sm.datasets.sunspots.load_pandas().data["SUNACTIVITY"]
    >>> print(f"{bandwidths.bw_isj(x):.3f}")
    4.874
    >>> print(f"{bandwidths.bw_normal_reference(x):.3f}")
    13.420

    The bandwidth is available in ``KDEUnivariate`` with ``bw="isj"``.

    >>> kde = sm.nonparametric.KDEUnivariate(x).fit(bw="isj")
    >>> print(f"{kde.bw:.3f}")
    4.874
    """
    if kernel is not None and not isinstance(kernel, kernels.Gaussian):
        raise ValueError(
            "The improved Sheather-Jones bandwidth is only available for the "
            "Gaussian kernel."
        )
    return _bw_isj(x)


# Least Squares Cross-Validation
# Helper Functions


bandwidth_funcs = {
    "scott": bw_scott,
    "silverman": bw_silverman,
    "normal_reference": bw_normal_reference,
    "isj": bw_isj,
}


@Substitution(", ".join(sorted(bandwidth_funcs.keys())))
def select_bandwidth(x, bw, kernel):
    """
    Selects bandwidth for a selection rule bw

    This is a wrapper around existing bandwidth selection rules.

    Parameters
    ----------
    x : array_like
        Array for which to get the bandwidth
    bw : str
        Name of the bandwidth selection rule, currently supported are:
        %s
    kernel : CustomKernel instance
        Passed through to the selected bandwidth rule. Used by the
        'normal_reference' rule and by 'isj', which requires the Gaussian
        kernel; ignored by 'scott' and 'silverman'.

    Returns
    -------
    bw : float
        The estimate of the bandwidth
    """
    bw = bw.lower()
    if bw not in bandwidth_funcs:
        raise ValueError(f"Bandwidth {bw} not understood")
    bandwidth = bandwidth_funcs[bw](x, kernel)
    if np.any(bandwidth == 0):
        # eventually this can fall back on another selection criterion.
        err = (
            "Selected KDE bandwidth is 0. Cannot estimate density. "
            "Either provide the bandwidth during initialization or use "
            "an alternative method."
        )
        raise RuntimeError(err)
    else:
        return bandwidth
