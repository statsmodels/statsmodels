from statsmodels.compat.pandas import Substitution

import numpy as np
from scipy.stats import norm, scoreatpercentile

from statsmodels.sandbox.nonparametric import kernels
from statsmodels.tools.validation import float_like


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
# Least Squares Cross-Validation
# Helper Functions


bandwidth_funcs = {
    "scott": bw_scott,
    "silverman": bw_silverman,
    "normal_reference": bw_normal_reference,
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
        Passed through to the selected bandwidth rule. Used only by the
        'normal_reference' rule; ignored by 'scott' and 'silverman'.

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
