"""
Support and standalone functions for Robust Linear Models

References
----------
PJ Huber.  'Robust Statistics' John Wiley and Sons, Inc., New York, 1981.

R Venables, B Ripley. 'Modern Applied Statistics in S'
    Springer, New York, 2002.

C Croux, PJ Rousseeuw, 'Time-efficient algorithms for two highly robust
estimators of scale' Computational statistics. Physica, Heidelberg, 1992.
"""

import numpy as np
from scipy import stats
from scipy.stats import norm as Gaussian

from statsmodels.tools import tools
from statsmodels.tools.validation import array_like, float_like

from . import norms
from ._qn import _qn

GAUSSIAN_3_4 = Gaussian.ppf(3 / 4.0)
GAUSSIAN_IQR = GAUSSIAN_3_4 - Gaussian.ppf(1 / 4)
ONE_OVER_SQRT2_GAUSSIAN_5_8 = 1 / (np.sqrt(2) * Gaussian.ppf(5 / 8))


class Holder:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def mad(a, c=GAUSSIAN_3_4, axis=0, center=np.median):
    # c \approx .6745
    """
    The Median Absolute Deviation along given axis of an array

    Parameters
    ----------
    a : array_like
        Input array.
    c : float, optional
        The normalization constant.  Defined as scipy.stats.norm.ppf(3/4.),
        which is approximately 0.6745.
    axis : int, optional
        The default is 0. Can also be None.
    center : callable or float
        If a callable is provided, such as the default `np.median` then it
        is expected to be called center(a). The axis argument will be applied
        via np.apply_over_axes. Otherwise, provide a float.

    Returns
    -------
    mad : float
        `mad` = median(abs(`a` - center))/`c`
    """
    a = array_like(a, "a", ndim=None)
    c = float_like(c, "c")
    if not a.size:
        center_val = 0.0
    elif callable(center):
        if axis is not None:
            center_val = np.apply_over_axes(center, a, axis)
        else:
            center_val = center(a.ravel())
    else:
        center_val = float_like(center, "center")
    err = (np.abs(a - center_val)) / c
    if not err.size:
        if axis is None or err.ndim == 1:
            return np.nan
        else:
            shape = list(err.shape)
            shape.pop(axis)
            return np.empty(shape)
    return np.median(err, axis=axis)


def iqr(a, c=GAUSSIAN_IQR, axis=0):
    """
    The normalized interquartile range along given axis of an array

    Parameters
    ----------
    a : array_like
        Input array.
    c : float, optional
        The normalization constant, used to get consistent estimates of the
        standard deviation at the normal distribution.  Defined as
        scipy.stats.norm.ppf(3/4.) - scipy.stats.norm.ppf(1/4.), which is
        approximately 1.349.
    axis : int, optional
        The default is 0. Can also be None.

    Returns
    -------
    The normalized interquartile range
    """
    a = array_like(a, "a", ndim=None)
    c = float_like(c, "c")

    if a.ndim == 0:
        raise ValueError("a should have at least one dimension")
    elif a.size == 0:
        return np.nan
    else:
        quantiles = np.quantile(a, [0.25, 0.75], axis=axis)
        return np.squeeze(np.diff(quantiles, axis=0) / c)


def qn_scale(a, c=ONE_OVER_SQRT2_GAUSSIAN_5_8, axis=0):
    """
    Computes the Qn robust estimator of scale

    The Qn scale estimator is a more efficient alternative to the MAD.
    The Qn scale estimator of an array a of length n is defined as
    c * {abs(a[i] - a[j]): i<j}_(k), for k equal to [n/2] + 1 choose 2. Thus,
    the Qn estimator is the k-th order statistic of the absolute differences
    of the array. The optional constant is used to normalize the estimate
    as explained below. The implementation follows the algorithm described
    in Croux and Rousseeuw (1992).

    Parameters
    ----------
    a : array_like
        Input array.
    c : float, optional
        The normalization constant. The default value is used to get consistent
        estimates of the standard deviation at the normal distribution.
    axis : int, optional
        The default is 0.

    Returns
    -------
    {float, ndarray}
        The Qn robust estimator of scale
    """
    a = array_like(a, "a", ndim=None, dtype=np.float64, contiguous=True, order="C")
    c = float_like(c, "c")
    if a.ndim == 0:
        raise ValueError("a should have at least one dimension")
    elif a.size == 0:
        return np.nan
    else:
        out = np.apply_along_axis(_qn, axis=axis, arr=a, c=c)
        if out.ndim == 0:
            return float(out)
        return out


def _qn_naive(a, c=ONE_OVER_SQRT2_GAUSSIAN_5_8):
    """
    A naive implementation of the Qn robust estimator of scale, used solely
    to test the faster, more involved one

    Parameters
    ----------
    a : array_like
        Input array.
    c : float, optional
        The normalization constant, used to get consistent estimates of the
        standard deviation at the normal distribution.  Defined as
        1/(np.sqrt(2) * scipy.stats.norm.ppf(5/8)), which is 2.219144.

    Returns
    -------
    The Qn robust estimator of scale
    """
    a = np.squeeze(a)
    n = a.shape[0]
    if a.size == 0:
        return np.nan
    else:
        h = int(n // 2 + 1)
        k = int(h * (h - 1) / 2)
        idx = np.triu_indices(n, k=1)
        diffs = np.abs(a[idx[0]] - a[idx[1]])
        output = np.partition(diffs, kth=k - 1)[k - 1]
        output = c * output
        return output


class Huber:
    """
    Huber's proposal 2 for estimating location and scale jointly.

    Parameters
    ----------
    c : float, optional
        Threshold used in threshold for chi=psi**2.  Default value is 1.5.
    tol : float, optional
        Tolerance for convergence.  Default value is 1e-08.
    maxiter : int, optional0
        Maximum number of iterations.  Default value is 30.
    norm : statsmodels.robust.norms.RobustNorm, optional
        A robust norm used in M estimator of location. If None,
        the location estimator defaults to a one-step
        fixed point version of the M-estimator using Huber's T.

    call
        Return joint estimates of Huber's scale and location.

    Examples
    --------
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> chem_data = np.array([2.20, 2.20, 2.4, 2.4, 2.5, 2.7, 2.8, 2.9, 3.03,
    ...        3.03, 3.10, 3.37, 3.4, 3.4, 3.4, 3.5, 3.6, 3.7, 3.7, 3.7, 3.7,
    ...        3.77, 5.28, 28.95])
    >>> sm.robust.scale.huber(chem_data)
    (array(3.2054980819923693), array(0.67365260010478967))
    """

    def __init__(self, c=1.5, tol=1.0e-08, maxiter=30, norm=None):
        self.c = c
        self.maxiter = maxiter
        self.tol = tol
        self.norm = norm
        tmp = 2 * Gaussian.cdf(c) - 1
        self.gamma = tmp + c**2 * (1 - tmp) - 2 * c * Gaussian.pdf(c)

    def __call__(self, a, mu=None, initscale=None, axis=0):
        """
        Compute Huber's proposal 2 estimate of scale, using an optional
        initial value of scale and an optional estimate of mu. If mu
        is supplied, it is not reestimated.

        Parameters
        ----------
        a : ndarray
            1d array
        mu : float or None, optional
            If the location mu is supplied then it is not reestimated.
            Default is None, which means that it is estimated.
        initscale : float or None, optional
            A first guess on scale.  If initscale is None then the standardized
            median absolute deviation of a is used.

        Notes
        -----
        `Huber` minimizes the function

        sum(psi((a[i]-mu)/scale)**2)

        as a function of (mu, scale), where

        psi(x) = np.clip(x, -self.c, self.c)
        """
        a = np.asarray(a)
        if mu is None:
            n = a.shape[axis] - 1
            mu = np.median(a, axis=axis)
            est_mu = True
        else:
            n = a.shape[axis]
            est_mu = False

        if initscale is None:
            scale = mad(a, axis=axis)
        else:
            scale = initscale
        scale = tools.unsqueeze(scale, axis, a.shape)
        mu = tools.unsqueeze(mu, axis, a.shape)
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
                if self.norm is None:
                    # This is a one-step fixed-point estimator
                    # if self.norm == norms.HuberT
                    # It should be faster than using norms.HuberT
                    nmu = (
                        np.clip(a, mu - self.c * scale, mu + self.c * scale).sum(axis)
                        / a.shape[axis]
                    )
                else:
                    nmu = norms.estimate_location(
                        a, scale, self.norm, axis, mu, self.maxiter, self.tol
                    )
            else:
                # Effectively, do nothing
                nmu = mu.squeeze()
            nmu = tools.unsqueeze(nmu, axis, a.shape)

            subset = np.less_equal(np.abs((a - mu) / scale), self.c)

            scale_num = np.sum(
                subset * (a - nmu) ** 2 + (1 - subset) * (scale * self.c) ** 2, axis
            )
            scale_denom = n * self.gamma
            nscale = np.sqrt(scale_num / scale_denom)
            nscale = tools.unsqueeze(nscale, axis, a.shape)

            test1 = np.all(np.less_equal(np.abs(scale - nscale), nscale * self.tol))
            test2 = np.all(np.less_equal(np.abs(mu - nmu), nscale * self.tol))
            if not (test1 and test2):
                mu = nmu
                scale = nscale
            else:
                return nmu.squeeze(), nscale.squeeze()
        raise ValueError(
            "joint estimation of location and scale failed "
            "to converge in %d iterations" % self.maxiter
        )


huber = Huber()


class HuberScale:
    r"""
    Huber's scaling for fitting robust linear models.

    Huber's scale is intended to be used as the scale estimate in the
    IRLS algorithm and is slightly different than the `Huber` class.

    Parameters
    ----------
    d : float, optional
        d is the tuning constant for Huber's scale.  Default is 2.5
    tol : float, optional
        The convergence tolerance
    maxiter : int, optiona
        The maximum number of iterations.  The default is 30.

    Methods
    -------
    call
        Return's Huber's scale computed as below

    Notes
    -----
    Huber's scale is the iterative solution to

    scale_(i+1)**2 = 1/(n*h)*sum(chi(r/sigma_i)*sigma_i**2

    where the Huber function is

    chi(x) = (x**2)/2       for \|x\| < d
    chi(x) = (d**2)/2       for \|x\| >= d

    and the Huber constant h = (n-p)/n*(d**2 + (1-d**2)*
    scipy.stats.norm.cdf(d) - .5 - d*sqrt(2*pi)*exp(-0.5*d**2)
    """

    def __init__(self, d=2.5, tol=1e-08, maxiter=30):
        self.d = d
        self.tol = tol
        self.maxiter = maxiter

    def __call__(self, df_resid, nobs, resid):
        h = (
            df_resid
            / nobs
            * (
                self.d**2
                + (1 - self.d**2) * Gaussian.cdf(self.d)
                - 0.5
                - self.d / (np.sqrt(2 * np.pi)) * np.exp(-0.5 * self.d**2)
            )
        )
        s = mad(resid)

        def subset(x):
            return np.less(np.abs(resid / x), self.d)

        def chi(s):
            return subset(s) * (resid / s) ** 2 / 2 + (1 - subset(s)) * (self.d**2 / 2)

        scalehist = [np.inf, s]
        niter = 1
        while (
            np.abs(scalehist[niter - 1] - scalehist[niter]) > self.tol
            and niter < self.maxiter
        ):
            nscale = np.sqrt(
                1 / (nobs * h) * np.sum(chi(scalehist[-1])) * scalehist[-1] ** 2
            )
            scalehist.append(nscale)
            niter += 1
            # TODO: raise on convergence failure?
        return scalehist[-1]


hubers_scale = HuberScale()


class MScale:
    """M-scale estimation.

    experimental interface, arguments and options will still change.

    Parameters
    ----------
    chi_func : callable
        The rho or chi function for the moment condition for estimating scale.
    scale_bias : float
        Factor in moment condition to obtain fisher consistency of the scale
        estimate at the normal distribution.
    """

    def __init__(self, chi_func, scale_bias):
        self.chi_func = chi_func
        self.scale_bias = scale_bias

    def __repr__(self):
        return repr(self.chi_func)

    def __call__(self, data, **kwds):
        return self.fit(data, **kwds)

    def fit(self, data, start_scale="mad", maxiter=100, rtol=1e-6, atol=1e-8):
        """
        Estimate M-scale using iteration.

        Parameters
        ----------
        data : array-like
            Data, currently assumed to be 1-dimensional.
        start_scale : string or float.
            Starting value of scale or method to compute the starting value.
            Default is using 'mad', no other string options are available.
        maxiter : int
            Maximum number of iterations.
        rtol : float
            Relative convergence tolerance.
        atol : float
            Absolute onvergence tolerance.

        Returns
        -------
        float : Scale estimate. The estimated variance is scale squared.
        Todo: switch to Holder instance with more information.

        """

        scale = _scale_iter(
            data,
            scale0=start_scale,
            maxiter=maxiter,
            rtol=rtol,
            atol=atol,
            meef_scale=self.chi_func,
            scale_bias=self.scale_bias,
        )

        return scale


def scale_trimmed(data, alpha, center="median", axis=0, distr=None, distargs=None):
    """scale estimate based on symmetrically trimmed sample

    The scale estimate is robust to a fraction alpha of outliers on each
    tail.
    The scale is normalized to correspond to a reference distribution, which
    is the normal distribution by default.

    Parameters
    ----------
    data : array_like
        dataset, by default (axis=0) observations are assumed to be in rows
        and variables in columns.
    alpha : float in interval (0, 1)
        Trimming fraction in each tail. The floor(nobs * alpha) smallest
        observations are trimmed, and the same number of the largest
        observations are trimmed. scale estimate is base on a fraction
        (1 - 2 * alpha) of observations.
    center : 'median', 'mean', 'tmean' or number
        `center` defines how the trimmed sample is centered. 'median' and
        'mean' are calculated on the full sample. `tmean` is the trimmed
        mean, calculated with the trimmed sample. If `center` is array_like
        then it needs to be scalar or correspond to the shape of the data
        reduced by axis.
    axis : int, default is 0
        axis along which scale is estimated.
    distr : None, 'raw' or a distribution instance
        Default if distr is None is the normal distribution `scipy.stats.norm`.
        This is the reference distribution to normalize the scale.
        Note: This cannot be a frozen instance, since it does not have an
        `expect` method.
        If distr is 'raw', then the scale is not normalized.
    distargs :
        Arguments for the distribution.

    Returns
    -------
    scale : float or array
        the estimated scale normalized for the reference distribution.

    Examples
    --------
    for normal distribution

    >>> np.random.seed(1)
    >>> x = 2 * np.random.randn(100)
    >>> scale_trimmed(x, 0.1)
    1.7479516739879672

    for t distribution
    >>> xt = stats.t.rvs(3, size=1000, scale=2)
    >>> print scale_trimmed(xt, alpha, distr=stats.t, distargs=(3,))
    2.06574778599

    compare to standard deviation of sample
    >>> xt.std()
    3.1457788359130481

    """

    if distr is None:
        distr = stats.norm
        if distargs is None:
            distargs = ()

    x = np.array(data)  # make copy for inplace sort
    if axis is None:
        x = x.ravel()
        axis = 0

    # TODO: latest numpy has partial sort
    x.sort(axis)
    nobs = x.shape[axis]

    if distr == "raw":
        c_inv = 1
    else:
        bound = distr.ppf(1 - alpha, *distargs)
        c_inv = distr.expect(lambda x: x * x, lb=-bound, ub=bound, args=distargs)

    cut_idx = np.floor(nobs * alpha).astype(int)
    sl = [slice(None, None, None)] * x.ndim
    sl[axis] = slice(cut_idx, -cut_idx)
    # x_trimmed = x[cut_idx:-cut_idx]
    # cut in axis
    x_trimmed = x[tuple(sl)]

    center_type = center
    if center in ["med", "median"]:
        center = np.median(x, axis=axis)
    elif center == "mean":
        center = np.mean(x, axis=axis)
    elif center == "tmean":
        center = np.mean(x_trimmed, axis=axis)
    else:
        # assume number
        center_type = "user"

    center_ndim = np.ndim(center)
    if (center_ndim > 0) and (center_ndim < x.ndim):
        center = np.expand_dims(center, axis)

    s_raw = ((x_trimmed - center) ** 2).sum(axis)
    scale = np.sqrt(s_raw / nobs / c_inv)

    res = Holder(
        scale=scale,
        center=center,
        center_type=center_type,
        trim_idx=cut_idx,
        nobs=nobs,
        distr=distr,
        scale_correction=1.0 / c_inv,
    )
    return res


def _weight_mean(x, c):
    """Tukey-biweight, bisquare weights used in tau scale.

    Parameters
    ----------
    x : ndarray
        Data
    c : float
        Parameter for bisquare weights

    Returns
    -------
    ndarray : weights
    """
    x = np.asarray(x)
    w = (1 - (x / c) ** 2) ** 2 * (np.abs(x) <= c)
    return w


def _winsor(x, c):
    """Winsorized squared data used in tau scale.

    Parameters
    ----------
    x : ndarray
        Data
    c : float
        threshold

    Returns
    -------
    winsorized squared data, ``np.minimum(x**2, c**2)``
    """
    return np.minimum(x**2, c**2)


def scale_tau(
    data,
    cm=4.5,
    cs=3,
    weight_mean=_weight_mean,
    weight_scale=_winsor,
    normalize=True,
    ddof=0,
):
    """Tau estimator of univariate scale.

    Experimental, API will change

    Parameters
    ----------
    data : array_like, 1-D or 2-D
        If data is 2d, then the location and scale estimates
        are calculated for each column
    cm : float
        constant used in call to weight_mean
    cs : float
        constant used in call to weight_scale
    weight_mean : callable
        function to calculate weights for weighted mean
    weight_scale : callable
        function to calculate scale, "rho" function
    normalize : bool
        rescale the scale estimate so it is consistent when the data is
        normally distributed. The computation assumes winsorized (truncated)
        variance.

    Returns
    -------
    mean : nd_array
        robust mean
    std : nd_array
        robust estimate of scale (standard deviation)

    Notes
    -----
    Uses definition of Maronna and Zamar 2002, with weighted mean and
    trimmed variance.
    The normalization has been added to match R robustbase.
    R robustbase uses by default ddof=0, with option to set it to 2.

    References
    ----------
    .. [1] Maronna, Ricardo A, and Ruben H Zamar. “Robust Estimates of Location
       and Dispersion for High-Dimensional Datasets.” Technometrics 44, no. 4
       (November 1, 2002): 307-17. https://doi.org/10.1198/004017002188618509.
    """

    x = np.asarray(data)
    nobs = x.shape[0]

    med_x = np.median(x, 0)
    xdm = x - med_x
    mad_x = np.median(np.abs(xdm), 0)
    wm = weight_mean(xdm / mad_x, cm)
    mean = (wm * x).sum(0) / wm.sum(0)
    var = mad_x**2 * weight_scale((x - mean) / mad_x, cs).sum(0) / (nobs - ddof)

    cf = 1
    if normalize:
        c = cs * stats.norm.ppf(0.75)
        cf = 2 * ((1 - c**2) * stats.norm.cdf(c) - c * stats.norm.pdf(c) + c**2) - 1
    # return Holder(loc=mean, scale=np.sqrt(var / cf))
    return mean, np.sqrt(var / cf)


debug = 0


def _scale_iter(
    data,
    scale0="mad",
    maxiter=100,
    rtol=1e-6,
    atol=1e-8,
    meef_scale=None,
    scale_bias=None,
    iter_method="rho",
    ddof=0,
):
    """iterative scale estimate base on "rho" function"""
    x = np.asarray(data)
    nobs = x.shape[0]
    if scale0 == "mad":
        scale0 = mad(x, center=0)

    for _ in range(maxiter):
        x_scaled = x / scale0
        if iter_method == "rho":
            scale = scale0 * np.sqrt(
                np.sum(meef_scale(x / scale0)) / scale_bias / (nobs - ddof)
            )
        else:
            weights_scale = meef_scale(x_scaled) / (1e-50 + x_scaled**2)
            scale2 = (weights_scale * x**2).sum() / (nobs - ddof)
            scale2 /= scale_bias
            scale = np.sqrt(scale2)
        if debug:
            print(scale)
        if np.allclose(scale, scale0, atol=atol, rtol=rtol):
            break
        scale0 = scale

    return scale
