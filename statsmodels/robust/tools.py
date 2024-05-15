"""
Created on Mar. 11, 2024 10:41:37 p.m.

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from scipy import stats, integrate, optimize

from statsmodels.tools.testing import Holder


def _var_normal(norm):
    """Variance factor for asymptotic relative efficiency of mean M-estimator.

    The reference distribution is the standard normal distribution.
    This assumes that the psi function is continuous.

    Relative efficiency is 1 / var_normal

    Parameters
    ----------
    norm : instance of a RobustNorm subclass.
        Norm for which variance for relative efficiency is computed.

    Returns
    -------
    Variance factor.

    Notes
    -----
    This function does not verify that the assumption on the psi function and
    it's derivative hold.

    Examples
    --------
    The following computes the relative efficiency of an M-estimator for the
    mean using HuberT norm. At the default tuning parameter, the relative
    efficiency is 95%.

    >>> import statsmodels.robust import norms
    >>> v = _var_normal(norms.HuberT())
    >>> eff = 1 / v
    >>> v, eff
    (1.0526312909084732, 0.9500002599551741)

    Notes
    -----
    S-estimator for mean and regression also have the same variance and
    efficiency computation as M-estimators. Therefore, this function can
    be used also for S-estimators and other estimators that .

    Reference
    ---------
    Menenez et al., but it's also in all text books for robust statistics.


    """
    num = stats.norm.expect(lambda x: norm.psi(x)**2)
    denom = stats.norm.expect(lambda x: norm.psi_deriv(x))**2
    return num / denom


def _var_normal_jump(norm):
    """Variance factor for asymptotic relative efficiency of mean M-estimator.

    The reference distribution is the standard normal distribution.
    This allows for the case when the psi function is not continuous, i.e.
    has jumps as in TrimmedMean norm.

    Relative efficiency is 1 / var_normal

    Parameters
    ----------
    norm : instance of a RobustNorm subclass.
        Norm for which variance for relative efficiency is computed.

    Returns
    -------
    Variance factor.

    Notes
    -----
    This function does not verify that the assumption on the psi function and
    it's derivative hold.

    Examples
    --------

    >>> import statsmodels.robust import norms
    >>> v = _var_normal_jump(norms.HuberT())
    >>> eff = 1 / v
    >>> v, eff
    (1.0526312908510451, 0.950000260007003)

    Reference
    ---------
    Menenez et al., but it's also in all text books for robust statistics.


    """
    num = stats.norm.expect(lambda x: norm.psi(x)**2)

    def func(x):
        # derivative normal pdf
        # d/dx(exp(-x^2/2)/sqrt(2 π)) = -(e^(-x^2/2) x)/sqrt(2 π)
        return norm.psi(x) * (- x * np.exp(-x**2/2) / np.sqrt(2 * np.pi))

    denom = integrate.quad(func, -np.inf, np.inf)[0]
    return num / denom**2


def _get_tuning_param(norm, eff, kwd="c", kwargs=None, use_jump=False,
                      bracket=None,
                      ):
    """Tuning parameter for RLM norms for required relative efficiency.

    Parameters
    ----------
    norm : instance of RobustNorm subclass
    eff : float in (0, 1)
        Required asymptotic relative efficiency compared to least squares
        at the normal reference distribution. For example, ``eff=0.95`` for
        95% efficiency.
    kwd : str
        Name of keyword for tuning parameter.
    kwargs : dict or None
        Dict for other keyword parameters.
    use_jump : bool
        If False (default), then use computation that require continuous
        psi function.
        If True, then use computation then the psi function can have jump
        discontinuities.
    bracket : None or tuple
        Bracket with lower and upper bounds to use for scipy.optimize.brentq.
        If None, than a default bracket, currently [0.1, 10], is used.

    Returns
    -------
    Float : Value of tuning parameter to achieve asymptotic relative
        efficiency.

    """
    kwds = {} if kwargs is None else kwargs
    if bracket is None:
        bracket = [0.1, 10]

    if not use_jump:
        def func(c):
            # kwds.update({kwd: c})
            # return _var_normal(norm(**kwds)) - 1 / eff
            norm._set_tuning_param(c, inplace=True)
            return _var_normal(norm) - 1 / eff
    else:
        def func(c):
            norm._set_tuning_param(c, inplace=True)
            return _var_normal_jump(norm(**kwds) - 1 / eff)

    res = optimize.brentq(func, *bracket)
    return res


def tuning_s_estimator_mean(norm, breakdown=None):
    """Tuning parameter and scale bias correction for S-estimators of mean.

    The reference distribution is the normal distribution.
    This requires a (hard) redescending norm, i.e. with finite max rho.

    Parameters
    ----------
    norm : instance of RobustNorm subclass
    breakdown : float or iterable of float in (0, 0.5]
        Desired breakdown point between 0 and 0.5.
        Default if breakdown is None is a list of breakdown points.

    Returns
    -------
    Holder instance with the following attributes :

     - `breakdown` : breakdown point
     - `eff` : relative efficiency
     - `param` : tuning parameter for norm
     - `scale_bias` : correction term for Fisher consistency.

    Notes
    -----
    Based on Rousseeuw and Leroy (1987). See table 19, p. 142 that can be
    replicated by this function for TukeyBiweight norm.
    Note, the results of this function are based computation without rounding
    to decimal precision, and differ in some cases in the last digit from
    the table by Rousseeuw and Leroy.

    Numerical expectation and root finding based on scipy integrate and
    optimize.

    TODO: more options for details, numeric approximation and root finding.
    There is currently no feasibility check in functions.

    Reference
    ---------
    Rousseeuw and Leroy book


    """
    if breakdown is None:
        bps = [0.5, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.1, 0.05]
    else:
        # allow for scalar bp
        try:
            _ = iter(breakdown)
            bps = breakdown
        except TypeError:
            bps = [breakdown]

    def func(c):
        norm_ = norm
        norm_._set_tuning_param(c, inplace=True)
        bp = stats.norm.expect(lambda x : norm_.rho(x)) / norm_.max_rho()
        return bp

    res = []
    for bp in bps:
        c_bp = optimize.brentq(lambda c0: func(c0) - bp, 0.1, 10)
        norm._set_tuning_param(c_bp, inplace=True)  # inplace modification
        eff = 1 / _var_normal(norm)
        b = stats.norm.expect(lambda x : norm.rho(x))
        res.append([bp, eff, c_bp, b])

    if np.size(bps) > 1:
        res = np.asarray(res).T
    else:
        # use one list
        res = res[0]

    res2 = Holder(
        breakdown=res[0],
        eff=res[1],
        param=res[2],
        scale_bias=res[3],
        all=res,
        )

    return res2


def scale_bias_cov_biw(c, k_vars):
    """Multivariate scale bias correction for TukeyBiweight norm.

    This uses the chisquare distribution as reference distribution for the
    squared Mahalanobis distance.
    """
    p = k_vars # alias for formula
    chip, chip2, chip4, chip6 = stats.chi2.cdf(c**2, [p, p + 2, p + 4, p + 6])
    b = p / 2 * chip2 -  p * (p + 2) / (2 * c**2) * chip4
    b += p * (p + 2) * (p + 4) / (6 * c**4) * chip6 + c**2 / 6 * (1 - chip)
    return b, b / (c**2 / 6)


def scale_bias_cov(norm, k_vars):
    """Multivariate scale bias correction.


    Parameter
    ---------
    norm : norm instance
        The rho function of the norm is used in the moment condition for
        estimating scale.
    k_vars : int
        Number of random variables in the multivariate data.

    Returns
    -------
    scale_bias: float
    breakdown_point : float
        Breakdown point computed as scale bias divided by max rho.
    """

    rho = lambda x: (norm.rho(np.sqrt(x)))  # noqa
    scale_bias = stats.chi2.expect(rho, args=(k_vars,))
    return scale_bias, scale_bias / norm.max_rho()


def tuning_s_cov(norm, k_vars, breakdown_point=0.5, limits=()):
    """Tuning parameter for multivariate S-estimator given breakdown point.
    """
    from .norms import TukeyBiweight  # avoid circular import

    if not limits:
        limits = (0.5, 30)

    if isinstance(norm, TukeyBiweight):
        def func(c):
            return scale_bias_cov_biw(c, k_vars)[1] - breakdown_point
    else:
        norm = norm._set_tuning_param(2., inplace=False)  # create copy
        def func(c):
            norm._set_tuning_param(c, inplace=True)
            return scale_bias_cov(norm, k_vars)[1] - breakdown_point

    p_tune = optimize.brentq(func, limits[0], limits[1])
    return p_tune


#  ##### tables

tukeybiweight_bp = {
    # breakdown point : (tuning parameter, efficiency, scale bias)
    0.50: (1.547645, 0.286826, 0.199600),
    0.45: (1.756059, 0.369761, 0.231281),
    0.40: (1.987965, 0.461886, 0.263467),
    0.35: (2.251831, 0.560447, 0.295793),
    0.30: (2.560843, 0.661350, 0.327896),
    0.25: (2.937015, 0.759040, 0.359419),
    0.20: (3.420681, 0.846734, 0.390035),
    0.15: (4.096255, 0.917435, 0.419483),
    0.10: (5.182361, 0.966162, 0.447614),
    0.05: (7.545252, 0.992424, 0.474424),
    }

tukeybiweight_eff = {
    #efficiency : (tuning parameter, breakdown point)
    0.65: (2.523102, 0.305646),
    0.70: (2.697221, 0.280593),
    0.75: (2.897166, 0.254790),
    0.80: (3.136909, 0.227597),
    0.85: (3.443690, 0.197957),
    0.90: (3.882662, 0.163779),
    0.95: (4.685065, 0.119414),
    0.97: (5.596823, 0.087088),
    0.98: (5.920719, 0.078604),
    0.99: (7.041392, 0.056969),
    }

# relative efficiency for M and MM estimator of multivariate location
# Table 2 from Kudraszow and Maronna JMA 2011
# k in [1, 2, 3, 4, 5, 10], eff in [0.8, 0.9, 0.95
# TODO: need to replace with more larger table and more digits.
tukeybiweight_mvmean_eff = {
        (1, 0.8): 3.14, (2, 0.8): 3.51, (3, 0.8): 3.82, (4, 0.8): 4.1,
        (5, 0.8): 4.34, (10, 0.8): 5.39,
        (1, 0.9): 3.88, (2, 0.9): 4.28, (3, 0.9): 4.62, (4, 0.9): 4.91,
        (5, 0.9): 5.18, (10, 0.9): 6.38,
        (1, 0.95): 4.68, (2, 0.95): 5.12, (3, 0.95): 5.48, (4, 0.95): 5.76,
        (5, 0.95): 6.1, (10, 0.95): 7.67,
        }
