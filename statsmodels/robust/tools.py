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
            kwds.update({kwd: c})
            return _var_normal(norm(**kwds)) - 1 / eff
    else:
        def func(c):
            kwds.update({kwd: c})
            return _var_normal_jump(norm(**kwds) - 1 / eff)

    res = optimize.brentq(func, *bracket)
    return res


def tuning_s_estimator_mean(norm, breakdown=None):
    """Tuning parameter and scale bias correction for S-estimators of mean.

    The reference distribution is the normal distribution.

    Parameters
    ----------
    norm : RobustNorm subclass
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
        norm_ = norm(c=c)
        bp = stats.norm.expect(lambda x : norm_.rho(x)) / norm_.rho(norm_.c)
        return bp

    res = []
    for bp in bps:
        c_bp = optimize.brentq(lambda c0: func(c0) - bp, 0.5, 10)
        norm_ = norm(c=c_bp)
        eff = 1 / _var_normal(norm_)
        b = stats.norm.expect(lambda x : norm_.rho(x))
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
