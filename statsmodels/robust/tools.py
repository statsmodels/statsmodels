"""
Created on Mar. 11, 2024 10:41:37 p.m.

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from scipy import integrate, optimize, stats

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
    num = stats.norm.expect(lambda x: norm.psi(x) ** 2)
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
            return _var_normal_jump(norm) - 1 / eff

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
        bp = stats.norm.expect(lambda x: norm_.rho(x)) / norm_.max_rho()
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
    p = k_vars  # alias for formula
    chip, chip2, chip4, chip6 = stats.chi2.cdf(c**2, [p, p + 2, p + 4, p + 6])
    b = p / 2 * chip2 - p * (p + 2) / (2 * c**2) * chip4
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

    def rho(x):
        return norm.rho(np.sqrt(x))
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


def eff_mvmean(norm, k_vars):
    """Efficiency for M-estimator of multivariate mean at normal distribution.

    This also applies to estimators that are locally equivalent to an
    M-estimator such as S- and MM-estimators.

    Parameters
    ----------
    norm : instance of norm class
    k_vars : int
        Number of variables in multivariate random variable, i.e. dimension.

    Returns
    -------
    eff : float
        Asymptotic relative efficiency of mean at normal distribution.
    alpha : float
        Numerical integral. Efficiency is beta**2 / alpha
    beta : float
        Numerical integral.

    Notes
    -----
    This implements equ. (5.3) p. 1671 in Lopuhaä 1989

    References
    ----------

    .. [1] Lopuhaä, Hendrik P. 1989. “On the Relation between S-Estimators
       and M-Estimators of Multivariate Location and Covariance.”
       The Annals of Statistics 17 (4): 1662–83.

    """
    k = k_vars  # shortcut

    def f_alpha(d):
        return norm.psi(d) ** 2 / k

    def f_beta(d):
        return (1 - 1 / k) * norm.weights(d) + 1 / k * norm.psi_deriv(d)

    alpha = stats.chi(k).expect(f_alpha)
    beta = stats.chi(k).expect(f_beta)
    return beta**2 / alpha, alpha, beta


def eff_mvshape(norm, k_vars):
    """Efficiency of M-estimator of multivariate shape at normal distribution.

    This also applies to estimators that are locally equivalent to an
    M-estimator such as S- and MM-estimators.

    Parameters
    ----------
    norm : instance of norm class
    k_vars : int
        Number of variables in multivariate random variable, i.e. dimension.

    Returns
    -------
    eff : float
        Asymptotic relative efficiency of mean at normal distribution.
    alpha : float
        Numerical integral. Efficiency is beta**2 / alpha
    beta : float
        Numerical integral.

    Notes
    -----
    This implements sigma_1 in equ. (5.5) p. 1671 in Lopuhaä 1989.
    Efficiency of shape is approximately 1 / sigma1.

    References
    ----------

    .. [1] Lopuhaä, Hendrik P. 1989. “On the Relation between S-Estimators
       and M-Estimators of Multivariate Location and Covariance.”
       The Annals of Statistics 17 (4): 1662–83.

    """

    k = k_vars  # shortcut

    def f_a(d):
        return k * (k + 2) * norm.psi(d) ** 2 * d**2

    def f_b(d):
        return norm.psi_deriv(d) * d**2 + (k + 1) * norm.psi(d) * d

    a = stats.chi(k).expect(f_a)
    b = stats.chi(k).expect(f_b)
    return b**2 / a, a, b


def tuning_m_cov_eff(norm, k_vars, efficiency=0.95, eff_mean=True, limits=()):
    """Tuning parameter for multivariate M-estimator given efficiency.

    This also applies to estimators that are locally equivalent to an
    M-estimator such as S- and MM-estimators.

    Parameters
    ----------
    norm : instance of norm class
    k_vars : int
        Number of variables in multivariate random variable, i.e. dimension.
    efficiency : float < 1
        Desired asymptotic relative efficiency of mean estimator.
        Default is 0.95.
    eff_mean : bool
        If eff_mean is true (default), then tuning parameter is to achieve
        efficiency of mean estimate.
        If eff_mean is fale, then tuning parameter is to achieve efficiency
        of shape estimate.
    limits : tuple
        Limits for rootfinding with scipy.optimize.brentq.
        In some cases the interval limits for rootfinding can be too small
        and not cover the root. Current default limits are (0.5, 30).

    Returns
    -------
    float : Tuning parameter for the norm to achieve desired efficiency.
        Asymptotic relative efficiency of mean at normal distribution.

    Notes
    -----
    This uses numerical integration and rootfinding and will be
    relatively slow.
    """
    if not limits:
        limits = (0.5, 30)

    # make copy of norm
    norm = norm._set_tuning_param(1, inplace=False)

    if eff_mean:
        def func(c):
            norm._set_tuning_param(c, inplace=True)
            return eff_mvmean(norm, k_vars)[0] - efficiency
    else:
        def func(c):
            norm._set_tuning_param(c, inplace=True)
            return eff_mvshape(norm, k_vars)[0] - efficiency

    p_tune = optimize.brentq(func, limits[0], limits[1])
    return p_tune


# relative efficiency for M and MM estimator of multivariate location
# Table 2 from Kudraszow and Maronna JMA 2011
# k in [1, 2, 3, 4, 5, 10], eff in [0.8, 0.9, 0.95
# TODO: need to replace with more larger table and more digits.
# (4, 0.95): 5.76 -. 5.81 to better match R rrcov and numerical integration
tukeybiweight_mvmean_eff_km = {
        (1, 0.8): 3.14, (2, 0.8): 3.51, (3, 0.8): 3.82, (4, 0.8): 4.1,
        (5, 0.8): 4.34, (10, 0.8): 5.39,
        (1, 0.9): 3.88, (2, 0.9): 4.28, (3, 0.9): 4.62, (4, 0.9): 4.91,
        (5, 0.9): 5.18, (10, 0.9): 6.38,
        (1, 0.95): 4.68, (2, 0.95): 5.12, (3, 0.95): 5.48, (4, 0.95): 5.81,
        (5, 0.95): 6.1, (10, 0.95): 7.67,
        }

_table_biweight_mvmean_eff = np.array([
    [2.89717, 3.13691, 3.44369, 3.88266, 4.68506, 5.59682, 7.04139],
    [3.26396, 3.51006, 3.82643, 4.2821, 5.12299, 6.0869, 7.62344],
    [3.5721, 3.82354, 4.14794, 4.61754, 5.49025, 6.49697, 8.10889],
    [3.84155, 4.09757, 4.42889, 4.91044, 5.81032, 6.85346, 8.52956],
    [4.08323, 4.34327, 4.68065, 5.17267, 6.09627, 7.17117, 8.90335],
    [4.30385, 4.56746, 4.91023, 5.41157, 6.35622, 7.45933, 9.24141],
    [4.50783, 4.77466, 5.12228, 5.63199, 6.59558, 7.72408, 9.5512],
    [4.69828, 4.96802, 5.32006, 5.83738, 6.81817, 7.96977, 9.83797],
    [4.87744, 5.14986, 5.506, 6.03022, 7.02679, 8.19958, 10.10558],
    [5.04704, 5.32191, 5.68171, 6.21243, 7.22354, 8.41594, 10.35696],
    [5.20839, 5.48554, 5.84879, 6.38547, 7.41008, 8.62071, 10.59439],
    [5.36254, 5.64181, 6.00827, 6.55051, 7.58772, 8.81538, 10.81968],
    [5.51034, 5.7916, 6.16106, 6.7085, 7.75752, 9.00118, 11.03428],
    [5.65249, 5.9356, 6.3079, 6.86021, 7.92034, 9.17908, 11.23939],
    ])

_table_biweight_mvshape_eff = np.array([
    [3.57210, 3.82354, 4.14794, 4.61754, 5.49025, 6.49697, 8.10889],
    [3.84155, 4.09757, 4.42889, 4.91044, 5.81032, 6.85346, 8.52956],
    [4.08323, 4.34327, 4.68065, 5.17267, 6.09627, 7.17117, 8.90335],
    [4.30385, 4.56746, 4.91023, 5.41157, 6.35622, 7.45933, 9.24141],
    [4.50783, 4.77466, 5.12228, 5.63199, 6.59558, 7.72408, 9.55120],
    [4.69828, 4.96802, 5.32006, 5.83738, 6.81817, 7.96977, 9.83797],
    [4.87744, 5.14986, 5.50600, 6.03022, 7.02679, 8.19958, 10.10558],
    [5.04704, 5.32191, 5.68171, 6.21243, 7.22354, 8.41594, 10.35696],
    [5.20839, 5.48554, 5.84879, 6.38547, 7.41008, 8.62071, 10.59439],
    [5.36254, 5.64181, 6.00827, 6.55051, 7.58772, 8.81538, 10.81968],
    [5.51034, 5.79160, 6.16106, 6.70849, 7.75752, 9.00118, 11.03428],
    [5.65249, 5.93560, 6.30790, 6.86021, 7.92034, 9.17908, 11.23939],
    [5.78957, 6.07443, 6.44938, 7.00630, 8.07692, 9.34991, 11.43603],
    [5.92206, 6.20858, 6.58604, 7.14731, 8.22785, 9.51437, 11.62502],
    ])


def _convert_to_dict_mvmean_effs(eff_mean=True):
    effs_mvmean = [0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
    ks = list(range(1, 15))
    if eff_mean:
        table = _table_biweight_mvmean_eff
    else:
        table = _table_biweight_mvshape_eff
    tp = {}
    for i, k in enumerate(ks):
        for j, eff in enumerate(effs_mvmean):
            tp[(k, eff)] = table[i, j]

    return tp


tukeybiweight_mvmean_eff_d = _convert_to_dict_mvmean_effs(eff_mean=True)
tukeybiweight_mvshape_eff_d = _convert_to_dict_mvmean_effs(eff_mean=False)


def tukeybiweight_mvmean_eff(k, eff, eff_mean=True):
    """tuning parameter for biweight norm to achieve efficiency for mv-mean.

    Uses values from precomputed table if available, otherwise computes it
    numerically and adds it to the module global dict.

    """
    if eff_mean:
        table_dict = tukeybiweight_mvmean_eff_d
    else:
        table_dict = tukeybiweight_mvshape_eff_d

    try:
        tp = table_dict[(k, eff)]
    except KeyError:
        # compute and cache
        from .norms import TukeyBiweight  # avoid circular import
        norm = TukeyBiweight(c=1)
        tp = tuning_m_cov_eff(norm, k, efficiency=eff, eff_mean=eff_mean)
        table_dict[(k, eff)] = tp
    return tp
