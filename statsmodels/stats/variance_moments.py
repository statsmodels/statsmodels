"""
Functions for variance and higher moments

Created on May 9, 2022 11:36:55 AM

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from scipy import stats

from statsmodels.stats.base import HolderTuple
from statsmodels.stats.robust_compare import trim_mean
from statsmodels.stats._inference_tools import _mover_confint


def kurtosis1(y, method=None, center="mean", return_results=False):
    """estimators for excess kurtosis

    b2, G2, b2 from Joanes and Gill 1998
    trimmed mean in 4th moment computation is based on Bonett 2006

    """
    options = {"method": method,
               "center": center,
               }
    y = np.asarray(y)
    n = nobs = y.shape[0]
    mean = y.mean(0)
    if center == "mean":
        center = mean
    elif center in ["tmean", "trimmed"]:
        proportiontocut = 0.5 / np.sqrt(n - 4)
        center = trim_mean(y, proportiontocut, axis=0)
    else:
        raise ValueError

    var_ = ((y - mean)**2).sum() / nobs
    m4 = ((y - center)**4).sum() / nobs
    k = m4 / var_**2  # kurtosis, not excess-kurtosis
    if method == "g2":
        kurt = k - 3
    elif method == "G2":
        kurt = (n - 1) / (n - 2) / (n - 3) * (k - 3)
    elif method == "b2":
        kurt = ((n - 1) / n)**2 * k - 3
    else:
        raise ValueError(f"method {method} not recognized")

    if not return_results:
        return kurt
    else:
        res = HolderTuple(
            tuple_=("kurtosis",),
            kurtosis=kurt,
            mean=mean,
            center=center,
            options=options
            )
        return res


def kurtosis(y, method=None, center="mean", return_results=False):
    """Estimators for excess kurtosis

    g2, G2, b2 from Joanes and Gill 1998

    """
    options = {"method": method,
               "center": center,
               }

    if not isinstance(y, tuple):
        y = np.asarray(y)
        n = y.shape[0]  # nobs
        mean = y.mean(0)
        if center == "mean":
            center = mean
        elif center in ["tmean", "trimmed"]:
            proportiontocut = 0.5 / np.sqrt(n - 4)
            center = trim_mean(y, proportiontocut, axis=0)
        else:
            raise ValueError

        var_ = ((y - mean)**2).sum(0) / n
        m4 = ((y - center)**4).sum(0) / n

    else:
        y_tuple = [np.asarray(yi) for yi in y]

        y_dm = np.concatenate([yi - yi.mean(0) for yi in y_tuple], axis=0)
        if center == "mean":
            y_dc = y_dm = np.concatenate([yi - yi.mean(0) for yi in y_tuple],
                                         axis=0)
        elif center in ["tmean", "trimmed"]:
            y_dtm = []
            for yi in y_tuple:
                proportiontocut = 0.5 / np.sqrt(len(yi) - 4)
                center = trim_mean(yi, proportiontocut, axis=0)
                y_dtm.append(yi - center)
            y_dc = np.concatenate(y_dtm)
        else:
            raise ValueError

        n = y_dm.shape[0]  # nobs total
        var_ = (y_dm**2).sum(0) / n
        m4 = (y_dc**4).sum(0) / n

    k = m4 / var_**2  # kurtosis, not excess-kurtosis
    if method == "g2":
        kurt = k - 3
    elif method == "G2":
        kurt = (n - 1) / (n - 2) / (n - 3) * (k - 3)
    elif method == "b2":
        kurt = ((n - 1) / n)**2 * k - 3
    else:
        raise ValueError(f"method {method} not recognized")

    if not return_results:
        return kurt
    else:
        res = HolderTuple(
            tuple_=("kurtosis",),
            kurtosis=kurt,
            mean=mean,
            center=center,
            options=options
            )
        return res


def _var_variance(y, center="mean", var_=None):
    """variance of variance using fourth moment

    This computes
    sum((y_i - m)^2 - s^2)^2 / n
    which is equivalent to
    (g2 + 3 - 1) * var_y**2

    based on
    Yuan, Bentler and Zhang 2005 p. 248
    """

    # if not isinstance(y, tuple):
    y = np.asarray(y)
    n = y.shape[0]  # nobs
    mean = y.mean(0)
    if center == "mean":
        center = mean
    elif center in ["tmean", "trimmed"]:
        proportiontocut = 0.5 / np.sqrt(n - 4)
        center = trim_mean(y, proportiontocut, axis=0)
    else:
        raise ValueError

    if var_ is None:
        var_ = ((y - mean)**2).sum(0) / n
    # m4 = ((y - center)**4).sum(0) / n
    mv = (((y - center)**2 - var_)**2).sum(0) / n
    return mv


def test_variance(y, value, method=None, alpha=0.05, ddof=0,
                  center_kurt="mean", alpha_bonett=0.05):
    """

    for ddof:
    Bonett, Douglas G. 2005. “Robust Confidence Interval for a Residual
    Standard Deviation.” Journal of Applied Statistics 32 (10): 1089–94.
    https://doi.org/10.1080/02664760500165339.


    """
    # some copy-paste code duplication with confint_variance
    options = {"method": method,
               "center_kurt": center_kurt,
               }
    y = np.asarray(y)
    n = y.shape[0]  # nobs
    # todo maybe reuse from kurtosis function
    var_y = np.var(y, axis=0, ddof=1 + ddof)

    if method == "bonett":
        crit = stats.norm.isf(alpha_bonett / 2)
        # Bonett 2005, 2006
        # use kurtosis, not excess kurtosis
        kurtne = kurtosis(y, method="g2", center="tmean") + 3
        if ddof == 0:
            c = n / (n - crit)  # small sample adjustment to location
            std_logvar = c * np.sqrt((kurtne - (n - 3) / n) / (n - 1))
        else:
            c = n / (n - crit * (n - 2) / (n - 1 - ddof))
            std_logvar = c * np.sqrt((kurtne - (n - 3) / n) / (n - ddof))
            # Note: last term is not (n - 1 - ddof) in Benett 2005
        effect = np.log(c * var_y) - np.log(value)
        statistic = effect / std_logvar
        distribution = "normal"
    elif method in ["normal", "nonrobust"]:
        df = n - 1 - ddof  # Bonett 2005
        ss = df * var_y
        statistic = ss / value
        # TODO: check this maybe wrong
        pvalue = stats.chi2.sf(statistic, df)
        distribution = "chi2"
    elif method in ["wald", "score"]:
        if method == "wald":
            var_y_ = var_y
        else:
            var_y_ = value
        # based on Yuan, Bentler and Zhang 2005 p. 248
        std = np.sqrt(_var_variance(y, center=center_kurt, var_=var_y_))
        effect = var_y - value
        statistic = effect / std
        distribution = "normal"
    else:
        raise ValueError(f'method "{method}" not recognized')

    res = HolderTuple(
        statistic=statistic,
        pvalue=pvalue,
        distribution=distribution,
        # kurtosis=kurt,
        # mean=mean,
        # center=center,
        options=options
        )
    return res


def confint_variance(y, method=None, alpha=0.05, ddof=0, center_kurt="mean"):
    """

    for ddof:
    Bonett, Douglas G. 2005. “Robust Confidence Interval for a Residual
    Standard Deviation.” Journal of Applied Statistics 32 (10): 1089–94.
    https://doi.org/10.1080/02664760500165339.


    """
    y = np.asarray(y)
    n = y.shape[0]  # nobs
    # todo maybe reuse from kurtosis function
    var_y = np.var(y, axis=0, ddof=1 + ddof)
    crit = stats.norm.isf(alpha / 2)
    if method == "bonett":
        # Bonett 2005, 2006
        # use kurtosis, not excess kurtosis
        kurtne = kurtosis(y, method="g2", center="tmean") + 3
        if ddof == 0:
            c = n / (n - crit)  # small sample adjustment to location
            std_logvar = c * np.sqrt((kurtne - (n - 3) / n) / (n - 1))
        else:
            c = n / (n - crit * (n - 2) / (n - 1 - ddof))
            std_logvar = c * np.sqrt((kurtne - (n - 3) / n) / (n - ddof))
            # Note: last term is not (n - 1 - ddof) in Benett 2005
        center_logci = np.log(c * var_y)
        half_logci = crit * std_logvar
        low = np.exp(center_logci - half_logci)
        upp = np.exp(center_logci + half_logci)
    elif method in ["normal", "nonrobust"]:
        df = n - 1 - ddof  # Bonett 2005
        ss = df * var_y
        low = ss / stats.chi2.isf(alpha / 2, df)
        upp = ss / stats.chi2.ppf(alpha / 2, df)
    elif method == "wald":
        # based on Yuan, Bentler and Zhang 2005 p. 248
        std = np.sqrt(_var_variance(y, center=center_kurt, var_=var_y))
        center_ci = var_y
        half_ci = crit * std
        low = center_ci - half_ci
        upp = center_ci + half_ci
    else:
        raise ValueError(f'method "{method}" not recognized')

    return (low, upp)


def power_variance(var_null, var_alternative, nobs, method=None,
                   alpha=0.05, alternative="two-sided"):
    s0, s1 = var_null, var_alternative
    n = nobs

    if method == "normal":
        pow_ = 0
        if alternative in ['two-sided', '2s', 'larger']:
            upp = s0 / s1 * stats.chi2.isf(alpha / 2, n - 1)  # upp
            pow_ += stats.chi2.sf(upp, n - 1)
        if alternative in ['two-sided', '2s', 'smaller']:
            low = s0 / s1 * stats.chi2.ppf(alpha / 2, n - 1)  # low
            pow_ += stats.chi2.cdf(low, n-1)
    else:
        raise ValueError(f'method "{method}" not recognized')

    return pow_

    # pow_ = 0
    # if alternative in ['two-sided', '2s', 'larger']:
        # crit = stats.norm.isf(alpha_)
        # pow_ = stats.norm.sf(crit - d*np.sqrt(nobs)/sigma)
    # if alternative in ['two-sided', '2s', 'smaller']:
        # crit = stats.norm.ppf(alpha_)
        # pow_ += stats.norm.cdf(crit - d*np.sqrt(nobs)/sigma)


def test_variances_2indep(y1, y2, method=None, compare="ratio", value=None,
                          alternative="two-sided", pooled_kurtosis=True):
    pass


def confint_variances_2indep(y1, y2, method=None, compare="ratio", alpha=0.05,
                             pooled_kurtosis=True):
    """
    Confidence interval comparing variances from two independent samples


    References
    ----------
    Bonett 2006
    Minitab documentation
    """
    var_y1 = np.var(y1, axis=0, ddof=1)
    var_y2 = np.var(y2, axis=0, ddof=1)
    n1 = len(y1)
    n2 = len(y2)
    crit = stats.norm.isf(alpha / 2)
    if compare == "ratio":
        if method == "bonett":
            # Bonett 2005, 2006
            # use kurtosis, not excess kurtosis
            if pooled_kurtosis:
                k1 = k2 = kurtosis((y1, y2), method="g2", center="tmean") + 3
            else:
                k1 = kurtosis(y1, method="g2", center="tmean") + 3
                k2 = kurtosis(y1, method="g2", center="tmean") + 3

            c = n1 / (n1 - crit) / n2 * (n2 - crit)
            std_logvar = np.sqrt(
                (k1 - (n1 - 3) / n1) / (n1 - 1) +
                (k2 - (n2 - 3) / n2) / (n2 - 1)
                )
            center_logci = np.log(c * var_y1 / var_y2)
            half_logci = crit * std_logvar
            low = np.exp(center_logci - half_logci)
            upp = np.exp(center_logci + half_logci)

        elif method in ["normal", "nonrobust"]:
            # 2 sample F-test
            df1 = n1 - 1
            df2 = n2 - 1
            var_ratio = var_y1 / var_y2
            # TODO: check sequence df1, df2 is correct
            low = var_ratio / stats.f.isf(alpha / 2, df1, df2)
            upp = var_ratio / stats.f.ppf(alpha / 2, df1, df2)

        elif method == "mover":
            method_p = "bonett"  # method_mover
            ci1 = confint_variance(y1, method=method_p, alpha=alpha)
            ci2 = confint_variance(y2, method=method_p, alpha=alpha)

            ci = _mover_confint(var_y1, var_y2, ci1, ci2, contrast="ratio")
            low, upp = ci

        else:
            raise ValueError(f'method "{method}" not recognized')

    elif compare == "diff":
        if method == "herbert":
            # Bonett 2005, 2006, Herbert et al 2011
            # use kurtosis, not excess kurtosis
            if pooled_kurtosis:
                k1 = k2 = kurtosis((y1, y2), method="g2", center="tmean") + 3
            else:
                k1 = kurtosis(y1, method="g2", center="tmean") + 3
                k2 = kurtosis(y1, method="g2", center="tmean") + 3

            std_diff = np.sqrt(
                var_y1**2 * (k1 - (n1 - 3) / (n1 - 1)) / n1 +
                var_y2**2 * (k2 - (n2 - 3) / (n2 - 1)) / n2
                )
            center_ci = var_y1 - var_y2
            half_ci = crit * std_diff
            low = center_ci - half_ci
            upp = center_ci + half_ci

        elif method == "mover":
            method_p = "bonett"  # method_mover
            ci1 = confint_variance(y1, method=method_p, alpha=alpha)
            ci2 = confint_variance(y2, method=method_p, alpha=alpha)

            ci = _mover_confint(var_y1, var_y2, ci1, ci2, contrast="diff")
            low, upp = ci

        else:
            raise ValueError(f'method "{method}" not recognized')

    else:
        raise ValueError(f'compare "{compare}" is not recognized')

    return (low, upp)


def tost_variances_2indep(y1, y2, method=None, compare="ratio", alpha=0.05,
                          pooled_kurtosis=True):
    pass


def power_variances_2indep(ratio_null, ratio_alternative, nobs1, nobs2,
                           method=None, alpha=0.05, alternative="two-sided"):
    r0, r1 = ratio_null, ratio_alternative
    n1, n2 = nobs1, nobs2
    if method == "normal":
        pow_ = 0
        if alternative in ['two-sided', '2s', 'larger']:
            upp = r0 / r1 * stats.f.isf(alpha / 2, n1-1, n2-1)  # upp
            pow_ += stats.f.sf(upp, n1-1, n2-1)
        if alternative in ['two-sided', '2s', 'smaller']:
            low = r0 / r1 * stats.f.ppf(alpha / 2, n1-1, n2-1)  # low
            pow_ += stats.f.cdf(low, n1-1, n2-1)
    else:
        raise ValueError(f'method "{method}" not recognized')

    return pow_
