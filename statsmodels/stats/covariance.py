"""

Author: Josef Perktold
License: BSD-3
"""

from typing import NamedTuple

import numpy as np
from scipy import integrate, stats

from statsmodels.tools.validation import bool_like

pi2 = np.pi**2
pi2i = 1. / pi2


def _term_integrate(rho):
    # needs other terms for spearman rho var calculation
    # TODO: streamline calculation and save to linear interpolation, maybe
    sin, cos = np.sin, np.cos

    def f1(t, x):
        return np.arcsin(sin(x) / (1 + 2 * cos(2 * x)))

    def f2(t, x):
        return np.arcsin(sin(2 * x) / np.sqrt(1 + 2 * cos(2 * x)))

    def f3(t, x):
        return np.arcsin(sin(2 * x) / (2 * np.sqrt(cos(2 * x))))

    def f4(t, x):
        return np.arcsin((3 * sin(x) - sin(3 * x)) / (4 * cos(2 * x)))

    fact = pi2i * (f1(None, rho) +
                   2 * pi2i * f2(None, rho) +
                   f3(None, rho) +
                   0.5 * f4(None, rho))

    return fact


class TransformCorrNormalResult(NamedTuple):
    """
    Result of :func:`transform_corr_normal` when a NamedTuple is returned.

    Parameters
    ----------
    corr : ndarray
        Correlation matrix, consistent with the correlation for a
        multivariate normal distribution.
    var : ndarray or None
        Asymptotic variance of the normalized correlation. ``None`` when
        the variance was not computed, i.e. when ``return_var=False``.
    """

    corr: np.ndarray
    var: np.ndarray


def transform_corr_normal(
    corr, method, return_var=False, possdef=True, *, result_object: bool | None = None
):
    """
    Transform correlation matrix to be consistent at normal distribution

    Parameters
    ----------
    corr : array_like
        correlation matrix, either Pearson, Gaussian-rank, Spearman, Kendall
        or quadrant correlation matrix
    method : str
        type of covariance matrix
        supported types are 'pearson', 'gauss_rank', 'kendal', 'spearman' and
        'quadrant'
    return_var : bool, optional
        If true, then the asymptotic variance of the normalized correlation
        is also returned. The variance of the spearman correlation requires
        numerical integration which is calculated with scipy's odeint.
    possdef : bool, optional
        Not implemented yet. Check whether resulting correlation matrix for
        positive semidefinite and return a positive semidefinite
        approximation if not.
    result_object : bool, optional
        Flag controlling whether a ``TransformCorrNormalResult`` NamedTuple
        is returned. When ``return_var=True`` a
        ``TransformCorrNormalResult`` is always returned; it holds the same
        two elements as the legacy tuple, so it unpacks and indexes
        identically. When ``return_var=False`` a bare correlation matrix is
        returned unless ``result_object=True``, which yields a
        ``TransformCorrNormalResult`` with ``var`` set to ``None``.

    Returns
    -------
    TransformCorrNormalResult or ndarray
        When ``return_var=True`` (or ``result_object=True``), a NamedTuple
        with fields:

        corr : ndarray
            correlation matrix, consistent with correlation for a
            multivariate normal distribution
        var : ndarray or None
            asymptotic variance of the correlation. ``None`` when
            ``return_var`` is False, since it is not computed in that case.

        ``TransformCorrNormalResult`` has the same length and contents as
        the plain ``(corr_n, var)`` tuple it replaces, so it unpacks and
        indexes identically. See
        :class:`~statsmodels.stats.covariance.TransformCorrNormalResult`.

        When ``return_var=False`` and ``result_object`` is not True, a bare
        correlation matrix is returned instead.

    Notes
    -----
    Pearson and Gaussian-rank correlation are consistent at the normal
    distribution and will be returned without changes.

    The other correlation matrices are not guaranteed to be positive
    semidefinite in small sample after conversion, even if the underlying
    untransformed correlation matrix is positive (semi)definite. Croux and
    Dehon mention that nobs / k_vars should be larger than 3 for kendall and
    larger than 2 for spearman.

    References
    ----------
    .. [1] Boudt, Kris, Jonathan Cornelissen, and Christophe Croux. “The
       Gaussian Rank Correlation Estimator: Robustness Properties.”
       Statistics and Computing 22, no. 2 (April 5, 2011): 471-83.
       https://doi.org/10.1007/s11222-011-9237-0.
    .. [2] Croux, Christophe, and Catherine Dehon. “Influence Functions of the
       Spearman and Kendall Correlation Measures.”
       Statistical Methods & Applications 19, no. 4 (May 12, 2010): 497-515.
       https://doi.org/10.1007/s10260-010-0142-z.

    """
    result_object = bool_like(result_object, "result_object", optional=True)
    method = method.lower()
    rho = np.asarray(corr)

    var = None  # initialize

    if method in ["pearson", "gauss_rank"]:
        corr_n = corr
        if return_var:
            var = (1 - rho**2)**2

    elif method.startswith("kendal"):
        corr_n = np.sin(np.pi / 2 * corr)
        if return_var:
            var = (1 - rho**2) * np.pi**2 * (
                  1./9 - 4 / np.pi**2 * np.arcsin(rho / 2)**2)

    elif method == "quadrant":
        corr_n = np.sin(np.pi / 2 * corr)
        if return_var:
            var = (1 - rho**2) * (np.pi**2 / 4 - np.arcsin(rho)**2)

    elif method.startswith("spearman"):
        corr_n = 2 * np.sin(np.pi / 6 * corr)
        # not clear which rho is in formula, should be normalized rho,
        # but original corr coefficient seems to match results in articles
        # rho = corr_n
        if return_var:
            # odeint only works if grid of rho is large, i.e., many points
            # e.g., rho = np.linspace(0, 1, 101)
            rho = np.atleast_1d(rho)
            idx = np.argsort(rho)
            rhos = rho[idx]
            rhos = np.concatenate(([0], rhos))
            t = np.arcsin(rhos / 2)
            # drop np namespace here
            sin, cos = np.sin, np.cos
            var = (1 - rho**2 / 4) * pi2 / 9  # leading factor

            def f1(t, x):
                return np.arcsin(sin(x) / (1 + 2 * cos(2 * x)))

            def f2(t, x):
                return np.arcsin(sin(2 * x) / np.sqrt(1 + 2 * cos(2 * x)))

            def f3(t, x):
                return np.arcsin(sin(2 * x) / (2 * np.sqrt(cos(2 * x))))

            def f4(t, x):
                return np.arcsin((3 * sin(x) - sin(3 * x)) / (4 * cos(2 * x)))

            # todo check dimension, odeint return column (n, 1) array
            hmax = 1e-1
            rf1 = integrate.odeint(f1 , 0, t=t, hmax=hmax).squeeze()
            rf2 = integrate.odeint(f2 , 0, t=t, hmax=hmax).squeeze()
            rf3 = integrate.odeint(f3 , 0, t=t, hmax=hmax).squeeze()
            rf4 = integrate.odeint(f4 , 0, t=t, hmax=hmax).squeeze()
            fact = 1 + 144 * (-9 / 4. * pi2i * np.arcsin(rhos / 2)**2 +
                              pi2i * rf1 +
                              2 * pi2i * rf2 + pi2i * rf3 +
                              0.5 * pi2i * rf4)
            # fact = 1 - 9 / 4 * pi2i * np.arcsin(rhos / 2)**2
            fact2 = np.zeros_like(var) * np.nan
            fact2[idx] = fact[1:]
            var *= fact2
    else:
        raise ValueError("method not recognized")

    # TransformCorrNormalResult has exactly the same length and contents as
    # the legacy (corr_n, var) tuple, so it unpacks and indexes identically
    # and is always used when return_var is True.  When return_var is False a
    # bare correlation matrix is returned, as before; pass
    # result_object=True to always get a TransformCorrNormalResult, with
    # var left as None since it is only computed when requested.
    if result_object or return_var:
        return TransformCorrNormalResult(corr_n, var)
    return corr_n


def corr_rank(data):
    """
    Spearman rank correlation

    Simplified version of scipy.stats.spearmanr.

    Parameters
    ----------
    data : array_like
        2-D data with observations in rows and variables in columns.

    Returns
    -------
    corr : ndarray
        correlation matrix
    """
    x = np.asarray(data)
    axisout = 0
    ar = np.apply_along_axis(stats.rankdata, axisout, x)
    corr = np.corrcoef(ar, rowvar=False)
    return corr


def corr_normal_scores(data):
    """
    Gaussian rank (normal scores) correlation

    Status: unverified, subject to change

    Parameters
    ----------
    data : array_like
        2-D data with observations in rows and variables in columns

    Returns
    -------
    corr : ndarray
        correlation matrix

    References
    ----------
    .. [1] Boudt, Kris, Jonathan Cornelissen, and Christophe Croux. “The
       Gaussian Rank Correlation Estimator: Robustness Properties.”
       Statistics and Computing 22, no. 2 (April 5, 2011): 471-83.
       https://doi.org/10.1007/s11222-011-9237-0.
    """
    # TODO: a full version should be same as scipy spearmanr
    # I think that's not true the croux et al articles mention different
    # results
    # needs verification for the p-value calculation
    x = np.asarray(data)
    nobs = x.shape[0]
    axisout = 0
    ar = np.apply_along_axis(stats.rankdata, axisout, x)
    ar = stats.norm.ppf(ar / (nobs + 1))
    corr = np.corrcoef(ar, rowvar=axisout)
    return corr


def corr_quadrant(data, transform=np.sign, normalize=False):
    """
    Quadrant correlation

    Status: unverified, subject to change

    Parameters
    ----------
    data : array_like
        2-D data with observations in rows and variables in columns
    transform : callable, optional
        Function used to transform the demeaned data before computing the
        correlation. Default is ``np.sign``.
    normalize : bool, optional
        If True, normalize the resulting matrix by the standard deviations
        so that it is a proper correlation matrix. Default is False.

    Returns
    -------
    corr : ndarray
        correlation matrix

    References
    ----------
    .. [1] Croux, Christophe, and Catherine Dehon. “Influence Functions of the
       Spearman and Kendall Correlation Measures.”
       Statistical Methods & Applications 19, no. 4 (May 12, 2010): 497-515.
       https://doi.org/10.1007/s10260-010-0142-z.
    """

    # try also with tanh transform, a starting corr for DetXXX
    # tanh produces a cov not a corr
    x = np.asarray(data)
    nobs = x.shape[0]
    med = np.median(x, 0)
    x_dm = transform(x - med)
    corr = x_dm.T.dot(x_dm) / nobs
    if normalize:
        std = np.sqrt(np.diag(corr))
        corr /= std
        corr /= std[:, None]
    return corr
