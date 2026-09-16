"""
Multiple Testing and P-Value Correction


Author: Josef Perktold
License: BSD-3

"""
from statsmodels.compat.scipy import SP_LT_112

from typing import NamedTuple

import numpy as np

from statsmodels.stats._knockoff import RegressionFDR

__all__ = [
    "LocalFDRCorrectionResult",
    "NullDistribution",
    "RegressionFDR",
    "fdrcorrection",
    "fdrcorrection_twostage",
    "local_fdr",
    "local_fdr_correction",
    "multipletests",
]

from statsmodels.tools.validation import (
    array_like,
    bool_like,
    float_like,
    string_like,
)

# ==============================================
#
# Part 1: Multiple Tests and P-Value Correction
#
# ==============================================


def _ecdf(x):
    """
    No frills empirical cdf used in fdrcorrection

    Parameters
    ----------
    x : ndarray
        The data to use
    """
    nobs = len(x)
    return np.arange(1, nobs + 1) / float(nobs)


multitest_methods_names = {
    "b": "Bonferroni",
    "s": "Sidak",
    "h": "Holm",
    "hs": "Holm-Sidak",
    "sh": "Simes-Hochberg",
    "ho": "Hommel",
    "fdr_bh": "FDR Benjamini-Hochberg",
    "fdr_by": "FDR Benjamini-Yekutieli",
    "fdr_tsbh": "FDR 2-stage Benjamini-Hochberg",
    "fdr_tsbky": "FDR 2-stage Benjamini-Krieger-Yekutieli",
    "fdr_gbs": "FDR adaptive Gavrilov-Benjamini-Sarkar",
    "lfdr": "lfdr support line procedure",
}

_alias_list = [
    ["b", "bonf", "bonferroni"],
    ["s", "sidak"],
    ["h", "holm"],
    ["hs", "holm-sidak"],
    ["sh", "simes-hochberg"],
    ["ho", "hommel"],
    ["fdr_bh", "fdr_i", "fdr_p", "fdri", "fdrp"],
    ["fdr_by", "fdr_n", "fdr_c", "fdrn", "fdrcorr"],
    ["fdr_tsbh", "fdr_2sbh"],
    ["fdr_tsbky", "fdr_2sbky", "fdr_twostage"],
    ["fdr_gbs"],
    ["lfdr", "lfdr_sl"],
]


if SP_LT_112:
    # lfdr does not work with scipy < 1.12 because it uses isotonic_regression
    del multitest_methods_names["lfdr"]
    _alias_list.remove(["lfdr", "lfdr_sl"])


multitest_alias = {}
for _alias_sub_list in _alias_list:
    _formal_name = _alias_sub_list[0]
    for _alias in _alias_sub_list:
        multitest_alias[_alias] = _formal_name


def multipletests(
    pvals, alpha=0.05, method="hs", maxiter=1, is_sorted=False, returnsorted=False
):
    """
    Test results and p-value correction for multiple tests

    Parameters
    ----------
    pvals : array_like, 1-d
        uncorrected p-values.   Must be 1-dimensional.
    alpha : float, optional
        FWER, family-wise error rate, e.g., 0.1
    method : str, optional
        Method used for testing and adjustment of pvalues. Can be either the
        full name or initial letters. Available methods are:

        - `bonferroni` : one-step correction
        - `sidak` : one-step correction
        - `holm-sidak` : step down method using Sidak adjustments
        - `holm` : step-down method using Bonferroni adjustments
        - `simes-hochberg` : step-up method  (independent)
        - `hommel` : closed method based on Simes tests (non-negative)
        - `fdr_bh` : Benjamini/Hochberg  (non-negative)
        - `fdr_by` : Benjamini/Yekutieli (negative)
        - `fdr_tsbh` : two stage fdr correction (non-negative)
        - `fdr_tsbky` : two stage fdr correction (non-negative)
        - `fdr_gbs` : adaptive step-down (Gavrilov, Benjamini, Sarkar)
        - `lfdr` : support line

    maxiter : int or bool, optional
        Maximum number of iterations for two-stage fdr, `fdr_tsbh` and
        `fdr_tsbky`. It is ignored by all other methods.
        maxiter=1 (default) corresponds to the two stage method.
        maxiter=-1 corresponds to full iterations which is maxiter=len(pvals).
        maxiter=0 uses only a single stage fdr correction using a 'bh' or 'bky'
        prior fraction of assumed true hypotheses.
    is_sorted : bool, optional
        If False (default), the p_values will be sorted, but the corrected
        pvalues are in the original order. If True, then it assumed that the
        pvalues are already sorted in ascending order.
    returnsorted : bool, optional
         not tested, return sorted p-values instead of original sequence

    Returns
    -------
    reject : ndarray, bool
        true for hypothesis that can be rejected for given alpha
    pvals_corrected : ndarray
        p-values corrected for multiple tests
    alphacSidak : float
        corrected alpha for Sidak method
    alphacBonf : float
        corrected alpha for Bonferroni method

    Notes
    -----
    There may be API changes for this function in the future.

    Except for 'fdr_twostage', the p-value correction is independent of the
    alpha specified as argument. In these cases the corrected p-values
    can also be compared with a different alpha. In the case of 'fdr_twostage',
    the corrected p-values are specific to the given alpha, see
    ``fdrcorrection_twostage``.

    The 'fdr_gbs' procedure is not verified against another package, p-values
    are derived from scratch and are not derived in the reference. In Monte
    Carlo experiments the method worked correctly and maintained the false
    discovery rate.

    All procedures that are included, control FWER or FDR in the independent
    case, and most are robust in the positively correlated case.

    `fdr_gbs`: high power, fdr control for independent case and only small
    violation in positively correlated case

    **Timing**:

    Most of the time with large arrays is spent in `argsort`. When
    we want to calculate the p-value for several methods, then it is more
    efficient to presort the pvalues, and put the results back into the
    original order outside of the function.

    Method='hommel' is very slow for large arrays, since it requires the
    evaluation of n partitions, where n is the number of p-values.
    """
    import gc

    pvals = np.asarray(pvals)
    alphaf = alpha  # Notation ?

    if not is_sorted:
        sortind = np.argsort(pvals)
        pvals = np.take(pvals, sortind)

    ntests = len(pvals)
    if ntests > 0:
        alphacSidak = 1 - np.power((1.0 - alphaf), 1.0 / ntests)
        alphacBonf = alphaf / float(ntests)
    else:
        # Nothing to correct. Skip the division by zero above and let the
        # per-method branches operate on the empty array, which produces empty
        # results. The corrected alphas are undefined without any tests.
        alphacSidak = np.nan
        alphacBonf = np.nan
    if method.lower() in ["b", "bonf", "bonferroni"]:
        reject = pvals <= alphacBonf
        pvals_corrected = pvals * float(ntests)

    elif method.lower() in ["s", "sidak"]:
        reject = pvals <= alphacSidak
        pvals_corrected = -np.expm1(ntests * np.log1p(-pvals))

    elif method.lower() in ["hs", "holm-sidak"]:
        alphacSidak_all = 1 - np.power((1.0 - alphaf), 1.0 / np.arange(ntests, 0, -1))
        notreject = pvals > alphacSidak_all
        del alphacSidak_all

        nr_index = np.nonzero(notreject)[0]
        if nr_index.size == 0:
            # nonreject is empty, all rejected
            notrejectmin = len(pvals)
        else:
            notrejectmin = np.min(nr_index)
        notreject[notrejectmin:] = True
        reject = ~notreject
        del notreject

        # It's eqivalent to 1 - np.power((1. - pvals),
        #                           np.arange(ntests, 0, -1))
        # but prevents the issue of the floating point precision
        pvals_corrected_raw = -np.expm1(np.arange(ntests, 0, -1) * np.log1p(-pvals))
        pvals_corrected = np.maximum.accumulate(pvals_corrected_raw)
        del pvals_corrected_raw

    elif method.lower() in ["h", "holm"]:
        notreject = pvals > alphaf / np.arange(ntests, 0, -1)
        nr_index = np.nonzero(notreject)[0]
        if nr_index.size == 0:
            # nonreject is empty, all rejected
            notrejectmin = len(pvals)
        else:
            notrejectmin = np.min(nr_index)
        notreject[notrejectmin:] = True
        reject = ~notreject
        pvals_corrected_raw = pvals * np.arange(ntests, 0, -1)
        pvals_corrected = np.maximum.accumulate(pvals_corrected_raw)
        del pvals_corrected_raw
        gc.collect()

    elif method.lower() in ["sh", "simes-hochberg"]:
        alphash = alphaf / np.arange(ntests, 0, -1)
        reject = pvals <= alphash
        rejind = np.nonzero(reject)
        if rejind[0].size > 0:
            rejectmax = np.max(np.nonzero(reject))
            reject[:rejectmax] = True
        pvals_corrected_raw = np.arange(ntests, 0, -1) * pvals
        pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
        del pvals_corrected_raw

    elif method.lower() in ["ho", "hommel"]:
        # we need a copy because we overwrite it in a loop
        a = pvals.copy()
        for m in range(ntests, 1, -1):
            cim = np.min(m * pvals[-m:] / np.arange(1, m + 1.0))
            a[-m:] = np.maximum(a[-m:], cim)
            a[:-m] = np.maximum(a[:-m], np.minimum(m * pvals[:-m], cim))
        pvals_corrected = a
        reject = a <= alphaf

    elif method.lower() in ["fdr_bh", "fdr_i", "fdr_p", "fdri", "fdrp"]:
        # delegate, call with sorted pvals
        reject, pvals_corrected = fdrcorrection(
            pvals, alpha=alpha, method="indep", is_sorted=True
        )
    elif method.lower() in ["fdr_by", "fdr_n", "fdr_c", "fdrn", "fdrcorr"]:
        # delegate, call with sorted pvals
        reject, pvals_corrected = fdrcorrection(
            pvals, alpha=alpha, method="n", is_sorted=True
        )
    elif method.lower() in ["fdr_tsbky", "fdr_2sbky", "fdr_twostage"]:
        # delegate, call with sorted pvals
        reject, pvals_corrected = fdrcorrection_twostage(
            pvals, alpha=alpha, method="bky", maxiter=maxiter, is_sorted=True
        )[:2]
    elif method.lower() in ["fdr_tsbh", "fdr_2sbh"]:
        # delegate, call with sorted pvals
        reject, pvals_corrected = fdrcorrection_twostage(
            pvals, alpha=alpha, method="bh", maxiter=maxiter, is_sorted=True
        )[:2]

    elif method.lower() == "fdr_gbs":
        # adaptive stepdown in Gavrilov, Benjamini, Sarkar, Annals of Statistics 2009
        #        notreject = pvals > alphaf / np.arange(ntests, 0, -1) # alphacSidak
        #        notrejectmin = np.min(np.nonzero(notreject))
        #        notreject[notrejectmin:] = True
        #        reject = ~notreject

        ii = np.arange(1, ntests + 1)
        q = (ntests + 1.0 - ii) / ii * pvals / (1.0 - pvals)
        pvals_corrected_raw = np.maximum.accumulate(q)  # up requirementd

        pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
        del pvals_corrected_raw
        reject = pvals_corrected <= alpha

    elif method.lower() in ["lfdr", "lfdr_sl"]:
        pvals_corrected = local_fdr_correction(pvals, is_sorted=True).lfdr
        reject = pvals_corrected <= alpha

    else:
        raise ValueError("method not recognized")

    if pvals_corrected is not None:  # not necessary anymore
        pvals_corrected[pvals_corrected > 1] = 1
    if is_sorted or returnsorted:
        return reject, pvals_corrected, alphacSidak, alphacBonf
    else:
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[sortind] = pvals_corrected
        del pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[sortind] = reject
        return reject_, pvals_corrected_, alphacSidak, alphacBonf


def fdrcorrection(pvals, alpha=0.05, method="indep", is_sorted=False):
    """
    pvalue correction for false discovery rate

    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests.

    Parameters
    ----------
    pvals : array_like, 1d
        Set of p-values of the individual tests.
    alpha : float, optional
        Family-wise error rate. Defaults to ``0.05``.
    method : {'i', 'indep', 'p', 'poscorr', 'n', 'negcorr'}, optional
        Which method to use for FDR correction.
        ``{'i', 'indep', 'p', 'poscorr'}`` all refer to ``fdr_bh``
        (Benjamini/Hochberg for independent or positively
        correlated tests). ``{'n', 'negcorr'}`` both refer to ``fdr_by``
        (Benjamini/Yekutieli for general or negatively correlated tests).
        Defaults to ``'indep'``.
    is_sorted : bool, optional
        If False (default), the p_values will be sorted, but the corrected
        pvalues are in the original order. If True, then it assumed that the
        pvalues are already sorted in ascending order.

    Returns
    -------
    rejected : ndarray, bool
        True if a hypothesis is rejected, False if not
    pvals_corrected : ndarray
        pvalues adjusted for multiple hypothesis testing to limit FDR

    Notes
    -----
    If there is prior information on the fraction of true hypothesis, then alpha
    should be set to ``alpha * m/m_0`` where m is the number of tests,
    given by the p-values, and m_0 is an estimate of the true hypothesis.
    (see Benjamini, Krieger and Yekutieli)

    The two-step method of Benjamini, Krieger and Yekutieli that estimates the number
    of false hypotheses will be available (soon).

    Method names can be abbreviated to first letter, 'i' or 'p' for fdr_bh and 'n' for
    fdr_by.

    **Benjamini-Hochberg procedure** (see Benjamini and Hochberg, 1995)

    Define pvals as:

        ``pvals`` = ``pval_1`` <= ``pval_2`` <= ... <= ``pval_k`` ... <= ``pval_(m-1)`` <= ``pval_m``

    Compute raw adjusted p-values as:

       ``raw_adj_pval_k`` = ``pval_k`` * ``m``/``k``, where

    - ``raw_adj_pval_k`` is the adjusted ``pval_k`` BEFORE a final correction,
    - ``pval_k`` is the p-value under consideration,
    - ``m`` is the total number of p-values, and
    - ``k`` is the rank of ``pval_k``.

    Perform a final correction to make sure that adjusted p-values are monotonic:

    The final correction is to make sure that ``adj_pval_k`` is less than or
    equal to ``adj_pval_(k+1)``. This procedure starts at the last p-value
    (``raw_adj_pval_m``) and proceeds until the first p-value (``raw_adj_pval_1``).

    Both methods exposed via this function (Benjamini/Hochberg, Benjamini/Yekutieli)
    are also available in the function ``multipletests``, as ``method="fdr_bh"`` and
    ``method="fdr_by"``, respectively.

    See Also
    --------
    multipletests

    """
    pvals = np.asarray(pvals)
    assert pvals.ndim == 1, "pvals must be 1-dimensional, that is of shape (n,)"

    if not is_sorted:
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
    else:
        pvals_sorted = pvals  # alias

    method = string_like(
        method,
        "method",
        options=("i", "indep", "p", "poscorr", "n", "negcorr"),
        lower=False,
    )
    if method in ["i", "indep", "p", "poscorr"]:
        ecdffactor = _ecdf(pvals_sorted)
    else:  # method in ("n", "negcorr")
        cm = np.sum(1.0 / np.arange(1, len(pvals_sorted) + 1))  # corrected this
        ecdffactor = _ecdf(pvals_sorted) / cm
    #    elif method in ['n', 'negcorr']:
    #        cm = np.sum(np.arange(len(pvals)))
    #        ecdffactor = ecdf(pvals_sorted)/cm
    reject = pvals_sorted <= ecdffactor * alpha
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
        reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    # adjust raw adjusted p-values to make them monotonic
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    del pvals_corrected_raw
    pvals_corrected[pvals_corrected > 1] = 1
    if not is_sorted:
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        del pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[pvals_sortind] = reject
        return reject_, pvals_corrected_
    else:
        return reject, pvals_corrected


class LocalFDRCorrectionResult(NamedTuple):
    """
    Result of :func:`local_fdr_correction`.

    Parameters
    ----------
    fdr : ndarray
        The estimated tail false discovery rate for each input p-value, in
        the same order as the p-values passed to `local_fdr_correction`.
    lfdr : ndarray
        The estimated local false discovery rate for each input p-value, in
        the same order as the p-values passed to `local_fdr_correction`.
    """

    fdr: np.ndarray
    lfdr: np.ndarray


def local_fdr_correction(pvals, null_proportion=1.0, is_sorted=False):
    r"""
    Estimate local and tail false discovery rates for a list of p-values.

    Fits a monotone (non-increasing) estimate of the marginal density of
    the p-values using the Grenander estimator. Combined with an estimate
    of the proportion of true null hypotheses, this yields, by Bayes' rule,
    empirical Bayes estimates of the tail and local false discovery rates.

    Parameters
    ----------
    pvals : array_like, 1d
        List of p-values of the individual tests.
    null_proportion : float, optional
        Estimate of :math:`\pi_0`, the proportion of true null hypotheses.
        Defaults to the conservative choice ``1.0``, i.e. all hypotheses
        are assumed null.
    is_sorted : bool, optional
        If False (default), the p-values will be sorted, but the estimated
        FDR values are returned in the original order. If True, then it is
        assumed that the p-values are already sorted in ascending order.

    Returns
    -------
    LocalFDRCorrectionResult
        A namedtuple with the estimated tail false discovery rates
        (``fdr``) and estimated local false discovery rates (``lfdr``).

    See Also
    --------
    local_fdr : Local FDR estimation for Z-scores using Poisson regression.
    fdrcorrection : Benjamini-Hochberg/Benjamini-Yekutieli p-value
        correction.

    Notes
    -----
    Let :math:`t \in [0, 1]` denote a p-value threshold and
    :math:`\hat{\pi}_0` the `null_proportion`. Let :math:`\hat{F}` be the
    Grenander estimate of the empirical distribution function of `pvals`,
    given by the least concave majorant (LCM) of the empirical cdf, and let
    :math:`\hat{f}` be its density estimate, given by the left-hand slope
    of the LCM. The tail and local false discovery rates are then estimated
    by Bayes' rule as

    .. math::

        \widehat{Fdr}(t) = \min\left(1, \hat{\pi}_0 \frac{t}{\hat{F}(t)}
            \right) \approx \Pr(\text{null} \mid p \leq t)

    .. math::

        \widehat{fdr}(t) = \min\left(1, \frac{\hat{\pi}_0}{\hat{f}(t)}
            \right) \approx \Pr(\text{null} \mid p = t)

    Tied p-values are treated as repeated observations at a single support
    point of the empirical distribution function.

    This method assumes that the p-values are independent, are uniformly
    distributed under the null, and have non-increasing densities under
    the alternative.

    References
    ----------
    .. [*] U Grenander (1956). On the theory of mortality measurement: part
       II. Scandinavian Actuarial Journal, 39, 125-153.

    .. [*] B Efron, R Tibshirani, J D Storey, and V Tusher (2001). Empirical
       Bayes analysis of a microarray experiment. Journal of the American
       Statistical Association, 96:456, 1151-1160.

    .. [*] B Efron (2007). Size, Power and False Discovery Rates. The
       Annals of Statistics, 35:4, 1351-1377.

    .. [*] K Strimmer (2008). A unified approach to false discovery rate
       estimation. BMC Bioinformatics, 9, 303.

    .. [*] J A Soloff, D Xiang, and W Fithian (2024). The edge of
       discovery: Controlling the local false discovery rate at the
       margin. The Annals of Statistics, 52:2, 580-601.

    Examples
    --------
    >>> from statsmodels.stats.multitest import local_fdr_correction
    >>> import numpy as np
    >>> pvals = np.random.rand(30)
    >>> lfdr = local_fdr_correction(pvals).lfdr
    """
    try:
        from scipy.optimize import isotonic_regression
    except ImportError as imp_err:
        raise ImportError(
            "SciPy 1.12 or greater is required to provide the function "
            "isotonic_regression in order to use local FDR."
        ) from imp_err

    pvals = array_like(pvals, "pvals", maxdim=1, ndim=1, dtype=float)
    null_proportion = float_like(null_proportion, "null_proportion")
    is_sorted = bool_like(is_sorted, "is_sorted")

    nobs = len(pvals)

    if not is_sorted:
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
    else:
        pvals_sorted = pvals  # alias

    # tied p-values share a support point of the empirical cdf, so the
    # Grenander fit is computed on the distinct values weighted by their
    # multiplicities (ties otherwise produce zero-width, zero-weight gaps)
    uniq_pvals, counts = np.unique(pvals_sorted, return_counts=True)

    # compute left-hand slopes of least concave majorant of empirical cdf
    gaps = np.diff(uniq_pvals, prepend=0)
    # Ensure arrays passed to isotonic_regression are contiguous, aligned, and writeable to avoid
    # potential issues with memory layout and performance on Windows
    requirements = ("C_CONTIGUOUS", "ALIGNED", "OWNDATA", "WRITEABLE", "ENSUREARRAY")
    y = np.require(counts / (nobs * gaps), dtype=float, requirements=requirements)
    weights = np.require(gaps, dtype=float, requirements=requirements)

    # Special case when y is empty, which can happen if pvals is empty.
    # In that case, we should avoid calling isotonic_regression and
    # just set slopes_uniq to an empty array.
    if y.size:
        slope_reg = isotonic_regression(y, weights=weights, increasing=False)
        slopes_uniq = slope_reg.x
    else:
        slopes_uniq = np.array([])

    # compute LCM of empirical cdf
    keep = np.ones(len(uniq_pvals), dtype=bool)
    keep[:-1] = ~np.isclose(slopes_uniq[:-1], slopes_uniq[1:])
    knots_ = np.hstack([0, uniq_pvals[keep]])
    heights_ = np.hstack([0, np.cumsum(counts)[keep] / nobs])
    lcm_cdf = np.interp(pvals_sorted, knots_, heights_)
    slopes = np.repeat(slopes_uniq, counts)

    # return fitted values in original order
    if not is_sorted:
        pvals_unsortind = pvals_sortind.argsort()
        slopes = np.take(slopes, pvals_unsortind)
        lcm_cdf = np.take(lcm_cdf, pvals_unsortind)
    lfdr = np.minimum(1, null_proportion / slopes)
    fdr = np.minimum(1, null_proportion * pvals / lcm_cdf)

    return LocalFDRCorrectionResult(fdr=fdr, lfdr=lfdr)


def fdrcorrection_twostage(
    pvals, alpha=0.05, method="bky", maxiter=1, iter=None, is_sorted=False
):
    """
    (iterated) two stage linear step-up procedure with estimation of number
    of true hypotheses

    Benjamini, Krieger and Yekutieli, procedure in Definition 6

    Parameters
    ----------
    pvals : array_like
        set of p-values of the individual tests.
    alpha : float, optional
        error rate
    method : {'bky', 'bh'}, optional
        see Notes for details

        * 'bky' - implements the procedure in Definition 6 of Benjamini, Krieger
           and Yekutieli 2006
        * 'bh' - the two stage method of Benjamini and Hochberg

    maxiter : int or bool, optional
        Maximum number of iterations.
        maxiter=1 (default) corresponds to the two stage method.
        maxiter=-1 corresponds to full iterations which is maxiter=len(pvals).
        maxiter=0 uses only a single stage fdr correction using a 'bh' or 'bky'
        prior fraction of assumed true hypotheses.
        Boolean maxiter is allowed for backwards compatibility with the
        deprecated ``iter`` keyword.
        maxiter=False is two-stage fdr (maxiter=1)
        maxiter=True is full iteration (maxiter=-1 or maxiter=len(pvals))
    iter : None, optional
        Removed keyword that is kept only for backwards compatibility.
        Passing anything other than the default ``None`` raises a
        ``TypeError``; use ``maxiter`` instead.
    is_sorted : bool, optional
        If False (default), the p_values will be sorted, but the corrected
        pvalues are in the original order. If True, then it assumed that the
        pvalues are already sorted in ascending order.

    Returns
    -------
    rejected : ndarray, bool
        True if a hypothesis is rejected, False if not
    pvals_corrected : ndarray
        pvalues adjusted for multiple hypotheses testing to limit FDR
    m0 : int
        ntest - rej, estimated number of true (not rejected) hypotheses
    alpha_stages : list of floats
        A list of alphas that have been used at each stage

    Notes
    -----
    The returned corrected p-values are specific to the given alpha, they
    cannot be used for a different alpha.

    The returned corrected p-values are from the last stage of the fdr_bh
    linear step-up procedure (fdrcorrection0 with method='indep') corrected
    for the estimated fraction of true hypotheses.
    This means that the rejection decision can be obtained with
    ``pval_corrected <= alpha``, where ``alpha`` is the original significance
    level.
    (Note: This has changed from earlier versions (<0.5.0) of statsmodels.)

    BKY described several other multi-stage methods, which would be easy to implement.
    However, in their simulation the simple two-stage method (with iter=False) was the
    most robust to the presence of positive correlation

    TODO: What should be returned?

    """
    pvals = np.asarray(pvals)

    if iter is not None:
        raise TypeError(
            "iter keyword is not longer allowed, use maxiter keyword instead."
        )

    if maxiter in [-1, None]:
        maxiter = len(pvals)
    # otherwise we use maxiter unchanged

    if not is_sorted:
        pvals_sortind = np.argsort(pvals)
        pvals = np.take(pvals, pvals_sortind)

    method = string_like(method, "method", options=("bky", "bh"), lower=False)
    ntests = len(pvals)
    if method == "bky":
        fact = 1.0 + alpha
        alpha_prime = alpha / fact
    else:  # method == "bh"
        fact = 1.0
        alpha_prime = alpha

    alpha_stages = [alpha_prime]
    rej, pvalscorr = fdrcorrection(
        pvals, alpha=alpha_prime, method="indep", is_sorted=True
    )
    r1 = rej.sum()
    if r1 in (0, ntests):
        # return rej, pvalscorr * fact, ntests - r1, alpha_stages
        reject = rej
        pvalscorr *= fact
        ri = r1
    else:
        ri_old = ri = r1
        ntests0 = ntests  # needed if maxiter=0
        # while True:
        for it in range(maxiter):
            ntests0 = 1.0 * ntests - ri_old
            alpha_star = alpha_prime * ntests / ntests0
            alpha_stages.append(alpha_star)
            # print ntests0, alpha_star
            rej, pvalscorr = fdrcorrection(
                pvals, alpha=alpha_star, method="indep", is_sorted=True
            )
            ri = rej.sum()
            if (it >= maxiter - 1) or ri == ri_old:
                break
            if ri < ri_old:
                # prevent cycles and endless loops
                raise RuntimeError(" oops - should not be here")
            ri_old = ri

        # make adjustment to pvalscorr to reflect estimated number of Non-Null cases
        # decision is then pvalscorr < alpha  (or <=)
        pvalscorr *= ntests0 * 1.0 / ntests
        if method == "bky":
            pvalscorr *= 1.0 + alpha

    pvalscorr[pvalscorr > 1] = 1
    if not is_sorted:
        pvalscorr_ = np.empty_like(pvalscorr)
        pvalscorr_[pvals_sortind] = pvalscorr
        del pvalscorr
        reject = np.empty_like(rej)
        reject[pvals_sortind] = rej
        return reject, pvalscorr_, ntests - ri, alpha_stages
    else:
        return rej, pvalscorr, ntests - ri, alpha_stages


def local_fdr(zscores, null_proportion=1.0, null_pdf=None, deg=7, nbins=30, alpha=0):
    """
    Calculate local FDR values for a list of Z-scores

    Parameters
    ----------
    zscores : ndarray
        A vector of Z-scores
    null_proportion : float, optional
        The assumed proportion of true null hypotheses
    null_pdf : callable, optional
        The density of null Z-scores; if None, use standard normal
    deg : int, optional
        The maximum exponent in the polynomial expansion of the
        density of non-null Z-scores
    nbins : int, optional
        The number of bins for estimating the marginal density
        of Z-scores.
    alpha : float, optional
        Use Poisson ridge regression with parameter alpha to estimate
        the density of non-null Z-scores.

    Returns
    -------
    fdr : ndarray
        A vector of FDR values

    References
    ----------
    .. [*] B Efron (2008).  Microarrays, Empirical Bayes, and the Two-Groups
       Model.  Statistical Science 23:1, 1-22.

    Examples
    --------
    Basic use (the null Z-scores are taken to be standard normal):

    >>> from statsmodels.stats.multitest import local_fdr
    >>> import numpy as np
    >>> zscores = np.random.randn(30)
    >>> fdr = local_fdr(zscores)

    Use a Gaussian null distribution estimated from the data:

    >>> from statsmodels.stats.multitest import NullDistribution
    >>> null = NullDistribution(zscores)
    >>> fdr = local_fdr(zscores, null_pdf=null.pdf)
    """

    from statsmodels.genmod.generalized_linear_model import GLM, families
    from statsmodels.regression.linear_model import OLS

    # Bins for Poisson modeling of the marginal Z-score density
    minz = min(zscores)
    maxz = max(zscores)
    bins = np.linspace(minz, maxz, nbins)

    # Bin counts
    zhist = np.histogram(zscores, bins)[0]

    # Bin centers
    zbins = (bins[:-1] + bins[1:]) / 2

    # The design matrix at bin centers
    dmat = np.vander(zbins, deg + 1)

    # Rescale the design matrix
    sd = dmat.std(0)
    ii = sd > 1e-8
    dmat[:, ii] /= sd[ii]

    start = OLS(np.log(1 + zhist), dmat).fit().params

    # Poisson regression
    if alpha > 0:
        md = GLM(zhist, dmat, family=families.Poisson()).fit_regularized(
            L1_wt=0, alpha=alpha, start_params=start
        )
    else:
        md = GLM(zhist, dmat, family=families.Poisson()).fit(start_params=start)

    # The design matrix for all Z-scores
    dmat_full = np.vander(zscores, deg + 1)
    dmat_full[:, ii] /= sd[ii]

    # The height of the estimated marginal density of Z-scores,
    # evaluated at every observed Z-score.
    fz = md.predict(dmat_full) / (len(zscores) * (bins[1] - bins[0]))

    # The null density.
    if null_pdf is None:
        f0 = np.exp(-0.5 * zscores**2) / np.sqrt(2 * np.pi)
    else:
        f0 = null_pdf(zscores)

    # The local FDR values
    fdr = null_proportion * f0 / fz

    fdr = np.clip(fdr, 0, 1)

    return fdr


class NullDistribution:
    """
    Estimate a Gaussian distribution for the null Z-scores

    The observed Z-scores consist of both null and non-null values.
    The fitted distribution of null Z-scores is Gaussian, but may have
    non-zero mean and/or non-unit scale.

    Parameters
    ----------
    zscores : ndarray
        The observed Z-scores.
    null_lb : float, optional
        Z-scores between `null_lb` and `null_ub` are all considered to be
        true null hypotheses.
    null_ub : float, optional
        See `null_lb`.
    estimate_mean : bool, optional
        If True, estimate the mean of the distribution.  If False, the
        mean is fixed at zero.
    estimate_scale : bool, optional
        If True, estimate the scale of the distribution.  If False, the
        scale parameter is fixed at 1.
    estimate_null_proportion : bool, optional
        If True, estimate the proportion of true null hypotheses (i.e.
        the proportion of z-scores with expected value zero).  If False,
        this parameter is fixed at 1.

    Attributes
    ----------
    mean : float
        The estimated mean of the empirical null distribution
    sd : float
        The estimated standard deviation of the empirical null distribution
    null_proportion : float
        The estimated proportion of true null hypotheses among all hypotheses

    References
    ----------
    .. [*] B Efron (2008).  Microarrays, Empirical Bayes, and the Two-Groups
       Model.  Statistical Science 23:1, 1-22.

    Notes
    -----
    See also:

    http://nipy.org/nipy/labs/enn.html#nipy.algorithms.statistics.empirical_pvalue.NormalEmpiricalNull.fdr
    """

    def __init__(
        self,
        zscores,
        null_lb=-1,
        null_ub=1,
        estimate_mean=True,
        estimate_scale=True,
        estimate_null_proportion=False,
    ):

        # Extract the null z-scores
        ii = np.flatnonzero((zscores >= null_lb) & (zscores <= null_ub))
        if len(ii) == 0:
            raise RuntimeError("No Z-scores fall between null_lb and null_ub")
        zscores0 = zscores[ii]

        # Number of Z-scores, and null Z-scores
        n_zs, n_zs0 = len(zscores), len(zscores0)

        # Unpack and transform the parameters to the natural scale, hold
        # parameters fixed as specified.
        def xform(params):

            mean = 0.0
            sd = 1.0
            prob = 1.0

            ii = 0
            if estimate_mean:
                mean = params[ii]
                ii += 1
            if estimate_scale:
                sd = np.exp(params[ii])
                ii += 1
            if estimate_null_proportion:
                prob = 1 / (1 + np.exp(-params[ii]))

            return mean, sd, prob

        from scipy.stats.distributions import norm

        def fun(params):
            """
            Negative log-likelihood of z-scores

            The implementation follows section 4 from Efron 2008.

            Parameters
            ----------
            params : ndarray
                Vector of three parameters, packed as ``mean`` (the location
                parameter), ``logscale`` (log of the scale parameter), and
                ``logitprop`` (logit of the proportion of true nulls).

            Returns
            -------
            float
                The negative log-likelihood evaluated at `params`.
            """

            d, s, p = xform(params)

            # Mass within the central region
            central_mass = norm.cdf((null_ub - d) / s) - norm.cdf((null_lb - d) / s)

            # Probability that a Z-score is null and is in the central region
            cp = p * central_mass

            # Binomial term
            rval = n_zs0 * np.log(cp) + (n_zs - n_zs0) * np.log(1 - cp)

            # Truncated Gaussian term for null Z-scores
            zv = (zscores0 - d) / s
            rval += np.sum(-(zv**2) / 2) - n_zs0 * np.log(s)
            rval -= n_zs0 * np.log(central_mass)

            return -rval

        # Estimate the parameters
        from scipy.optimize import minimize

        # starting values are mean = 0, scale = 1, p0 ~ 1
        mz = minimize(fun, np.r_[0.0, 0, 3], method="Nelder-Mead")
        mean, sd, prob = xform(mz["x"])

        self.mean = mean
        self.sd = sd
        self.null_proportion = prob

    # The fitted null density function
    def pdf(self, zscores):
        """
        Evaluates the fitted empirical null Z-score density

        Parameters
        ----------
        zscores : scalar or ndarray
            The point or points at which the density is to be
            evaluated.

        Returns
        -------
        scalar or ndarray
            The empirical null Z-score density evaluated at the given
            points.
        """

        zval = (zscores - self.mean) / self.sd
        return np.exp(-0.5 * zval**2 - np.log(self.sd) - 0.5 * np.log(2 * np.pi))
