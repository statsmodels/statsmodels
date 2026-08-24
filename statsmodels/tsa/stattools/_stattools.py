"""Statistical tools for time series analysis"""

from __future__ import annotations

from statsmodels.compat.python import lzip
from statsmodels.compat.scipy import _next_regular

from dataclasses import dataclass
from typing import ClassVar, Literal, NamedTuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import correlate

from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.stats._results_store import ResultsStore
from statsmodels.stats.base import LimitedIterationMixin
from statsmodels.tools.sm_exceptions import (
    CollinearityWarning,
    InfeasibleTestError,
    InterpolationWarning,
    MissingDataError,
    ValueWarning,
)
from statsmodels.tools.tools import add_constant
from statsmodels.tools.validation import (
    array_like,
    bool_like,
    float_like,
    int_like,
    string_like,
)
from statsmodels.tsa._bds import bds
from statsmodels.tsa._innovations import innovations_algo, innovations_filter
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
from statsmodels.tsa.tsatools import add_trend, lagmat, lagmat2ds

ArrayLike1D = np.ndarray | pd.Series | list[float]

__all__ = [
    "ADFullerResult",
    "AcfResult",
    "BreakvarHeteroskedasticityResult",
    "CcfResult",
    "CointResult",
    "JackknifeResult",
    "KPSSResult",
    "LevinsonDurbinPacfResult",
    "LevinsonDurbinResult",
    "PacfBurgResult",
    "PacfResult",
    "PccfResult",
    "QStatResult",
    "RURResult",
    "acf",
    "acovf",
    "adfuller",
    "bds",
    "block_jackknife",
    "breakvar_heteroskedasticity_test",
    "ccf",
    "ccovf",
    "coint",
    "grangercausalitytests",
    "innovations_algo",
    "innovations_filter",
    "kpss",
    "lagmat",
    "levinson_durbin",
    "levinson_durbin_pacf",
    "pacf",
    "pacf_burg",
    "pacf_ols",
    "pacf_yw",
    "pccf",
    "q_stat",
    "range_unit_root_test",
    "zivot_andrews",
]

SQRTEPS = np.sqrt(np.finfo(np.double).eps)


def _autolag(
    mod,
    endog,
    exog,
    startlag,
    maxlag,
    method,
    modargs=(),
    fitargs=(),
    regresults=False,
):
    """
    Returns the results for the lag length that maximizes the info criterion

    Parameters
    ----------
    mod : Model class
        Model estimator class
    endog : array_like
        nobs array containing endogenous variable
    exog : array_like
        nobs by (startlag + maxlag) array containing lags and possibly other
        variables
    startlag : int
        The first zero-indexed column to hold a lag.  See Notes.
    maxlag : int
        The highest lag order for lag length selection.
    method : {"aic", "bic", "t-stat"}
        aic - Akaike Information Criterion
        bic - Bayes Information Criterion
        t-stat - Based on last lag
    modargs : tuple, optional
        args to pass to model.  See notes.
    fitargs : tuple, optional
        args to pass to fit.  See notes.
    regresults : bool, optional
        Flag indicating to return optional return results

    Returns
    -------
    icbest : float
        Best information criteria.
    bestlag : int
        The lag length that maximizes the information criterion.
    results : dict, optional
        Dictionary containing all estimation results

    Notes
    -----
    Does estimation like mod(endog, exog[:,:i], *modargs).fit(*fitargs)
    where i goes from lagstart to lagstart+maxlag+1.  Therefore, lags are
    assumed to be in contiguous columns from low to high lag length with
    the highest lag in the last column.
    """
    # TODO: can tcol be replaced by maxlag + 2?
    # TODO: This could be changed to laggedRHS and exog keyword arguments if
    #    this will be more general.

    method = string_like(method, "method", options=("aic", "bic", "t-stat"))
    results = {}
    for lag in range(startlag, startlag + maxlag + 1):
        mod_instance = mod(endog, exog[:, :lag], *modargs)
        results[lag] = mod_instance.fit()

    if method == "aic":
        icbest, bestlag = min((v.aic, k) for k, v in results.items())
    elif method == "bic":
        icbest, bestlag = min((v.bic, k) for k, v in results.items())
    else:  # method == "t-stat"
        # stop = stats.norm.ppf(.95)
        stop = 1.6448536269514722
        # Default values to ensure that always set
        bestlag = startlag + maxlag
        icbest = 0.0
        for lag in range(startlag + maxlag, startlag - 1, -1):
            icbest = np.abs(results[lag].tvalues[-1])
            bestlag = lag
            if np.abs(icbest) >= stop:
                # Break for first lag with a significant t-stat
                break

    if not regresults:
        return icbest, bestlag
    else:
        return icbest, bestlag, results


@dataclass(frozen=True, slots=True, repr=False)
class ADFullerResult(LimitedIterationMixin[float]):
    """
    Result of :func:`adfuller`.

    Parameters
    ----------
    statistic : float
        The test statistic.
    pvalue : float
        MacKinnon's approximate p-value based on MacKinnon (1994, 2010).
    lags : int
        The number of lags used.
    nobs : int
        The number of observations used for the ADF regression and
        calculation of the critical values.
    critical_values : dict[str, float]
        Critical values for the test statistic at the 1 %, 5 %, and 10 %
        levels. Based on MacKinnon (2010).
    icbest : float or None
        The maximized information criterion if autolag is not None,
        otherwise None.
    resstore : ResultsStore or None
        A dummy class with results attached as attributes, if ``store`` was
        True, otherwise None.

    Notes
    -----
    Unpacks as ``statistic, pvalue = result``. Other values are only
    accessible using attributes.
    """

    statistic: float
    pvalue: float
    lags: int
    nobs: int
    critical_values: dict[str, float]
    icbest: float | None
    resstore: ResultsStore | None

    _iter_fields: ClassVar[tuple[str, ...]] = ("statistic", "pvalue")

    def __repr__(self) -> str:
        return f"""\
{self.__class__.__name__}
ADF Statistic: {self.statistic:0.5f}
P-value: {self.pvalue:0.5f}
Used Lag: {self.lags}
Nobs: {self.nobs}
Critical Values: {self.critical_values}
"""


# See:
# Ng and Perron(2001), Lag length selection and the construction of unit root
# tests with good size and power, Econometrica, Vol 69 (6) pp 1519-1554
# TODO: include drift keyword, only valid with regression == "c"
# just changes the distribution of the test statistic to a t distribution
# TODO: autolag is untested
def adfuller(
    x,
    maxlag: int | None = None,
    regression="c",
    autolag="AIC",
    store=False,
    regresults=False,
    *,
    result_object: bool | None = None,
):
    """
    Augmented Dickey-Fuller unit root test

    The Augmented Dickey-Fuller test can be used to test for a unit root in a
    univariate process in the presence of serial correlation.

    Parameters
    ----------
    x : array_like, 1d
        The data series to test.
    maxlag : int, optional
        Maximum lag which is included in test, default value of
        12*(nobs/100)^{1/4} is used when ``None``.
    regression : {"c","ct","ctt","n"}, optional
        Constant and trend order to include in regression.

        * "c" : constant only (default).
        * "ct" : constant and trend.
        * "ctt" : constant, and linear and quadratic trend.
        * "n" : no constant, no trend.

    autolag : {"AIC", "BIC", "t-stat", None}, optional
        Method to use when automatically determining the lag length among the
        values 0, 1, ..., maxlag.

        * If "AIC" (default) or "BIC", then the number of lags is chosen
          to minimize the corresponding information criterion.
        * "t-stat" based choice of maxlag.  Starts with maxlag and drops a
          lag until the t-statistic on the last lag length is significant
          using a 5%-sized test.
        * If None, then the number of included lags is set to maxlag.
    store : bool, optional
        If True, then a result instance is returned additionally to
        the adf statistic. Default is False.
    regresults : bool, optional
        If True, the full regression results are returned. Default is False.
    result_object : bool, optional
        Flag indicating whether to return the results as an
        ``ADFullerResult`` instead of a plain tuple. If ``None``
        (the default), the current tuple-returning behavior is used and a
        ``FutureWarning`` is issued.

        .. deprecated:: 0.15.0

            In release 0.16.0 or after July 2027, whichever is later, the
            default will change to always return an ``ADFullerResult``.
            Set ``result_object=True`` to opt in now, or
            ``result_object=False`` to silence the warning and keep the
            current return type.

    Returns
    -------
    ADFullerResult
        If ``result_object=True``, a result object with fields ``statistic``,
        ``pvalue``, ``lags``, ``nobs``, ``critical_values``, ``icbest``,
        and ``resstore`` (``icbest``/``resstore`` are ``None`` when not
        computed). See :class:`~statsmodels.tsa.stattools.ADFullerResult`.

    Otherwise (the deprecated default), a plain tuple whose length depends
    on `store` and `autolag`, made up of a subset of:

    statistic : float
        The test statistic.
    pvalue : float
        MacKinnon's approximate p-value based on MacKinnon (1994, 2010).
    lags : int
        The number of lags used.
    nobs : int
        The number of observations used for the ADF regression and calculation
        of the critical values.
    critical_values : dict
        Critical values for the test statistic at the 1 %, 5 %, and 10 %
        levels. Based on MacKinnon (2010).
    icbest : float
        The maximized information criterion if autolag is not None.
    resstore : ResultsStore, optional
        A dummy class with results attached as attributes.

    Notes
    -----
    The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
    root, with the alternative that there is no unit root. If the pvalue is
    above a critical size, then we cannot reject that there is a unit root.

    The p-values are obtained through regression surface approximation from
    MacKinnon 1994, but using the updated 2010 tables. If the p-value is close
    to significant, then the critical values should be used to judge whether
    to reject the null.

    The autolag option and maxlag for it are described in Greene.

    See the notebook `Stationarity and detrending (ADF/KPSS)
    <../examples/notebooks/generated/stationarity_detrending_adf_kpss.html>`__
    for an overview.

    References
    ----------
    .. [1] W. Green.  "Econometric Analysis," 5th ed., Pearson, 2003.

    .. [2] Hamilton, J.D.  "Time Series Analysis".  Princeton, 1994.

    .. [3] MacKinnon, J.G. 1994.  "Approximate asymptotic distribution functions for
        unit-root and cointegration tests.  *Journal of Business and Economic
        Statistics* 12, 167-76.

    .. [4] MacKinnon, J.G. 2010. "Critical Values for Cointegration Tests."  Queen's
        University, Dept of Economics, Working Papers.  Available at
        http://ideas.repec.org/p/qed/wpaper/1227.html
    """
    x = array_like(x, "x", ndim=1)
    maxlag = int_like(maxlag, "maxlag", optional=True)
    regression = string_like(regression, "regression", options=("c", "ct", "ctt", "n"))
    autolag = string_like(
        autolag, "autolag", optional=True, options=("aic", "bic", "t-stat")
    )
    store = bool_like(store, "store")
    regresults = bool_like(regresults, "regresults")
    result_object = bool_like(result_object, "result_object", optional=True)

    if x.max() == x.min():
        raise ValueError("Invalid input, x is constant")

    if regresults:
        store = True

    nobs = x.shape[0]

    ntrend = len(regression) if regression != "n" else 0
    if maxlag is None:
        # from Greene referencing Schwert 1989
        maxlag = int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))
        # -1 for the diff
        maxlag = min(nobs // 2 - ntrend - 1, maxlag)
        if maxlag < 0:
            raise ValueError(
                "sample size is too short to use selected regression component"
            )
    elif maxlag > nobs // 2 - ntrend - 1:
        raise ValueError(
            "maxlag must be less than (nobs/2 - 1 - ntrend) "
            "where n trend is the number of included "
            "deterministic regressors"
        )
    xdiff = np.diff(x)
    xdall = lagmat(xdiff[:, None], maxlag, trim="both", original="in")
    nobs = xdall.shape[0]

    xdall[:, 0] = x[-nobs - 1 : -1]  # replace 0 xdiff with level of x
    xdshort = xdiff[-nobs:]

    if store:
        resstore = ResultsStore()
    if autolag:
        if regression != "n":
            fullRHS = add_trend(xdall, regression, prepend=True)
        else:
            fullRHS = xdall
        startlag = fullRHS.shape[1] - xdall.shape[1] + 1
        # 1 for level
        # search for lag length with smallest information criteria
        # Note: use the same number of observations to have comparable IC
        # aic and bic: smaller is better

        if not regresults:
            icbest, bestlag = _autolag(OLS, xdshort, fullRHS, startlag, maxlag, autolag)
        else:
            icbest, bestlag, alres = _autolag(
                OLS,
                xdshort,
                fullRHS,
                startlag,
                maxlag,
                autolag,
                regresults=regresults,
            )
            resstore.autolag_results = alres

        bestlag -= startlag  # convert to lag not column index

        # rerun ols with best autolag
        xdall = lagmat(xdiff[:, None], bestlag, trim="both", original="in")
        nobs = xdall.shape[0]
        xdall[:, 0] = x[-nobs - 1 : -1]  # replace 0 xdiff with level of x
        xdshort = xdiff[-nobs:]
        usedlag = bestlag
    else:
        usedlag = maxlag
        icbest = None
    if regression != "n":
        resols = OLS(xdshort, add_trend(xdall[:, : usedlag + 1], regression)).fit()
    else:
        resols = OLS(xdshort, xdall[:, : usedlag + 1]).fit()

    adfstat = float(resols.tvalues[0])
    #    adfstat = (resols.params[0]-1.0)/resols.bse[0]
    # the "asymptotically correct" z statistic is obtained as
    # nobs/(1-np.sum(resols.params[1:-(trendorder+1)])) (resols.params[0] - 1)
    # I think this is the statistic that is used for series that are integrated
    # for orders higher than I(1), ie., not ADF but cointegration tests.

    # Get approx p-value and critical values
    pvalue = mackinnonp(adfstat, regression=regression, N=1)
    critvalues = mackinnoncrit(N=1, regression=regression, nobs=nobs)
    critvalues = {
        "1%": critvalues[0],
        "5%": critvalues[1],
        "10%": critvalues[2],
    }
    if store:
        resstore.resols = resols
        resstore.maxlag = maxlag
        resstore.usedlag = usedlag
        resstore.adfstat = adfstat
        resstore.critvalues = critvalues
        resstore.nobs = nobs
        resstore.H0 = "The coefficient on the lagged level equals 1 - unit root"
        resstore.HA = "The coefficient on the lagged level < 1 - stationary"
        resstore.icbest = icbest
        resstore._str = "Augmented Dickey-Fuller Test Results"
    else:
        resstore = None

    if result_object is None:
        warnings.warn(
            "adfuller currently returns a plain tuple whose length depends "
            "on the store and autolag arguments. In release 0.16 or after "
            "July 2027, whichever is later, the default behavior will "
            "switch to always returning an ADFullerResult. Set "
            "result_object=True to switch now, or result_object=False "
            "to keep the current behavior and silence this warning.",
            FutureWarning,
            stacklevel=2,
        )
    if result_object:
        return ADFullerResult(
            adfstat, pvalue, usedlag, nobs, critvalues, icbest, resstore
        )
    if store:
        return adfstat, pvalue, critvalues, resstore
    elif not autolag:
        return adfstat, pvalue, usedlag, nobs, critvalues
    else:
        return adfstat, pvalue, usedlag, nobs, critvalues, icbest


def acovf(x, adjusted=False, demean=True, fft=True, missing="none", nlag=None):
    """
    Estimate autocovariances

    Parameters
    ----------
    x : array_like
        Time series data. Must be 1d.
    adjusted : bool, optional
        If True, then denominators is n-k, otherwise n.
    demean : bool, optional
        If True, then subtract the mean x from each element of x.
    fft : bool, optional
        If True, use FFT convolution.  This method should be preferred
        for long time series.
    missing : {"none", "raise", "conservative", "drop"}, optional
        Specifies how the NaNs are to be treated. "none" performs no checks.
        "raise" raises an exception if NaN values are found. "drop" removes
        the missing observations and then estimates the autocovariances
        treating the non-missing as contiguous. "conservative" computes the
        autocovariance using nan-ops so that nans are removed when computing
        the mean and cross-products that are used to estimate the
        autocovariance. When using "conservative", n is set to the number of
        non-missing observations.
    nlag : int, optional
        Limit the number of autocovariances returned.  Size of returned
        array is nlag + 1.  Setting nlag when fft is False uses a simple,
        direct estimator of the autocovariances that only computes the first
        nlag + 1 values. This can be much faster when the time series is long
        and only a small number of autocovariances are needed.

    Returns
    -------
    ndarray
        The estimated autocovariances.

    References
    ----------
    .. [1] Parzen, E., 1963. On spectral analysis with missing observations
           and amplitude modulation. Sankhya: The Indian Journal of
           Statistics, Series A, pp.383-392.
    """
    adjusted = bool_like(adjusted, "adjusted")
    demean = bool_like(demean, "demean")
    fft = bool_like(fft, "fft", optional=False)
    missing = string_like(
        missing, "missing", options=("none", "raise", "conservative", "drop")
    )
    nlag = int_like(nlag, "nlag", optional=True)

    x = array_like(x, "x", ndim=1)

    if missing == "none":
        deal_with_masked = False
    else:
        deal_with_masked = has_missing(x)
    notmask_bool = ~np.isnan(x)
    if deal_with_masked:
        if missing == "raise":
            raise MissingDataError("NaNs were encountered in the data")
        if missing == "drop" and notmask_bool.sum() == 0:
            raise ValueError("All observations are missing after dropping.")
        if missing == "conservative":
            # Must copy for thread safety
            x = x.copy()
            x[~notmask_bool] = 0
        else:  # "drop"
            x = x[notmask_bool]  # copies non-missing
        notmask_int = notmask_bool.astype(int)  # int

    if demean and deal_with_masked:
        # whether "drop" or "conservative":
        xo = x - x.sum() / notmask_int.sum()
        if missing == "conservative":
            xo[~notmask_bool] = 0
    elif demean:
        xo = x - x.mean()
    else:
        xo = x

    n = len(x)
    lag_len = nlag
    if nlag is None:
        lag_len = n - 1
    elif nlag > n - 1:
        raise ValueError("nlag must be smaller than nobs - 1")

    if not fft and nlag is not None:
        acov = np.empty(lag_len + 1)
        acov[0] = xo.dot(xo)
        for i in range(lag_len):
            acov[i + 1] = xo[i + 1 :].dot(xo[: -(i + 1)])
        if not deal_with_masked or missing == "drop":
            if adjusted:
                acov /= n - np.arange(lag_len + 1)
            else:
                acov /= n
        elif adjusted:
            divisor = np.empty(lag_len + 1, dtype=np.int64)
            divisor[0] = notmask_int.sum()
            for i in range(lag_len):
                divisor[i + 1] = notmask_int[i + 1 :].dot(notmask_int[: -(i + 1)])
            divisor[divisor == 0] = 1
            acov /= divisor
        else:  # biased, missing data but npt "drop"
            acov /= notmask_int.sum()
        return acov

    if adjusted and deal_with_masked and missing == "conservative":
        d = np.correlate(notmask_int, notmask_int, "full")
        d[d == 0] = 1
    elif adjusted:
        xi = np.arange(1, n + 1)
        d = np.hstack((xi, xi[:-1][::-1]))
    elif deal_with_masked:
        # biased and NaNs given and ("drop" or "conservative")
        d = notmask_int.sum() * np.ones(2 * n - 1)
    else:  # biased and no NaNs or missing=="none"
        d = n * np.ones(2 * n - 1)

    if fft:
        nobs = len(xo)
        n = _next_regular(2 * nobs + 1)
        Frf = np.fft.fft(xo, n=n)
        acov = np.fft.ifft(Frf * np.conjugate(Frf))[:nobs] / d[nobs - 1 :]
        acov = acov.real
    else:
        acov = np.correlate(xo, xo, "full")[n - 1 :] / d[n - 1 :]

    if deal_with_masked and notmask_bool.sum() == 0:
        acov[:] = np.nan

    if nlag is not None:
        # Copy to allow gc of full array rather than view
        return acov[: lag_len + 1].copy()
    return acov


class JackknifeResult(NamedTuple):
    """
    Result of :func:`block_jackknife`.

    Parameters
    ----------
    theta_jack : float or ndarray
        The bias-corrected jackknife point estimate.
    se : float or ndarray
        The jackknife standard error estimate.
    """

    theta_jack: float | np.ndarray
    se: float | np.ndarray


def block_jackknife(x, statistic, n_blocks=-1):
    """
    Delete-k (block) jackknife estimate of bias and standard error

    Computes the jackknife point estimate and standard error of a statistic
    by systematically leaving out contiguous blocks of the sample and
    recomputing the statistic, generalizing the classical delete-1 jackknife
    to allow for serial dependence in the data.

    Parameters
    ----------
    x : array_like
        The data series, 1-D.
    statistic : callable
        Function that takes an array and returns a scalar or array-valued
        estimate.
    n_blocks : int, optional
        The number of blocks to use. If -1 (default), uses ``n_blocks =
        len(x)``, i.e., the classical delete-1 jackknife.

    Returns
    -------
    JackknifeResult
        A result object with fields:

        theta_jack : float or ndarray
            The bias-corrected jackknife point estimate.
        se : float or ndarray
            The jackknife standard error estimate.

    Notes
    -----
    For data with serial dependence, ``n_blocks`` should be chosen small
    enough (equivalently, block size large enough) that observations in
    different blocks are approximately uncorrelated; this is a modeling
    choice left to the user and is not verified automatically.

    This implementation performs single-order bias correction only. A
    known residual bias of order O(1/n^2) remains uncorrected; an iterated
    jackknife (Schucany, Gray & Owen 1971) would address this at added
    computational cost. Additionally, non-overlapping block placement
    introduces some sensitivity to the arbitrary choice of block
    boundaries; overlapping/moving-block schemes (Kunsch 1989) would
    reduce this. Both are natural extensions but out of scope here.

    To avoid O(n_blocks * n) cumulative memory allocation, a single
    internal buffer is allocated once and reused across all delete-block
    computations. As a result, ``statistic`` must not modify the array
    it receives in place; doing so will silently corrupt results in
    later iterations.

    References
    ----------
    .. [1] Quenouille, M.H. (1949). "Approximate tests of correlation in
       time-series." Journal of the Royal Statistical Society, Series B,
       11: 68-84.
    .. [2] Tukey, J.W. (1958). "Bias and confidence in not-quite large
       samples." Annals of Mathematical Statistics, 29: 614.
    .. [3] Kunsch, H.R. (1989). "The jackknife and the bootstrap for
       general stationary observations." Annals of Statistics, 17:
       1217-1241.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.normal(size=100)
    >>> theta_jack, se = block_jackknife(x, np.mean, n_blocks=10)
    """
    x = array_like(x, "x", ndim=1)
    n = x.shape[0]

    if n_blocks == -1:
        n_blocks = n
    n_blocks = int_like(n_blocks, "n_blocks")

    if n_blocks <= 1:
        raise ValueError("n_blocks must be greater than 1.")
    if n_blocks > n:
        raise ValueError("n_blocks cannot exceed the number of observations.")
    if not callable(statistic):
        raise ValueError("statistic must be callable.")

    block_sizes = np.full(n_blocks, n // n_blocks)
    block_sizes[: n % n_blocks] += 1
    block_bounds = np.cumsum(block_sizes)

    theta_full = np.asarray(statistic(x))

    max_reduced_len = n - block_sizes.min()
    buffer = np.empty(max_reduced_len, dtype=float)

    theta_delete = np.empty((n_blocks,) + theta_full.shape, dtype=float)
    start = 0
    for b in range(n_blocks):
        end = block_bounds[b]
        reduced_len = n - (end - start)

        buffer[:start] = x[:start]
        buffer[start:reduced_len] = x[end:]

        theta_delete[b] = statistic(buffer[:reduced_len])
        start = end

    pseudo_values = n_blocks * theta_full - (n_blocks - 1) * theta_delete
    theta_jack = np.mean(pseudo_values, axis=0)
    variance = np.sum((pseudo_values - theta_jack) ** 2, axis=0) / (
        n_blocks * (n_blocks - 1)
    )
    se = np.sqrt(variance)

    if theta_full.shape == ():
        theta_jack = float(theta_jack)
        se = float(se)

    return JackknifeResult(theta_jack, se)


class QStatResult(NamedTuple):
    """
    Result of :func:`q_stat`.

    Parameters
    ----------
    statistic : ndarray
        Ljung-Box Q-statistic for autocorrelation parameters.
    pvalue : ndarray
        P-value of the Q statistic.
    """

    statistic: np.ndarray
    pvalue: np.ndarray


def q_stat(x, nobs):
    """
    Compute Ljung-Box Q Statistic

    Parameters
    ----------
    x : array_like
        Array of autocorrelation coefficients.  Can be obtained from acf.
    nobs : int
        Number of observations in the entire sample (ie., not just the length
        of the autocorrelation function results).

    Returns
    -------
    QStatResult
        A result object with fields:

        statistic : ndarray
            Ljung-Box Q-statistic for autocorrelation parameters.
        pvalue : ndarray
            P-value of the Q statistic.

    See Also
    --------
    statsmodels.stats.diagnostic.acorr_ljungbox
        Ljung-Box Q-test for autocorrelation in time series based
        on a time series rather than the estimated autocorrelation
        function.

    Notes
    -----
    Designed to be used with acf.
    """
    x = array_like(x, "x")
    nobs = int_like(nobs, "nobs")

    ret = (
        nobs * (nobs + 2) * np.cumsum((1.0 / (nobs - np.arange(1, len(x) + 1))) * x**2)
    )
    chi2 = stats.chi2.sf(ret, np.arange(1, len(x) + 1))
    return QStatResult(ret, chi2)


# NOTE: Changed unbiased to False
# see for example
# http://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm
class AcfResult(NamedTuple):
    """
    Result of :func:`acf`.

    Parameters
    ----------
    acf : ndarray
        The autocorrelation function for lags 0, 1, ..., nlags. Shape
        (nlags+1,).
    confint : ndarray or None
        Confidence intervals for the ACF at lags 0, 1, ..., nlags. Shape
        (nlags + 1, 2). ``None`` unless ``alpha`` was not None.
    qstat : ndarray or None
        The Ljung-Box Q-statistic for autocorrelation parameters. ``None``
        unless ``qstat`` was True.
    pvalues : ndarray or None
        The p-values associated with the Q-statistics of the Ljung-Box
        autocorrelation test. ``None`` unless ``qstat`` was True.
    """

    acf: np.ndarray
    confint: np.ndarray | None
    qstat: np.ndarray | None
    pvalues: np.ndarray | None


def acf(
    x,
    adjusted=False,
    nlags=None,
    qstat=False,
    fft=True,
    alpha=None,
    bartlett_confint=True,
    missing="none",
    *,
    result_object: bool | None = None,
):
    """
    Calculate the autocorrelation function

    Parameters
    ----------
    x : array_like
       The time series data.
    adjusted : bool, optional
       If True, then denominators for autocovariance are n-k, otherwise n.
    nlags : int, optional
        Number of lags to return autocorrelation for. If not provided,
        uses min(10 * np.log10(nobs), nobs - 1). The returned value
        includes lag 0 (ie., 1) so size of the acf vector is (nlags + 1,).
    qstat : bool, optional
        If True, returns the Ljung-Box q statistic for each autocorrelation
        coefficient.  See q_stat for more information.
    fft : bool, optional
        If True, computes the ACF via FFT.
    alpha : float, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        Bartlett's formula.
    bartlett_confint : bool, optional
        Confidence intervals for ACF values are generally placed at 2
        standard errors around r_k. The formula used for standard error
        depends upon the situation. If the autocorrelations are being used
        to test for randomness of residuals as part of the ARIMA routine,
        the standard errors are determined assuming the residuals are white
        noise. The approximate formula for any lag is that standard error
        of each r_k = 1/sqrt(N). See section 9.4 of [2] for more details on
        the 1/sqrt(N) result. For more elementary discussion, see section 5.3.2
        in [3].
        For the ACF of raw data, the standard error at a lag k is
        found as if the right model was an MA(k-1). This allows the possible
        interpretation that if all autocorrelations past a certain lag are
        within the limits, the model might be an MA of order defined by the
        last significant autocorrelation. In this case, a moving average
        model is assumed for the data and the standard errors for the
        confidence intervals should be generated using Bartlett's formula.
        For more details on Bartlett formula result, see section 7.2 in [2].
    missing : {"none", "raise", "conservative", "drop"}, optional
        Specifies how the NaNs are to be treated. "none" performs no checks.
        "raise" raises an exception if NaN values are found. "drop" removes
        the missing observations and then estimates the autocovariances
        treating the non-missing as contiguous. "conservative" computes the
        autocovariance using nan-ops so that nans are removed when computing
        the mean and cross-products that are used to estimate the
        autocovariance. When using "conservative", n is set to the number of
        non-missing observations.
    result_object : bool, optional
        Flag indicating whether to return the results as an ``AcfResult``
        instead of a plain tuple. ``AcfResult`` always carries all four
        fields, matching the legacy tuple's contents only when both
        ``qstat`` is True and ``alpha`` is not None; that combination is
        always returned, with no warning. Requesting only one of
        ``qstat`` or ``alpha`` still returns the shorter legacy tuple by
        default and issues a ``FutureWarning``, because ``AcfResult``
        would change how many values are returned. Ignored when
        ``qstat`` is False and ``alpha`` is None, since ``acf`` returns a
        single array in that case.

        .. deprecated:: 0.15.0

            In release 0.16.0 or after July 2027, whichever is later, the
            default will change to always return an ``AcfResult``. Set
            ``result_object=True`` to opt in now, or
            ``result_object=False`` to silence the warning and keep the
            current return type.

    Returns
    -------
    acf : ndarray
        The autocorrelation function for lags 0, 1, ..., nlags. Shape
        (nlags+1,). Returned directly (not part of a tuple) unless
        ``qstat`` is True or ``alpha`` is not None.
    AcfResult
        A result object with fields ``acf``, ``confint``, ``qstat`` and
        ``pvalues`` (each of the latter three is ``None`` when it was not
        computed). See :class:`~statsmodels.tsa.stattools.AcfResult`.

        This is returned whenever ``result_object=True``. It is also
        returned by default when both ``qstat`` is True and ``alpha`` is
        not None, because ``AcfResult`` then has exactly the same four
        values as the legacy tuple; that case is adopted silently.
        Requesting only one of ``qstat`` or ``alpha`` still returns the
        shorter legacy tuple below and warns, since ``AcfResult`` would
        change how many values are returned.
    confint : ndarray, optional
        Confidence intervals for the ACF at lags 0, 1, ..., nlags. Shape
        (nlags + 1, 2). Returned (as part of a plain tuple, the
        deprecated default) if alpha is not None. The confidence
        intervals are centered on the estimated ACF values. This behavior
        differs from plot_acf which centers the confidence intervals on 0.
    qstat : ndarray, optional
        The Ljung-Box Q-Statistic for lags 1, 2, ..., nlags (excludes lag
        zero). Returned (as part of a plain tuple, the deprecated
        default) if qstat is True.
    pvalues : ndarray, optional
        The p-values associated with the Q-statistics for lags 1, 2, ...,
        nlags (excludes lag zero). Returned (as part of a plain tuple,
        the deprecated default) if qstat is True.

    Notes
    -----
    The acf at lag 0 (ie., 1) is returned.

    For very long time series it is recommended to use fft convolution instead.
    When fft is False uses a simple, direct estimator of the autocovariances
    that only computes the first nlag + 1 values. This can be much faster when
    the time series is long and only a small number of autocovariances are
    needed.

    If adjusted is true, the denominator for the autocovariance is adjusted
    for the loss of data.

    References
    ----------
    .. [1] Parzen, E., 1963. On spectral analysis with missing observations
       and amplitude modulation. Sankhya: The Indian Journal of
       Statistics, Series A, pp.383-392.
    .. [2] Brockwell and Davis, 1987. Time Series Theory and Methods
    .. [3] Brockwell and Davis, 2010. Introduction to Time Series and
       Forecasting, 2nd edition.

    See Also
    --------
    statsmodels.tsa.stattools.acf
        Estimate the autocorrelation function.
    statsmodels.graphics.tsaplots.plot_acf
        Plot autocorrelations and confidence intervals.
    """
    adjusted = bool_like(adjusted, "adjusted")
    nlags = int_like(nlags, "nlags", optional=True)
    qstat = bool_like(qstat, "qstat")
    fft = bool_like(fft, "fft", optional=False)
    alpha = float_like(alpha, "alpha", optional=True)
    result_object = bool_like(result_object, "result_object", optional=True)
    missing = string_like(
        missing, "missing", options=("none", "raise", "conservative", "drop")
    )
    x = array_like(x, "x")
    nobs = x.shape[0]
    if nlags is None:
        nlags = min(int(10 * np.log10(nobs)), nobs - 1)
    if missing in ("drop", "conservative"):
        # "drop" removes the NaNs and "conservative" zeroes them out, and
        # acovf uses the non-missing count as the divisor in both cases
        # (see the missing docstring), so nobs must match.
        nobs = int(np.sum(~np.isnan(x)))
        if nobs == 0:
            raise ValueError("All observations are missing after dropping.")

    avf = acovf(x, adjusted=adjusted, demean=True, fft=fft, missing=missing)
    acf = avf[: nlags + 1] / avf[0]

    confint = None
    if alpha is not None:
        if bartlett_confint:
            varacf = np.ones_like(acf) / nobs
            varacf[0] = 0
            varacf[1] = 1.0 / nobs
            varacf[2:] *= 1 + 2 * np.cumsum(acf[1:-1] ** 2)
        else:
            varacf = 1.0 / nobs
        interval = stats.norm.ppf(1 - alpha / 2.0) * np.sqrt(varacf)
        confint = np.array(lzip(acf - interval, acf + interval))

    qstat_vals = pvalue = None
    if qstat:
        qstat_vals, pvalue = q_stat(acf[1:], nobs=nobs)  # drop lag 0

    # AcfResult always carries all four fields, matching the legacy tuple's
    # contents only when both qstat and alpha were requested; in that case
    # it is adopted silently.  Requesting just one of the two still returns
    # the shorter legacy tuple and warns.  Only the qstat/alpha paths return
    # more than one value today, so the single-output path stays quiet --
    # warning there would fire for every internal use of acf.
    same_arity_as_legacy = qstat and alpha is not None
    if (
        result_object is None
        and (qstat or alpha is not None)
        and not same_arity_as_legacy
    ):
        warnings.warn(
            "acf currently returns a plain tuple whose length depends on "
            "the qstat and alpha arguments. In release 0.16 or after "
            "July 2027, whichever is later, the default behavior will "
            "switch to always returning an AcfResult. Set "
            "result_object=True to switch now, or result_object=False "
            "to keep the current behavior and silence this warning.",
            FutureWarning,
            stacklevel=2,
        )
    if result_object or same_arity_as_legacy:
        return AcfResult(acf, confint, qstat_vals, pvalue)

    if not (qstat or alpha):
        return acf
    elif not qstat:
        return acf, confint
    return acf, qstat_vals, pvalue


def pacf_yw(
    x: ArrayLike1D,
    nlags: int | None = None,
    method: Literal["adjusted", "mle"] = "adjusted",
) -> np.ndarray:
    """
    Partial autocorrelation estimated with non-recursive yule_walker

    Parameters
    ----------
    x : array_like
        The observations of time series for which pacf is calculated.
    nlags : int, optional
        Number of lags to return autocorrelation for. If not provided,
        uses min(10 * np.log10(nobs), nobs - 1).
    method : {"adjusted", "mle"}, optional
        The method for the autocovariance calculations in yule walker.

    Returns
    -------
    ndarray
        The partial autocorrelations, maxlag+1 elements.

    See Also
    --------
    statsmodels.tsa.stattools.pacf
        Partial autocorrelation estimation.
    statsmodels.tsa.stattools.pacf_ols
        Partial autocorrelation estimation using OLS.
    statsmodels.tsa.stattools.pacf_burg
        Partial autocorrelation estimation using Burg's method.

    Notes
    -----
    This solves yule_walker for each desired lag and contains
    currently duplicate calculations.
    """
    x = array_like(x, "x")
    nlags = int_like(nlags, "nlags", optional=True)
    nobs = x.shape[0]
    if nlags is None:
        nlags = max(min(int(10 * np.log10(nobs)), nobs - 1), 1)
    method = string_like(method, "method", options=("adjusted", "mle"))
    pacf = [1.0]
    with warnings.catch_warnings():
        warnings.simplefilter("once", ValueWarning)
        pacf.extend(
            [
                yule_walker(x, k, method=method, result_object=False)[0][-1]
                for k in range(1, nlags + 1)
            ]
        )
    return np.array(pacf)


class PacfBurgResult(NamedTuple):
    """
    Result of :func:`pacf_burg`.

    Parameters
    ----------
    pacf : ndarray
        Partial autocorrelations for lags 0, 1, ..., nlag.
    sigma2 : ndarray
        Residual variance estimates where the value in position m is the
        residual variance in an AR model that includes m lags.
    """

    pacf: np.ndarray
    sigma2: np.ndarray


def pacf_burg(
    x: ArrayLike1D, nlags: int | None = None, demean: bool = True
) -> PacfBurgResult:
    """
    Calculate Burg's partial autocorrelation estimator

    Parameters
    ----------
    x : array_like
        Observations of time series for which pacf is calculated.
    nlags : int, optional
        Number of lags to return autocorrelation for. If not provided,
        uses min(10 * np.log10(nobs), nobs - 1).
    demean : bool, optional
        Flag indicating to demean that data. Set to False if x has been
        previously demeaned.

    Returns
    -------
    PacfBurgResult
        A result object with fields:

        pacf : ndarray
            Partial autocorrelations for lags 0, 1, ..., nlag.
        sigma2 : ndarray
            Residual variance estimates where the value in position m is
            the residual variance in an AR model that includes m lags.

    See Also
    --------
    statsmodels.tsa.stattools.pacf
        Partial autocorrelation estimation.
    statsmodels.tsa.stattools.pacf_yw
         Partial autocorrelation estimation using Yule-Walker.
    statsmodels.tsa.stattools.pacf_ols
        Partial autocorrelation estimation using OLS.

    References
    ----------
    .. [1] Brockwell, P.J. and Davis, R.A., 2016. Introduction to time series
        and forecasting. Springer.
    """
    x = array_like(x, "x")
    if demean:
        x = x - x.mean()
    nobs = x.shape[0]
    p = nlags if nlags is not None else min(int(10 * np.log10(nobs)), nobs - 1)
    p = max(p, 1)
    if p > nobs - 1:
        raise ValueError("nlags must be smaller than nobs - 1")
    d = np.zeros(p + 1)
    d[0] = 2 * x.dot(x)
    pacf = np.zeros(p + 1)
    u = x[::-1].copy()
    v = x[::-1].copy()
    d[1] = u[:-1].dot(u[:-1]) + v[1:].dot(v[1:])
    pacf[1] = 2 / d[1] * v[1:].dot(u[:-1])
    last_u = np.empty_like(u)
    last_v = np.empty_like(v)
    for i in range(1, p):
        last_u[:] = u
        last_v[:] = v
        u[1:] = last_u[:-1] - pacf[i] * last_v[1:]
        v[1:] = last_v[1:] - pacf[i] * last_u[:-1]
        d[i + 1] = (1 - pacf[i] ** 2) * d[i] - v[i] ** 2 - u[-1] ** 2
        pacf[i + 1] = 2 / d[i + 1] * v[i + 1 :].dot(u[i:-1])
    sigma2 = (1 - pacf**2) * d / (2.0 * (nobs - np.arange(0, p + 1)))
    pacf[0] = 1  # Insert the 0 lag partial autocorrel

    return PacfBurgResult(pacf, sigma2)


def pacf_ols(
    x: ArrayLike1D,
    nlags: int | None = None,
    efficient: bool = True,
    adjusted: bool = False,
) -> np.ndarray:
    """
    Calculate partial autocorrelations via OLS

    Parameters
    ----------
    x : array_like
        Observations of time series for which pacf is calculated.
    nlags : int, optional
        Number of lags to return autocorrelation for. If not provided,
        uses min(10 * np.log10(nobs), nobs - 1).
    efficient : bool, optional
        If true, uses the maximum number of available observations to compute
        each partial autocorrelation. If not, uses the same number of
        observations to compute all pacf values.
    adjusted : bool, optional
        Adjust each partial autocorrelation by n / (n - lag).

    Returns
    -------
    ndarray
        The partial autocorrelations, (maxlag,) array corresponding to lags
        0, 1, ..., maxlag.

    See Also
    --------
    statsmodels.tsa.stattools.pacf
        Partial autocorrelation estimation.
    statsmodels.tsa.stattools.pacf_yw
         Partial autocorrelation estimation using Yule-Walker.
    statsmodels.tsa.stattools.pacf_burg
        Partial autocorrelation estimation using Burg's method.

    Notes
    -----
    This solves a separate OLS estimation for each desired lag using method in
    [1]_. Setting efficient to True has two effects. First, it uses
    `nobs - lag` observations of estimate each pacf.  Second, it re-estimates
    the mean in each regression. If efficient is False, then the data are first
    demeaned, and then `nobs - maxlag` observations are used to estimate each
    partial autocorrelation.

    The inefficient estimator appears to have better finite sample properties.
    This option should only be used in time series that are covariance
    stationary.

    OLS estimation of the pacf does not guarantee that all pacf values are
    between -1 and 1.

    References
    ----------
    .. [1] Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
       Time series analysis: forecasting and control. John Wiley & Sons, p. 66
    """
    x = array_like(x, "x")
    nlags = int_like(nlags, "nlags", optional=True)
    efficient = bool_like(efficient, "efficient")
    adjusted = bool_like(adjusted, "adjusted")
    nobs = x.shape[0]
    if nlags is None:
        nlags = max(min(int(10 * np.log10(nobs)), nobs // 2), 1)
    if nlags > nobs // 2:
        raise ValueError(f"nlags must be smaller than nobs // 2 ({nobs // 2})")
    pacf = np.empty(nlags + 1)
    pacf[0] = 1.0
    if efficient:
        _lagmat_result = lagmat(x, nlags, original="sep")
        xlags, x0 = _lagmat_result.lags, _lagmat_result.leads
        xlags = add_constant(xlags)
        for k in range(1, nlags + 1):
            params = np.linalg.lstsq(xlags[k:, : k + 1], x0[k:], rcond=None)[0]
            pacf[k] = np.squeeze(params[-1])
    else:
        x = x - np.mean(x)
        # Create a single set of lags for multivariate OLS
        _lagmat_result = lagmat(x, nlags, original="sep", trim="both")
        xlags, x0 = _lagmat_result.lags, _lagmat_result.leads
        for k in range(1, nlags + 1):
            params = np.linalg.lstsq(xlags[:, :k], x0, rcond=None)[0]
            # Last coefficient corresponds to PACF value (see [1])
            pacf[k] = np.squeeze(params[-1])
    if adjusted:
        pacf *= nobs / (nobs - np.arange(nlags + 1))
    return pacf


class PacfResult(NamedTuple):
    """
    Result of :func:`pacf`.

    Parameters
    ----------
    pacf : ndarray
        Partial autocorrelations for lags 0, 1, ..., nlags.
    confint : ndarray or None
        Confidence intervals for the PACF at lags 0, 1, ..., nlags. Shape
        (nlags + 1, 2). ``None`` when ``alpha`` is None.
    """

    pacf: np.ndarray
    confint: np.ndarray | None


def pacf(
    x: ArrayLike1D,
    nlags: int | None = None,
    method: Literal[
        "yw",
        "ywa",
        "ywadjusted",
        "yw_adjusted",
        "ywm",
        "ywmle",
        "yw_mle",
        "ols",
        "ols-inefficient",
        "ols-adjusted",
        "ld",
        "lda",
        "ldadjusted",
        "ld_adjusted",
        "ldb",
        "ldbiased",
        "ld_biased",
        "burg",
    ] = "ywadjusted",
    alpha: float | None = None,
    *,
    result_object: bool | None = None,
) -> np.ndarray | PacfResult:
    """
    Partial autocorrelation estimate

    Parameters
    ----------
    x : array_like
        Observations of time series for which pacf is calculated.
    nlags : int, optional
        Number of lags to return autocorrelation for. If not provided,
        uses min(10 * np.log10(nobs), nobs // 2 - 1). The returned value
        includes lag 0 (ie., 1) so size of the pacf vector is (nlags + 1,).
    method : str, optional
        Specifies which method for the calculations to use.

        - "yw", "ywa", "ywadjusted" or "yw_adjusted" : Yule-Walker with
          sample-size adjustment in denominator for acovf. Default.
        - "ywm", "ywmle" or "yw_mle" : Yule-Walker without adjustment.
        - "ols" : regression of time series on lags of it and on constant.
        - "ols-inefficient" : regression of time series on lags using a single
          common sample to estimate all pacf coefficients.
        - "ols-adjusted" : regression of time series on lags with a bias
          adjustment.
        - "ld", "lda", "ldadjusted" or "ld_adjusted" : Levinson-Durbin
          recursion with bias correction.
        - "ldb", "ldbiased" or "ld_biased" : Levinson-Durbin recursion
          without bias correction.
        - "burg" :  Burg's partial autocorrelation estimator.

    alpha : float, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        1/sqrt(len(x)).
    result_object : bool, optional
        Flag controlling whether a :class:`PacfResult` is returned. When
        ``alpha`` is not None a :class:`PacfResult` is always returned; it
        holds the same two values as the legacy tuple it replaces. When
        ``alpha`` is None a bare array is returned unless
        ``result_object=True``, which additionally yields a
        :class:`PacfResult` with ``confint`` set to ``None``.

    Returns
    -------
    PacfResult or ndarray
        When ``alpha`` is not None (or ``result_object=True``), a
        result object with fields:

        pacf : ndarray
            The partial autocorrelations for lags 0, 1, ..., nlags. Shape
            (nlags+1,).
        confint : ndarray or None
            Confidence intervals for the PACF at lags 0, 1, ..., nlags.
            Shape (nlags + 1, 2). ``None`` when ``alpha`` is None.

        See :class:`~statsmodels.tsa.stattools.PacfResult`.

        When ``alpha`` is None a bare ndarray of partial autocorrelations
        is returned instead.

    See Also
    --------
    statsmodels.tsa.stattools.acf
        Estimate the autocorrelation function.
    statsmodels.tsa.stattools.pacf_yw
         Partial autocorrelation estimation using Yule-Walker.
    statsmodels.tsa.stattools.pacf_ols
        Partial autocorrelation estimation using OLS.
    statsmodels.tsa.stattools.pacf_burg
        Partial autocorrelation estimation using Burg's method.
    statsmodels.graphics.tsaplots.plot_pacf
        Plot partial autocorrelations and confidence intervals.

    Notes
    -----
    Based on simulation evidence across a range of low-order ARMA models,
    the best methods based on root MSE are Yule-Walker (MLW), Levinson-Durbin
    (MLE) and Burg, respectively. The estimators with the lowest bias included
    included these three in addition to OLS and OLS-adjusted.

    Yule-Walker (adjusted) and Levinson-Durbin (adjusted) performed
    consistently worse than the other options.
    """
    result_object = bool_like(result_object, "result_object", optional=True)
    nlags = int_like(nlags, "nlags", optional=True)
    methods = (
        "ols",
        "ols-inefficient",
        "ols-adjusted",
        "yw",
        "ywa",
        "ld",
        "ywadjusted",
        "yw_adjusted",
        "ywm",
        "ywmle",
        "yw_mle",
        "lda",
        "ldadjusted",
        "ld_adjusted",
        "ldb",
        "ldbiased",
        "ld_biased",
        "burg",
    )
    x = array_like(x, "x", maxdim=2)
    method = string_like(method, "method", options=methods)
    alpha = float_like(alpha, "alpha", optional=True)

    nobs = x.shape[0]
    if nlags is None:
        nlags = min(int(10 * np.log10(nobs)), nobs // 2 - 1)
    nlags = max(nlags, 1)
    if nlags > x.shape[0] // 2:
        raise ValueError(
            "Can only compute partial correlations for lags up to 50% of the "
            f"sample size. The requested nlags {nlags} must be < "
            f"{x.shape[0] // 2}."
        )
    if method in ("ols", "ols-inefficient", "ols-adjusted"):
        efficient = "inefficient" not in method
        adjusted = "adjusted" in method
        ret = pacf_ols(x, nlags=nlags, efficient=efficient, adjusted=adjusted)
    elif method in ("yw", "ywa", "ywadjusted", "yw_adjusted"):
        ret = pacf_yw(x, nlags=nlags, method="adjusted")
    elif method in ("ywm", "ywmle", "yw_mle"):
        ret = pacf_yw(x, nlags=nlags, method="mle")
    elif method in ("ld", "lda", "ldadjusted", "ld_adjusted"):
        acv = acovf(x, adjusted=True, fft=False)
        ret = levinson_durbin(acv, nlags=nlags, isacov=True).pacf
    elif method == "burg":
        ret = pacf_burg(x, nlags=nlags, demean=True).pacf
    # inconsistent naming with ywmle
    else:  # method in ("ldb", "ldbiased", "ld_biased")
        acv = acovf(x, adjusted=False, fft=False)
        ret = levinson_durbin(acv, nlags=nlags, isacov=True).pacf
    confint = None
    if alpha is not None:
        varacf = 1.0 / len(x)  # for all lags >=1
        interval = stats.norm.ppf(1.0 - alpha / 2.0) * np.sqrt(varacf)
        confint = np.array(lzip(ret - interval, ret + interval))
        confint[0] = ret[0]  # fix confidence interval for lag 0 to varpacf=0

    # PacfResult is always used when alpha is not None.  When alpha is
    # None a bare array is returned, as before; pass result_object=True
    # to always get a PacfResult.
    if result_object or alpha is not None:
        return PacfResult(ret, confint)
    return ret


def ccovf(x, y, adjusted=True, demean=True, fft=True):
    """
    Calculate the cross-covariance between two series

    Parameters
    ----------
    x, y : array_like
       The time series data to use in the calculation.
    adjusted : bool, optional
       If True, then denominators for cross-covariance are the number of
       overlapping observations at each lag k, min(m, n-k), otherwise n.
    demean : bool, optional
        Flag indicating whether to demean x and y.
    fft : bool, optional
        If True, use FFT convolution.  This method should be preferred
        for long time series.

    Returns
    -------
    ndarray
        The estimated cross-covariance function: the element at index k
        is the covariance between {x[k], x[k+1], ..., x[n]} and
        {y[0], y[1], ..., y[m-k]}, where n and m are the lengths of x and y,
        respectively.
    """
    x = array_like(x, "x")
    y = array_like(y, "y")
    adjusted = bool_like(adjusted, "adjusted")
    demean = bool_like(demean, "demean")
    fft = bool_like(fft, "fft", optional=False)

    n = len(x)
    m = len(y)
    if demean:
        xo = x - x.mean()
        yo = y - y.mean()
    else:
        xo = x
        yo = y
    if adjusted:
        d = np.minimum(np.arange(n, 0, -1), m)
    else:
        d = n

    method = "fft" if fft else "direct"
    # When y is shorter than x, the denominator counts must follow len(y)
    # to match the actual number of overlapping observations at each lag.
    return correlate(xo, yo, "full", method=method)[m - 1 :] / d


class CcfResult(NamedTuple):
    """
    Result of :func:`ccf`.

    Parameters
    ----------
    ccf : ndarray
        The cross-correlation function of x and y: the element at index k
        is the correlation between {x[k], x[k+1], ..., x[n]} and
        {y[0], y[1], ..., y[m-k]}, where n and m are the lengths of x and
        y, respectively.
    confint : ndarray or None
        Confidence intervals for the CCF at lags 0, 1, ..., nlags-1. Shape
        (nlags, 2). ``None`` when ``alpha`` is None.
    """

    ccf: np.ndarray
    confint: np.ndarray | None


def ccf(
    x,
    y,
    adjusted=True,
    fft=True,
    *,
    nlags=None,
    alpha=None,
    result_object: bool | None = None,
):
    """
    The cross-correlation function

    Parameters
    ----------
    x, y : array_like
        The time series data to use in the calculation.
    adjusted : bool, optional
        If True, then denominators for cross-covariance are the number of
        overlapping observations at each lag k, min(m, n-k), otherwise n.
    fft : bool, optional
        If True, use FFT convolution.  This method should be preferred
        for long time series.
    nlags : int, optional
        Number of lags to return cross-correlations for. If not provided,
        the number of lags equals len(x).
    alpha : float, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        1/sqrt(len(x)).
    result_object : bool, optional
        Flag controlling whether a :class:`CcfResult` is returned. When
        ``alpha`` is not None a :class:`CcfResult` is always returned; it
        holds the same two values as the legacy tuple it replaces. When
        ``alpha`` is None a bare array is returned unless
        ``result_object=True``, which additionally yields a
        :class:`CcfResult` with ``confint`` set to ``None``.

    Returns
    -------
    CcfResult or ndarray
        When ``alpha`` is not None (or ``result_object=True``), a
        result object with fields:

        ccf : ndarray
            The cross-correlation function of x and y: the element at
            index k is the correlation between {x[k], x[k+1], ..., x[n]}
            and {y[0], y[1], ..., y[m-k]}, where n and m are the lengths
            of x and y, respectively.
        confint : ndarray or None
            Confidence intervals for the CCF at lags 0, 1, ..., nlags-1
            using the level given by alpha and the standard deviation
            calculated as 1/sqrt(len(x)) [1]_. Shape (nlags, 2). ``None``
            when ``alpha`` is None.

        See :class:`~statsmodels.tsa.stattools.CcfResult`.

        When ``alpha`` is None a bare ndarray of cross-correlations is
        returned instead.

    See Also
    --------
    statsmodels.tsa.stattools.pccf
        Partial cross-correlation function.
    statsmodels.tsa.stattools.acf
        Autocorrelation function.
    statsmodels.tsa.stattools.pacf
        Partial autocorrelation function.
    statsmodels.graphics.tsaplots.plot_ccf
        Plot cross-correlations and confidence intervals.

    Notes
    -----
    If adjusted is True, the denominator for the cross-correlation is adjusted.

    References
    ----------
    .. [1] Brockwell and Davis, 2016. Introduction to Time Series and
       Forecasting, 3rd edition, p. 242.
    """
    x = array_like(x, "x")
    y = array_like(y, "y")
    adjusted = bool_like(adjusted, "adjusted")
    fft = bool_like(fft, "fft", optional=False)
    result_object = bool_like(result_object, "result_object", optional=True)

    cvf = ccovf(x, y, adjusted=adjusted, demean=True, fft=fft)
    ret = cvf / (np.std(x) * np.std(y))
    ret = ret[:nlags]

    confint = None
    if alpha is not None:
        interval = stats.norm.ppf(1.0 - alpha / 2.0) / np.sqrt(len(x))
        confint = ret.reshape(-1, 1) + interval * np.array([-1, 1])

    # CcfResult is always used when alpha is not None.  When alpha is
    # None a bare array is returned, as before; pass result_object=True
    # to always get a CcfResult.
    if result_object or alpha is not None:
        return CcfResult(ret, confint)
    return ret


def _pccf_yw(x, y, nlags, adjusted=False):
    """PCCF via multivariate Levinson-Durbin recursion."""
    nobs = len(x)
    xo = x - x.mean()
    yo = y - y.mean()

    gamma = np.empty((nlags + 1, 2, 2))
    for h in range(nlags + 1):
        if h == 0:
            gamma[h] = [[xo.dot(xo), yo.dot(xo)], [xo.dot(yo), yo.dot(yo)]]
        else:
            gamma[h] = [
                [xo[h:].dot(xo[:-h]), yo[h:].dot(xo[:-h])],
                [xo[h:].dot(yo[:-h]), yo[h:].dot(yo[:-h])],
            ]
        gamma[h] /= nobs - h if adjusted else nobs

    if np.linalg.matrix_rank(gamma[0]) < 2:
        return np.full(nlags, np.nan)

    sig_f = gamma[0].copy()
    sig_b = gamma[0].copy()

    phi_prev = [None] * (nlags + 1)
    psi_prev = [None] * (nlags + 1)

    pccf_vals = np.empty(nlags)

    for s in range(1, nlags + 1):
        delta_f = gamma[s].copy()
        delta_b = gamma[s].T.copy()
        for j in range(1, s):
            delta_f -= phi_prev[j] @ gamma[s - j]
            delta_b -= psi_prev[j] @ gamma[s - j].T

        v_f = sig_f[0, 0]
        v_b = sig_b[1, 1]
        if not np.isfinite(v_f) or not np.isfinite(v_b):
            pccf_vals[s - 1 :] = np.nan
            break
        tol = np.finfo(np.float64).eps * gamma[0].trace()
        if v_f <= tol or v_b <= tol:
            pccf_vals[s - 1 :] = np.nan
            break

        d_f = np.sqrt(v_f)
        d_b = np.sqrt(v_b)

        pccf_vals[s - 1] = delta_f[0, 1] / (d_f * d_b)

        try:
            phi_ss = np.linalg.solve(sig_b.T, delta_f.T).T
            psi_ss = np.linalg.solve(sig_f.T, delta_b.T).T
        except np.linalg.LinAlgError:
            pccf_vals[s - 1 :] = np.nan
            break

        phi_new = [None] * (nlags + 1)
        psi_new = [None] * (nlags + 1)
        phi_new[s] = phi_ss
        psi_new[s] = psi_ss
        for j in range(1, s):
            phi_new[j] = phi_prev[j] - phi_ss @ psi_prev[s - j]
            psi_new[j] = psi_prev[j] - psi_ss @ phi_prev[s - j]

        sig_f = sig_f - phi_ss @ delta_b
        sig_b = sig_b - psi_ss @ delta_f
        phi_prev = phi_new
        psi_prev = psi_new

    return pccf_vals


def _pccf_ols(x, y, nlags):
    """PCCF via OLS regression on intervening observations."""
    nobs = len(x)
    pccf_vals = []

    for h in range(1, nlags + 1):
        if h == 1:
            resid_x = x[: nobs - 1]
            resid_y = y[1:nobs]
        else:
            indices = np.arange(0, nobs - h)
            n_carriers = 2 * (h - 1) + 1
            if len(indices) <= n_carriers:
                pccf_vals.append(np.nan)
                continue

            carriers = add_constant(
                np.column_stack(
                    [x[indices + j] for j in range(1, h)]
                    + [y[indices + j] for j in range(1, h)]
                )
            )
            targets = np.column_stack([x[indices], y[indices + h]])
            coeffs = np.linalg.lstsq(carriers, targets, rcond=None)[0]
            resids = targets - carriers.dot(coeffs)
            resid_x = resids[:, 0]
            resid_y = resids[:, 1]

        if resid_x.size < 2 or np.ptp(resid_x) == 0 or np.ptp(resid_y) == 0:
            pccf_vals.append(np.nan)
        else:
            pccf_vals.append(np.corrcoef(resid_x, resid_y)[0, 1])

    return np.array(pccf_vals)


class PccfResult(NamedTuple):
    """
    Result of :func:`pccf`.

    Parameters
    ----------
    pccf : ndarray
        The partial cross-correlation function for lags 1, 2, ..., nlags.
    confint : ndarray or None
        Confidence intervals for the PCCF at lags 1, 2, ..., nlags. Shape
        (nlags, 2). ``None`` when ``alpha`` is None.
    """

    pccf: np.ndarray
    confint: np.ndarray | None


def pccf(
    x: ArrayLike1D,
    y: ArrayLike1D,
    *,
    nlags: int | None = None,
    method: Literal[
        "yw",
        "ywa",
        "ywadjusted",
        "yw_adjusted",
        "ywm",
        "ywmle",
        "yw_mle",
        "ols",
    ] = "ywm",
    alpha: float | None = None,
    result_object: bool | None = None,
) -> np.ndarray | PccfResult:
    """
    Partial Cross-Correlation Function

    Computes the (1,2) element of the partial lag correlation
    matrix P(s) for the bivariate series (x, y) at lags
    1, 2, ..., nlags. At lag h, this is the correlation between
    x_t and y_{t+h} after removing the linear dependence on all
    intervening observations of both series.

    Parameters
    ----------
    x, y : array_like
        The time series data to use in the calculation.
    nlags : int, optional
        Number of lags to return partial cross-correlations for.
        If not provided, uses
        min(10 * np.log10(nobs), nobs // 2 - 1).
    method : str, optional
        Specifies which method for the calculations to use.

        - "ywm", "ywmle" or "yw_mle" : Yule-Walker via the
          multivariate Levinson-Durbin recursion without sample-size
          adjustment in the autocovariance denominator. Default.
        - "yw", "ywa", "ywadjusted" or "yw_adjusted" : Yule-Walker
          via the multivariate Levinson-Durbin recursion with
          sample-size adjustment in the autocovariance denominator.
        - "ols" : OLS regression of x_t and y_{t+h} on all
          intervening observations.
    alpha : float, optional
        If a number is given, the confidence intervals for the
        given level are returned. For instance if alpha=.05,
        95 % confidence intervals are returned where the standard
        deviation is 1/sqrt(n).
    result_object : bool, optional
        Flag controlling whether a :class:`PccfResult` is returned. When
        ``alpha`` is not None a :class:`PccfResult` is always returned;
        it holds the same two values as the legacy tuple it replaces.
        When ``alpha`` is None a bare array is returned unless
        ``result_object=True``, which additionally yields a
        :class:`PccfResult` with ``confint`` set to ``None``.

    Returns
    -------
    PccfResult or ndarray
        When ``alpha`` is not None (or ``result_object=True``), a
        result object with fields:

        pccf : ndarray
            The partial cross-correlation function for lags
            1, 2, ..., nlags.
        confint : ndarray or None
            Confidence intervals for the PCCF at lags 1, 2, ..., nlags
            using the level given by alpha. Shape (nlags, 2). ``None``
            when ``alpha`` is None.

        See :class:`~statsmodels.tsa.stattools.PccfResult`.

        When ``alpha`` is None a bare ndarray of partial
        cross-correlations is returned instead.

    See Also
    --------
    statsmodels.tsa.stattools.ccf
        Cross-correlation function.
    statsmodels.tsa.stattools.acf
        Autocorrelation function.
    statsmodels.tsa.stattools.pacf
        Partial autocorrelation function.
    statsmodels.graphics.tsaplots.plot_pccf
        Plot partial cross-correlations and confidence intervals.

    Notes
    -----
    The PCCF at lag h is the (1,2) element of the partial lag
    correlation matrix P(h) defined by Heyse and Wei (1985) [2]_.

    pccf(x, y) is not symmetric: pccf(x, y) != pccf(y, x) in
    general. The (1,2) element measures the partial association
    from x to y, while pccf(y, x) measures the reverse direction.

    The Yule-Walker methods use the multivariate Levinson-Durbin
    recursion on the bivariate autocovariance matrix sequence. For
    stationary series, this is asymptotically equivalent to OLS by
    the Frisch-Waugh-Lovell theorem [3]_, but is O(n * nlags)
    versus O(n * nlags^3) for OLS. The two methods may differ
    substantially on non-stationary (e.g., trending) data.

    The "ols" method computes pccf(h) as the sample correlation
    between the backward residual (x_t regressed on the
    intervening observations) and the forward residual (y_{t+h}
    regressed on the same intervening observations). Both
    regressions include an intercept. For h=1, there are no
    intervening observations, so pccf(1) is the correlation between
    x[:-1] and y[1:].

    For a bivariate VAR(p) process, P(h) = 0 for h > p,
    providing a useful identification tool.

    If the series are perfectly collinear or constant, the sample
    autocovariance matrix is singular and the Yule-Walker methods
    return NaN for all lags.

    The OLS method returns NaN for lags where either residual series
    has zero variance.

    The confidence intervals use the asymptotic standard deviation
    1/sqrt(n), following Wei (2006, Section 11.2) [1]_.

    References
    ----------
    .. [1] Wei, W. W. S. (2006). Time Series Analysis:
       Univariate and Multivariate Methods, 2nd edition.
       Pearson.
    .. [2] Heyse, J. F. and Wei, W. W. S. (1985). Inverse
       and partial lag autocorrelation for vector time series.
       American Statistical Association Proceedings of Business
       and Economic Statistics Section, pp. 233-237.
    .. [3] Frisch, R. and Waugh, F. V. (1933). Partial time
       regressions as compared with individual trends.
       Econometrica, 1(4), 387-401.

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.tsa.stattools import pccf
    >>> rng = np.random.default_rng(12345)
    >>> x = rng.standard_normal(100)
    >>> y = 0.5 * x + rng.standard_normal(100)
    >>> result = pccf(x, y, nlags=5)
    >>> result_with_ci = pccf(x, y, nlags=5, alpha=0.05, result_object=True)
    >>> result_with_ci.pccf.shape
    (5,)
    >>> result_with_ci.confint.shape
    (5, 2)
    """
    x = array_like(x, "x")
    y = array_like(y, "y")
    nlags = int_like(nlags, "nlags", optional=True)
    methods = (
        "ols",
        "yw",
        "ywa",
        "ywadjusted",
        "yw_adjusted",
        "ywm",
        "ywmle",
        "yw_mle",
    )
    method = string_like(method, "method", options=methods)
    alpha = float_like(alpha, "alpha", optional=True)
    result_object = bool_like(result_object, "result_object", optional=True)

    nobs = len(x)
    if len(y) != nobs:
        raise ValueError("x and y must have the same length")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise MissingDataError("x and y must not contain NaN or inf values")

    if nlags is None:
        nlags = min(int(10 * np.log10(nobs)), nobs // 2 - 1)
        nlags = max(nlags, 1)
    elif nlags <= 0:
        raise ValueError("nlags must be a positive integer")
    if nlags > nobs // 2:
        raise ValueError(
            "Can only compute partial correlations for lags up to 50% "
            f"of the sample size. The requested nlags {nlags} must be "
            f"<= {nobs // 2}."
        )

    if method == "ols":
        ret = _pccf_ols(x, y, nlags)
    else:
        adjusted = method in ("yw", "ywa", "ywadjusted", "yw_adjusted")
        ret = _pccf_yw(x, y, nlags, adjusted=adjusted)

    confint = None
    if alpha is not None:
        interval = stats.norm.ppf(1.0 - alpha / 2.0) / np.sqrt(nobs)
        confint = ret.reshape(-1, 1) + interval * np.array([-1, 1])

    # PccfResult is always used when alpha is not None.  When alpha is
    # None a bare array is returned, as before; pass result_object=True
    # to always get a PccfResult.
    if result_object or alpha is not None:
        return PccfResult(ret, confint)
    return ret


# moved from sandbox.tsa.examples.try_ld_nitime, via nitime
# TODO: check what to return, for testing and trying out returns everything
class LevinsonDurbinResult(NamedTuple):
    """
    Result of :func:`levinson_durbin`.

    Parameters
    ----------
    sigma_v : float
        The estimate of the error variance.
    arcoefs : ndarray
        The estimate of the autoregressive coefficients for a model
        including nlags.
    pacf : ndarray
        The partial autocorrelation function.
    sigma : ndarray
        The entire sigma array from intermediate result, last value is
        sigma_v.
    phi : ndarray
        The entire phi array from intermediate result, last column
        contains autoregressive coefficients for AR(nlags).
    """

    sigma_v: float
    arcoefs: np.ndarray
    pacf: np.ndarray
    sigma: np.ndarray
    phi: np.ndarray


def levinson_durbin(s, nlags=10, isacov=False):
    """
    Levinson-Durbin recursion for autoregressive processes

    Parameters
    ----------
    s : array_like
        If isacov is False, then this is the time series. If isacov is true
        then this is interpreted as autocovariance starting with lag 0.
    nlags : int, optional
        The largest lag to include in recursion or order of the autoregressive
        process.
    isacov : bool, optional
        Flag indicating whether the first argument, s, contains the
        autocovariances or the data series.

    Returns
    -------
    LevinsonDurbinResult
        A result object with fields:

        sigma_v : float
            The estimate of the error variance.
        arcoefs : ndarray
            The estimate of the autoregressive coefficients for a model
            including nlags.
        pacf : ndarray
            The partial autocorrelation function.
        sigma : ndarray
            The entire sigma array from intermediate result, last value is
            sigma_v.
        phi : ndarray
            The entire phi array from intermediate result, last column
            contains autoregressive coefficients for AR(nlags).

    Notes
    -----
    This function returns currently all results, but maybe we drop sigma and
    phi from the returns.

    If this function is called with the time series (isacov=False), then the
    sample autocovariance function is calculated with the default options
    (biased, no fft).
    """
    s = array_like(s, "s")
    nlags = int_like(nlags, "nlags")
    isacov = bool_like(isacov, "isacov")

    order = nlags

    if isacov:
        sxx_m = s
    else:
        sxx_m = acovf(s, fft=False)[: order + 1]  # not tested

    phi = np.zeros((order + 1, order + 1), "d")
    sig = np.zeros(order + 1)
    # initial points for the recursion
    phi[1, 1] = sxx_m[1] / sxx_m[0]
    sig[1] = sxx_m[0] - phi[1, 1] * sxx_m[1]
    for k in range(2, order + 1):
        phi[k, k] = (sxx_m[k] - np.dot(phi[1:k, k - 1], sxx_m[1:k][::-1])) / sig[k - 1]
        for j in range(1, k):
            phi[j, k] = phi[j, k - 1] - phi[k, k] * phi[k - j, k - 1]
        sig[k] = sig[k - 1] * (1 - phi[k, k] ** 2)

    sigma_v = sig[-1]
    arcoefs = phi[1:, -1]
    pacf_ = np.diag(phi).copy()
    pacf_[0] = 1.0
    return LevinsonDurbinResult(sigma_v, arcoefs, pacf_, sig, phi)  # return everything


class LevinsonDurbinPacfResult(NamedTuple):
    """
    Result of :func:`levinson_durbin_pacf`.

    Parameters
    ----------
    arcoefs : ndarray
        AR coefficients computed from the partial autocorrelations.
    acf : ndarray
        The acf computed from the partial autocorrelations. Contains the
        autocorrelations corresponding to lags 0, 1, ..., p.
    """

    arcoefs: np.ndarray
    acf: np.ndarray


def levinson_durbin_pacf(pacf, nlags=None):
    """
    Levinson-Durbin algorithm that returns the acf and ar coefficients

    Parameters
    ----------
    pacf : array_like
        Partial autocorrelation array for lags 0, 1, ... p.
    nlags : int, optional
        Number of lags in the AR model.  If omitted, returns coefficients from
        an AR(p) and the first p autocorrelations.

    Returns
    -------
    LevinsonDurbinPacfResult
        A result object with fields:

        arcoefs : ndarray
            AR coefficients computed from the partial autocorrelations.
        acf : ndarray
            The acf computed from the partial autocorrelations. Array
            returned contains the autocorrelations corresponding to lags
            0, 1, ..., p.

    References
    ----------
    .. [1] Brockwell, P.J. and Davis, R.A., 2016. Introduction to time series
        and forecasting. Springer.
    """
    pacf = array_like(pacf, "pacf")
    nlags = int_like(nlags, "nlags", optional=True)
    pacf = np.squeeze(np.asarray(pacf))

    if pacf[0] != 1:
        raise ValueError(
            "The first entry of the pacf corresponds to lags 0 and so must be 1."
        )
    pacf = pacf[1:]
    n = pacf.shape[0]
    if nlags is not None:
        if nlags > n:
            raise ValueError(
                "Must provide at least as many values from the "
                "pacf as the number of lags."
            )
        pacf = pacf[:nlags]
        n = pacf.shape[0]

    acf = np.zeros(n + 1)
    acf[1] = pacf[0]
    nu = np.cumprod(1 - pacf**2)
    arcoefs = pacf.copy()
    for i in range(1, n):
        prev = arcoefs[: -(n - i)].copy()
        arcoefs[: -(n - i)] = prev - arcoefs[i] * prev[::-1]
        acf[i + 1] = arcoefs[i] * nu[i - 1] + prev.dot(acf[1 : -(n - i)][::-1])
    acf[0] = 1
    return LevinsonDurbinPacfResult(arcoefs, acf)


class BreakvarHeteroskedasticityResult(NamedTuple):
    """
    Result of :func:`breakvar_heteroskedasticity_test`.

    Parameters
    ----------
    statistic : float or ndarray
        Test statistic(s) H(h).
    pvalue : float or ndarray
        p-value(s) of test statistic(s).
    """

    statistic: float | np.ndarray
    pvalue: float | np.ndarray


def breakvar_heteroskedasticity_test(
    resid, subset_length=1 / 3, alternative="two-sided", use_f=True
):
    r"""
    Test for heteroskedasticity of residuals

    Tests whether the sum-of-squares in the first subset of the sample is
    significantly different than the sum-of-squares in the last subset
    of the sample. Analogous to a Goldfeld-Quandt test. The null hypothesis
    is of no heteroskedasticity.

    Parameters
    ----------
    resid : array_like
        Residuals of a time series model.
        The shape is 1d (nobs,) or 2d (nobs, nvars).
    subset_length : int or float, optional
        Length of the subsets to test (h in Notes below).
        If a float in 0 < subset_length < 1, it is interpreted as fraction.
        Default is 1/3.
    alternative : {"increasing", "decreasing", "two-sided"}, optional
        This specifies the alternative for the p-value calculation. Default
        is two-sided.
    use_f : bool, optional
        Whether or not to compare against the asymptotic distribution
        (chi-squared) or the approximate small-sample distribution (F).
        Default is True (i.e., default is to compare against an F
        distribution).

    Returns
    -------
    BreakvarHeteroskedasticityResult
        A result object with fields:

        statistic : float or ndarray
            Test statistic(s) H(h).
        pvalue : float or ndarray
            p-value(s) of test statistic(s).

    Notes
    -----
    The null hypothesis is of no heteroskedasticity. That means different
    things depending on which alternative is selected:

    - Increasing: Null hypothesis is that the variance is not increasing
        throughout the sample; that the sum-of-squares in the later
        subsample is *not* greater than the sum-of-squares in the earlier
        subsample.
    - Decreasing: Null hypothesis is that the variance is not decreasing
        throughout the sample; that the sum-of-squares in the earlier
        subsample is *not* greater than the sum-of-squares in the later
        subsample.
    - Two-sided: Null hypothesis is that the variance is not changing
        throughout the sample. Both that the sum-of-squares in the earlier
        subsample is not greater than the sum-of-squares in the later
        subsample *and* that the sum-of-squares in the later subsample is
        not greater than the sum-of-squares in the earlier subsample.

    For :math:`h = [T/3]`, the test statistic is:

    .. math::

        H(h) = \sum_{t=T-h+1}^T  \tilde v_t^2
        \Bigg / \sum_{t=1}^{h} \tilde v_t^2

    This statistic can be tested against an :math:`F(h,h)` distribution.
    Alternatively, :math:`h H(h)` is asymptotically distributed according
    to :math:`\chi_h^2`; this second test can be applied by passing
    `use_f=False` as an argument.

    See section 5.4 of [1]_ for the above formula and discussion, as well
    as additional details.

    References
    ----------
    .. [1] Harvey, Andrew C. 1990. *Forecasting, Structural Time Series*
            *Models and the Kalman Filter.* Cambridge University Press.
    """
    alternative = string_like(
        alternative,
        "alternative",
        options=("increasing", "decreasing", "two-sided"),
        lower=True,
        deprecated={
            "i": "increasing",
            "inc": "increasing",
            "d": "decreasing",
            "dec": "decreasing",
            "2": "two-sided",
            "2-sided": "two-sided",
        },
        removed_after="0.16",
    )
    squared_resid = np.asarray(resid, dtype=float) ** 2
    if squared_resid.ndim == 1:
        squared_resid = squared_resid.reshape(-1, 1)
    nobs = len(resid)

    if 0 < subset_length < 1:
        h = int(np.round(nobs * subset_length))
    elif type(subset_length) is int and subset_length >= 1:
        h = subset_length

    numer_resid = squared_resid[-h:]
    numer_dof = (~np.isnan(numer_resid)).sum(axis=0)
    numer_squared_sum = np.nansum(numer_resid, axis=0)
    for i, dof in enumerate(numer_dof):
        if dof < 2:
            warnings.warn(
                f"Later subset of data for variable {i:d}"
                " has too few non-missing observations to"
                " calculate test statistic.",
                stacklevel=2,
            )
            numer_squared_sum[i] = np.nan

    denom_resid = squared_resid[:h]
    denom_dof = (~np.isnan(denom_resid)).sum(axis=0)
    denom_squared_sum = np.nansum(denom_resid, axis=0)
    for i, dof in enumerate(denom_dof):
        if dof < 2:
            warnings.warn(
                f"Early subset of data for variable {i:d}"
                " has too few non-missing observations to"
                " calculate test statistic.",
                stacklevel=2,
            )
            denom_squared_sum[i] = np.nan

    test_statistic = numer_squared_sum / denom_squared_sum

    # Under the null the two sums of squares are independent and, divided by
    # the common variance, are chi2 with dfn and dfd degrees of freedom.  The
    # ratio of their *means* is F(dfn, dfd), so H(h) -- a ratio of *sums* --
    # must be rescaled by dfd / dfn before being referred to that
    # distribution.  The two agree when dfn == dfd, which is the usual case;
    # they differ only when missing observations leave the two subsets with
    # different numbers of usable residuals.
    #
    # The chi2 form is the dfd -> oo limit: the denominator sum converges to
    # dfd * sigma**2, so dfd * H(h) -> chi2(dfn).
    if use_f:

        def tail_pvalues(stat, dfn, dfd):
            """Lower- and upper-tail p-values for the statistic."""
            scaled = stat * dfd / dfn
            return stats.f.cdf(scaled, dfn, dfd), stats.f.sf(scaled, dfn, dfd)

    else:

        def tail_pvalues(stat, dfn, dfd):
            """Lower- and upper-tail p-values for the statistic."""
            scaled = stat * dfd
            return stats.chi2.cdf(scaled, dfn), stats.chi2.sf(scaled, dfn)

    # Calculate the one- or two-sided p-values.  Inverting the statistic for
    # the "decreasing" alternative swaps the roles of the two subsets, so the
    # degrees of freedom have to swap with it.
    if alternative == "decreasing":
        test_statistic = 1.0 / test_statistic
        numer_dof, denom_dof = denom_dof, numer_dof
    lower, upper = tail_pvalues(test_statistic, numer_dof, denom_dof)
    if alternative == "two-sided":
        p_value = 2 * np.minimum(lower, upper)
    else:  # "increasing" or "decreasing"
        p_value = upper

    if len(test_statistic) == 1:
        return BreakvarHeteroskedasticityResult(test_statistic[0], p_value[0])

    return BreakvarHeteroskedasticityResult(test_statistic, p_value)


def grangercausalitytests(x, maxlag, addconst=True):
    """
    Four tests for Granger non-causality of 2 time series

    All four tests give similar results. `params_ftest` and `ssr_ftest` are
    equivalent based on F test which is identical to lmtest:grangertest in R.

    Parameters
    ----------
    x : array_like
        The data for testing whether the time series in the second column Granger
        causes the time series in the first column. Missing values are not
        supported.
    maxlag : int or sequence of int
        If an integer, computes the test for all lags up to maxlag. If a
        sequence, computes the tests only for the lags in maxlag.
    addconst : bool, optional
        Include a constant in the model.

    Returns
    -------
    dict
        All test results, dictionary keys are the number of lags. For each
        lag the values are a tuple, with the first element a dictionary with
        test statistic, pvalues, degrees of freedom, the second element are
        the OLS estimation results for the restricted model, the unrestricted
        model and the restriction (contrast) matrix for the parameter f_test.

    Notes
    -----
    TODO: convert to class and attach results properly

    The Null hypothesis for grangercausalitytests is that the time series in
    the second column, x2, does NOT Granger cause the time series in the first
    column, x1. Grange causality means that past values of x2 have a
    statistically significant effect on the current value of x1, taking past
    values of x1 into account as regressors. We reject the null hypothesis
    that x2 does not Granger cause x1 if the pvalues are below a desired size
    of the test.

    The null hypothesis for all four test is that the coefficients
    corresponding to past values of the second time series are zero.

    `params_ftest`, `ssr_ftest` are based on F distribution

    `ssr_chi2test`, `lrtest` are based on chi-square distribution

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Granger_causality

    .. [2] Greene: Econometric Analysis

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from statsmodels.tsa.stattools import grangercausalitytests
    >>> import numpy as np
    >>> data = sm.datasets.macrodata.load_pandas()
    >>> data = data.data[["realgdp", "realcons"]].pct_change().dropna()

    All lags up to 4

    >>> gc_res = grangercausalitytests(data, 4)

    Only lag 4

    >>> gc_res = grangercausalitytests(data, [4])
    """
    x = array_like(x, "x", ndim=2)
    if not np.isfinite(x).all():
        raise ValueError("x contains NaN or inf values.")
    addconst = bool_like(addconst, "addconst")

    try:
        maxlag = int_like(maxlag, "maxlag")
        if maxlag <= 0:
            raise ValueError("maxlag must be a positive integer")
        lags = np.arange(1, maxlag + 1)
    except TypeError as exc:
        lags = np.array([int(lag) for lag in maxlag])
        maxlag = lags.max()
        if lags.min() <= 0 or lags.size == 0:
            raise ValueError(
                "maxlag must be a non-empty list containing only positive integers"
            ) from exc

    if x.shape[0] <= 3 * maxlag + int(addconst):
        raise ValueError(
            f"Insufficient observations. Maximum allowable lag is {int((x.shape[0] - int(addconst)) / 3) - 1}"
        )

    resli = {}

    for mlg in lags:
        result = {}
        mxlg = mlg

        # create lagmat of both time series
        dta = lagmat2ds(x, mxlg, trim="both", dropex=1)

        # add constant
        if addconst:
            dtaown = add_constant(dta[:, 1 : (mxlg + 1)], prepend=False)
            dtajoint = add_constant(dta[:, 1:], prepend=False)
            if (
                dtajoint.shape[1] == (dta.shape[1] - 1)
                or (dtajoint.max(0) == dtajoint.min(0)).sum() != 1
            ):
                raise InfeasibleTestError(
                    "The x values include a column with constant values and so"
                    " the test statistic cannot be computed."
                )
        else:
            raise NotImplementedError("Not Implemented")
            # dtaown = dta[:, 1:mxlg]
            # dtajoint = dta[:, 1:]

        # Run ols on both models without and with lags of second variable
        res2down = OLS(dta[:, 0], dtaown).fit()
        res2djoint = OLS(dta[:, 0], dtajoint).fit()

        # print results
        # for ssr based tests see:
        # http://support.sas.com/rnd/app/examples/ets/granger/index.htm
        # the other tests are made-up

        # Granger Causality test using ssr (F statistic)
        if res2djoint.model.k_constant:
            tss = res2djoint.centered_tss
        else:
            tss = res2djoint.uncentered_tss
        if (
            tss == 0
            or res2djoint.ssr == 0
            or np.isnan(res2djoint.rsquared)
            or (res2djoint.ssr / tss) < np.finfo(float).eps
            or res2djoint.params.shape[0] != dtajoint.shape[1]
        ):
            raise InfeasibleTestError(
                "The Granger causality test statistic cannot be computed "
                "because the VAR has a perfect fit of the data."
            )
        fgc1 = (
            (res2down.ssr - res2djoint.ssr)
            / res2djoint.ssr
            / mxlg
            * res2djoint.df_resid
        )
        result["ssr_ftest"] = (
            fgc1,
            stats.f.sf(fgc1, mxlg, res2djoint.df_resid),
            res2djoint.df_resid,
            mxlg,
        )

        # Granger Causality test using ssr (ch2 statistic)
        fgc2 = res2down.nobs * (res2down.ssr - res2djoint.ssr) / res2djoint.ssr
        result["ssr_chi2test"] = (fgc2, stats.chi2.sf(fgc2, mxlg), mxlg)

        # likelihood ratio test pvalue:
        lr = -2 * (res2down.llf - res2djoint.llf)
        result["lrtest"] = (lr, stats.chi2.sf(lr, mxlg), mxlg)

        # F test that all lag coefficients of exog are zero
        rconstr = np.column_stack(
            (np.zeros((mxlg, mxlg)), np.eye(mxlg, mxlg), np.zeros((mxlg, 1)))
        )
        ftres = res2djoint.f_test(rconstr)
        result["params_ftest"] = (
            np.squeeze(ftres.fvalue)[()],
            np.squeeze(ftres.pvalue)[()],
            ftres.df_denom,
            ftres.df_num,
        )

        resli[mxlg] = (result, [res2down, res2djoint, rconstr])

    return resli


class CointResult(NamedTuple):
    """
    Result of :func:`coint`.

    Parameters
    ----------
    coint_t : float
        The t-statistic of unit-root test on residuals.
    pvalue : float
        MacKinnon's approximate, asymptotic p-value based on MacKinnon
        (1994).
    critical_values : ndarray
        Critical values for the test statistic at the 1 %, 5 %, and 10 %
        levels based on regression curve. This depends on the number of
        observations.
    """

    coint_t: float
    pvalue: float
    critical_values: np.ndarray


def coint(
    y0,
    y1,
    trend="c",
    method="aeg",
    maxlag=None,
    autolag: str | None = "aic",
    return_results=None,
):
    """
    Test for no-cointegration of a univariate equation

    The null hypothesis is no cointegration. Variables in y0 and y1 are
    assumed to be integrated of order 1, I(1).

    This uses the augmented Engle-Granger two-step cointegration test.
    Constant or trend is included in 1st stage regression, i.e., in
    cointegrating equation.

    **Warning:** The autolag default has changed compared to statsmodels 0.8.
    In 0.8 autolag was always None, no the keyword is used and defaults to
    "aic". Use `autolag=None` to avoid the lag search.

    Parameters
    ----------
    y0 : array_like
        The first element in cointegrated system. Must be 1-d.
    y1 : array_like
        The remaining elements in cointegrated system.
    trend : {"c", "ct", "ctt", "n"}, optional
        The trend term included in regression for cointegrating equation.

        * "c" : constant (default).
        * "ct" : constant and linear trend.
        * "ctt" : constant and quadratic trend.
        * "n" : no constant.

    method : {"aeg"}, optional
        Only "aeg" (augmented Engle-Granger) is available.
    maxlag : int, optional
        Argument for `adfuller`, largest or given number of lags.
    autolag : {"AIC", "BIC", "t-stat", None}, optional
        Argument for `adfuller`, lag selection criterion.

        * If None, then maxlag lags are used without lag search.
        * If "AIC" (default) or "BIC", then the number of lags is chosen
          to minimize the corresponding information criterion.
        * "t-stat" based choice of maxlag.  Starts with maxlag and drops a
          lag until the t-statistic on the last lag length is significant
          using a 5%-sized test.
    return_results : bool, optional
        For future compatibility, currently only tuple available.
        If True, then a results instance is returned. Otherwise, a tuple
        with the test outcome is returned. Set `return_results=False` to
        avoid future changes in return.

    Returns
    -------
    CointResult
        A result object with fields:

        coint_t : float
            The t-statistic of unit-root test on residuals.
        pvalue : float
            MacKinnon's approximate, asymptotic p-value based on MacKinnon
            (1994).
        critical_values : ndarray
            Critical values for the test statistic at the 1 %, 5 %, and
            10 % levels based on regression curve. This depends on the
            number of observations.

    Notes
    -----
    The Null hypothesis is that there is no cointegration, the alternative
    hypothesis is that there is cointegrating relationship. If the pvalue is
    small, below a critical size, then we can reject the hypothesis that there
    is no cointegrating relationship.

    P-values and critical values are obtained through regression surface
    approximation from MacKinnon 1994 and 2010.

    If the two series are almost perfectly collinear, then computing the
    test is numerically unstable. However, the two series will be cointegrated
    under the maintained assumption that they are integrated. In this case
    the t-statistic will be set to -inf and the pvalue to zero.

    TODO: We could handle gaps in data by dropping rows with nans in the
    Auxiliary regressions. Not implemented yet, currently assumes no nans
    and no gaps in time series.

    References
    ----------
    .. [1] MacKinnon, J.G. 1994  "Approximate Asymptotic Distribution Functions
       for Unit-Root and Cointegration Tests." Journal of Business & Economics
       Statistics, 12.2, 167-76.
    .. [2] MacKinnon, J.G. 2010.  "Critical Values for Cointegration Tests."
       Queen's University, Dept of Economics Working Papers 1227.
       http://ideas.repec.org/p/qed/wpaper/1227.html
    """
    y0 = array_like(y0, "y0")
    y1 = array_like(y1, "y1", ndim=2)
    trend = string_like(trend, "trend", options=("c", "n", "ct", "ctt"))
    string_like(method, "method", options=("aeg",))
    maxlag = int_like(maxlag, "maxlag", optional=True)
    autolag = string_like(
        autolag, "autolag", optional=True, options=("aic", "bic", "t-stat")
    )
    return_results = bool_like(return_results, "return_results", optional=True)

    nobs, k_vars = y1.shape
    k_vars += 1  # add 1 for y0

    if trend == "n":
        xx = y1
    else:
        xx = add_trend(y1, trend=trend, prepend=False)

    res_co = OLS(y0, xx).fit()

    if res_co.rsquared < 1 - 100 * SQRTEPS:
        res_adf = adfuller(
            res_co.resid,
            maxlag=maxlag,
            autolag=autolag,
            regression="n",
            result_object=False,
        )
    else:
        warnings.warn(
            "y0 and y1 are (almost) perfectly colinear."
            "Cointegration test is not reliable in this case.",
            CollinearityWarning,
            stacklevel=2,
        )
        # Edge case where series are too similar
        res_adf = (-np.inf,)

    # no constant or trend, see egranger in Stata and MacKinnon
    if trend == "n":
        crit = [np.nan] * 3  # 2010 critical values not available
    else:
        crit = mackinnoncrit(N=k_vars, regression=trend, nobs=nobs - 1)
        #  nobs - 1, the -1 is to match egranger in Stata, I do not know why.
        #  TODO: check nobs or df = nobs - k

    pval_asy = mackinnonp(res_adf[0], regression=trend, N=k_vars)
    return CointResult(res_adf[0], pval_asy, crit)


@dataclass(frozen=True, slots=True)
class DieboldMarianoResult(LimitedIterationMixin[float]):
    """
    Result of :func:`diebold_mariano_test`.

    Parameters
    ----------
    statistic : float
        The Diebold-Mariano test statistic. Asymptotically standard normal
        under the null of equal predictive accuracy, or Student's t with
        ``nobs - 1`` degrees of freedom when the Harvey et al. (1997)
        small-sample correction is applied.
    pvalue : float
        The two-sided p-value for the null of equal predictive accuracy.
    lags : int
        The number of lags used in the Newey-West estimator of the long-run
        variance of the loss differential.
    harvey_adj_factor : float or None
        The finite-sample adjustment factor of Harvey et al. (1997) that
        was applied to ``statistic``. ``None`` unless ``harvey_adj`` was
        True.

    Notes
    -----
    Unpacks as ``statistic, pvalue = result``. The other two fields are only
    available through attribute access.
    """

    statistic: float
    pvalue: float
    lags: int
    harvey_adj_factor: float | None

    _iter_fields: ClassVar[tuple[str, ...]] = ("statistic", "pvalue")


def diebold_mariano_test(
    y,
    forecast_a,
    forecast_b,
    *,
    lags=None,
    criterion="mse",
    power=2,
    harvey_adj=False,
    horizon=1,
):
    """
    Performs a Diebold-Mariano test under the null hypothesis of
    equal predictive accuracy between two forecasts.

    Parameters
    ----------
    y : array_like
        Array of the observed variable.
    forecast_a : array_like
        Array of forecasted values from the first model.
    forecast_b : array_like
        Array of forecasted values from the second model.
    lags : int, optional
        The number of lags to include in the Newey-West (HAC) variance
        estimator used for the loss differential. Must be non-negative if
        provided. If not provided, ``max(horizon - 1, ceil(nobs ** (1/3)))``
        is used, so this also depends on ``horizon`` even when
        ``harvey_adj`` is False. See the Notes for details.
    criterion : str or callable, optional
        The loss function used to score each forecast. Default is 'mse'.
        Implemented criteria are 'mse', 'mad' (equivalently 'mae') and
        'mape'; 'poly' selects a generalized power loss whose exponent is
        set with ``power``. See the Notes for the definition of each
        criterion. Alternatively, ``criterion`` can be a callable that
        accepts two array_like arguments and returns an array of losses
        with signature ``loss = criterion(y, forecast)``, allowing
        problem-specific loss functions (e.g., QLIKE, see Examples).
    power : float, optional
        The exponent used to compute the loss when ``criterion='poly'``,
        i.e., the loss is ``|y - forecast| ** power``. Default is 2, which
        reproduces 'mse'. Ignored unless ``criterion='poly'``.
    harvey_adj : bool, optional
        Indicates if the Harvey-Leybourne-Newbold (1997) correction for
        small samples should be applied. Default is False. When True, the
        test statistic is rescaled and the p-value is computed from a
        Student's t distribution with ``nobs - 1`` degrees of freedom
        instead of the standard normal.
    horizon : int, optional
        The forecast horizon used to (1) form the default number of HAC
        lags and (2) compute the Harvey et al. (1997) adjustment when
        ``harvey_adj`` is True. Must be a positive integer. Default is 1.

    Returns
    -------
    DieboldMarianoResult
        A result object containing the DM test statistic, its p-value, and
        the Harvey et al. (1997) adjustment factor, if applicable.

    Notes
    -----
    The Diebold-Mariano (1995) test compares the predictive accuracy of two
    competing forecasts, ``forecast_a`` and ``forecast_b``, of the same
    series ``y``. Accuracy is measured with a loss function :math:`g(y, f)`
    (chosen via ``criterion``), and the test is based on the loss
    differential

    .. math::

        d_t = g(y_t, \\text{forecast}_{a,t}) - g(y_t, \\text{forecast}_{b,t}).

    Under the null hypothesis of equal predictive accuracy,
    :math:`E[d_t] = 0`. The test statistic is the t-statistic from
    regressing :math:`d_t` on a constant using a Newey-West (HAC)
    covariance estimator with ``lags`` lags, i.e.

    .. math::

        DM = \\frac{\\bar{d}}{\\sqrt{\\widehat{\\mathrm{avar}}(\\bar{d})}},

    where :math:`\\bar{d}` is the sample mean of :math:`d_t` and
    :math:`\\widehat{\\mathrm{avar}}` is the HAC long-run variance
    estimator. When ``lags`` is not supplied, a bandwidth of
    ``max(horizon - 1, ceil(nobs ** (1/3)))`` is used. This differs from
    some presentations of the DM test that always use ``horizon - 1``
    lags; pass ``lags=horizon - 1`` explicitly to reproduce that
    parameterization. Under the null, ``statistic`` is asymptotically
    standard normal.

    Because ``forecast_a`` and ``forecast_b`` are typically generated from
    overlapping information sets (e.g., multi-step-ahead forecasts), the
    loss differential is often serially correlated even under the null,
    which is why a HAC estimator rather than the usual OLS standard error
    is used.

    For small samples, Harvey, Leybourne and Newbold (1997) propose
    rescaling the statistic by

    .. math::

        DM^{HLN} = \\sqrt{\\frac{T + 1 - 2h + h(h - 1) / T}{T}}\\, DM,

    where :math:`T` is the number of observations and :math:`h` is the
    forecast ``horizon``, and comparing ``DM^{HLN}`` to a Student's t
    distribution with :math:`T - 1` degrees of freedom rather than the
    standard normal. This correction is enabled with ``harvey_adj=True``.

    The built-in criteria are:

    * ``'mse'``: squared error loss, :math:`g(y, f) = (y - f)^2`. Penalizes
      large errors more heavily than small ones; the corresponding DM test
      answers "which forecast has lower mean squared error?"
    * ``'mad'`` / ``'mae'``: mean absolute (deviation) error loss,
      :math:`g(y, f) = |y - f|`. More robust to outliers than 'mse' since
      errors are not squared.
    * ``'mape'``: mean absolute percentage error loss,
      :math:`g(y, f) = |(y - f) / y|`. Expresses errors relative to the
      level of ``y``, which is useful when comparing series of different
      scales, but is undefined when any element of ``y`` is zero and can
      be dominated by observations where ``y`` is close to zero.
    * ``'poly'``: generalized power loss, :math:`g(y, f) = |y - f|^p`
      where :math:`p` is set with ``power``. Setting ``power=2`` is
      equivalent to 'mse' and ``power=1`` is equivalent to 'mad'.

    Any other loss can be supplied directly through ``criterion`` as a
    callable, for example an asymmetric loss or a loss appropriate for
    strictly positive series such as QLIKE (see Examples).

    References
    ----------

    .. [1] Diebold, Francis X., and Roberto S. Mariano. "Comparing predictive
       accuracy." Journal of Business & Economic Statistics 13, no. 3
       (1995): 253-263.
    .. [2] Harvey, David, Stephen Leybourne, and Paul Newbold. "Testing the
       equality of prediction mean squared errors." International Journal
       of Forecasting 13, no. 2 (1997): 281-291.

    Examples
    --------
    Comparing two forecasts of a strictly positive series (e.g., realized
    variance) using the QLIKE loss, which is standard in the volatility
    forecasting literature and only defined for non-negative ``y`` and
    strictly positive forecasts:

    >>> import numpy as np
    >>> from statsmodels.tsa.stattools import diebold_mariano_test
    >>> rng = np.random.default_rng(0)
    >>> y = rng.standard_normal(200) ** 2
    >>> scale = rng.chisquare(5, size=y.shape) / 5
    >>> forecast_a = (0.9 * scale * y)
    >>> forecast_b = scale * y

    >>> def qlike(y, forecast):
    ...     ratio = y / forecast
    ...     return ratio - np.log(ratio) - 1

    >>> res = diebold_mariano_test(y, forecast_a, forecast_b, criterion=qlike)
    >>> res.statistic, res.pvalue  # doctest: +SKIP
    """

    y = array_like(y, "y", ndim=1, maxdim=1, dtype=float)
    forecast_a = array_like(forecast_a, "forecast_a", ndim=1, maxdim=1, dtype=float)
    forecast_b = array_like(forecast_b, "forecast_b", ndim=1, maxdim=1, dtype=float)
    lags = int_like(lags, "lags", optional=True)
    harvey_adj = bool_like(harvey_adj, "harvey_adj")
    power = float_like(power, "power", optional=True)
    horizon = int_like(horizon, "horizon")

    if horizon < 1:
        raise ValueError("horizon must be a positive integer.")
    if lags is not None and lags < 0:
        raise ValueError("lags must be a non-negative integer.")

    t = len(y)
    if forecast_a.shape[0] != t or forecast_b.shape[0] != t:
        raise ValueError("y, forecast_a and forecast_b must all have equal length.")

    if lags is None:
        lags = int(max(horizon - 1, np.ceil(t ** (1 / 3))))

    if isinstance(criterion, str):
        criterion = string_like(
            criterion, "criterion", options=("mse", "mad", "mae", "mape", "poly")
        )
        if criterion == "mse":

            def criterion_func(y, f):
                return (y - f) ** 2

        elif criterion == "mape":

            def criterion_func(y, f):
                return np.abs((y - f) / y)

        elif criterion in ("mae", "mad"):

            def criterion_func(y, f):
                return np.abs(y - f)

        else:  # criterion == "poly"

            def criterion_func(y, f):
                return np.abs(y - f) ** power

    else:
        criterion_func = criterion

    # calculate d based on criterion
    loss_1 = criterion_func(y, forecast_a)
    loss_2 = criterion_func(y, forecast_b)
    d = loss_1 - loss_2

    # calculate test statistic as the t-stat of a constant-only HAC regression
    res = OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    dm_stat = float(res.tvalues[0])

    # Harvey et. al (1997) small-sample adjustment
    if harvey_adj:
        adj_factor = np.sqrt(
            (t + 1 - (2 * horizon) + (horizon * (horizon - 1) / t)) / t
        )
        dm_stat = adj_factor * dm_stat
        p_value = 2 * stats.t.cdf(-np.abs(dm_stat), df=t - 1)
    else:
        adj_factor = None
        p_value = 2 * stats.norm.cdf(-np.abs(dm_stat))

    return DieboldMarianoResult(dm_stat, float(p_value), lags, adj_factor)


def has_missing(data):
    """
    Returns True if `data` contains missing entries, otherwise False

    Parameters
    ----------
    data : array_like
        The data to check for missing entries.

    Returns
    -------
    bool
        True if data contains missing entries, otherwise False.
    """
    return np.isnan(np.sum(data))


@dataclass(frozen=True, slots=True, repr=False)
class KPSSResult(LimitedIterationMixin[float]):
    """
    Result of :func:`kpss`.

    Parameters
    ----------
    statistic : float
        The KPSS test statistic.
    pvalue : float
        The p-value of the test. The p-value is interpolated from Table 1
        in Kwiatkowski et al. (1992), and a boundary point is returned if
        the test statistic is outside the table of critical values, that
        is, if the p-value is outside the interval (0.01, 0.1).
    lags : int
        The truncation lag parameter.
    critical_values : dict[str, float]
        The critical values at 10%, 5%, 2.5% and 1%. Based on Kwiatkowski
        et al. (1992).
    resstore : ResultsStore or None
        An instance of a dummy class with results attached as attributes,
        if ``store`` was True, otherwise None.

    Notes
    -----
    Unpacks as ``statistic, pvalue = result``. Other values are only
    available through attribute access.
    """

    statistic: float
    pvalue: float
    lags: int
    critical_values: dict[str, float]
    resstore: ResultsStore | None

    _iter_fields: ClassVar[tuple[str, ...]] = ("statistic", "pvalue")

    def __repr__(self) -> str:
        return f"""\
{self.__class__.__name__}
KPSS Statistic: {self.statistic:0.5f}
P-value: {self.pvalue:0.5f}
Lags: {self.lags}
Critical Values: {self.critical_values}
"""


def kpss(
    x,
    regression: Literal["c", "ct"] = "c",
    nlags: Literal["auto", "legacy"] | int = "auto",
    store: bool = False,
    *,
    result_object: bool | None = None,
) -> (
        tuple[float, float, int, dict[str, float]]
        | tuple[float, float, dict[str, float], ResultsStore]
        | KPSSResult
):
    """
    Kwiatkowski-Phillips-Schmidt-Shin test for stationarity

    Computes the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for the null
    hypothesis that x is level or trend stationary.

    Parameters
    ----------
    x : array_like, 1d
        The data series to test.
    regression : {"c", "ct"}, optional
        The null hypothesis for the KPSS test.

        * "c" : The data is stationary around a constant (default).
        * "ct" : The data is stationary around a trend.
    nlags : {"auto", "legacy"} or int, optional
        Indicates the number of lags to be used. If "auto" (default), lags
        is calculated using the data-dependent method of Hobijn et al. (1998).
        See also Andrews (1991), Newey & West (1994), and Schwert (1989). If
        set to "legacy", uses int(12 * (n / 100)**(1 / 4)), as outlined in
        Schwert (1989).
    store : bool, optional
        If True, then a result instance is returned additionally to
        the KPSS statistic (default is False).
    result_object : bool, optional
        Flag indicating whether to return the results as a ``KPSSResult``
        instead of a plain tuple. If ``None`` (the default), the
        current tuple-returning behavior is used and a ``FutureWarning`` is
        issued.

        .. deprecated:: 0.15.0

            In release 0.16.0 or after July 2027, whichever is later, the
            default will change to always return a ``KPSSResult``. Set
            ``result_object=True`` to opt in now, or
            ``result_object=False`` to silence the warning and keep the
            current return type.

    Returns
    -------
    KPSSResult
        If ``result_object=True``, a result object with fields ``statistic``,
        ``pvalue``, ``lags``, ``critical_values``, and ``resstore``
        (``resstore`` is ``None`` when not computed). See
        :class:`~statsmodels.tsa.stattools.KPSSResult`.

    Otherwise (the deprecated default), a plain tuple made up of:

    statistic : float
        The KPSS test statistic.
    pvalue : float
        The p-value of the test. The p-value is interpolated from
        Table 1 in Kwiatkowski et al. (1992), and a boundary point
        is returned if the test statistic is outside the table of
        critical values, that is, if the p-value is outside the
        interval (0.01, 0.1).
    lags : int
        The truncation lag parameter. Only returned when ``store=False``;
        when ``store=True`` the lag is instead available as
        ``resstore.lags``.
    crit : dict
        The critical values at 10%, 5%, 2.5% and 1%. Based on
        Kwiatkowski et al. (1992).
    resstore : instance of ResultsStore
        Only returned when ``store=True`` (and, in that case, in place of
        ``lags``). An instance of a dummy class with results attached as
        attributes.

    Notes
    -----
    To estimate sigma^2 the Newey-West estimator is used. If lags is "legacy",
    the truncation lag parameter is set to int(12 * (n / 100) ** (1 / 4)),
    as outlined in Schwert (1989). The p-values are interpolated from
    Table 1 of Kwiatkowski et al. (1992). If the computed statistic is
    outside the table of critical values, then a warning message is
    generated.

    Missing values are not handled.

    See the notebook `Stationarity and detrending (ADF/KPSS)
    <../examples/notebooks/generated/stationarity_detrending_adf_kpss.html>`__
    for an overview.

    References
    ----------
    .. [1] Andrews, D.W.K. (1991). Heteroskedasticity and autocorrelation
       consistent covariance matrix estimation. Econometrica, 59: 817-858.

    .. [2] Hobijn, B., Frances, B.H., & Ooms, M. (2004). Generalizations of the
       KPSS-test for stationarity. Statistica Neerlandica, 52: 483-502.

    .. [3] Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., & Shin, Y. (1992).
       Testing the null hypothesis of stationarity against the alternative of a
       unit root. Journal of Econometrics, 54: 159-178.

    .. [4] Newey, W.K., & West, K.D. (1994). Automatic lag selection in
       covariance matrix estimation. Review of Economic Studies, 61: 631-653.

    .. [5] Schwert, G. W. (1989). Tests for unit roots: A Monte Carlo
       investigation. Journal of Business and Economic Statistics, 7 (2):
       147-159.
    """
    x = array_like(x, "x")
    regression = string_like(regression, "regression", options=("c", "ct"))
    store = bool_like(store, "store")
    result_object = bool_like(result_object, "result_object", optional=True)

    nobs = x.shape[0]
    hypo = regression

    # if m is not one, n != m * n
    if nobs != x.size:
        raise ValueError(f"x of shape {x.shape} not understood")

    if hypo == "ct":
        # p. 162 Kwiatkowski et al. (1992): y_t = beta * t + r_t + e_t,
        # where beta is the trend, r_t a random walk and e_t a stationary
        # error term.
        resids = OLS(x, add_constant(np.arange(1, nobs + 1))).fit().resid
        crit = [0.119, 0.146, 0.176, 0.216]
    else:  # hypo == "c"
        # special case of the model above, where beta = 0 (so the null
        # hypothesis is that the data is stationary around r_0).
        resids = x - x.mean()
        crit = [0.347, 0.463, 0.574, 0.739]

    if nlags is None:
        raise ValueError(
            "None is not a valid value for nlags. nlags must be an integer, 'auto' "
            "or 'legacy'."
        )
    if isinstance(nlags, str):
        nlags = string_like(nlags, "nlags", options=("auto", "legacy"))
    if nlags == "legacy":
        nlags = int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))
        nlags = min(nlags, nobs - 1)
    elif nlags == "auto":
        # autolag method of Hobijn et al. (1998)
        nlags = _kpss_autolag(resids, nobs)
        nlags = min(nlags, nobs - 1)
    else:
        nlags = int_like(nlags, "nlags", optional=False)

        if nlags >= nobs:
            raise ValueError(
                f"lags ({nlags}) must be < number of observations ({nobs})"
            )

    pvals = [0.10, 0.05, 0.025, 0.01]

    eta = np.sum(resids.cumsum() ** 2) / (nobs**2)  # eq. 11, p. 165
    s_hat = _sigma_est_kpss(resids, nobs, nlags)

    kpss_stat = eta / s_hat
    p_value = np.interp(kpss_stat, crit, pvals)

    warn_msg = (
        "The test statistic is outside of the range of p-values available in the "
        "look-up table. The actual p-value is {direction} than the p-value returned."
    )

    if p_value == pvals[-1]:
        warnings.warn(
            warn_msg.format(direction="smaller"),
            InterpolationWarning,
            stacklevel=2,
        )
    elif p_value == pvals[0]:
        warnings.warn(
            warn_msg.format(direction="greater"),
            InterpolationWarning,
            stacklevel=2,
        )

    crit_dict = {"10%": crit[0], "5%": crit[1], "2.5%": crit[2], "1%": crit[3]}

    if store:
        rstore = ResultsStore()
        rstore.lags = nlags
        rstore.nobs = nobs

        stationary_type = "level" if hypo == "c" else "trend"
        rstore.H0 = f"The series is {stationary_type} stationary"
        rstore.HA = f"The series is not {stationary_type} stationary"
    else:
        rstore = None

    if result_object is None:
        warnings.warn(
            "kpss currently returns a plain tuple whose length and layout "
            "depends on the store argument (and which silently drops "
            "`lags` when store=True). In release 0.16 or after July 2027, "
            "whichever is later, the default behavior will switch to "
            "always returning a KPSSResult. Set "
            "result_object=True to switch now, or result_object=False "
            "to keep the current behavior and silence this warning.",
            FutureWarning,
            stacklevel=2,
        )
    if result_object:
        return KPSSResult(kpss_stat, p_value, nlags, crit_dict, rstore)
    if store:
        return kpss_stat, p_value, crit_dict, rstore
    else:
        return kpss_stat, p_value, nlags, crit_dict


def _sigma_est_kpss(resids, nobs, lags):
    """
    Computes equation 10, p. 164 of Kwiatkowski et al. (1992)

    This is the consistent estimator for the variance.

    Parameters
    ----------
    resids : ndarray
        The residuals from the regression used to compute the test.
    nobs : int
        The number of observations.
    lags : int
        The number of lags to use in the covariance estimate.

    Returns
    -------
    float
        The estimated variance.
    """
    s_hat = np.sum(resids**2)
    for i in range(1, lags + 1):
        resids_prod = np.dot(resids[i:], resids[: nobs - i])
        s_hat += 2 * resids_prod * (1.0 - (i / (lags + 1.0)))
    return s_hat / nobs


def _kpss_autolag(resids, nobs):
    """
    Computes the number of lags for covariance matrix estimation in KPSS test

    Uses method of Hobijn et al (1998). See also Andrews (1991), Newey &
    West (1994), and Schwert (1989). Assumes Bartlett / Newey-West kernel.

    Parameters
    ----------
    resids : ndarray
        The residuals from the regression used to compute the test.
    nobs : int
        The number of observations.

    Returns
    -------
    int
        The number of lags to use in covariance matrix estimation.
    """
    covlags = int(np.power(nobs, 2.0 / 9.0))
    s0 = np.sum(resids**2) / nobs
    s1 = 0
    for i in range(1, covlags + 1):
        resids_prod = np.dot(resids[i:], resids[: nobs - i])
        resids_prod /= nobs / 2.0
        s0 += resids_prod
        s1 += i * resids_prod
    s_hat = s1 / s0
    pwr = 1.0 / 3.0
    gamma_hat = 1.1447 * np.power(s_hat * s_hat, pwr)
    autolags = int(gamma_hat * np.power(nobs, pwr))
    return autolags


@dataclass(frozen=True, slots=True, repr=False)
class RURResult(LimitedIterationMixin[float]):
    """
    Result of :func:`range_unit_root_test` when ``result_object=True``.

    Parameters
    ----------
    statistic : float
        The RUR test statistic.
    pvalue : float
        The p-value of the test. The p-value is interpolated from Table 1
        in Aparicio et al. (2006), and a boundary point is returned if the
        test statistic is outside the table of critical values, that is,
        if the p-value is outside the interval (0.01, 0.1).
    critical_values : dict[str, float]
        The critical values at 10%, 5%, 2.5% and 1%. Based on Aparicio et
        al. (2006).
    resstore : ResultsStore or None
        An instance of a dummy class with results attached as attributes,
        if ``store`` was True, otherwise None.

    Notes
    -----
    Unpacks as ``statistic, pvalue = result``. Other values are only
    accessible using attributes.
    """

    statistic: float
    pvalue: float
    critical_values: dict[str, float]
    resstore: ResultsStore | None

    _iter_fields: ClassVar[tuple[str, ...]] = ("statistic", "pvalue")

    def __repr__(self) -> str:
        return f"""\
{self.__class__.__name__}
RUR Statistic: {self.statistic:0.5f}
P-value: {self.pvalue:0.5f}
Critical Values: {self.critical_values}
"""


def range_unit_root_test(x, store=False, *, result_object: bool | None = None):
    """
    Range unit-root test for stationarity

    Computes the Range Unit-Root (RUR) test for the null
    hypothesis that x is stationary.

    Parameters
    ----------
    x : array_like, 1d
        The data series to test.
    store : bool, optional
        If True, then a result instance is returned additionally to
        the RUR statistic (default is False).
    result_object : bool, optional
        Flag indicating whether to return the results as a
        ``RURResult`` instead of a plain tuple. The legacy tuple (whose
        length depends on `store`) is returned by default and a
        ``FutureWarning`` is issued.

        .. deprecated:: 0.15.0

            In release 0.16.0 or after July 2027, whichever is later, the
            default will change to always return a
            ``RURResult``. Set ``result_object=True`` to opt
            in now, or ``result_object=False`` to silence the warning and
            keep the current return type.

    Returns
    -------
    RURResult
        If ``result_object=True``, a result object with fields
        ``statistic``, ``pvalue``, ``critical_values``, and ``resstore``
        (``resstore`` is ``None`` when not computed). See
        :class:`~statsmodels.tsa.stattools.RURResult`.

    Otherwise (the deprecated default), a plain tuple whose length depends
    on `store`, made up of a subset of:

    statistic : float
        The RUR test statistic.
    pvalue : float
        The p-value of the test. The p-value is interpolated from
        Table 1 in Aparicio et al. (2006), and a boundary point
        is returned if the test statistic is outside the table of
        critical values, that is, if the p-value is outside the
        interval (0.01, 0.1).
    crit : dict
        The critical values at 10%, 5%, 2.5% and 1%. Based on
        Aparicio et al. (2006).
    resstore : (optional) instance of ResultsStore
        An instance of a dummy class with results attached as attributes.

    Notes
    -----
    The p-values are interpolated from
    Table 1 of Aparicio et al. (2006). If the computed statistic is
    outside the table of critical values, then a warning message is
    generated.

    Missing values are not handled.

    References
    ----------
    .. [1] Aparicio, F., Escribano A., Sipols, A.E. (2006). Range Unit-Root (RUR)
        tests: robust against nonlinearities, error distributions, structural breaks
        and outliers. Journal of Time Series Analysis, 27 (4): 545-576.
    """
    x = array_like(x, "x")
    store = bool_like(store, "store")
    result_object = bool_like(result_object, "result_object", optional=True)

    nobs = x.shape[0]

    # if m is not one, n != m * n
    if nobs != x.size:
        raise ValueError(f"x of shape {x.shape} not understood")

    # Table from [1] has been replicated using 200,000 samples
    # Critical values for new n_obs values have been identified
    pvals = [0.01, 0.025, 0.05, 0.10, 0.90, 0.95]
    n = np.array([25, 50, 100, 150, 200, 250, 500, 1000, 2000, 3000, 4000, 5000])
    crit = np.array(
        [
            [0.6626, 0.8126, 0.9192, 1.0712, 2.4863, 2.7312],
            [0.7977, 0.9274, 1.0478, 1.1964, 2.6821, 2.9613],
            [0.9070, 1.0243, 1.1412, 1.2888, 2.8317, 3.1393],
            [0.9543, 1.0768, 1.1869, 1.3294, 2.8915, 3.2049],
            [0.9833, 1.0984, 1.2101, 1.3494, 2.9308, 3.2482],
            [0.9982, 1.1137, 1.2242, 1.3632, 2.9571, 3.2842],
            [1.0494, 1.1643, 1.2712, 1.4076, 3.0207, 3.3584],
            [1.0846, 1.1959, 1.2988, 1.4344, 3.0653, 3.4073],
            [1.1121, 1.2200, 1.3230, 1.4556, 3.0948, 3.4439],
            [1.1204, 1.2295, 1.3303, 1.4656, 3.1054, 3.4632],
            [1.1309, 1.2347, 1.3378, 1.4693, 3.1165, 3.4717],
            [1.1377, 1.2402, 1.3408, 1.4729, 3.1252, 3.4807],
        ]
    )

    # Interpolation for nobs
    inter_crit = np.zeros((1, crit.shape[1]))
    for i in range(crit.shape[1]):
        f = interp1d(n, crit[:, i])
        inter_crit[0, i] = f(nobs)

    # Calculate RUR stat
    xs = pd.Series(x)
    exp_max = xs.expanding(1).max().shift(1)
    exp_min = xs.expanding(1).min().shift(1)
    count = (xs > exp_max).sum() + (xs < exp_min).sum()

    rur_stat = count / np.sqrt(len(x))

    k = len(pvals) - 1
    for i in range(len(pvals) - 1, -1, -1):
        if rur_stat < inter_crit[0, i]:
            k = i
        else:
            break

    p_value = pvals[k]

    warn_msg = """\
The test statistic is outside of the range of p-values available in the
look-up table. The actual p-value is {direction} than the p-value returned.
"""
    direction = ""
    if p_value == pvals[-1]:
        direction = "smaller"
    elif p_value == pvals[0]:
        direction = "larger"

    if direction:
        warnings.warn(
            warn_msg.format(direction=direction),
            InterpolationWarning,
            stacklevel=2,
        )

    crit_dict = {
        "10%": inter_crit[0, 3],
        "5%": inter_crit[0, 2],
        "2.5%": inter_crit[0, 1],
        "1%": inter_crit[0, 0],
    }

    if store:
        rstore = ResultsStore()
        rstore.nobs = nobs

        rstore.H0 = "The series is not stationary"
        rstore.HA = "The series is stationary"
    else:
        rstore = None

    if result_object is None:
        warnings.warn(
            "range_unit_root_test currently returns a plain tuple whose "
            "length depends on the store argument. In release 0.16 or "
            "after July 2027, whichever is later, the default behavior "
            "will switch to always returning a RURResult. "
            "Set result_object=True to switch now, or "
            "result_object=False to keep the current behavior and "
            "silence this warning.",
            FutureWarning,
            stacklevel=2,
        )
    if result_object:
        return RURResult(rur_stat, p_value, crit_dict, rstore)
    if store:
        return rur_stat, p_value, crit_dict, rstore
    return rur_stat, p_value, crit_dict


class ZivotAndrewsUnitRoot:
    """Class wrapper for Zivot-Andrews structural-break unit-root test"""

    def __init__(self):
        """
        Critical values for the three different models specified for the
        Zivot-Andrews unit-root test.

        Notes
        -----
        The p-values are generated through Monte Carlo simulation using
        100,000 replications and 2000 data points.
        """
        self._za_critical_values = {}
        # constant-only model
        self._c = (
            (0.001, -6.78442),
            (0.100, -5.83192),
            (0.200, -5.68139),
            (0.300, -5.58461),
            (0.400, -5.51308),
            (0.500, -5.45043),
            (0.600, -5.39924),
            (0.700, -5.36023),
            (0.800, -5.33219),
            (0.900, -5.30294),
            (1.000, -5.27644),
            (2.500, -5.03340),
            (5.000, -4.81067),
            (7.500, -4.67636),
            (10.000, -4.56618),
            (12.500, -4.48130),
            (15.000, -4.40507),
            (17.500, -4.33947),
            (20.000, -4.28155),
            (22.500, -4.22683),
            (25.000, -4.17830),
            (27.500, -4.13101),
            (30.000, -4.08586),
            (32.500, -4.04455),
            (35.000, -4.00380),
            (37.500, -3.96144),
            (40.000, -3.92078),
            (42.500, -3.88178),
            (45.000, -3.84503),
            (47.500, -3.80549),
            (50.000, -3.77031),
            (52.500, -3.73209),
            (55.000, -3.69600),
            (57.500, -3.65985),
            (60.000, -3.62126),
            (65.000, -3.54580),
            (70.000, -3.46848),
            (75.000, -3.38533),
            (80.000, -3.29112),
            (85.000, -3.17832),
            (90.000, -3.04165),
            (92.500, -2.95146),
            (95.000, -2.83179),
            (96.000, -2.76465),
            (97.000, -2.68624),
            (98.000, -2.57884),
            (99.000, -2.40044),
            (99.900, -1.88932),
        )
        self._za_critical_values["c"] = np.asarray(self._c)
        # trend-only model
        self._t = (
            (0.001, -83.9094),
            (0.100, -13.8837),
            (0.200, -9.13205),
            (0.300, -6.32564),
            (0.400, -5.60803),
            (0.500, -5.38794),
            (0.600, -5.26585),
            (0.700, -5.18734),
            (0.800, -5.12756),
            (0.900, -5.07984),
            (1.000, -5.03421),
            (2.500, -4.65634),
            (5.000, -4.40580),
            (7.500, -4.25214),
            (10.000, -4.13678),
            (12.500, -4.03765),
            (15.000, -3.95185),
            (17.500, -3.87945),
            (20.000, -3.81295),
            (22.500, -3.75273),
            (25.000, -3.69836),
            (27.500, -3.64785),
            (30.000, -3.59819),
            (32.500, -3.55146),
            (35.000, -3.50522),
            (37.500, -3.45987),
            (40.000, -3.41672),
            (42.500, -3.37465),
            (45.000, -3.33394),
            (47.500, -3.29393),
            (50.000, -3.25316),
            (52.500, -3.21244),
            (55.000, -3.17124),
            (57.500, -3.13211),
            (60.000, -3.09204),
            (65.000, -3.01135),
            (70.000, -2.92897),
            (75.000, -2.83614),
            (80.000, -2.73893),
            (85.000, -2.62840),
            (90.000, -2.49611),
            (92.500, -2.41337),
            (95.000, -2.30820),
            (96.000, -2.25797),
            (97.000, -2.19648),
            (98.000, -2.11320),
            (99.000, -1.99138),
            (99.900, -1.67466),
        )
        self._za_critical_values["t"] = np.asarray(self._t)
        # constant + trend model
        self._ct = (
            (0.001, -38.17800),
            (0.100, -6.43107),
            (0.200, -6.07279),
            (0.300, -5.95496),
            (0.400, -5.86254),
            (0.500, -5.77081),
            (0.600, -5.72541),
            (0.700, -5.68406),
            (0.800, -5.65163),
            (0.900, -5.60419),
            (1.000, -5.57556),
            (2.500, -5.29704),
            (5.000, -5.07332),
            (7.500, -4.93003),
            (10.000, -4.82668),
            (12.500, -4.73711),
            (15.000, -4.66020),
            (17.500, -4.58970),
            (20.000, -4.52855),
            (22.500, -4.47100),
            (25.000, -4.42011),
            (27.500, -4.37387),
            (30.000, -4.32705),
            (32.500, -4.28126),
            (35.000, -4.23793),
            (37.500, -4.19822),
            (40.000, -4.15800),
            (42.500, -4.11946),
            (45.000, -4.08064),
            (47.500, -4.04286),
            (50.000, -4.00489),
            (52.500, -3.96837),
            (55.000, -3.93200),
            (57.500, -3.89496),
            (60.000, -3.85577),
            (65.000, -3.77795),
            (70.000, -3.69794),
            (75.000, -3.61852),
            (80.000, -3.52485),
            (85.000, -3.41665),
            (90.000, -3.28527),
            (92.500, -3.19724),
            (95.000, -3.08769),
            (96.000, -3.03088),
            (97.000, -2.96091),
            (98.000, -2.85581),
            (99.000, -2.71015),
            (99.900, -2.28767),
        )
        self._za_critical_values["ct"] = np.asarray(self._ct)

    def _za_crit(self, stat, model="c"):
        """
        Linear interpolation for Zivot-Andrews p-values and critical values

        Parameters
        ----------
        statistic : float
            The ZA test statistic
        model : {"c","t","ct"}, optional
            The model used when computing the ZA statistic. "c" is default.

        Returns
        -------
        pvalue : float
            The interpolated p-value
        cvdict : dict
            Critical values for the test statistic at the 1%, 5%, and 10%
            levels

        Notes
        -----
        The p-values are linear interpolated from the quantiles of the
        simulated ZA test statistic distribution
        """
        table = self._za_critical_values[model]
        pcnts = table[:, 0]
        stats = table[:, 1]
        # ZA cv table contains quantiles multiplied by 100
        pvalue = np.interp(stat, stats, pcnts) / 100.0
        cv = [1.0, 5.0, 10.0]
        crit_value = np.interp(cv, pcnts, stats)
        cvdict = {
            "1%": crit_value[0],
            "5%": crit_value[1],
            "10%": crit_value[2],
        }
        return pvalue, cvdict

    def _quick_ols(self, endog, exog):
        """
        Minimal implementation of LS estimator for internal use

        Parameters
        ----------
        endog : ndarray
            The dependent variable.
        exog : ndarray
            The independent variables.

        Returns
        -------
        ndarray
            The t-statistics of the estimated coefficients.
        """
        xpxi = np.linalg.inv(exog.T.dot(exog))
        xpy = exog.T.dot(endog)
        nobs, k_exog = exog.shape
        b = xpxi.dot(xpy)
        e = endog - exog.dot(b)
        sigma2 = e.T.dot(e) / (nobs - k_exog)
        return b / np.sqrt(np.diag(sigma2 * xpxi))

    def _format_regression_data(self, series, nobs, const, trend, cols, lags):
        """
        Create the endog/exog data for the auxiliary regressions
        from the original (standardized) series under test.

        Parameters
        ----------
        series : ndarray
            The standardized series under test.
        nobs : int
            The number of observations in series.
        const : float
            The normalized constant term value to use in exog.
        trend : ndarray
            The normalized trend values to use in exog.
        cols : int
            The number of non-lag columns in exog.
        lags : int
            The number of lagged difference terms to include in exog.

        Returns
        -------
        endog : ndarray
            The standardized first difference of series.
        exog : ndarray
            The regressor matrix for the auxiliary regression.
        """
        # first-diff y and standardize for numerical stability
        endog = np.diff(series, axis=0)
        endog /= np.sqrt(endog.T.dot(endog))
        series = series / np.sqrt(series.T.dot(series))
        # reserve exog space
        exog = np.zeros((endog[lags:].shape[0], cols + lags))
        exog[:, 0] = const
        # lagged y and dy
        exog[:, cols - 1] = series[lags : (nobs - 1)]
        exog[:, cols:] = lagmat(endog, lags, trim="none")[lags : exog.shape[0] + lags]
        return endog, exog

    def _update_regression_exog(
        self, exog, regression, period, nobs, const, trend, cols, lags
    ):
        """
        Update the exog array for the next regression

        Parameters
        ----------
        exog : ndarray
            The regressor matrix to update in place.
        regression : {"c", "t", "ct"}
            The trend order to include in the regression.
        period : int
            The candidate break period.
        nobs : int
            The number of observations in series.
        const : float
            The normalized constant term value to use in exog.
        trend : ndarray
            The normalized trend values to use in exog.
        cols : int
            The number of non-lag columns in exog.
        lags : int
            The number of lagged difference terms included in exog.

        Returns
        -------
        ndarray
            The updated regressor matrix.
        """
        cutoff = period - (lags + 1)
        if regression != "t":
            exog[:cutoff, 1] = 0
            exog[cutoff:, 1] = const
            exog[:, 2] = trend[(lags + 2) : (nobs + 1)]
            if regression == "ct":
                exog[:cutoff, 3] = 0
                exog[cutoff:, 3] = trend[1 : (nobs - period + 1)]
        else:
            exog[:, 1] = trend[(lags + 2) : (nobs + 1)]
            exog[: (cutoff - 1), 2] = 0
            exog[(cutoff - 1) :, 2] = trend[0 : (nobs - period + 1)]
        return exog

    def run(self, x, trim=0.15, maxlag=None, regression="c", autolag="AIC"):
        """
        Zivot-Andrews structural-break unit-root test

        The Zivot-Andrews test tests for a unit root in a univariate process
        in the presence of serial correlation and a single structural break.

        Parameters
        ----------
        x : array_like
            The data series to test.
        trim : float, optional
            The percentage of series at begin/end to exclude from break-period
            calculation in range [0, 0.333] (default=0.15).
        maxlag : int, optional
            The maximum lag which is included in test, default is
            12*(nobs/100)^{1/4} (Schwert, 1989).
        regression : {"c","t","ct"}, optional
            Constant and trend order to include in regression.

            * "c" : constant only (default).
            * "t" : trend only.
            * "ct" : constant and trend.
        autolag : {"AIC", "BIC", "t-stat", None}, optional
            The method to select the lag length when using automatic selection.

            * if None, then maxlag lags are used,
            * if "AIC" (default) or "BIC", then the number of lags is chosen
              to minimize the corresponding information criterion,
            * "t-stat" based choice of maxlag.  Starts with maxlag and drops a
              lag until the t-statistic on the last lag length is significant
              using a 5%-sized test.

        Returns
        -------
        statistic : float
            The test statistic.
        pvalue : float
            The pvalue based on MC-derived critical values.
        cvdict : dict
            The critical values for the test statistic at the 1%, 5%, and 10%
            levels.
        baselag : int
            The number of lags used for period regressions.
        bpidx : int
            The index of x corresponding to endogenously calculated break period
            with values in the range [0..nobs-1].

        Notes
        -----
        H0 = unit root with a single structural break

        Algorithm follows Baum (2004/2015) approximation to original
        Zivot-Andrews method. Rather than performing an autolag regression at
        each candidate break period (as per the original paper), a single
        autolag regression is run up-front on the base model (constant + trend
        with no dummies) to determine the best lag length. This lag length is
        then used for all subsequent break-period regressions. This results in
        significant run time reduction but also slightly more pessimistic test
        statistics than the original Zivot-Andrews method, although no attempt
        has been made to characterize the size/power trade-off.

        References
        ----------
        .. [1] Baum, C.F. (2004). "ZANDREWS: Stata module to calculate
           Zivot-Andrews unit root test in presence of structural break,"
           Statistical Software Components S437301, Boston College Department
           of Economics, revised 2015.

        .. [2] Schwert, G.W. (1989). Tests for unit roots: A Monte Carlo
           investigation. Journal of Business & Economic Statistics, 7:
           147-159.

        .. [3] Zivot, E., and Andrews, D.W.K. (1992). Further evidence on the
           great crash, the oil-price shock, and the unit-root hypothesis.
           Journal of Business & Economic Studies, 10: 251-270.
        """
        x = array_like(x, "x", dtype=np.double, ndim=1)
        trim = float_like(trim, "trim")
        maxlag = int_like(maxlag, "maxlag", optional=True)
        regression = string_like(regression, "regression", options=("c", "t", "ct"))
        autolag = string_like(
            autolag, "autolag", options=("aic", "bic", "t-stat"), optional=True
        )
        if trim < 0 or trim > (1.0 / 3.0):
            raise ValueError("trim value must be a float in range [0, 1/3)")
        nobs = x.shape[0]
        if autolag:
            adf_res = adfuller(
                x,
                maxlag=maxlag,
                regression="ct",
                autolag=autolag,
                result_object=False,
            )
            baselags = adf_res[2]
        elif maxlag:
            baselags = maxlag
        else:
            baselags = int(12.0 * np.power(nobs / 100.0, 1 / 4.0))
        trimcnt = int(nobs * trim)
        start_period = trimcnt
        end_period = nobs - trimcnt
        if regression == "ct":
            basecols = 5
        else:
            basecols = 4
        # normalize constant and trend terms for stability
        c_const = 1 / np.sqrt(nobs)
        t_const = np.arange(1.0, nobs + 2)
        t_const *= np.sqrt(3) / nobs ** (3 / 2)
        # format the auxiliary regression data
        endog, exog = self._format_regression_data(
            x, nobs, c_const, t_const, basecols, baselags
        )
        # iterate through the time periods
        stats = np.full(end_period + 1, np.inf)
        for bp in range(start_period + 1, end_period + 1):
            # update intercept dummy / trend / trend dummy
            exog = self._update_regression_exog(
                exog,
                regression,
                bp,
                nobs,
                c_const,
                t_const,
                basecols,
                baselags,
            )
            # check exog rank on first iteration
            if bp == start_period + 1:
                o = OLS(endog[baselags:], exog, hasconst=1).fit()
                if o.df_model < exog.shape[1] - 1:
                    raise ValueError(
                        "ZA: auxiliary exog matrix is not full rank.\n"
                        f"  cols (exc intercept) = {exog.shape[1] - 1}  rank = {o.df_model}"
                    )
                stats[bp] = o.tvalues[basecols - 1]
            else:
                stats[bp] = self._quick_ols(endog[baselags:], exog)[basecols - 1]
        # return best seen
        zastat = np.min(stats)
        bpidx = np.argmin(stats) - 1
        crit = self._za_crit(zastat, regression)
        pval = crit[0]
        cvdict = crit[1]
        return zastat, pval, cvdict, baselags, bpidx

    def __call__(self, x, trim=0.15, maxlag=None, regression="c", autolag="AIC"):
        return self.run(
            x, trim=trim, maxlag=maxlag, regression=regression, autolag=autolag
        )


zivot_andrews = ZivotAndrewsUnitRoot()
zivot_andrews.__doc__ = zivot_andrews.run.__doc__
