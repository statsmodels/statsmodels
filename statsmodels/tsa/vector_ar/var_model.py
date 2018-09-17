# -*- coding: utf-8 -*-
"""
Vector Autoregression (VAR) processes

References
----------
Lütkepohl, H., 2005. New introduction to multiple time series analysis.
Springer Science & Business Media.

Hamilton, J.D., 1994. Time series analysis. Princeton, NJ: Princeton
University Press.
"""

from __future__ import division, print_function
from statsmodels.compat.python import (range, lrange, string_types, StringIO,
                                       iteritems)

from collections import defaultdict

import numpy as np
try:
    from pandas.util._decorators import deprecate_kwarg
except ImportError:
    from pandas.util.decorators import deprecate_kwarg
import scipy.linalg
import scipy.stats as stats

import statsmodels.base.wrapper as wrap
import statsmodels.tsa.base.tsa_model as tsbase
import statsmodels.tsa.tsatools as tsa
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.linalg import logdet_symm
from statsmodels.tools.sm_exceptions import OutputWarning
from statsmodels.tools.tools import chain_dot, Bunch
from statsmodels.tsa.tsatools import vec, unvec, duplication_matrix
from statsmodels.tsa.vector_ar import output, plotting, util
from statsmodels.tsa.vector_ar.hypothesis_test_results import \
    CausalityTestResults, NormalityTestResults, WhitenessTestResults
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.output import VARSummary

# ----------------------------------------------------------------------------
# VAR process routines


def ma_rep(coefs, maxn=10):
    r"""
    MA representation of VAR(p) process

    Parameters
    ----------
    coefs : ndarray (p x k x k)
        VAR coefficients
    maxn : int
        Number of MA matrices to compute

    Returns
    -------
    ma_coefs : ndarray (maxn + 1 x k x k)
        Coefficient of MA process for lags 0 to maxn

    Notes
    -----
    VAR(p) process as

    .. math:: y_t = A_1 y_{t-1} + \ldots + A_p y_{t-p} + u_t

    can be equivalently represented as

    .. math:: y_t = \mu + \sum_{i=0}^\infty \Phi_i u_{t-i}

    The MA coefficients are :math:`\Phi_i` where :math:`\Phi_0 = I_k`.
    """
    p, k, k = coefs.shape
    phis = np.zeros((maxn + 1, k, k))
    phis[0] = np.eye(k)

    # recursively compute Phi matrices
    for i in range(1, maxn + 1):
        for j in range(1, i + 1):
            if j > p:
                break

            phis[i] += np.dot(phis[i - j], coefs[j - 1])

    return phis


@deprecate_kwarg('verbose', 'eigenvalues')
def is_stable(coefs, eigenvalues=False):
    """
    Test stability of a VAR(p)

    Parameters
    ----------
    coefs : ndarray (p x k x k)
        VAR coefficients
    eigenvalues : bool
        Flag indicating to also return eigenvalues

    Returns
    -------
    is_stable : bool
        Indicator that the maximum eigenvalue is less than 1
    eigenvals : array, optional
        Array of eigenvalues of the VAR(1) coefficient in the companion form

    Notes
    -----
    Computes the eigenvalues of the coefficient matrix in the the companion
    form of the VAR(p).
    """
    a_var1 = util.comp_matrix(coefs)
    eigs = np.linalg.eigvals(a_var1)
    stable = (np.abs(eigs) <= 1).all()
    if eigenvalues:
        return stable, eigs
    return stable


@deprecate_kwarg('sig_u', 'cov_resid')
def var_acf(coefs, cov_resid, nlags=None):
    """
    Compute the autocovariance function of a stable VAR(p) process

    Parameters
    ----------
    coefs : ndarray
        Coefficient matrices of the VAR (p, k,  k)
    cov_resid : ndarray
        Covariance of the white noise process residuals (k, k)
    nlags : int, optional
        Number of ACF lags. Default value is p.

    Returns
    -------
    acf : ndarray
        Autocorrelation and cross correlations (nlags, k, k)

    Notes
    -----
    Ref: Lütkepohl p.28-29
    """
    p, k, _ = coefs.shape
    if nlags is None:
        nlags = p

    # p x k x k, ACF for lags 0, ..., p-1
    result = np.zeros((nlags + 1, k, k))
    result[:p] = _var_acf(coefs, cov_resid)

    # yule-walker equations
    for h in range(p, nlags + 1):
        # compute ACF for lag=h
        # G(h) = A_1 G(h-1) + ... + A_p G(h-p)

        for j in range(p):
            result[h] += np.dot(coefs[j], result[h - j - 1])

    return result


def _var_acf(coefs, cov_resid):
    """
    Compute autocovariance function ACF_y(h) for h=1,...,p

    Parameters
    ----------
    coefs : ndarray (p x k x k)
        Coefficient matrices A_i
    cov_resid : ndarray (k x k)
        Covariance of white noise process u_t

    Notes
    -----
    Lütkepohl (2005) p.29
    """
    p, k, k2 = coefs.shape

    a = util.comp_matrix(coefs)
    # construct VAR(1) noise covariance
    companion_cov_resid = np.zeros((k * p, k * p))
    companion_cov_resid[:k, :k] = cov_resid

    # vec(ACF) = (I_(kp)^2 - kron(A, A))^-1 vec(Sigma_U)
    vec_acf = scipy.linalg.solve(np.eye((k * p) ** 2) - np.kron(a, a),
                                 vec(companion_cov_resid))

    acf = unvec(vec_acf)
    acf = acf[:k].T.reshape((p, k, k))

    return acf


def forecast(y, coefs, trend_coefs, steps, exog=None):
    """
    Produce linear minimum MSE forecast

    Parameters
    ----------
    y : ndarray (k_ar x neqs)
        The initial values to use for the forecasts.
    coefs : ndarray
        Each of the k_ar matrices are for lag 1, ... , lag k_ar. Where the
        columns are the variable and the rows are the equations so that
        coefs[i-1] is the estimated A_i matrix of the VAR. (k_ar, neqs, neqs)
        See VARResults Notes.
    trend_coefs : ndarray
        1d or 2d array. If 1d, should be of length neqs and is assumed to be
        a vector of constants. If 2d should be of shape k_trend x neqs.
    steps : int
        Number of steps ahead to forecast
    exog : ndarray
        The exogenous variables. Should include constant, trend, etc. as
        needed, including extrapolating out of sample.

    Returns
    -------
    forecasts : ndarray
        Forecast values (steps, neqs)

    Notes
    -----
    Lütkepohl p. 37

    Also used by DynamicVAR class
    """
    p = len(coefs)
    k = len(coefs[0])
    # initial value
    forcs = np.zeros((steps, k))
    if exog is not None and trend_coefs is not None:
        forcs += np.dot(exog, trend_coefs)
    # to make existing code (with trend_coefs=intercept and without exog) work:
    elif exog is None and trend_coefs is not None:
        forcs += trend_coefs

    # h=0 forecast should be latest observation
    # forcs[0] = y[-1]

    # make indices easier to think about
    for h in range(1, steps + 1):
        # y_t(h) = intercept + sum_1^p A_i y_t_(h-i)
        f = forcs[h - 1]
        for i in range(1, p + 1):
            # slightly hackish
            if h - i <= 0:
                # e.g. when h=1, h-1 = 0, which is y[-1]
                prior_y = y[h - i - 1]
            else:
                # e.g. when h=2, h-1=1, which is forcs[0]
                prior_y = forcs[h - i - 1]

            # i=1 is coefs[0]
            f = f + np.dot(coefs[i - 1], prior_y)

        forcs[h - 1] = f

    return forcs


@deprecate_kwarg('sig_u', 'cov_resid')
def forecast_cov(ma_coefs, cov_resid, steps):
    """
    Compute theoretical forecast error variance matrices

    Parameters
    ----------
    ma_coefs : ndarray
        Coefficients from the moving average representation (steps, k, k)
    cov_resid : ndarray
        Covariance of white noise process u_t (k, k)
    steps : int
        Number of steps ahead

    Notes
    -----
    .. math:: \mathrm{MSE}(h) = \sum_{i=0}^{h-1} \Phi \Sigma_u \Phi^T

    Returns
    -------
    forc_covs : ndarray
        Covariance of the forecast errors (steps, neqs, neqs)
    """
    k = len(cov_resid)
    forc_covs = np.zeros((steps, k, k))

    prior = np.zeros((k, k))
    for h in range(steps):
        # Sigma(h) = Sigma(h-1) + Phi Sig_u Phi'
        phi = ma_coefs[h]
        var = chain_dot(phi, cov_resid, phi.T)
        forc_covs[h] = prior = prior + var

    return forc_covs


mse = forecast_cov


def _forecast_vars(steps, ma_coefs, cov_resid):
    """_forecast_vars function used by VECMResults. Note that the definition
    of the local variable covs is the same as in VARProcess and as such it
    differs from the one in VARResults!

    Parameters
    ----------
    steps : int
        Number of period to compute forecast variances
    ma_coefs : ndarray
        MA coefficients (steps, neqs, neqs)
    cov_resid : ndarray
        Residual covariance (neqs, neqs)

    Returns
    -------
    covs : ndarray
        Forecast variances (steps, neqs)
    """
    covs = mse(ma_coefs, cov_resid, steps)
    # Take diagonal for each cov
    neqs = len(cov_resid)
    inds = np.arange(neqs)
    return covs[:, inds, inds]


def forecast_interval(y, coefs, trend_coefs, cov_resid, steps=5, alpha=0.05,
                      exog=1):
    """Construct forecast interval estimates assuming the y are Gaussian

    Parameters
    ----------
    y : ndarray (k_ar x neqs)
        The initial values to use for the forecasts.
    coefs : ndarray
        Each of the k_ar matrices are for lag 1, ... , lag k_ar. Where the
        columns are the variable and the rows are the equations so that
        coefs[i-1] is the estimated A_i matrix of the VAR. (k_ar, neqs, neqs)
        See VARResults Notes.
    trend_coefs : ndarray
        1d or 2d array. If 1d, should be of length neqs and is assumed to be
        a vector of constants. If 2d should be of shape k_trend x neqs.
    cov_resid : ndarray
        Residual covariance (neqs, neqs)
    steps : int
        Number of steps ahead to forecast
    alpha : float, optional
        The significance level for the confidence intervals.
    exog : ndarray
        The exogenous variables. Should include constant, trend, etc. as
        needed, including extrapolating out of sample.

    Returns
    -------
    point : ndarray
        Mean value of forecast
    lower : ndarray
        Lower bound of confidence interval
    upper : ndarray
        Upper bound of confidence interval
    """
    assert (0 < alpha < 1)
    q = util.norm_signif_level(alpha)

    point_forecast = forecast(y, coefs, trend_coefs, steps, exog)
    ma_coefs = ma_rep(coefs, steps)
    sigma = np.sqrt(_forecast_vars(steps, ma_coefs, cov_resid))

    forc_lower = point_forecast - q * sigma
    forc_upper = point_forecast + q * sigma

    return point_forecast, forc_lower, forc_upper


def var_loglike(resid, cov_resid, nobs):
    r"""
    VAR(p) log-likelihood assuming normality.

    Parameters
    ----------
    resid : ndarray
        The residuals of each variable (nobs, neqs)
    cov_resid : ndarray
        This is the maximum likelihood estimate for the equation by equation
        residual covariance. Each element i,j is the average product of the
        OLS residual for variable i and the OLS residual for variable j or
        np.dot(resid.T,resid)/nobs. There should be no correction for the
        degrees of freedom.
    nobs : int
        The number of observations used during fitting. Does not include
        the `k_ar` pre-sample observations in Y.

    Returns
    -------
    llf : float
        Loglikelihood

    Notes
    -----
    The loglikelihood function for the VAR(p) is

    .. math::

        -\left(\frac{nobs}{2}\right)
        \left(\ln\left|\Omega\right|-neqs \ln\left(2\pi\right)-K\right)

    where :math:`Omega` is the residual covariance.
    """
    logdet = logdet_symm(np.asarray(cov_resid))
    neqs = cov_resid.shape[0]
    part1 = - (nobs * neqs / 2) * np.log(2 * np.pi)
    part2 = - (nobs / 2) * (logdet + neqs)
    return part1 + part2


def _reordered(self, order):
    """
    Construct a VARResults with the variables reordered according to order

    Parameters
    ----------
    self : VARResult instance
    order: {List[int], List[str]}
        New order.  If integers, then interpreted as the index of the new
        result. If str, then interpreted as the order of the variable names in
        the new result.

    Returns
    -------
    result : VARResults
        Reordered result
    """
    # Create new arrays to hold rearranged results from .fit()
    endog = self.endog
    endog_lagged = self.endog_lagged
    params = self.params
    cov_resid = self.cov_resid
    names = self.names
    k_ar = self.k_ar
    endog_reorder = np.zeros_like(endog)
    endog_lagged_reorder = np.zeros_like(endog_lagged)
    params_reorder_inc = np.zeros_like(params)
    params_reorder = np.zeros_like(params)
    cov_resid_reorder_inc = np.zeros_like(cov_resid)
    cov_resid_reorder = np.zeros_like(cov_resid)

    num_end = len(self.params[0])
    names_new = []

    # Rearrange elements and fill in new arrays
    k = self.k_trend
    for i, c in enumerate(order):
        endog_reorder[:, i] = self.endog[:, c]
        if k > 0:
            params_reorder_inc[0, i] = params[0, i]
            endog_lagged_reorder[:, 0] = endog_lagged[:, 0]
        for j in range(k_ar):
            _params = self.params[c + j * num_end + k, :]
            params_reorder_inc[i + j * num_end + k, :] = _params
            lag = endog_lagged[:, c + j * num_end + k]
            endog_lagged_reorder[:, i + j * num_end + k] = lag
        cov_resid_reorder_inc[i, :] = cov_resid[c, :]
        names_new.append(names[c])
    for i, c in enumerate(order):
        params_reorder[:, i] = params_reorder_inc[:, c]
        cov_resid_reorder[:, i] = cov_resid_reorder_inc[:, c]

    return VARResults(endog=endog_reorder, endog_lagged=endog_lagged_reorder,
                      params=params_reorder, cov_resid=cov_resid_reorder,
                      lag_order=self.k_ar, model=self.model,
                      trend='c', names=names_new, dates=self.dates)


@deprecate_kwarg('P', 'p')
def orth_ma_rep(results, maxn=10, p=None):
    r"""
    Compute orthogonalized MA coefficient matrices

    Parameters
    ----------
    results : {VARResults, VECMResults}
        Results instance
    maxn : int
        Number of coefficient matrices to compute
    p : ndarray, optional
        Matrix such that results.cov_resid = pp'. Defaults to the Cholesky
        decomposition of results.cov_resid if not provided

    Returns
    -------
    coefs : ndarray (maxn x neqs x neqs)
    """
    if p is None:
        p = results._chol_cov_resid

    ma_mats = results.ma_rep(maxn=maxn)
    return np.array([np.dot(coefs, p) for coefs in ma_mats])


def test_normality(results, signif=0.05):
    """
    Test assumption of normal-distributed errors using Jarque-Bera-style
    omnibus Chi^2 test

    Parameters
    ----------
    results : {VARResults, VECMResults}
        Results instance
    signif : float
        The test's significance level.

    Returns
    -------
    result : NormalityTestResults
        Result instance containing the test statistic and related quantities

    Notes
    -----
    H0 (null) : data are generated by a Gaussian-distributed process

    References
    ----------
    .. [*] Lütkepohl, H., 2005. New introduction to multiple time series
       analysis. Springer Science & Business Media.

    .. [*] Kilian, L. & Demiroglu, U. (2000). "Residual-Based Tests for
       Normality in Autoregressions: Asymptotic Theory and Simulation
       Evidence." Journal of Business & Economic Statistics
    """
    resid_c = results.resid - results.resid.mean(0)
    sig = np.dot(resid_c.T, resid_c) / results.nobs
    p_inv = np.linalg.inv(np.linalg.cholesky(sig))

    w = np.dot(p_inv, resid_c.T)
    b1 = (w ** 3).sum(1)[:, None] / results.nobs
    b2 = (w ** 4).sum(1)[:, None] / results.nobs - 3

    lam_skew = results.nobs * np.dot(b1.T, b1) / 6
    lam_kurt = results.nobs * np.dot(b2.T, b2) / 24

    lam_omni = float(lam_skew + lam_kurt)
    omni_dist = stats.chi2(results.neqs * 2)
    omni_pvalue = float(omni_dist.sf(lam_omni))
    crit_omni = float(omni_dist.ppf(1 - signif))

    return NormalityTestResults(lam_omni, crit_omni, omni_pvalue,
                                results.neqs * 2, signif)


class LagOrderResults(object):
    """
    Results class for choosing a model's lag order.

    Parameters
    ----------
    ics : dict
        The keys are the strings ``"aic"``, ``"bic"``, ``"hqic"``, and
        ``"fpe"``. A corresponding value is a list of information criteria for
        various numbers of lags.
    selected_orders: dict
        The keys are the strings ``"aic"``, ``"bic"``, ``"hqic"``, and
        ``"fpe"``. The corresponding value is an integer specifying the number
        of lags chosen according to a given criterion (key).

    Notes
    -----
    In case of a VECM the shown lags are lagged differences.
    """

    def __init__(self, ics, selected_orders, vecm=None):
        if vecm is not None:
            # Deprecated in 0.10.0, remove in 0.11.0
            import warnings
            warnings.warn('vecm is deprecated and instance type is '
                          'automatically detected.', DeprecationWarning)
        # Circular import delay
        from statsmodels.tsa.vector_ar.vecm import VECMResults
        self.vecm = isinstance(self, VECMResults)
        self.title = ('VECM' if self.vecm else 'VAR') + ' Order Selection'
        self.title += ' (* highlights the minimums)'
        self.ics = ics
        self.selected_orders = selected_orders
        self.aic = selected_orders['aic']
        self.bic = selected_orders['bic']
        self.hqic = selected_orders['hqic']
        self.fpe = selected_orders['fpe']

    def summary(self):
        """
        Summary of information criteria

        Returns
        -------
        summ : SimpleTable
            Table that supports printing results or export
        """
        cols = sorted(self.ics)  # ["aic", "bic", "hqic", "fpe"]
        str_data = [['{:10.4g}'.format(v) for v in self.ics[c]] for c in cols]
        str_data = np.array(str_data, dtype=object).T
        # mark minimum with an asterisk
        for i, col in enumerate(cols):
            idx = int(self.selected_orders[col]), i
            str_data[idx] += '*'
        return SimpleTable(str_data, [col.upper() for col in cols],
                           lrange(len(str_data)), title=self.title)

    def __str__(self):
        return "<" + self.__module__ + "." + self.__class__.__name__ \
               + " object. Selected orders are: AIC -> " + str(self.aic) \
               + ", BIC -> " + str(self.bic) \
               + ", FPE -> " + str(self.fpe) \
               + ", HQIC -> " + str(self.hqic) + ">"


# ----------------------------------------------------------------------------
# VARProcess class: for known or unknown VAR process


class VAR(tsbase.TimeSeriesModel):
    r"""
    Fit VAR(p) process and do lag order selection

    Parameters
    ----------
    endog : array-like
        2-d endogenous response variable. The independent variable.
    exog : array-like
        2-d exogenous variable.
    dates : array-like
        must match number of rows of endog
    freq : str, optional
        The frequency of the time-series. A Pandas offset or 'B', 'D', 'W',
        'M', 'A', or 'Q'. This is optional if dates are given.
    missing : str
        Available options are 'none', 'drop', and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default is 'none.'

    Notes
    -----
    A VAR(p) process has dynamics described by

    .. math:: y_t = A_1 y_{{t-1}} + \ldots + A_p y_{{t-p}} + u_t

    References
    ----------
    .. [*] Lütkepohl, H., 2005. New introduction to multiple time series
       analysis. Springer Science & Business Media.
    """
    # TODO: Use existing strings in signature for missing and freq?
    def __init__(self, endog, exog=None, dates=None, freq=None,
                 missing='none'):
        super(VAR, self).__init__(endog, exog, dates, freq, missing=missing)
        if self.endog.ndim == 1:
            raise ValueError("Only gave one variable to VAR")
        self.neqs = self.endog.shape[1]
        self.n_totobs = len(endog)

    def predict(self, params, start=None, end=None, lags=1, trend='c'):
        """
        In-sample predictions or forecasts

        Parameters
        ----------
        params : ndarray
            Model parameters including trend.
        start : label
            The key at which to start prediction. Depending on the underlying
            model's index, may be an integer, a date (string, datetime object,
            pd.Timestamp, or pd.Period object), or some other object in the
            model's row labels.
        end : label
            The key at which to end prediction (note that this key will be
            *included* in prediction). Depending on the underlying
            model's index, may be an integer, a date (string, datetime object,
            pd.Timestamp, or pd.Period object), or some other object in the
            model's row labels.
        lags : int
            Number of lags in the model
        trend : str
            Order of the trend

        Returns
        -------
        predictions : ndarray
            Array containing predictions. (ninterval, neqs) where ninterval is
            the number of observations between start and end.
        """
        params = np.array(params)

        if start is None:
            start = lags

        # Handle start, end
        start, end, out_of_sample, prediction_index = (
            self._get_prediction_index(start, end))

        if end < start:
            raise ValueError("end is before start")
        if end == start + out_of_sample:
            return np.array([])

        k_trend = util.get_trendorder(trend)
        k = self.neqs
        k_ar = lags

        predictedvalues = np.zeros((end + 1 - start + out_of_sample, k))
        if k_trend != 0:
            intercept = params[:k_trend]
            predictedvalues += intercept
        else:
            intercept = np.empty((0,))

        y = self.endog
        x = util.get_var_endog(y, lags, trend=trend, has_constant='raise')
        fittedvalues = np.dot(x, params)

        fv_start = start - k_ar
        pv_end = min(len(predictedvalues), len(fittedvalues) - fv_start)
        fv_end = min(len(fittedvalues), end - k_ar + 1)
        predictedvalues[:pv_end] = fittedvalues[fv_start:fv_end]

        if not out_of_sample:
            return predictedvalues

        # fit out of sample
        y = y[-k_ar:]
        coefs = params[k_trend:].reshape((k_ar, k, k)).swapaxes(1, 2)
        predictedvalues[pv_end:] = forecast(y, coefs, intercept, out_of_sample)
        return predictedvalues

    def fit(self, maxlags=None, method='ols', ic=None, trend='c',
            verbose=None):
        """
        Fit the VAR model

        Parameters
        ----------
        maxlags : int
            Maximum number of lags to check for order selection, defaults to
            12 * (nobs/100.)**(1./4), see select_order function
        method : {'ols'}
            Estimation method to use
        ic : {'aic', 'fpe', 'hqic', 'bic', None}
            Information criterion to use for VAR order selection.
            aic : Akaike
            fpe : Final prediction error
            hqic : Hannan-Quinn
            bic : Bayesian/Schwarz
        trend : str {"c", "ct", "ctt", "nc"}
            "c" - add constant
            "ct" - constant and trend
            "ctt" - constant, linear and quadratic trend
            "nc" - co constant, no trend
            Note that these are prepended to the columns of the dataset.

        Returns
        -------
        est : VARResults
            Results instance

        Notes
        -----
        Lütkepohl pp. 146-153
        """
        # todo: this code is only supporting deterministic terms as exog.
        # This means that all exog-variables have lag 0. If dealing with
        # different exogs is necessary, a `lags_exog`-parameter might make
        # sense (e.g. a sequence of ints specifying lags).
        # Alternatively, leading zeros for exog-variables with smaller number
        # of lags than the maximum number of exog-lags might work.

        if verbose is not None:
            # Deprecated in 0.10.0, remove in 0.11.0
            import warnings
            warnings.warn('verbose is deprecated.', DeprecationWarning)

        lags = maxlags

        if trend not in ['c', 'ct', 'ctt', 'nc']:
            raise ValueError("trend '{}' not supported for VAR".format(trend))

        if ic is not None:
            selections = self.select_order(maxlags=maxlags)
            if not hasattr(selections, ic):
                raise ValueError("%s not recognized, must be among %s"
                                 % (ic, sorted(selections)))
            lags = getattr(selections, ic)
        else:
            if lags is None:
                lags = 1

        k_trend = util.get_trendorder(trend)
        self.exog_names = util.make_lag_names(self.endog_names, lags, k_trend)
        self.nobs = self.n_totobs - lags

        # add exog to data.xnames (necessary because the length of xnames also
        # determines the allowed size of VARResults.params)
        if self.exog is not None:
            x_names_to_add = [("exog%d" % i)
                              for i in range(self.exog.shape[1])]
            self.data.xnames = (self.data.xnames[:k_trend] +
                                x_names_to_add +
                                self.data.xnames[k_trend:])

        return self._estimate_var(lags, trend=trend)

    def _estimate_var(self, lags, offset=0, trend='c'):
        """
        lags : int
            Lags of the endogenous variable.
        offset : int
            Periods to drop from beginning-- for order selection so it's an
            apples-to-apples comparison
        trend : string or None
            As per above
        """
        # have to do this again because select_order doesn't call fit
        self.k_trend = k_trend = util.get_trendorder(trend)

        if offset < 0:  # pragma: no cover
            raise ValueError('offset must be >= 0')

        nobs = self.n_totobs - lags - offset
        endog = self.endog[offset:]
        exog = None if self.exog is None else self.exog[offset:]
        z = util.get_var_endog(endog, lags, trend=trend,
                               has_constant='raise')
        if exog is not None:
            # TODO: currently only deterministic terms supported (exoglags==0)
            # and since exoglags==0, x will be an array of size 0.
            x = util.get_var_endog(exog[-nobs:], 0, trend="nc",
                                   has_constant="raise")
            x_inst = exog[-nobs:]
            x = np.column_stack((x, x_inst))
            del x_inst  # free memory
            temp_z = z
            z = np.empty((x.shape[0], x.shape[1] + z.shape[1]))
            z[:, :self.k_trend] = temp_z[:, :self.k_trend]
            z[:, self.k_trend:self.k_trend + x.shape[1]] = x
            z[:, self.k_trend + x.shape[1]:] = temp_z[:, self.k_trend:]
            del temp_z, x  # free memory
        # the following modification of z is necessary to get the same results
        # as JMulTi for the constant-term-parameter...
        for i in range(self.k_trend):
            if (np.diff(z[:, i]) == 1).all():  # modify the trend-column
                z[:, i] += lags
            # make the same adjustment for the quadratic term
            if (np.diff(np.sqrt(z[:, i])) == 1).all():
                z[:, i] = (np.sqrt(z[:, i]) + lags) ** 2

        y_sample = endog[lags:]
        # Lütkepohl p75, about 5x faster than stated formula
        params = np.linalg.lstsq(z, y_sample, rcond=1e-15)[0]
        resid = y_sample - np.dot(z, params)

        # Unbiased estimate of covariance matrix $\Sigma_u$ of the white noise
        # process $u$
        # equivalent definition
        # .. math:: \frac{1}{T - Kp - 1} Y^\prime (I_T - Z (Z^\prime Z)^{-1}
        # Z^\prime) Y
        # Ref: Lütkepohl p.75
        # df_resid right now is T - Kp - 1, which is a suggested correction

        avobs = len(y_sample)
        if exog is not None:
            k_trend += exog.shape[1]
        df_resid = avobs - (self.neqs * lags + k_trend)

        sse = np.dot(resid.T, resid)
        omega = sse / df_resid

        varfit = VARResults(endog, z, params, omega, lags,
                            names=self.endog_names, trend=trend,
                            dates=self.data.dates, model=self, exog=self.exog)
        return VARResultsWrapper(varfit)

    def select_order(self, maxlags=None, trend="c"):
        """
        Compute lag order selections based on each of the available information
        criteria

        Parameters
        ----------
        maxlags : int
            if None, defaults to int(12 * (nobs/100.)**(1./4))
        trend : str {"nc", "c", "ct", "ctt"}
            * "nc" - no deterministic terms
            * "c" - constant term
            * "ct" - constant and linear term
            * "ctt" - constant, linear, and quadratic term

        Returns
        -------
        selections : LagOrderResults
        """
        if maxlags is None:
            # TODO: Consider using a function for this common lag length
            maxlags = int(round(12 * (len(self.endog) / 100.)**(1 / 4.)))

        ics = defaultdict(list)
        p_min = 0 if self.exog is not None or trend != "nc" else 1
        for p in range(p_min, maxlags + 1):
            # exclude some periods to same amount of data used for each lag
            # order
            result = self._estimate_var(p, offset=maxlags - p, trend=trend)

            for k, v in iteritems(result.info_criteria):
                ics[k].append(v)

        selected_orders = dict((k, np.array(v).argmin() + p_min)
                               for k, v in iteritems(ics))

        return LagOrderResults(ics, selected_orders)


class VARProcess(object):
    """
    Representation of a VAR(p) with known parameters

    Parameters
    ----------
    coefs : ndarray (p x neqs x neqs)
        Each of the p matrices are for lag 1, ... , lag p. Where the columns
        are the variable and the rows are the equations so that coefs[i-1] is
        the estimated A_i matrix. See Notes.
    coefs_exog : ndarray
        parameters for trend and user provided exog
    cov_resid : ndarray (neqs x neqs)
        The covariance matrix of the residuals. :math:`\Sigma_u` in the Notes.
    names : sequence (length k)
    _params_info : dict
        internal dict to provide information about the composition of `params`,
        specifically `k_trend` (trend order) and `k_exog_user` (the number of
        exog variables provided by the user).
        If it is None, then coefs_exog are assumed to be for the intercept and
        trend.

    Returns
    -------
    **Attributes**:
    """
    def __init__(self, coefs, coefs_exog, cov_resid, names=None, _params_info=None):
        self.k_ar = len(coefs)
        self.neqs = coefs.shape[1]
        self.coefs = coefs
        self.coefs_exog = coefs_exog
        # TODO: Verify this claim
        # TODO: Note reshaping 1-D coefs_exog to 2_D makes unit tests fail
        self.cov_resid = cov_resid
        self.names = names

        if _params_info is None:
            _params_info = {}
        self.k_exog_user = _params_info.get('k_exog_user', 0)
        if self.coefs_exog is not None:
            k_ex = self.coefs_exog.shape[0] if self.coefs_exog.ndim != 1 else 1
            k_c = k_ex - self.k_exog_user
        else:
            k_c = 0
        self.k_trend = _params_info.get('k_trend', k_c)
        # TODO: we need to distinguish exog including trend and exog_user
        self.k_exog = self.k_trend + self.k_exog_user

        if self.k_trend > 0:
            if coefs_exog.ndim == 2:
                self.intercept = coefs_exog[:, 0]
            else:
                self.intercept = coefs_exog
        else:
            self.intercept = np.zeros(self.neqs)

    def get_eq_index(self, name):
        """Return integer position of requested equation name"""
        return util.get_index(self.model.endog_names, name)

    def __str__(self):
        output = ('VAR(%d) process for %d-dimensional response y_t'
                  % (self.k_ar, self.neqs))
        output += '\nstable: %s' % self.is_stable()
        output += '\nmean: %s' % self.mean()

        return output

    @deprecate_kwarg('verbose', 'eigenvalues')
    def is_stable(self, eigenvalues=False):
        """Determine stability based on model coefficients

        Parameters
        ----------
        eigenvalues : bool
            Print eigenvalues of the VAR(1) companion

        Returns
        -------
        is_stable : bool
            Indicator that the maximum eigenvalue is less than 1
        eigenvals : array, optional
            Array of eigenvalues of the VAR(1) coefficient in the companion
            form

        Notes
        -----
        Checks if det(I - Az) = 0 for any mod(z) <= 1, so all the eigenvalues
        of the companion matrix must lie outside the unit circle
        """
        return is_stable(self.coefs, eigenvalues=eigenvalues)

    def simulate_var(self, steps=None, offset=None, seed=None):
        """
        Simulate the VAR(p) process for the desired number of steps

        Parameters
        ----------
        steps : None or int
            number of observations to simulate, this includes the initial
            observations to start the autoregressive process.
            If offset is not None, then exog of the model are used if they were
            provided in the model
        offset : None or ndarray (steps, neqs)
            If not None, then offset is added as an observation specific
            intercept to the autoregression. If it is None and either trend
            (including intercept) or exog were used in the VAR model, then
            the linear predictor of those components will be used as offset.
            This should have the same number of rows as steps, and the same
            number of columns as endogenous variables (neqs).
        seed : {None, integer, np.random.RandomState}
            If seed is not None, then it will be used with for the random
            variables generated by numpy.random. If a RandomState is provided
            then the random numbers will be produced using the method of the
            RandomState.

        Returns
        -------
        endog_simulated : nd_array
            Endog of the simulated VAR process
        """
        steps_ = None
        if offset is None:
            if self.k_exog_user > 0 or self.k_trend > 1:
                # if more than intercept
                # endog_lagged contains all regressors, trend, exog_user
                # and lagged endog, trimmed initial observations
                coefs = self.coefs_exog
                offset = self.endog_lagged[:, :self.k_exog].dot(coefs.T)
                steps_ = self.endog_lagged.shape[0]
            else:
                offset = self.intercept
        else:
            steps_ = offset.shape[0]

        # default, but over written if exog or offset are used
        if steps is None:
            if steps_ is None:
                steps = 1000
            else:
                steps = steps_
        else:
            if steps_ is not None and steps != steps_:
                raise ValueError('if exog or offset are used, then steps must'
                                 'be equal to their length or None')

        y = util.varsim(self.coefs, offset, self.cov_resid, steps=steps,
                        seed=seed)
        return y

    def plotsim(self, steps=None, offset=None, seed=None):
        """Deprecated.  Use plot_sim."""
        import warnings
        warnings.warn("plotsim is deprecated and will be removed in 0.11.0. "
                      "Use plot_sim.", DeprecationWarning)

        return self.plot_sim(steps, offset, seed)

    def plot_sim(self, steps=None, offset=None, seed=None):
        """
        Plot a simulation from the VAR(p) process.

        Parameters
        ----------
        steps : int
            The number of observations to simulate.
        offset : None or ndarray (steps, neqs)
            If not None, then offset is added as an observation specific
            intercept to the autoregression. If it is None and either trend
            (including intercept) or exog were used in the VAR model, then
            the linear predictor of those components will be used as offset.
            This should have the same number of rows as steps, and the same
            number of columns as endogenous variables (neqs).
        seed : None or integer
            If seed is not None, then it will be used with for the random
            variables generated by numpy.random. If a RandomState is provided
            then the random numbers will be produced using the method of the
            RandomState.
        """

        y = self.simulate_var(steps=steps, offset=offset, seed=seed)
        return plotting.plot_mts(y)

    def intercept_longrun(self):
        r"""
        Long run intercept of stable VAR process

        Lütkepohl eq. 2.1.23

        .. math:: \mu = (I - A_1 - \dots - A_p)^{-1} \alpha

        where :math:`\alpha` is the intercept (parameter of the constant) and
        A_1, ... A_p are the VAR coefficients.
        """
        return np.linalg.solve(self._char_mat, self.intercept)

    mean = intercept_longrun

    def ma_rep(self, maxn=10):
        r"""
        Compute MA representation of the VAR model

        Parameters
        ----------
        maxn : int
            Number of MA coefficients to compute

        Returns
        -------
        coefs : ndarray
            Moving average coefficients (maxn + 1, neqs, neqs). coefs[0] is an
            identity matrix.

        Notes
        -----
        These are the first `maxn` coefficient matrices of the
        MA(:math:`\infty`).
        """
        return ma_rep(self.coefs, maxn=maxn)

    def orth_ma_rep(self, maxn=10, p=None):
        r"""
        Compute orthogonalized MA coefficient matrices

        Parameters
        ----------
        maxn : int
            Number of coefficient matrices to compute
        p : ndarray, optional
            Matrix such that cov_resid = pp'. Defaults to the Cholesky
            decomposition of cov_resid if not provided

        Returns
        -------
        coefs : ndarray
            Array containing the coefficient (maxn x k x k)
        """
        return orth_ma_rep(self, maxn, p)

    def long_run_effects(self):
        """
        Compute long-run effect of a unit impulse

        Returns
        -------
        lr_effect : ndarray
            Long-run effect of an impulse

        Notes
        -----
        The long-run effect of an impulse is defined as the sum of all
        coefficients in the MA(:math:`\infty`) representation.

        .. math::

            \Psi_\infty = \sum_{i=0}^\infty \Phi_i

        where :math:`\Phi_i` is the MA coefficient at lag i.
        """
        return scipy.linalg.inv(self._char_mat)

    @cache_readonly
    def _chol_cov_resid(self):
        """Cholesky factor of the residual covariance"""
        return np.linalg.cholesky(self.cov_resid)

    @cache_readonly
    def _char_mat(self):
        """Characteristic matrix of the VAR"""
        return np.eye(self.neqs) - self.coefs.sum(0)

    def acf(self, nlags=None):
        """
        Autocovariance function

        Parameters
        ----------
        nlags : int or None
            The number of lags to include in the autocovariance function. The
            default is the number of lags included in the model.

        Returns
        -------
        acf : ndarray
            Autocovariance and cross covariances (nlags, neqs, neqs)
        """
        return var_acf(self.coefs, self.cov_resid, nlags=nlags)

    def acorr(self, nlags=None):
        """
        Autocorrelation function

        Parameters
        ----------
        nlags : int or None
            The number of lags to include in the autocovariance function. The
            default is the number of lags included in the model.

        Returns
        -------
        acorr : ndarray
            Autocorrelation and cross correlations (nlags, neqs, neqs)
        """
        return util.acf_to_acorr(self.acf(nlags=nlags))

    def plot_acorr(self, nlags=10, linewidth=8):
        """Plot theoretical autocorrelation function

        Parameters
        ----------
        nlags : int, optional
            Number of lags to plot
        linewidth : int, optional
            Width of the line in the plot

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure handle
        """
        fig = plotting.plot_full_acorr(self.acorr(nlags=nlags),
                                       linewidth=linewidth)
        return fig

    def forecast(self, y, steps, exog_future=None):
        """Minimum MSE forecasts for desired number of steps

        Parameters
        ----------
        y : ndarray (k_ar x neqs)
            Deprecated. The initial values to use for the forecasts. If None,
            the last k_ar values of the original endogenous variables are
            used. Use statsmodels.tsa.var.forecast instead if you need to
            give y values.
        steps : int
            The number of steps ahead to forecast.
        exog_future : ndarray
            Forecast values of the exogenous variable or future values of
            trends.

        Returns
        -------
        forecasts : ndarray
            The forecast values. (steps, neqs)

        Notes
        -----
        Lütkepohl pp 37-38
        """
        if self.exog is None and exog_future is not None:
            raise ValueError("No exog in model, so no exog_future supported "
                             "in forecast method.")
        if self.exog is not None and exog_future is None:
            raise ValueError("Please provide an exog_future argument to "
                             "the forecast method.")
        trend_coefs = None if self.coefs_exog.size == 0 else self.coefs_exog.T

        exogs = []
        if self.trend.startswith("c"):  # constant term
            exogs.append(np.ones(steps))
        exog_lin_trend = np.arange(self.n_totobs + 1,
                                   self.n_totobs + 1 + steps)
        if "t" in self.trend:
            exogs.append(exog_lin_trend)
        if "tt" in self.trend:
            exogs.append(exog_lin_trend ** 2)
        if exog_future is not None:
            exogs.append(exog_future)

        if exogs == []:
            exog_future = None
        else:
            exog_future = np.column_stack(exogs)
        return forecast(y, self.coefs, trend_coefs, steps, exog_future)

    def mse(self, steps):
        """
        Compute theoretical forecast error variance matrices

        Parameters
        ----------
        steps : int
            Number of steps ahead

        Returns
        -------
        forc_covs : ndarray
            Covariance of the forecast errors (steps, neqs, neqs)

        Notes
        -----
        .. math:: \mathrm{MSE}(h) = \sum_{i=0}^{h-1} \Phi_i \Sigma_u \Phi_i'

        where :math:`\Sigma_u` is the residual covariance and :math:`\Phi_i`
        are the coefficient from the MA(:math:`\infty`) representation.
        """
        ma_coefs = self.ma_rep(steps)
        return forecast_cov(ma_coefs, self.cov_resid, steps)

    forecast_cov = mse

    def _forecast_vars(self, steps):
        """Variance of forecast errors"""
        covs = self.forecast_cov(steps)

        # Take diagonal for each cov
        inds = np.arange(self.neqs)
        return covs[:, inds, inds]

    def forecast_interval(self, y, steps, alpha=0.05, exog_future=None):
        """
        Construct forecast interval estimates assuming residuals are Gaussian

        Parameters
        ----------
        y : {ndarray, None}
            The initial values to use for the forecasts. If None,
            the last k_ar values of the original endogenous variables are
            used.
        steps : int
            Number of steps ahead to forecast
        alpha : float, optional
            The significance level for the confidence intervals.
        exog_future : ndarray, optional
            Forecast values of the exogenous variables. Should include
            constant, trend, etc. as needed, including extrapolating out
            of sample.

        Returns
        -------
        point : ndarray
            Mean value of forecast
        lower : ndarray
            Lower bound of confidence interval
        upper : ndarray
            Upper bound of confidence interval

        Notes
        -----
        Lütkepohl pp. 39-40
        """
        if not 0 < alpha < 1:
            raise ValueError('alpha must be between 0 and 1')
        q = util.norm_signif_level(alpha)

        point_forecast = self.forecast(y, steps, exog_future=exog_future)
        sigma = np.sqrt(self._forecast_vars(steps))

        forc_lower = point_forecast - q * sigma
        forc_upper = point_forecast + q * sigma

        return point_forecast, forc_lower, forc_upper

    def to_vecm(self):
        """
        Transform a VAR to a VECM

        Returns
        -------
        params : Bunch
            Dict-like object with attributes Gamma and Pi

        Notes
        -----
        The parameters of the VECM are defined as

        .. math::

            \Pi = I_k - \sum_{i=1}^p A_i

        .. math::

            \Gamma_i = -\sum_{j=i+1}^p A_j
        """
        k = self.coefs.shape[1]
        p = self.coefs.shape[0]
        A = self.coefs
        pi = -(np.identity(k) - np.sum(A, 0))
        gamma = np.zeros((p - 1, k, k))
        for i in range(p - 1):
            gamma[i] = -(np.sum(A[i + 1:], 0))
        gamma = np.concatenate(gamma, 1)
        return Bunch(Gamma=gamma, Pi=pi)


# -------------------------------------------------------------------------------
# VARResults class and utilities


def _acovs_to_acorrs(acovs):
    """Convert 3-d array of autocovariances to autocorrelations"""
    sd = np.sqrt(np.diag(acovs[0]))
    return acovs / np.outer(sd, sd)


def _compute_acov(x, nlags=1):
    """Multivariate sample autocovariances"""
    x = x - x.mean(0)

    result = []
    for lag in range(nlags + 1):
        if lag > 0:
            r = np.dot(x[lag:].T, x[:-lag])
        else:
            r = np.dot(x.T, x)

        result.append(r)

    return np.array(result) / len(x)


class VARResults(VARProcess):
    """
    The results instances of a fitted VAR(p) model.

    Parameters
    ----------
    endog : ndarray
    endog_lagged : ndarray
    params : ndarray
    cov_resid : ndarray
    lag_order : int
    model : VAR model instance
    trend : {'nc', 'c', 'ct'}
    names : array-like
        List of names of the endogenous variables in order of appearance in
        `endog`.
    dates
    exog : array

    Attributes
    ----------
    coefs : ndarray
        Estimated A_i matrices, A_i = coefs[i-1] (p, neqs, neqs)
    cov_resid : ndarray
        Estimate of white noise process variance Var[u_t] (neqs, neqs)
    dates : {None, DatetimeIndex}
        Dates of observations
    endog : ndarray
        Endogenous data (nobs, neqs)
    endog_lagged : ndarray
        Endogenous data (nobs, p x neqs + k_exog).  Contains all lagged
        variables plus any trend or exogenous terms. TODO: Check exogenous
    intercept : ndarray
        Estimated intercepts (neqs, )
    k_ar : int
        Order of VAR process
    k_trend : int
        Number of trend terms in the model, including the constant
    model : VAR
        The model instance
    names : list
        Variable names
    neqs : int
        Number of variables (equations)
    nobs : int
        Number of observations used to estimate parameters after
        accounting for lags
    n_totobs : int
        Number of observations ignoring losses for lags
    params : ndarray
        Stacked A_i matrices and intercept in stacked form [int A_1 ... A_p]
        (neqs x k_ar + 1, neqs)
    trend : str
        String name of included trend terms (e.g., 'c' for a constant)
    """
    _model_type = 'VAR'

    def __init__(self, endog, endog_lagged, params, cov_resid, lag_order,
                 model=None, trend='c', names=None, dates=None, exog=None):

        self.model = model
        self.endog = endog
        self.endog_lagged = endog_lagged
        self.dates = dates

        self.n_totobs, neqs = self.endog.shape
        self.nobs = self.n_totobs - lag_order
        self.trend = trend
        k_trend = util.get_trendorder(trend)
        self.exog_names = util.make_lag_names(names, lag_order, k_trend, exog)
        self.params = params
        self.exog = exog

        # Initialize VARProcess parent class
        # construct coefficient matrices
        # Each matrix needs to be transposed
        endog_start = k_trend
        if exog is not None:
            k_exog_user = exog.shape[1]
            endog_start += k_exog_user
        else:
            k_exog_user = 0
        reshaped = self.params[endog_start:]
        reshaped = reshaped.reshape((lag_order, neqs, neqs))
        # Need to transpose each coefficient matrix
        coefs = reshaped.swapaxes(1, 2).copy()

        self.coefs_exog = params[:endog_start].T
        self.k_exog = self.coefs_exog.shape[1]
        self.k_exog_user = k_exog_user

        # maybe change to params class, distinguish exog_all versus exog_user
        # see issue #4535
        _params_info = {'k_trend': k_trend,
                        'k_exog_user': k_exog_user,
                        'k_ar': lag_order}
        super(VARResults, self).__init__(coefs, self.coefs_exog, cov_resid,
                                         names=names,
                                         _params_info=_params_info)

    def plot(self):
        """Plot input time series"""
        return plotting.plot_mts(self.endog, names=self.names,
                                 index=self.dates)

    @property
    def df_model(self):
        """
        Number of estimated parameters, including the intercept / trends
        """
        return self.neqs * self.k_ar + self.k_exog

    @property
    def df_resid(self):
        """
        Number of observations minus number of estimated parameters
        """
        return self.nobs - self.df_model

    @cache_readonly
    def fittedvalues(self):
        """
        The predicted in-sample values of the response variables of the model.
        """
        return np.dot(self.endog_lagged, self.params)

    @cache_readonly
    def resid(self):
        """
        Residuals of response variable resulting from estimated coefficients.
        """
        return self.endog[self.k_ar:] - self.fittedvalues

    def sample_acov(self, nlags=1):
        """
        Sample autocovariances of endog

        Parameters
        ----------
        nlags : int
            The number of lags to include. Does not count the zero lag, which
            will be returned.

        Returns
        -------
        acov : ndarray
            The autocovariace including the zero lag. Shape is
            (nlags + 1, neqs, neqs).
        """
        return _compute_acov(self.endog[self.k_ar:], nlags=nlags)

    def sample_acorr(self, nlags=1):
        """
        Sample autocorrelations of endog

        Parameters
        ----------
        nlags : int
            The number of lags to include. Does not count the zero lag, which
            will be returned.


        Returns
        -------
        acorr : ndarray
            The autocorrelation including the zero lag. Shape is
            (nlags + 1, neqs, neqs).
        """
        acovs = self.sample_acov(nlags=nlags)
        return _acovs_to_acorrs(acovs)

    def plot_sample_acorr(self, nlags=10, linewidth=8, **plot_kwargs):
        """
        Plot sample autocorrelation function

        Parameters
        ----------
        nlags : int
            The number of lags to use in compute the autocorrelation. Does
            not count the zero lag, which will be returned.
        linewidth : int
            The linewidth for the plots.
        plot_kwargs : kwargs
            Will be passed to `matplotlib.pyplot.axvlines`

        Returns
        -------
        fig : matplotlib.Figure
            The figure that contains the plot axes.
        """
        fig = plotting.plot_full_acorr(self.sample_acorr(nlags=nlags),
                                       linewidth=linewidth,
                                       xlabel=self.model.endog_names,
                                       **plot_kwargs)
        return fig

    def resid_acov(self, nlags=1):
        """
        Compute centered sample residual autocovariance (including lag 0)

        Parameters
        ----------
        nlags : int
            The number of lags to use in compute the autocovariace. Does
            not count the zero lag, which will be returned.

        Returns
        -------
        acov : ndarray
            The autocovariance for the residuals. The shape is
            (nlags + 1, neqs, neqs).
        """
        return _compute_acov(self.resid, nlags=nlags)

    def resid_acorr(self, nlags=1):
        """
        Compute sample autocorrelation (including lag 0)

        Parameters
        ----------
        nlags : int
            The number of lags to use in compute the autocorrelation. Does
            not count the zero lag, which will be returned.

        Returns
        -------
        acorr : ndarray
            The autocorrelation for the residuals. The shape is
            (nlags + 1, neqs, neqs).
        """
        acovs = self.resid_acov(nlags=nlags)
        return _acovs_to_acorrs(acovs)

    @cache_readonly
    def resid_corr(self):
        """
        Centered residual correlation matrix
        """
        return self.resid_acorr(0)[0]

    @property
    def sigma_u(self):
        """
        Debiased estimate of noise process covariance

        Notes
        -----
        Deprecated. Use cov_resid
        """
        # Deprecated in 0.10.0, remove in 0.11.0
        import warnings
        warnings.warn('sigma_u has been deprecated in favor of cov_resid',
                      DeprecationWarning)
        return self.cov_resid

    @property
    def sigma_u_mle(self):
        """
        Debiased estimate of noise process covariance

        Notes
        -----
        Deprecated. Use cov_resid
        """
        # Deprecated in 0.10.0, remove in 0.11.0
        import warnings
        warnings.warn('sigma_u_mle has been deprecated in favor of '
                      'cov_resid_mle', DeprecationWarning)
        return self.cov_resid

    @cache_readonly
    def cov_resid_mle(self):
        """
        Maximum likelihood estimate of noise process covariance

        Notes
        -----
        Has a finite sample bias. Differs from cov_resid by ratio
        df_resid / nobs
        """
        return self.cov_resid * self.df_resid / self.nobs

    @cache_readonly
    def cov_params(self):
        """Estimated variance-covariance of model coefficients

        Notes
        -----
        Covariance of vec(B), where B is the matrix

        [params_for_deterministic_terms, A_1, ..., A_p] with the shape (q, q)
        where q = neqs x (neqs x k_ar + number_of_deterministic_terms)

        Adjusted to be an unbiased estimator

        Ref: Lütkepohl p.74-75
        """
        z = self.endog_lagged
        return np.kron(scipy.linalg.inv(np.dot(z.T, z)), self.cov_resid)

    def cov_ybar(self):
        """
        Deprecated.  Use cov_sample_mean.
        """
        return self.cov_sample_mean()

    def cov_sample_mean(self):
        r"""Asymptotically consistent estimate of covariance of the sample mean

        .. math::

            \sqrt(T) (\bar{y} - \mu) \rightarrow
                                               {\cal N}(0, \Sigma_{\bar{y}}) \\

            \Sigma_{\bar{y}} = B \Sigma_u B^\prime, \text{where }
                               B = (I_K - A_1 - \cdots - A_p)^{-1}

        Notes
        -----
        Lütkepohl Proposition 3.3
        """
        a_inv = scipy.linalg.inv(np.eye(self.neqs) - self.coefs.sum(0))
        return chain_dot(a_inv, self.cov_resid, a_inv.T)

    # ------------------------------------------------------------
    # Estimation-related things

    @cache_readonly
    def _zz(self):
        # Z'Z
        return np.dot(self.endog_lagged.T, self.endog_lagged)

    @property
    def _cov_params_ex_trend(self):
        """
        Estimated covariance matrix of model coefficients w/o exog
        """
        # drop exog
        idx = self.k_exog * self.neqs
        return self.cov_params[idx:, idx:]

    @cache_readonly
    def _cov_of_cov_resid(self):
        """
        Estimated covariance matrix of vech(cov_resid)
        """
        d_k = tsa.duplication_matrix(self.neqs)
        d_k_inv = np.linalg.pinv(d_k)

        sigxsig = np.kron(self.cov_resid, self.cov_resid)
        return 2 * chain_dot(d_k_inv, sigxsig, d_k_inv.T)

    @cache_readonly
    def llf(self):
        """VAR(p) log-likelihood"""
        return var_loglike(self.resid, self.cov_resid_mle, self.nobs)

    @cache_readonly
    def stderr(self):
        """
        Standard errors of coefficients, reshaped to match in size
        """
        stderr = np.sqrt(np.diag(self.cov_params))
        return stderr.reshape((self.df_model, self.neqs), order='C')

    bse = stderr  # statsmodels interface?

    @cache_readonly
    def stderr_endog_lagged(self):
        """
        Standard errors of the parameters on the lagged endogenous variables

        Notes
        -----
        Compliment of stderr_exog
        """
        start = self.k_exog
        return self.stderr[start:]

    @cache_readonly
    def stderr_exog(self):
        """
        Standard errors of the exogenous regressors including trends.

        Notes
        -----
        Compliment of stderr_endog_lagged
        """
        end = self.k_exog
        return self.stderr[:end]

    @cache_readonly
    def stderr_dt(self):
        """
        Deprecated. Use stderr_exog.
        """
        import warnings
        warnings.warn('stderr_dt is deprecated.  Use stderr_exog.',
                      DeprecationWarning)
        return self.stderr_exog

    @cache_readonly
    def tvalues(self):
        """
        t-statistics of coefficients

        Notes
        -----
        Degree of freedom is `df_resid`
        """
        return self.params / self.stderr

    @cache_readonly
    def tvalues_endog_lagged(self):
        """
        t-values of the parameters on the lagged endogenous variables

        Notes
        -----
        Compliment of tvalues_exog
        """
        start = self.k_exog
        return self.tvalues[start:]

    @cache_readonly
    def tvalues_exog(self):
        """
        t-values of the parameters on the exogenous variables

        Notes
        -----
        Compliment of tvalues_endog_lagged
        """
        end = self.k_exog
        return self.tvalues[:end]

    @cache_readonly
    def tvalues_dt(self):
        """Deprecated.  Use tvalues_exog."""
        import warnings
        warnings.warn('tvalues_dt is deprecated.  Use tvalues_exog.',
                      DeprecationWarning)
        return self.tvalues_exog

    @cache_readonly
    def pvalues(self):
        """
        Two-sided p-values for model coefficients from the Normal distribution
        """
        # return stats.t.sf(np.abs(self.tvalues), self.df_resid)*2
        return 2 * stats.norm.sf(np.abs(self.tvalues))

    @cache_readonly
    def pvalues_endog_lagged(self):
        """
        Two-sided p-values of the parameters on the lagged endogenous variables

        Notes
        -----
        Compliment of pvalues_exog
        """
        start = self.k_exog
        return self.pvalues[start:]

    @cache_readonly
    def pvalues_exog(self):
        """
        Two-sided p-values of the parameters on the exogenous variables

        Notes
        -----
        Compliment of pvalues_endog_lagged
        """
        end = self.k_exog
        return self.pvalues[:end]

    @cache_readonly
    def pvalues_dt(self):
        """Deprecated.  Use pvalues_exog."""
        import warnings
        warnings.warn('pvalues_dt is deprecated.  Use pvalues_exog.',
                      DeprecationWarning)
        return self.pvalues_exog

    def plot_forecast(self, steps, alpha=0.05, plot_stderr=True):
        """
        Plot forecasts

        Parameters
        ----------
        steps : int
            The number of steps ahead to forecast.
        alpha : float
            The significance level for the confidence intervals.
        plot_stderr : bool
            Whether or not to plot the standard error bars.

        Returns
        -------
        fig : `matplotlib.figure`
            The figure that contains the axes
        """
        mid, lower, upper = self.forecast_interval(self.endog[-self.k_ar:],
                                                   steps, alpha=alpha)
        fig = plotting.plot_var_forc(self.endog, mid, lower, upper,
                                     names=self.model.endog_names,
                                     plot_stderr=plot_stderr)
        return fig

    # Forecast error covariance functions

    def forecast_cov(self, steps=1, method='mse'):
        r"""Compute forecast covariance matrices for desired number of steps

        Parameters
        ----------
        steps : int

        Notes
        -----
        .. math:: \Sigma_{\hat y}(h) = \Sigma_y(h) + \Omega(h) / T

        Ref: Lütkepohl pp. 96-97

        Returns
        -------
        forc_covs : ndarray
            Covariance of the forecast errors (steps, neqs, neqs)
        """
        fc_cov = self.mse(steps)
        if method == 'mse':
            pass
        elif method == 'auto':
            if self.k_exog == 1 and self.k_trend < 2:
                # currently only supported if no exog and trend in ['nc', 'c']
                fc_cov += self._omega_forc_cov(steps) / self.nobs
                import warnings
                warnings.warn('forecast cov takes parameter uncertainty into'
                              'account', OutputWarning)
        else:
            raise ValueError("method has to be either 'mse' or 'auto'")

        return fc_cov

    # Monte Carlo irf standard errors
    @deprecate_kwarg('T', 'horizon')
    def irf_errband_mc(self, orth=False, repl=1000, horizon=10,
                       signif=0.05, seed=None, burn=100, cum=False):
        """
        Compute Monte Carlo integrated error bands assuming normally
        distributed for impulse response functions

        Parameters
        ----------
        orth: bool, default False
            Compute orthogonalized impulse response error bands
        repl: int
            number of Monte Carlo replications to perform
        horizon: int, default 10
            number of impulse response periods
        signif: float (0 < signif <1)
            Significance level for error bars, defaults to 95% CI
        seed : {None, integer, np.random.RandomState}
            If seed is not None, then it will be used with for the random
            variables generated by numpy.random. If a RandomState is provided
            then the random numbers will be produced using the method of the
            RandomState.
        burn: int
            number of initial observations to discard for simulation
        cum: bool, default False
            produce cumulative irf error bands

        Notes
        -----
        Lütkepohl (2005) Appendix D

        Returns
        -------
        Tuple of lower and upper arrays of ma_rep monte carlo standard errors
        """
        ma_coll = self.irf_resim(orth=orth, repl=repl, horizon=horizon,
                                 seed=seed, burn=burn, cum=cum)

        ma_sort = np.sort(ma_coll, axis=0)  # sort to get quantiles
        # python 2: round returns float
        low_idx = int(round(signif / 2 * repl) - 1)
        upp_idx = int(round((1 - signif / 2) * repl) - 1)
        lower = ma_sort[low_idx, :, :, :]
        upper = ma_sort[upp_idx, :, :, :]
        return lower, upper

    @deprecate_kwarg('T', 'horizon')
    def irf_resim(self, orth=False, repl=1000, horizon=10, seed=None,
                  burn=100, cum=False):
        """
        Simulates impulse response function, returning an array of simulations.
        Used for Sims-Zha error band calculation.

        Parameters
        ----------
        orth: bool, default False
            Compute orthoganalized impulse response error bands
        repl: int
            number of Monte Carlo replications to perform
        horizon: int, default 10
            number of impulse response periods
        seed : {None, integer, np.random.RandomState}
            If seed is not None, then it will be used with for the random
            variables generated by numpy.random. If a RandomState is provided
            then the random numbers will be produced using the method of the
            RandomState.
        burn: int
            number of initial observations to discard for simulation
        cum: bool, default False
            produce cumulative irf error bands

        Notes
        -----
        .. [*] Sims, Christoper A., and Tao Zha. 1999. "Error Bands for Impulse
           Response." Econometrica 67: 1113-1155.

        Returns
        -------
        Array of simulated impulse response functions

        """
        neqs = self.neqs
        # mean = self.mean()
        k_ar = self.k_ar
        coefs = self.coefs
        cov_resid = self.cov_resid
        intercept = self.intercept
        # df_model = self.df_model
        nobs = self.nobs

        ma_coll = np.zeros((repl, horizon + 1, neqs, neqs))

        def fill_coll(simulation):
            ret = VAR(simulation, exog=self.exog).fit(maxlags=k_ar,
                                                      trend=self.trend)
            if orth:
                rep = ret.orth_ma_rep(maxn=horizon)
            else:
                rep = ret.ma_rep(maxn=horizon)
            return rep.cumsum(axis=0) if cum else rep

        for i in range(repl):
            # discard first hundred to eliminate correct for starting bias
            sim = util.varsim(coefs, intercept, cov_resid,
                              seed=seed, steps=nobs+burn)
            sim = sim[burn:]
            ma_coll[i, :, :, :] = fill_coll(sim)

        return ma_coll

    def _omega_forc_cov(self, steps):
        # Approximate MSE matrix \Omega(h) as defined in Lut p97
        g = self._zz
        g_inv = scipy.linalg.inv(g)
        b = self._bmat_forc_cov()

        # memoized values speedup
        cache_b_power = {}
        cache_bi_ginv_bjt_g = {}
        cache_phi_cov_phit = {}

        def b_power(power):
            if power not in cache_b_power:
                cache_b_power[power] = np.linalg.matrix_power(b, power)

            return cache_b_power[power]

        def bi_ginv_bjt_g(i, j):
            if (i, j) not in cache_bi_ginv_bjt_g:
                b_i = b_power(i)
                b_j = b_power(j)
                prod = np.trace(chain_dot(b_i.T, g_inv, b_j, g))
                cache_bi_ginv_bjt_g[(i, j)] = prod
            return cache_bi_ginv_bjt_g[(i, j)]

        def _phi_cov_phit(i, j):
            if (i, j) not in cache_phi_cov_phit:
                prod = chain_dot(phis[i], cov_resid, phis[j].T)
                cache_phi_cov_phit[(i, j)] = prod
            return cache_phi_cov_phit[(i, j)]

        phis = self.ma_rep(steps)
        cov_resid = self.cov_resid

        omegas = np.zeros((steps, self.neqs, self.neqs))
        for h in range(1, steps + 1):
            if h == 1:
                omegas[h - 1] = self.df_model * self.cov_resid
                continue

            om = omegas[h - 1]
            for i in range(h):
                for j in range(h):
                    # Replaced by memoized version
                    # b_i = b_power(h - 1 - i)
                    # b_j = b_power(h - 1 - j)
                    # mult = np.trace(chain_dot(b_i.T, g_inv, b_j, g))
                    mult = bi_ginv_bjt_g(h - 1 - i, h - 1 - j)
                    # Replaced by memoized version
                    # om += mult * chain_dot(phis[i], cov_resid, phis[j].T)
                    om += mult * _phi_cov_phit(i, j)
            omegas[h - 1] = om

        return omegas

    def _bmat_forc_cov(self):
        # B as defined on p. 96 of Lut
        upper = np.zeros((self.k_exog, self.df_model))
        upper[:, :self.k_exog] = np.eye(self.k_exog)

        lower_dim = self.neqs * (self.k_ar - 1)
        eye = np.eye(lower_dim)
        lower = np.column_stack((np.zeros((lower_dim, self.k_exog)), eye,
                                 np.zeros((lower_dim, self.neqs))))

        return np.vstack((upper, self.params.T, lower))

    def summary(self):
        """
        Summary of estimates

        Returns
        -------
        summary : VARSummary
            A `statsmodels.tsa.vector_ar.output.VARSummary` class.
        """
        return VARSummary(self)

    def irf(self, periods=10, var_decomp=None, var_order=None):
        """Analyze impulse responses to shocks in system

        Parameters
        ----------
        periods : int
            The number of periods for which to get the impulse responses.
        var_decomp : ndarray (neqs x neqs), lower triangular
            Must satisfy `cov_resid` = P P', where P is the passed matrix.
            If P is None, defaults to Cholesky decomposition of `cov_resid`.
        var_order : sequence
            Alternate variable order for Cholesky decomposition

        Returns
        -------
        irf : IRAnalysis
            A `statmodels.tsa.vector_ar.irf.IRAnalysis` instance.
        """
        if var_order is not None:
            raise NotImplementedError('alternate variable order not '
                                      'implemented (yet)')

        return IRAnalysis(self, P=var_decomp, periods=periods)

    def fevd(self, periods=10, var_decomp=None):
        """
        Compute forecast error variance decomposition ("fevd")

        Parameters
        ----------
        periods : int
            The number of periods for which to give the FEVD.
        var_decomp : ndarray (neqs x neqs), lower triangular
            Must satisfy `cov_resid` = P P', where P is the passed matrix.
            If P is None, defaults to Cholesky decomposition of `cov_resid`.

        Returns
        -------
        fevd : FEVD
            A `statsmodels.tsa.vector_ar.var_model.FEVD` instance.
        """
        return FEVD(self, p=var_decomp, periods=periods)

    def reorder(self, order):
        """
        Reorder variables for structural specification
        """
        if len(order) != len(self.params[0, :]):
            raise ValueError("Reorder specification length should match "
                             "number of endogenous variables")
        # This converts order to list of integers if given as strings
        if isinstance(order[0], string_types):
            order_new = []
            for name in order:
                order_new.append(self.names.index(name))
            order = order_new
        return _reordered(self, order)

    # ------------------------------------------------------------------------
    # VAR Diagnostics: Granger-causality, whiteness of resids, normality, etc.

    # TODO: Rescue?
    # def test_causality_all(self, kind='F', signif=0.05):
    #     """
    #     Returns a DataFrame with tests for all equations and variables.
    #
    #     Parameters
    #     ----------
    #     kind : str {'F', 'Wald'}
    #         Perform F-test or Wald (Chi-sq) test
    #     signif : float, default 5%
    #         Significance level for computing critical values for test,
    #         defaulting to standard 0.95 level
    #
    #     Returns
    #     -------
    #     tbl : DataFrame
    #         A hierarchical index DataFrame with tests for each equation
    #         for each variable.
    #
    #     Notes
    #     -----
    #     If an F-test is requested, then the degrees of freedom given in the
    #     results table will be the denominator degrees of freedom. The
    #     """
    #     kind = kind.lower()
    #     if kind == 'f':
    #         columns = ['F', 'df1', 'df2', 'prob(>F)']
    #     elif kind == 'wald':
    #         columns = ['chi2', 'df', 'prob(>chi2)']
    #     else:
    #         raise ValueError("kind %s not understood" % kind)
    #     from pandas import DataFrame, MultiIndex
    #     table = DataFrame(np.zeros((9,len(columns))), columns=columns)
    #     index = []
    #     variables = self.model.endog_names
    #     i = 0
    #     for vari in variables:
    #         others = []
    #         for j, ex_vari in enumerate(variables):
    #             if vari == ex_vari: # don't want to test this
    #                 continue
    #             others.append(ex_vari)
    #             res = self.test_causality(vari, ex_vari, kind=kind,
    #                                       verbose=False)
    #             if kind == 'f':
    #                 row = (res['statistic'],) + res['df'] + (res['pvalue'],)
    #             else:
    #                 row = (res['statistic'], res['df'], res['pvalue'])
    #             table.ix[[i], columns] = row
    #             i += 1
    #             index.append([vari, ex_vari])
    #         res = self.test_causality(vari, others, kind=kind, verbose=False)
    #         if kind == 'f':
    #             row = (res['statistic'],) + res['df'] + (res['pvalue'],)
    #         else:
    #             row = (res['statistic'], res['df'], res['pvalue'])
    #         table.ix[[i], columns] = row
    #         index.append([vari, 'ALL'])
    #         i += 1
    #     table.index = MultiIndex.from_tuples(index, names=['Equation',
    #                                                        'Excluded'])
    #
    #     return table

    def test_causality(self, caused, causing=None, kind='f', signif=0.05):
        """
        Test Granger causality

        Parameters
        ----------
        caused : int or str or sequence of int or str
            If int or str, test whether the variable specified via this index
            (int) or name (str) is Granger-caused by the variable(s) specified
            by `causing`.
            If a sequence of int or str, test whether the corresponding
            variables are Granger-caused by the variable(s) specified
            by `causing`.
        causing : int or str or sequence of int or str or None, default: None
            If int or str, test whether the variable specified via this index
            (int) or name (str) is Granger-causing the variable(s) specified by
            `caused`.
            If a sequence of int or str, test whether the corresponding
            variables are Granger-causing the variable(s) specified by
            `caused`.
            If None, `causing` is assumed to be the complement of `caused`.
        kind : {'f', 'wald'}
            Perform F-test or Wald (chi-sq) test
        signif : float, default 5%
            Significance level for computing critical values for test,
            defaulting to standard 0.05 level

        Notes
        -----
        Null hypothesis is that there is no Granger-causality for the indicated
        variables. The degrees of freedom in the F-test are based on the
        number of variables in the VAR system, that is, degrees of freedom
        are equal to the number of equations in the VAR times degree of freedom
        of a single equation.

        Test for Granger-causality as described in chapter 7.6.3 of [1]_.
        Test H0: "`causing` does not Granger-cause the remaining variables of
        the system" against  H1: "`causing` is Granger-causal for the
        remaining variables".

        Returns
        -------
        results : CausalityTestResults
            A class holding the test's results.

        References
        ----------
        .. [*] Lütkepohl, H., 2005. New introduction to multiple time series
           analysis. Springer Science & Business Media.
        """
        if not (0 < signif < 1):
            raise ValueError("signif has to be between 0 and 1")

        allowed_types = (string_types, int)

        if isinstance(caused, allowed_types):
            caused = [caused]
        if not all(isinstance(c, allowed_types) for c in caused):
            raise TypeError("caused has to be of type string or int (or a "
                            "sequence of these types).")
        caused = [self.names[c] if type(c) == int else c for c in caused]
        caused_ind = [util.get_index(self.names, c) for c in caused]

        if causing is not None:
            if isinstance(causing, allowed_types):
                causing = [causing]
            if not all(isinstance(c, allowed_types) for c in causing):
                raise TypeError("causing has to be of type string or int (or "
                                "a sequence of these types) or None.")
            causing = [self.names[c] if type(c) == int else c for c in causing]
            causing_ind = [util.get_index(self.names, c) for c in causing]
        else:  # causing is None
            causing_ind = [i for i in range(self.neqs) if i not in caused_ind]
            causing = [self.names[c] for c in caused_ind]

        k, p = self.neqs, self.k_ar

        # number of restrictions
        num_restr = len(causing) * len(caused) * p
        num_det_terms = self.k_exog

        # Make restriction matrix
        c = np.zeros((num_restr, k * num_det_terms + k ** 2 * p), dtype=float)
        cols_det = k * num_det_terms
        row = 0
        for j in range(p):
            for ing_ind in causing_ind:
                for ed_ind in caused_ind:
                    c[row, cols_det + ed_ind + k * ing_ind + k ** 2 * j] = 1
                    row += 1

        # Lütkepohl 3.6.5
        c_b = np.dot(c, vec(self.params.T))
        middle = scipy.linalg.inv(chain_dot(c, self.cov_params, c.T))

        # wald statistic
        lam_wald = statistic = chain_dot(c_b, middle, c_b)

        if kind.lower() == 'wald':
            df = num_restr
            dist = stats.chi2(df)
        elif kind.lower() == 'f':
            statistic = lam_wald / num_restr
            df = (num_restr, k * self.df_resid)
            dist = stats.f(*df)
        else:
            raise ValueError('kind %s not recognized' % kind)

        pvalue = dist.sf(statistic)
        crit_value = dist.ppf(1 - signif)

        return CausalityTestResults(causing, caused, statistic, crit_value,
                                    pvalue, df, signif, test="granger",
                                    method=kind)

    def test_inst_causality(self, causing, signif=0.05):
        """
        Test for instantaneous causality

        Parameters
        ----------
        causing :
            If int or str, test whether the corresponding variable is causing
            the variable(s) specified in caused.
            If sequence of int or str, test whether the corresponding variables
            are causing the variable(s) specified in caused.
        signif : float between 0 and 1, default 5 %
            Significance level for computing critical values for test,
            defaulting to standard 0.05 level

        Returns
        -------
        results : CausalityTestResults
            A class holding the test's results.

        Notes
        -----
        Test for instantaneous causality as described in chapters 3.6.3 and
        7.6.4 of [1]_.
        Test H0: "No instantaneous causality between caused and causing"
        against H1: "Instantaneous causality between caused and causing
        exists".

        Instantaneous causality is a symmetric relation (i.e. if causing is
        "instantaneously causing" caused, then also caused is "instantaneously
        causing" causing), thus the naming of the parameters (which is chosen
        to be in accordance with test_granger_causality()) may be misleading.

        This method is not returning the same result as JMulTi. This is because
        the test is based on a VAR(k_ar) model in statsmodels (in accordance to
        pp. 104, 320-321 in [1]_) whereas JMulTi seems to be using a
        VAR(k_ar+1) model.

        References
        ----------
        .. [*] Lütkepohl, H., 2005. New introduction to multiple time series
           analysis. Springer Science & Business Media.
        """
        if not (0 < signif < 1):
            raise ValueError("signif has to be between 0 and 1")

        allowed_types = (string_types, int)
        if isinstance(causing, allowed_types):
            causing = [causing]
        if not all(isinstance(c, allowed_types) for c in causing):
            raise TypeError("causing has to be of type string or int (or a " +
                            "a sequence of these types).")
        causing = [self.names[c] if type(c) == int else c for c in causing]
        causing_ind = [util.get_index(self.names, c) for c in causing]

        caused_ind = [i for i in range(self.neqs) if i not in causing_ind]
        caused = [self.names[c] for c in caused_ind]

        # Note: JMulTi seems to be using k_ar+1 instead of k_ar
        k, t = self.neqs, self.nobs

        num_restr = len(causing) * len(caused)  # called N in Lütkepohl

        cov_resid = self.cov_resid
        vech_cov_resid = util.vech(cov_resid)
        sig_mask = np.zeros(cov_resid.shape)
        # set =1 twice to ensure, that all the ones needed are below the main
        # diagonal:
        sig_mask[causing_ind, caused_ind] = 1
        sig_mask[caused_ind, causing_ind] = 1
        vech_sig_mask = util.vech(sig_mask)
        inds = np.nonzero(vech_sig_mask)[0]

        # Make restriction matrix
        c = np.zeros((num_restr, len(vech_cov_resid)), dtype=float)
        for row in range(num_restr):
            c[row, inds[row]] = 1
        c_s = np.dot(c, vech_cov_resid)
        d = np.linalg.pinv(duplication_matrix(k))
        c_d = np.dot(c, d)
        middle = scipy.linalg.inv(chain_dot(c_d, np.kron(cov_resid, cov_resid),
                                            c_d.T)) / 2

        wald_statistic = t * chain_dot(c_s.T, middle, c_s)
        df = num_restr
        dist = stats.chi2(df)

        pvalue = dist.sf(wald_statistic)
        crit_value = dist.ppf(1 - signif)

        return CausalityTestResults(causing, caused, wald_statistic,
                                    crit_value, pvalue, df, signif,
                                    test="inst", method="wald")

    def test_whiteness(self, nlags=10, signif=0.05, adjusted=False):
        """
        Residual whiteness tests using Portmanteau test

        Parameters
        ----------
        nlags : int
            The number of lags for the autocorrelations.
        signif : float, between 0 and 1
            Significance level for the test
        adjusted : bool, optional
            Flag indicating to use a degree-of-freedom adjustment

        Returns
        -------
        results : WhitenessTestResults

        Notes
        -----
        Test the whiteness of the residuals using the Portmanteau test as
        described in [1]_, chapter 4.4.3.

        References
        ----------
        .. [*] Lütkepohl, H., 2005. New introduction to multiple time series
           analysis. Springer Science & Business Media.
        """
        statistic = 0
        u = np.asarray(self.resid)
        acov_list = _compute_acov(u, nlags)
        cov0_inv = scipy.linalg.inv(acov_list[0])
        for t in range(1, nlags + 1):
            ct = acov_list[t]
            to_add = np.trace(chain_dot(ct.T, cov0_inv, ct, cov0_inv))
            if adjusted:
                to_add /= (self.nobs - t)
            statistic += to_add
        statistic *= self.nobs ** 2 if adjusted else self.nobs
        df = self.neqs ** 2 * (nlags - self.k_ar)
        dist = stats.chi2(df)
        pvalue = dist.sf(statistic)
        crit_value = dist.ppf(1 - signif)

        return WhitenessTestResults(statistic, crit_value, pvalue, df, signif,
                                    nlags, adjusted)

    def plot_acorr(self, nlags=10, resid=True, linewidth=8):
        """
        Plot autocorrelation of sample (endog) or residuals

        Sample (Y) or Residual autocorrelations are plotted together with the
        standard :math:`2 / \sqrt{T}` bounds.

        Parameters
        ----------
        nlags : int
            number of lags to display (excluding 0)
        resid: boolean
            If True, then the autocorrelation of the residuals is plotted
            If False, then the autocorrelation of endog is plotted.
        linewidth : int
            width of vertical bars

        Returns
        -------
        fig : matplotlib.figure.Figure

        """
        if resid:
            acorrs = self.resid_acorr(nlags)
        else:
            acorrs = self.sample_acorr(nlags)

        bound = 2 / np.sqrt(self.nobs)

        fig = plotting.plot_full_acorr(acorrs[1:],
                                       xlabel=np.arange(1, nlags + 1),
                                       err_bound=bound,
                                       linewidth=linewidth)
        fig.suptitle(r"ACF plots for residuals with $2 / \sqrt{T}$ bounds ")
        return fig

    def test_normality(self, signif=0.05):
        """
        Test assumption of normality in resids with Jarque-Bera-style test

        Parameters
        ----------
        signif : float
            Test significance threshold

        Returns
        -------
        result : NormalityTestResults

        Notes
        -----
        H0 (null) : data are generated by a Gaussian-distributed process
        """
        return test_normality(self, signif=signif)

    @cache_readonly
    def detomega(self):
        """
        detomega is deprecated. Use `det_cov_resid`
        """
        import warnings
        warnings.warn("detomega is deprecated and will be removed in 0.11.0. "
                      "Use det_cov_resid.", DeprecationWarning)
        return self.det_cov_resid

    @cache_readonly
    def det_cov_resid(self):
        r"""
        Returns determinant of the cov_resid with degrees of freedom correction

        .. math::

            \hat \Omega = \frac{T}{T - Kp - 1} \hat \Omega_{\mathrm{MLE}}

        where :math:`\Omega` is the covariance of the residuals
        """
        return scipy.linalg.det(self.cov_resid)

    @cache_readonly
    def info_criteria(self):
        """
        Information criteria for order selection

        Returns
        -------
        ic : dict-like
            Container with all model information criteria
        """
        nobs = self.nobs
        neqs = self.neqs
        lag_order = self.k_ar
        free_params = lag_order * neqs ** 2 + neqs * self.k_exog

        ld = logdet_symm(self.cov_resid_mle)

        # See Lütkepohl pp. 146-150

        aic = ld + (2. / nobs) * free_params
        bic = ld + (np.log(nobs) / nobs) * free_params
        hqic = ld + (2. * np.log(np.log(nobs)) / nobs) * free_params
        fpe = ((nobs + self.df_model) / self.df_resid) ** neqs * np.exp(ld)
        res = Bunch()
        res.update({'aic': aic, 'bic': bic, 'hqic': hqic, 'fpe': fpe})

        return res

    @property
    def aic(self):

        """
        Akaike information criterion

        ln(det(cov_resid_mle)) + (2 / nobs) * k_ar * k_trend * neqs ** 2

        Notes
        -----
        Uses the definition from Lütkepohl.
        """
        return self.info_criteria['aic']

    @property
    def fpe(self):
        """Final Prediction Error (FPE)

        det(cov_resid_mle) * ((nobs + df_model)/df_resid)**neqs

        Notes
        -----
        Uses the definition from Lütkepohl.
        """
        return self.info_criteria['fpe']

    @property
    def hqic(self):
        """
        Hannan-Quinn criterion

        ln(det(cov_resid_mle)) + 2*ln(ln(nobs))/nobs * k_ar * k_trend * neqs**2

        Notes
        -----
        Uses the definition from Lütkepohl.
        """

        return self.info_criteria['hqic']

    @property
    def bic(self):
        """
        Bayesian/Schwarz information criterion

        ln(det(cov_resid_mle)) + ln(nobs)/nobs * k_ar * k_trend * neqs ** 2

        Notes
        -----
        Uses the definition from Lütkepohl.
        """

        return self.info_criteria['bic']

    @cache_readonly
    def roots(self):
        """
        Roots of the VAR model

        Notes
        -----
        The roots of the VAR process are the solution to
        (I - coefs[0]*z - coefs[1]*z**2 ... - coefs[p-1]*z**k_ar) = 0.
        Note that the inverse roots are returned, and stability requires that
        the roots lie outside the unit circle.
        """
        neqs = self.neqs
        k_ar = self.k_ar
        p = neqs * k_ar
        arr = np.zeros((p, p))
        arr[:neqs, :] = np.column_stack(self.coefs)
        arr[neqs:, :-neqs] = np.eye(p - neqs)
        roots = np.linalg.eig(arr)[0] ** -1
        idx = np.argsort(np.abs(roots))[::-1]  # sort by reverse modulus
        return roots[idx]


class VARResultsWrapper(wrap.ResultsWrapper):
    _attrs = {'bse': 'columns_eq', 'cov_params': 'cov',
              'params': 'columns_eq', 'pvalues': 'columns_eq',
              'tvalues': 'columns_eq', 'cov_resid': 'cov_eq',
              'cov_resid_mle': 'cov_eq',
              'stderr': 'columns_eq'}
    _wrap_attrs = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {}
    _tsbase_methods = tsbase.TimeSeriesResultsWrapper._wrap_methods
    _wrap_methods = wrap.union_dicts(_tsbase_methods, _methods)
    _wrap_methods.pop('cov_params')  # not yet a method in VARResults


wrap.populate_wrapper(VARResultsWrapper, VARResults)  # noqa:E305


class FEVD(object):
    """
    Compute and plot Forecast error variance decomposition and asymptotic
    standard errors

    Parameters
    ----------
    results : VARResults
        Results of a VAR estimation
    p :
        A square root of the residual covariance.
    periods : int, optional
        Number of periods to compute the decomposition

    Notes
    -----
    p must satisfy

    .. math:

        \Omega = p p'

    where :math:`\Omega` is the residuals covariance
    """
    @deprecate_kwarg('P', 'p')
    def __init__(self, results, p=None, periods=None):
        self.periods = periods

        self.results = results
        self.neqs = results.neqs
        self.endog_names = results.model.endog_names

        self.irfobj = results.irf(var_decomp=p, periods=periods)
        self.orth_irfs = self.irfobj.orth_irfs

        # cumulative impulse responses
        irfs = (self.orth_irfs[:periods] ** 2).cumsum(axis=0)

        rng = lrange(self.neqs)
        mse = results.mse(periods)[:, rng, rng]

        # lag x equation x component
        fevd = np.empty_like(irfs)

        for i in range(periods):
            fevd[i] = (irfs[i].T / mse[i]).T

        # switch to equation x lag x component
        self.decomp = fevd.swapaxes(0, 1)

    def summary(self):
        buf = StringIO()

        rng = lrange(self.periods)
        for i in range(self.neqs):
            ppm = output.pprint_matrix(self.decomp[i], rng, self.endog_names)

            buf.write('FEVD for %s\n' % self.endog_names[i])
            buf.write(ppm + '\n')

        print(buf.getvalue())

    def cov(self):
        """Compute asymptotic standard errors

        Returns
        -------
        """
        raise NotImplementedError

    def plot(self, periods=None, figsize=(10, 10), **plot_kwds):
        """Plot the forecast error variance decompositions

        Parameters
        ----------
        periods : int, default None
            Defaults to number originally specified. Can be at most that number
        figsize : tuple
            The figure size
        plot_kwds : kwargs
            Keyword arguments that

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure instance that contains the axes.
        """
        import matplotlib.pyplot as plt

        k = self.neqs
        periods = periods or self.periods

        fig, axes = plt.subplots(nrows=k, figsize=figsize)
        fig.suptitle('Forecast error variance decomposition (FEVD)')

        colors = [str(c) for c in np.arange(k, dtype=float) / k]
        ticks = np.arange(periods)

        limits = self.decomp.cumsum(2)

        for i in range(k):
            ax = axes[i]

            this_limits = limits[i].T

            handles = []

            for j in range(k):
                lower = this_limits[j - 1] if j > 0 else 0
                upper = this_limits[j]
                handle = ax.bar(ticks, upper - lower, bottom=lower,
                                color=colors[j], label=self.endog_names[j],
                                **plot_kwds)

                handles.append(handle)

            ax.set_title(self.endog_names[i])

        # just use the last axis to get handles for plotting
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        plotting.adjust_subplots(right=0.85)
        return fig
