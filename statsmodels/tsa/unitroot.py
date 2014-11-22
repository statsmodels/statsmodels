from __future__ import division
import warnings

from numpy import (diff, ceil, power, squeeze, sqrt, sum, cumsum, int32, int64,
                   interp, abs, log, sort, polyval)
from numpy.linalg import pinv
from scipy.stats import norm

from statsmodels.compat import lmap, range, long
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.tsatools import lagmat, add_trend
from statsmodels.tsa.stattools import _autolag, cov_nw
from statsmodels.tsa.critical_values.dickey_fuller import *
from statsmodels.tsa.critical_values.kpss import *
from statsmodels.tsa.critical_values.dfgls import *
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.sm_exceptions import (InvalidLengthWarning,
                                             invalid_length_doc)


deprecation_doc = """
{old_func} has been deprecated.  Please use {new_func}.
"""

__all__ = ['ADF', 'DFGLS', 'PhillipsPerron', 'KPSS', 'VarianceRatio',
           'adfuller', 'kpss_crit', 'mackinnoncrit', 'mackinnonp']

TREND_MAP = {None: 'nc', 0: 'c', 1: 'ct', 2: 'ctt'}

TREND_DESCRIPTION = {'nc': 'No Trend',
                     'c': 'Constant',
                     'ct': 'Constant and Linear Time Trend',
                     'ctt': 'Constant, Linear and Quadratic Time Trends'}


def _df_select_lags(y, trend, max_lags, method):
    """
    Helper method to determine the best lag length in DF-like regressions

    Parameters
    ----------
    y : array-like, (nobs,)
        The data for the lag selection exercise
    trend : str, {'nc','c','ct','ctt'}
        The trend order
    max_lags : int
        The maximum number of lags to check.  This setting affects all
        estimation since the sample is adjusted by max_lags when
        fitting the models
    method : str, {'AIC','BIC','t-stat'}
        The method to use when estimating the model

    Returns
    -------
    best_ic : float
        The information criteria at the selected lag
    best_lag : int
        The selected lag
    all_res : list
        List of OLS results from fitting max_lag + 1 models

    Notes
    -----
    See statsmodels.tsa.tsatools._autolag for details.  If max_lags is None, the
    default value of 12 * (nobs/100)**(1/4) is used.
    """
    nobs = y.shape[0]
    delta_y = diff(y)

    if max_lags is None:
        max_lags = int(ceil(12. * power(nobs / 100., 1 / 4.)))

    rhs = lagmat(delta_y[:, None], max_lags, trim='both', original='in')
    nobs = rhs.shape[0]
    rhs[:, 0] = y[-nobs - 1:-1]  # replace 0 with level of y
    lhs = delta_y[-nobs:]

    if trend != 'nc':
        full_rhs = add_trend(rhs, trend, prepend=True)
    else:
        full_rhs = rhs

    start_lag = full_rhs.shape[1] - rhs.shape[1] + 1
    # TODO: Remove all_res after adfuller deprecation
    ic_best, best_lag, all_res = _autolag(OLS, lhs, full_rhs, start_lag,
                                          max_lags, method, regresults=True)
    # To get the correct number of lags, subtract the start_lag since
    # lags 0,1,...,start_lag-1 were not actual lags, but other variables
    best_lag -= start_lag
    return ic_best, best_lag, all_res


def _estimate_df_regression(y, trend, lags):
    """Helper function that estimates the core (A)DF regression

    Parameters
    ----------
    y : array-like, (nobs,)
        The data for the lag selection
    trend : str, {'nc','c','ct','ctt'}
        The trend order
    lags : int
        The number of lags to include in the ADF regression

    Returns
    -------
    ols_res : OLSResults
        A results class object produced by OLS.fit()

    Notes
    -----
    See statsmodels.regression.linear_model.OLS for details on the results
    returned
    """
    delta_y = diff(y)

    rhs = lagmat(delta_y[:, None], lags, trim='both', original='in')
    nobs = rhs.shape[0]
    lhs = rhs[:, 0].copy()  # lag-0 values are lhs, Is copy() necessary?
    rhs[:, 0] = y[-nobs - 1:-1]  # replace lag 0 with level of y

    if trend != 'nc':
        rhs = add_trend(rhs[:, :lags + 1], trend)

    return OLS(lhs, rhs).fit()


def ensure_1d(y, var_name=None):
    """Returns a 1d array if the input is squeezable to 1d. Otherwise raises
    and error

    Parameters
    ----------
    y : array-like
        The array to squeeze to 1d, or raise an error if not compatible

    Returns
    -------
    y_1d : array
        A 1d version of the input, returned as an array
    """
    y = squeeze(asarray(y))
    if y.ndim != 1:
        if var_name is None:
            var_name = 'Input'
        err_msg = '{var_name} must be 1d or squeezable to 1d.'
        raise ValueError(err_msg.format(var_name=var_name))
    return y


class UnitRootTest(object):
    """Base class to be used for inheritance in unit root tests"""

    def __init__(self, y, lags, trend, valid_trends):
        self._y = ensure_1d(y)
        self._delta_y = diff(y)
        self._nobs = self._y.shape[0]
        self._lags = None
        self.lags = lags
        self._valid_trends = valid_trends
        self._trend = ''
        self.trend = trend
        self._stat = None
        self._critical_values = None
        self._pvalue = None
        self.trend = trend
        self._null_hypothesis = 'The process contains a unit root.'
        self._alternative_hypothesis = 'The process is weakly stationary.'
        self._test_name = None
        self._title = None
        self._summary_text = None

    def __str__(self):
        return self.summary().__str__()

    def __repr__(self):
        return str(type(self)) + '\n"""\n' + self.__str__() + '\n"""'

    def _repr_html_(self):
        """Display as HTML for IPython notebook.
        """
        return self.summary().as_html()

    def _compute_statistic(self):
        """This is the core routine that computes the test statistic, computes
        the p-value and constructs the critical values.
        """
        raise NotImplementedError("Subclass must implement")

    def _reset(self):
        """Resets the unit root test so that it will be recomputed
        """
        self._stat = None

    def _compute_if_needed(self):
        """Checks whether the statistic needs to be computed, and computed if
        needed
        """
        if self._stat is None:
            self._compute_statistic()

    @property
    def null_hypothesis(self):
        """The null hypothesis
        """
        return self._null_hypothesis

    @property
    def alternative_hypothesis(self):
        """The alternative hypothesis
        """
        return self._alternative_hypothesis

    @property
    def nobs(self):
        """The number of observations used when computing the test statistic.
        Accounts for loss of data due to lags for regression-based tests."""
        return self._nobs

    @property
    def valid_trends(self):
        """List of valid trend terms."""
        return self._valid_trends

    @property
    def pvalue(self):
        """Returns the p-value for the test statistic
        """
        self._compute_if_needed()
        return self._pvalue

    @property
    def stat(self):
        """The test statistic for a unit root
        """
        self._compute_if_needed()
        return self._stat

    @property
    def critical_values(self):
        """Dictionary containing critical values specific to the test, number of
        observations and included deterministic trend terms.
        """
        self._compute_if_needed()
        return self._critical_values

    def summary(self):
        """Summary of test, containing statistic, p-value and critical values
        """
        table_data = [('Test Statistic', '{0:0.3f}'.format(self.stat)),
                      ('P-value', '{0:0.3f}'.format(self.pvalue)),
                      ('Lags', '{0:d}'.format(self.lags))]
        title = self._title

        if not title:
            title = self._test_name + " Results"
        table = SimpleTable(table_data, stubs=None, title=title, colwidths=18,
                            datatypes=[0, 1], data_aligns=("l", "r"))

        smry = Summary()
        smry.tables.append(table)

        cv_string = 'Critical Values: '
        cv = self._critical_values.keys()
        g = lambda x: float(x.split('%')[0])
        cv_numeric = array(lmap(g, cv))
        cv_numeric = sort(cv_numeric)
        for val in cv_numeric:
            p = str(int(val)) + '%'
            cv_string += '{0:0.2f}'.format(self._critical_values[p])
            cv_string += ' (' + p + ')'
            if val != cv_numeric[-1]:
                cv_string += ', '

        extra_text = ['Trend: ' + TREND_DESCRIPTION[self._trend],
                      cv_string,
                      'Null Hypothesis: ' + self.null_hypothesis,
                      'Alternative Hypothesis: ' + self.alternative_hypothesis]

        smry.add_extra_txt(extra_text)
        if self._summary_text:
            smry.add_extra_txt(self._summary_text)
        return smry

    @property
    def lags(self):
        """Sets or gets the number of lags used in the model.
        When tests use DF-type regressions, lags is the number of lags in the
        regression model.  When tests use long-run variance estimators, lags
        is the number of lags used in the long-run variance estimator.
        """
        self._compute_if_needed()
        return self._lags

    @lags.setter
    def lags(self, value):
        types = (int, long, int32, int64)
        if value is not None and not isinstance(value, types) or \
                (isinstance(value, types) and value < 0):
            raise ValueError('lags must be a non-negative integer or None')
        if self._lags != value:
            self._reset()
        self._lags = value

    @property
    def y(self):
        """Returns the data used in the test statistic
        """
        return self._y

    @property
    def trend(self):
        """Sets or gets the deterministic trend term used in the test. See
        valid_trends for a list of supported trends
        """
        return self._trend

    @trend.setter
    def trend(self, value):
        if value not in self.valid_trends:
            raise ValueError('trend not understood')
        if self._trend != value:
            self._reset()
            self._trend = value


class ADF(UnitRootTest):
    """
    Augmented Dickey-Fuller unit root test

    Parameters
    ----------
    y : array-like, (nobs,)
        The data to test for a unit root
    lags : int, non-negative, optional
        The number of lags to use in the ADF regression.  If omitted or None,
        `method` is used to automatically select the lag length with no more
        than `max_lags` are included.
    trend : str, {'nc', 'c', 'ct', 'ctt'}, optional
        The trend component to include in the ADF test
        'nc' - No trend components
        'c' - Include a constant (Default)
        'ct' - Include a constant and linear time trend
        'ctt' - Include a constant and linear and quadratic time trends
    max_lags : int, non-negative, optional
        The maximum number of lags to use when selecting lag length
    method : str, {'AIC', 'BIC', 't-stat'}, optional
        The method to use when selecting the lag length
        'AIC' - Select the minimum of the Akaike IC
        'BIC' - Select the minimum of the Schwarz/Bayesian IC
        't-stat' - Select the minimum of the Schwarz/Bayesian IC

    Attributes
    ----------
    stat
    pvalue
    critical_values
    null_hypothesis
    alternative_hypothesis
    summary
    regression
    valid_trends
    y
    trend
    lags

    Notes
    -----
    The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
    root, with the alternative that there is no unit root. If the pvalue is
    above a critical size, then the null cannot be rejected that there
    and the series appears to be a unit root.

    The p-values are obtained through regression surface approximation from
    MacKinnon (1994) using the updated 2010 tables.
    If the p-value is close to significant, then the critical values should be
    used to judge whether to reject the null.

    The autolag option and maxlag for it are described in Greene.

    Examples
    --------
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.macrodata.load().data
    >>> inflation = np.diff(np.log(data['cpi']))
    >>> adf = sm.tsa.ADF(inflation)
    >>> adf.stat
    -3.093111891727883
    >>> adf.pvalue
    0.027067156654784364
    >>> adf.lags
    2L
    >>> adf.trend='ct'
    >>> adf.stat
    -3.2111220810567316
    >>> adf.pvalue
    0.082208183131829649

    References
    ----------
    Greene, W. H. 2011. Econometric Analysis. Prentice Hall: Upper Saddle River,
    New Jersey.

    Hamilton, J. D. 1994. Time Series Analysis. Princeton: Princeton
    University Press.

    P-Values (regression surface approximation)
    MacKinnon, J.G. 1994.  "Approximate asymptotic distribution functions for
    unit-root and cointegration tests.  `Journal of Business and Economic
    Statistics` 12, 167-76.

    Critical values
    MacKinnon, J.G. 2010. "Critical Values for Cointegration Tests."  Queen's
    University, Dept of Economics, Working Papers.  Available at
    http://ideas.repec.org/p/qed/wpaper/1227.html
    """

    def __init__(self, y, lags=None, trend='c',
                 max_lags=None, method='AIC'):
        valid_trends = ('nc', 'c', 'ct', 'ctt')
        super(ADF, self).__init__(y, lags, trend, valid_trends)
        self._max_lags = max_lags
        self._method = method
        self._test_name = 'Augmented Dickey-Fuller'
        self._regression = None
        # TODO: Remove when adfuller is deprecation
        self._ic_best = None  # For compat with adfuller
        self._autolag_results = None  # For compat with adfuller

    def _select_lag(self):
        ic_best, best_lag, all_res = _df_select_lags(self._y,
                                                     self._trend,
                                                     self._max_lags,
                                                     self._method)
        # TODO: Remove when adfuller is deprecated
        self._autolag_results = all_res
        self._ic_best = ic_best
        self._lags = best_lag

    def _compute_statistic(self):
        if self._lags is None:
            self._select_lag()
        y, trend, lags = self._y, self._trend, self._lags
        resols = _estimate_df_regression(y, trend, lags)
        self._regression = resols
        self._stat = stat = resols.tvalues[0]
        self._nobs = int(resols.nobs)
        self._pvalue = mackinnonp(stat, regression=trend,
                                  num_unit_roots=1)
        critical_values = mackinnoncrit(num_unit_roots=1,
                                        regression=trend,
                                        nobs=resols.nobs)
        self._critical_values = {"1%": critical_values[0],
                                 "5%": critical_values[1],
                                 "10%": critical_values[2]}

    @property
    def regression(self):
        """Returns the OLS regression results from the ADF model estimated
        """
        self._compute_if_needed()
        return self._regression

    @property
    def max_lags(self):
        """Sets or gets the maximum lags used when automatically selecting lag
        length"""
        return self._max_lags

    @max_lags.setter
    def max_lags(self, value):
        self._max_lags = value


class DFGLS(UnitRootTest):
    """
    Elliott, Rothenberg and Stock's GLS version of the Dickey-Fuller test

    Parameters
    ----------
    y : array-like, (nobs,)
        The data to test for a unit root
    lags : int, non-negative, optional
        The number of lags to use in the ADF regression.  If omitted or None,
        `method` is used to automatically select the lag length with no more
        than `max_lags` are included.
    trend : str, {'c', 'ct'}, optional
        The trend component to include in the ADF test
        'nc' - No trend components
        'c' - Include a constant (Default)
        'ct' - Include a constant and linear time trend
        'ctt' - Include a constant and linear and quadratic time trends
    max_lags : int, non-negative, optional
        The maximum number of lags to use when selecting lag length
    method : str, {'AIC', 'BIC', 't-stat'}, optional
        The method to use when selecting the lag length
        'AIC' - Select the minimum of the Akaike IC
        'BIC' - Select the minimum of the Schwarz/Bayesian IC
        't-stat' - Select the minimum of the Schwarz/Bayesian IC

    Attributes
    ----------
    stat
    pvalue
    critical_values
    null_hypothesis
    alternative_hypothesis
    summary
    regression
    valid_trends
    y
    trend
    lags

    Notes
    -----
    The null hypothesis of the Dickey-Fuller GLS is that there is a unit
    root, with the alternative that there is no unit root. If the pvalue is
    above a critical size, then the null cannot be rejected that there
    and the series appears to be a unit root.

    DFGLS differs from the ADF test in that an initial GLS detrending step
    is used before a trend-less ADF regression is run.

    Critical values and p-values when trend is 'c' are identical to
    the ADF.  When trend is set to 'ct, they are from ...

    Examples
    --------
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.macrodata.load().data
    >>> inflation = np.diff(np.log(data['cpi']))
    >>> dfgls = sm.tsa.DFGLS(inflation)
    >>> dfgls.stat
    -2.7610943027494161
    >>> dfgls.pvalue
    0.0059024528170065768
    >>> dfgls.lags
    2L
    >>> dfgls.trend = 'ct'
    >>> dfgls.stat
    -2.90355133529604
    >>> dfgls.pvalue
    0.044662612387444081

    References
    ----------
    Elliott, G. R., T. J. Rothenberg, and J. H. Stock. 1996. Efficient tests
    for an autoregressive unit root. Econometrica 64: 813-836
    """

    def __init__(self, y, lags=None, trend='c',
                 max_lags=None, method='AIC'):
        valid_trends = ('c', 'ct')
        super(DFGLS, self).__init__(y, lags, trend, valid_trends)
        self._max_lags = max_lags
        self._method = method
        self._regression = None
        self._test_name = 'Dickey-Fuller GLS'
        if trend == 'c':
            self._c = -7.0
        else:
            self._c = -13.5

    def _compute_statistic(self):
        """Core routine to estimate DF-GLS test statistic"""
        # 1. GLS detrend
        trend, c = self._trend, self._c

        nobs = self._y.shape[0]
        ct = c / nobs
        z = add_trend(nobs=nobs, trend=trend)

        delta_z = z.copy()
        delta_z[1:, :] = delta_z[1:, :] - (1 + ct) * delta_z[:-1, :]
        delta_y = self._y.copy()[:, None]
        delta_y[1:] = delta_y[1:] - (1 + ct) * delta_y[:-1]
        detrend_coef = pinv(delta_z).dot(delta_y)
        y = self._y
        y_detrended = y - z.dot(detrend_coef).ravel()

        # 2. determine lag length, if needed
        if self._lags is None:
            max_lags, method = self._max_lags, self._method
            icbest, bestlag, all_res = _df_select_lags(y_detrended, 'nc',
                                                       max_lags, method)
            self._lags = bestlag

        # 3. Run Regression
        lags = self._lags

        resols = _estimate_df_regression(y_detrended,
                                         lags=lags,
                                         trend='nc')
        self._regression = resols
        self._nobs = int(resols.nobs)
        self._stat = resols.tvalues[0]
        self._pvalue = mackinnonp(self._stat,
                                  regression=trend,
                                  dist_type='DFGLS')
        critical_values = mackinnoncrit(regression=trend,
                                        nobs=self._nobs,
                                        dist_type='DFGLS')
        self._critical_values = {"1%": critical_values[0],
                                 "5%": critical_values[1],
                                 "10%": critical_values[2]}

    @UnitRootTest.trend.setter
    def trend(self, value):
        if value not in self.valid_trends:
            raise ValueError('trend not understood')
        if self._trend != value:
            self._reset()
            self._trend = value
        if value == 'c':
            self._c = -7.0
        else:
            self._c = -13.5

    @property
    def regression(self):
        """Returns the OLS regression results from the ADF model estimated
        """
        self._compute_if_needed()
        return self._regression

    @property
    def max_lags(self):
        """Sets or gets the maximum lags used when automatically selecting lag
        length"""
        return self._max_lags

    @max_lags.setter
    def max_lags(self, value):
        self._max_lags = value


class PhillipsPerron(UnitRootTest):
    """
    Phillips-Perron unit root test

    Parameters
    ----------
    y : array-like, (nobs,)
        The data to test for a unit root
    lags : int, non-negative, optional
        The number of lags to use in the Newey-West estimator of the long-run
        covariance.  If omitted or None, the lag length is set automatically to
        12 * (nobs/100) ** (1/4)
    trend : str, {'nc', 'c', 'ct'}, optional
        The trend component to include in the ADF test
            'nc' - No trend components
            'c' - Include a constant (Default)
            'ct' - Include a constant and linear time trend
    test_type : str, {'tau', 'rho'}
        The test to use when computing the test statistic. 'tau' is based on
        the t-stat and 'rho' uses a test based on nobs times the re-centered
        regression coefficient

    Attributes
    ----------
    stat
    pvalue
    critical_values
    test_type
    null_hypothesis
    alternative_hypothesis
    summary
    valid_trends
    y
    trend
    lags

    Notes
    -----
    The null hypothesis of the Phillips-Perron (PP) test is that there is a
    unit root, with the alternative that there is no unit root. If the pvalue
    is above a critical size, then the null cannot be rejected that there
    and the series appears to be a unit root.

    Unlike the ADF test, the regression estimated includes only one lag of
    the dependant variable, in addition to trend terms. Any serial
    correlation in the regression errors is accounted for using a long-run
    variance estimator (currently Newey-West).

    The p-values are obtained through regression surface approximation from
    MacKinnon (1994) using the updated 2010 tables.
    If the p-value is close to significant, then the critical values should be
    used to judge whether to reject the null.

    Examples
    --------
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.macrodata.load().data
    >>> inflation = np.diff(np.log(data['cpi']))
    >>> pp = sm.tsa.PhillipsPerron(inflation)
    >>> pp.stat
    -8.1355784802409303
    >>> pp.pvalue
    1.061467301832819e-12
    >>> pp.lags
    15
    >>> pp.trend = 'ct'
    >>> pp.stat
    -8.2021582107367514
    >>> pp.pvalue
    2.4367363200875479e-11
    >>> pp.test_type = 'rho'
    >>> pp.stat
    -120.32706602359212
    >>> pp.pvalue
    0.0

    References
    ----------
    Hamilton, J. D. 1994. Time Series Analysis. Princeton: Princeton
    University Press.

    Newey, W. K., and K. D. West. 1987. A simple, positive semidefinite,
    heteroskedasticity and autocorrelation consistent covariance matrix.
    Econometrica 55, 703-708.

    Phillips, P. C. B., and P. Perron. 1988. Testing for a unit root in
    time series regression. Biometrika 75, 335-346.

    P-Values (regression surface approximation)
    MacKinnon, J.G. 1994.  "Approximate asymptotic distribution functions for
    unit-root and cointegration tests.  `Journal of Business and Economic
    Statistics` 12, 167-76.

    Critical values
    MacKinnon, J.G. 2010. "Critical Values for Cointegration Tests."  Queen's
    University, Dept of Economics, Working Papers.  Available at
    http://ideas.repec.org/p/qed/wpaper/1227.html
    """

    def __init__(self, y, lags=None, trend='c', test_type='tau'):
        valid_trends = ('nc', 'c', 'ct')
        super(PhillipsPerron, self).__init__(y, lags, trend, valid_trends)
        self._test_type = test_type
        self._stat_rho = None
        self._stat_tau = None
        self._test_name = 'Phillips-Perron Test'
        self._lags = lags

    def _compute_statistic(self):
        """Core routine to estimate PP test statistics"""
        # 1. Estimate Regression
        y, trend = self._y, self._trend
        nobs = y.shape[0]

        if self._lags is None:
            self._lags = int(ceil(12. * power(nobs / 100., 1 / 4.)))
        lags = self._lags

        rhs = y[:-1, None]
        lhs = y[1:, None]
        if trend != 'nc':
            rhs = add_trend(rhs, trend)

        resols = OLS(lhs, rhs).fit()
        k = rhs.shape[1]
        n, u = resols.nobs, resols.resid
        lam2 = cov_nw(u, lags, demean=False)
        lam = sqrt(lam2)
        # 2. Compute components
        s2 = u.dot(u) / (n - k)
        s = sqrt(s2)
        gamma0 = s2 * (n - k) / n
        sigma = resols.bse[0]
        sigma2 = sigma ** 2.0
        rho = resols.params[0]
        # 3. Compute statistics
        self._stat_tau = sqrt(gamma0 / lam2) * ((rho - 1) / sigma) \
                         - 0.5 * ((lam2 - gamma0) / lam) * (n * sigma / s)
        self._stat_rho = n * (rho - 1) \
                         - 0.5 * (n ** 2.0 * sigma2 / s2) * (lam2 - gamma0)

        self._nobs = int(resols.nobs)
        if self._test_type == 'rho':
            self._stat = self._stat_rho
            dist_type = 'ADF-z'
        else:
            self._stat = self._stat_tau
            dist_type = 'ADF-t'

        self._pvalue = mackinnonp(self._stat,
                                  regression=trend,
                                  dist_type=dist_type)
        critical_values = mackinnoncrit(regression=trend,
                                        nobs=n,
                                        dist_type=dist_type)
        self._critical_values = {"1%": critical_values[0],
                                 "5%": critical_values[1],
                                 "10%": critical_values[2]}

        self._title = self._test_name + ' (Z-' + self._test_type + ')'

    @property
    def test_type(self):
        """Gets or sets the test type returned by stat.
        Valid values are 'tau' or 'rho'"""
        return self._test_type

    @test_type.setter
    def test_type(self, value):
        if value not in ('rho', 'tau'):
            raise ValueError('stat must be either ''rho'' or ''tau''.')
        self._reset()
        self._test_type = value


class KPSS(UnitRootTest):
    """
    Kwiatkowski, Phillips, Schmidt and Shin (KPSS) stationarity test

    Parameters
    ----------
    y : array-like, (nobs,)
        The data to test for stationarity
    lags : int, non-negative, optional
        The number of lags to use in the Newey-West estimator of the long-run
        covariance.  If omitted or None, the lag length is set automatically to
        12 * (nobs/100) ** (1/4)
    trend : str, {'c', 'ct'}, optional
        The trend component to include in the ADF test
            'c' - Include a constant (Default)
            'ct' - Include a constant and linear time trend

    Attributes
    ----------
    stat
    pvalue
    critical_values
    null_hypothesis
    alternative_hypothesis
    summary
    valid_trends
    y
    trend
    lags

    Notes
    -----
    The null hypothesis of the KPSS test is that the series is weakly stationary
    and the alternative is that it is non-stationary. If the p-value
    is above a critical size, then the null cannot be rejected that there
    and the series appears stationary.

    The p-values and critical values were computed using an extensive simulation
    based on 100,000,000 replications using series with 2,000 observations.

    Examples
    --------
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.macrodata.load().data
    >>> inflation = np.diff(np.log(data['cpi']))
    >>> kpss = sm.tsa.KPSS(inflation)
    >>> kpss.stat
    0.28700886586311969
    >>> kpss.pvalue
    .14735012654422672
    >>> kpss.trend = 'ct'
    >>> kpss.stat
    0.20749526406020977
    >>> kpss.pvalue
    .012834648872163952

    References
    ----------
    Kwiatkowski, D.; Phillips, P. C. B.; Schmidt, P.; Shin, Y. (1992). "Testing
    the null hypothesis of stationarity against the alternative of a unit root".
    Journal of Econometrics 54 (1-3), 159-178
    """

    def __init__(self, y, lags=None, trend='c'):
        valid_trends = ('c', 'ct')
        super(KPSS, self).__init__(y, lags, trend, valid_trends)
        self._test_name = 'KPSS Stationarity Test'
        self._null_hypothesis = 'The process is weakly stationary.'
        self._alternative_hypothesis = 'The process contains a unit root.'

    def _compute_statistic(self):
        # 1. Estimate model with trend
        nobs, y, trend = self._nobs, self._y, self._trend
        z = add_trend(nobs=nobs, trend=trend)
        res = OLS(y, z).fit()
        # 2. Compute KPSS test
        u = res.resid
        if self._lags is None:
            self._lags = int(ceil(12. * power(nobs / 100., 1 / 4.)))
        lam = cov_nw(u, self._lags, demean=False)
        s = cumsum(u)
        self._stat = 1 / (nobs ** 2.0) * sum(s ** 2.0) / lam
        self._nobs = u.shape[0]
        self._pvalue, critical_values = kpss_crit(self._stat, trend)
        self._critical_values = {"1%": critical_values[0],
                                 "5%": critical_values[1],
                                 "10%": critical_values[2]}


class VarianceRatio(UnitRootTest):
    """
    Variance Ratio test of a random walk.

    Parameters
    ----------
    y : array-like, (nobs,)
        The data to test for a random walk
    lags : int, >=2
        The number of periods to used in the multi-period variance, which is the
        numerator of the test statistic.  Must be at least 2
    trend : str, {'nc', 'c'}, optional
        'c' allows for a non-zero drift in the random walk, while 'nc' requires
        that the increments to y are mean 0
    overlap : bool, optional
        Indicates whether to use all overlapping blocks.  Default is True.  If
        False, the number of observations in y minus 1 must be an exact multiple
        of lags.  If this condition is not satistifed, some values at the end of
        y will be discarded.
    robust : bool, optional
        Indicates whether to use heteroskedasticity robust inference. Default is
        True.
    debiased : bool, optional
        Indicates whether to use a debiased version of the test. Default is
        True. Only applicable if overlap is True.

    Attributes
    ----------
    stat
    pvalue
    critical_values
    null_hypothesis
    alternative_hypothesis
    summary
    valid_trends
    y
    trend
    lags
    overlap
    robust
    debiased

    Notes
    -----
    The null hypothesis of a VR is that the process is a random walk, possibly
    plus drift.  Rejection of the null with a positive test statistic indicates
    the presence of positive serial correlation in the time series.

    Examples
    --------
    >>> import datetime as dt
    >>> from matplotlib.finance import fetch_historical_yahoo as yahoo
    >>> csv = yahoo('^GSPC', dt.date(1950,1,1), dt.date(2010,1,1))
    >>> import pandas as pd
    >>> data = pd.DataFrame.from_csv(csv)
    >>> data = data[::-1]  # Reverse
    >>> data.resample('M',how='last')  # End of month
    >>> returns = data['Adj Close'].pct_change().dropna()
    >>> import statsmodels.api as sm
    >>> vr = sm.tsa.VarianceRatio(returns, lags=12)
    >>> vr.stat
    -23.021986263667511
    >>> vr.pvalue
    0.0

    References
    ----------
    Campbell, John Y., Lo, Andrew W. and MacKinlay, A. Craig. (1997) The
    Econometrics of Financial Markets. Princeton, NJ: Princeton University
    Press.

    """

    def __init__(self, y, lags=2, trend='c', debiased=True,
                 robust=True, overlap=True):
        if lags < 2:
            raise ValueError('lags must be an integer larger than 2')
        valid_trends = ('nc', 'c')
        super(VarianceRatio, self).__init__(y, lags, trend, valid_trends)
        self._test_name = 'Variance-Ratio Test'
        self._null_hypothesis = 'The process is a random walk.'
        self._alternative_hypothesis = 'The process is not a random walk.'
        self._robust = robust
        self._debiased = debiased
        self._overlap = overlap
        self._vr = None
        self._stat_variance = None
        quantiles = array([.01, .05, .1, .9, .95, .99])
        self._critical_values = {}
        self._summary_text = ''
        for q, cv in zip(quantiles, norm.ppf(quantiles)):
            self._critical_values[str(int(100 * q)) + '%'] = cv

    @property
    def vr(self):
        """The ratio of the long block lags-period variance
        to the 1-period variance"""
        self._compute_if_needed()
        return self._vr

    @property
    def overlap(self):
        """Sets of gets the indicator to use overlaping returns in the
        long-period vairance estimator"""
        return self._overlap

    @overlap.setter
    def overlap(self, value):
        self._reset()
        self._overlap = bool(value)

    @property
    def robust(self):
        """Sets of gets the indicator to use a heteroskedasticity robust
        variance estimator """
        return self._robust

    @robust.setter
    def robust(self, value):
        self._reset()
        self._robust = bool(value)

    @property
    def debiased(self):
        """Sets of gets the indicator to use debiased variances in the ratio"""
        return self._debiased

    @debiased.setter
    def debiased(self, value):
        self._reset()
        self._debiased = bool(value)

    def _compute_statistic(self):
        overlap, debiased, robust = self._overlap, self._debiased, self._robust
        y, nobs, q, trend = self._y, self._nobs, self._lags, self._trend

        nq = nobs - 1
        if not overlap:
            # Check length of y
            if nq % q != 0:
                extra = nq % q
                y = y[:-extra]
                warnings.warn(invalid_length_doc.format(var='y',
                                                        block=q,
                                                        drop=extra),
                              InvalidLengthWarning)

        nobs = y.shape[0]
        if trend == 'nc':
            mu = 0
        else:
            mu = (y[-1] - y[0]) / (nobs - 1)

        delta_y = diff(y)
        nq = delta_y.shape[0]
        sigma2_1 = sum((delta_y - mu) ** 2.0) / nq

        if not overlap:
            delta_y_q = y[q::q] - y[0:-q:q]
            sigma2_q = sum((delta_y_q - q * mu) ** 2.0) / nq
            self._summary_text = 'Computed with non-overlapping blocks'
        else:
            delta_y_q = y[q:] - y[:-q]
            sigma2_q = sum((delta_y_q - q * mu) ** 2.0) / (nq * q)
            self._summary_text = 'Computed with overlapping blocks'

        if debiased and overlap:
            sigma2_1 *= nq / (nq - 1)
            m = q * (nq - q + 1) * (1 - (q / nq))
            sigma2_q *= (nq * q) / m
            self._summary_text = 'Computed with overlapping blocks (de-biased)'

        if not overlap:
            self._stat_variance = 2.0 * (q - 1)
        elif not robust:
            self._stat_variance = (2 * (2 * q - 1) * (q - 1)) / (2 * q)
        else:
            z2 = (delta_y - mu) ** 2.0
            scale = sum(z2) ** 2.0
            theta = 0.0
            for k in range(1, q):
                delta = nq * z2[k:].dot(z2[:-k]) / scale
                theta += (1 - k / q) ** 2.0 * delta
            self._stat_variance = theta
        self._vr = sigma2_q / sigma2_1
        self._stat = sqrt(nq) * (self._vr - 1) / sqrt(self._stat_variance)
        self._pvalue = 2 - 2 * norm.cdf(abs(self._stat))


# Wrapper before deprecation
# See:
# Ng and Perron(2001), Lag length selection and the construction of unit root
# tests with good size and power, Econometrica, Vol 69 (6) pp 1519-1554
# TODO: include drift keyword, only valid with regression == "c" which will
# TODO: change the distribution of the test statistic to a t distribution
def adfuller(x, maxlag=None, regression="c", autolag='AIC',
             store=False, regresults=False):
    """
    Augmented Dickey-Fuller unit root test (Deprecated)

    The Augmented Dickey-Fuller test can be used to test for a unit root in a
    univariate process in the presence of serial correlation.

    Parameters
    ----------
    x : array_like, 1d
        data series
    maxlag : int
        Maximum lag which is included in test, default 12*(nobs/100)^{1/4}
    regression : str {'c','ct','ctt','nc'}
        Constant and trend order to include in regression
        * 'c' : constant only (default)
        * 'ct' : constant and trend
        * 'ctt' : constant, and linear and quadratic trend
        * 'nc' : no constant, no trend
    autolag : {'AIC', 'BIC', 't-stat', None}
        * if None, then maxlag lags are used
        * if 'AIC' (default) or 'BIC', then the number of lags is chosen
          to minimize the corresponding information criterium
        * 't-stat' based choice of maxlag.  Starts with maxlag and drops a
          lag until the t-statistic on the last lag length is significant at
          the 95 % level.
    store : bool
        If True, then an ADF instance is returned additionally to
        the adf statistic (default is False)
    regresults : bool
        If True, the full regression results are returned (default is False)

    Returns
    -------
    adf : float
        Test statistic
    pvalue : float
        MacKinnon's approximate p-value based on MacKinnon (1994)
    usedlag : int
        Number of lags used.
    nobs : int
        Number of observations used for the ADF regression and calculation of
        the critical values.
    critical values : dict
        Critical values for the test statistic at the 1 %, 5 %, and 10 %
        levels. Based on MacKinnon (2010)
    icbest : float
        The maximized information criterion if autolag is not None.
    regresults : RegressionResults instance
        The
    resstore : (optional) instance of ResultStore
        an instance of a dummy class with results attached as attributes

    Notes
    -----
    This function has been deprecated. Please use ADF instead.

    The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
    root, with the alternative that there is no unit root. If the pvalue is
    above a critical size, then we cannot reject that there is a unit root.

    The p-values are obtained through regression surface approximation from
    MacKinnon 1994, but using the updated 2010 tables.
    If the p-value is close to significant, then the critical values should be
    used to judge whether to accept or reject the null.

    The autolag option and maxlag for it are described in Greene.

    Examples
    --------
    see example script

    References
    ----------
    Greene, W. H. 2011. Econometric Analysis. Prentice Hall: Upper Saddle River,
    New Jersey.

    Hamilton, J. D. 1994. Time Series Analysis. Princeton: Princeton
    University Press.

    P-Values (regression surface approximation)
    MacKinnon, J.G. 1994.  "Approximate asymptotic distribution functions for
    unit-root and cointegration tests.  `Journal of Business and Economic
    Statistics` 12, 167-76.

    Critical values
    MacKinnon, J.G. 2010. "Critical Values for Cointegration Tests."  Queen's
    University, Dept of Economics, Working Papers.  Available at
    http://ideas.repec.org/p/qed/wpaper/1227.html

    """
    warnings.warn(deprecation_doc.format(old_func='adfuller', new_func='ADF'),
                  DeprecationWarning)
    lags = None
    if autolag is None:
        lags = maxlag
    adf = ADF(x, lags=lags, trend=regression, max_lags=maxlag, method=autolag)

    adfstat = adf.stat
    pvalue = adf.pvalue
    critvalues = adf.critical_values
    usedlag = adf.lags
    nobs = adf.nobs
    icbest = adf._ic_best
    resstore = adf
    if regresults:
        # Work around for missing properties
        setattr(resstore, 'autolag_results', resstore._autolag_results)
        setattr(resstore, 'usedlag', resstore.lags)
        return adfstat, pvalue, critvalues, resstore
    elif autolag:
        return adfstat, pvalue, usedlag, nobs, critvalues, icbest
    else:
        return adfstat, pvalue, usedlag, nobs, critvalues


def mackinnonp(stat, regression="c", num_unit_roots=1, dist_type='ADF-t'):
    """
    Returns MacKinnon's approximate p-value for test stat.

    Parameters
    ----------
    stat : float
        "T-value" from an Augmented Dickey-Fuller or DFGLS regression.
    regression : str {"c", "nc", "ct", "ctt"}
        This is the method of regression that was used.  Following MacKinnon's
        notation, this can be "c" for constant, "nc" for no constant, "ct" for
        constant and trend, and "ctt" for constant, trend, and trend-squared.
    num_unit_roots : int
        The number of series believed to be I(1).  For (Augmented) Dickey-
        Fuller N = 1.
    dist_type: str, {'ADF-t', 'ADF-z', 'DFGLS'}
        The test type to use when computing p-values.  Options include
        'ADF-t' - ADF t-stat based tests
        'ADF-z' - ADF z tests
        'DFGLS' - GLS detrended Dickey Fuller

    Returns
    -------
    p-value : float
        The p-value for the ADF statistic estimated using MacKinnon 1994.

    References
    ----------
    MacKinnon, J.G. 1994  "Approximate Asymptotic Distribution Functions for
        Unit-Root and Cointegration Tests." Journal of Business & Economics
        Statistics, 12.2, 167-76.

    Notes
    -----
    Most values are from MacKinnon (1994).  Values for DFGLS test statistics
    and the 'nc' version of the ADF z test statistic were computed following
    the methodology of MacKinnon (1994).
    """
    dist_type = dist_type.lower()
    if num_unit_roots > 1 and dist_type.lower() != 'adf-t':
        raise ValueError('Cointegration results (num_unit_roots > 1) are' +
                         'only available for ADF-t values')
    if dist_type == 'adf-t':
        maxstat = tau_max[regression][num_unit_roots - 1]
        minstat = tau_min[regression][num_unit_roots - 1]
        starstat = tau_star[regression][num_unit_roots - 1]
        small_p = tau_small_p[regression][num_unit_roots - 1]
        large_p = tau_large_p[regression][num_unit_roots - 1]
    elif dist_type == 'adf-z':
        maxstat = adf_z_max[regression]
        minstat = adf_z_min[regression]
        starstat = adf_z_star[regression]
        small_p = adf_z_small_p[regression]
        large_p = adf_z_large_p[regression]
    elif dist_type == 'dfgls':
        maxstat = dfgls_tau_max[regression]
        minstat = dfgls_tau_min[regression]
        starstat = dfgls_tau_star[regression]
        small_p = dfgls_small_p[regression]
        large_p = dfgls_large_p[regression]
    else:
        raise ValueError('Unknown test type {0}'.format(dist_type))

    if stat > maxstat:
        return 1.0
    elif stat < minstat:
        return 0.0
    if stat <= starstat:
        poly_coef = small_p
        if dist_type == 'adf-z':
            stat = log(abs(stat))  # Transform stat for small p ADF-z
    else:
        poly_coef = large_p
    return norm.cdf(polyval(poly_coef[::-1], stat))


def mackinnoncrit(num_unit_roots=1, regression="c", nobs=inf,
                  dist_type='ADF-t'):
    """
    Returns the critical values for cointegrating and the ADF test.

    In 2010 MacKinnon updated the values of his 1994 paper with critical values
    for the augmented Dickey-Fuller tests.  These new values are to be
    preferred and are used here.

    Parameters
    ----------
    num_unit_roots : int
        The number of series of I(1) series for which the null of
        non-cointegration is being tested.  For N > 12, the critical values
        are linearly interpolated (not yet implemented).  For the ADF test,
        N = 1.
    reg : str {'c', 'tc', 'ctt', 'nc'}
        Following MacKinnon (1996), these stand for the type of regression run.
        'c' for constant and no trend, 'tc' for constant with a linear trend,
        'ctt' for constant with a linear and quadratic trend, and 'nc' for
        no constant.  The values for the no constant case are taken from the
        1996 paper, as they were not updated for 2010 due to the unrealistic
        assumptions that would underlie such a case.
    nobs : int or np.inf
        This is the sample size.  If the sample size is numpy.inf, then the
        asymptotic critical values are returned.

    Returns
    -------
    crit_vals : array
        Three critical values corresponding to 1%, 5% and 10% cut-offs.

    Notes
    -----
    Results for ADF t-stats from MacKinnon (1994,2010).  Results for DFGLS and
    ADF z-tests use the same methodology as MacKinnon.

    References
    ----------
    MacKinnon, J.G. 1994  "Approximate Asymptotic Distribution Functions for
        Unit-Root and Cointegration Tests." Journal of Business & Economics
        Statistics, 12.2, 167-76.
    MacKinnon, J.G. 2010.  "Critical Values for Cointegration Tests."
        Queen's University, Dept of Economics Working Papers 1227.
        http://ideas.repec.org/p/qed/wpaper/1227.html
    """
    dist_type = dist_type.lower()
    valid_regression = ['c', 'ct', 'nc', 'ctt']
    if dist_type == 'dfgls':
        valid_regression = ['c', 'ct']
    if regression not in valid_regression:
        raise ValueError(
            "regression keyword {0} not understood".format(regression))

    if dist_type == 'adf-t':
        asymptotic_cv = tau_2010[regression][num_unit_roots - 1, :, 0]
        poly_coef = tau_2010[regression][num_unit_roots - 1, :, :].T
    elif dist_type == 'adf-z':
        poly_coef = adf_z_cv_approx[regression].T
        asymptotic_cv = adf_z_cv_approx[regression][:, 0]
    elif dist_type == 'dfgls':
        poly_coef = dfgls_cv_approx[regression].T
        asymptotic_cv = dfgls_cv_approx[regression][:, 0]
    else:
        raise ValueError('Unknown test type {0}'.format(dist_type))

    if nobs is inf:
        return asymptotic_cv
    else:
        # Flip so that highest power to lowest power
        return polyval(poly_coef[::-1], 1. / nobs)


def kpss_crit(stat, trend='c'):
    """
    Linear interpolation for KPSS p-values and critical values

    Parameters
    ----------
    stat : float
        The KPSS test statistic.
    trend : str, {'c','ct'}
        The trend used when computing the KPSS statistic

    Returns
    -------
    pvalue : float
        The interpolated p-value
    crit_val : array
        Three element array containing the 10%, 5% and 1% critical values,
        in order

    Notes
    -----
    The p-values are linear interpolated from the quantiles of the simulated
    KPSS test statistic distribution using 100,000,000 replications and 2000
    data points.
    """
    table = kpss_critical_values[trend]
    y = table[:, 0]
    x = table[:, 1]
    # kpss.py contains quantiles multiplied by 100
    pvalue = interp(stat, x, y) / 100.0
    cv = [1.0, 5.0, 10.0]
    crit_value = interp(cv, y[::-1], x[::-1])

    return pvalue, crit_value


if __name__ == '__main__':
    pass
