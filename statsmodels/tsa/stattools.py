"""
Statistical tools for time series analysis
"""
from __future__ import division
from statsmodels.compat.python import (iteritems, range, lrange, string_types, lzip,
                                zip, map, range)
import numpy as np
from numpy.linalg import LinAlgError
from scipy import stats
from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.tools.tools import add_constant, Bunch
from .tsatools import lagmat, lagmat2ds, add_trend
from statsmodels.tsa.arima_model import ARMA
from statsmodels.compat.scipy import _next_regular

__all__ = ['acovf', 'acf', 'pacf', 'pacf_yw', 'pacf_ols', 'ccovf', 'ccf',
           'periodogram', 'q_stat', 'arma_order_select_ic', 'cov_nw']


#NOTE: now in two places to avoid circular import
#TODO: I like the bunch pattern for this too.
class ResultsStore(object):
    def __str__(self):
        return self._str  # pylint: disable=E1101


def cov_nw(y, lags=0, demean=True, axis=0, ddof=0):
    """
    Computes Newey-West covariance for 1-d and 2-d arrays

    Parameters
    ----------
    y : array-like, 1d or 2d
        Values to use when computing the Newey-West covariance estimator.
        When u is 2d, default behavior is to treat columns as variables and
        rows as observations.
    lags : int, non-negative
        Number of lags to include in the Newey-West covariance estimator
    demean : bool
        Indicates whether to subtract the mean.  Default is True
    axis : int, (0, 1)
        The axis to use when y is 2d
    ddof : int, non-negative
        Degree of freedom correction for compatability with simple covariance
        estimators.  Default is 0.

    Returns
    -------
    cov : array
        The estimated covariance

    """
    z = y
    is_1d = False
    if axis > z.ndim:
        raise ValueError('axis must be less than the dimension of y')
    if z.ndim == 1:
        is_1d = True
        z = z[:, None]
    if axis == 1:
        z = z.T
    n = z.shape[0]
    if ddof > n:
        raise ValueError("ddof must be strictly smaller than the number of "
                         "observations")
    if lags > n:
        error = 'lags must be weakly smaller than the number of observations'
        raise ValueError(error)

    if demean:
        z = z - z.mean(0)
    cov = z.T.dot(z)
    for j in range(1, lags + 1):
        w = (1 - j / (lags + 1))
        gamma = z[j:].T.dot(z[:-j])
        cov += w * (gamma + gamma.T)
    cov = cov / (n - ddof)
    if is_1d:
        cov = float(cov)
    return cov


def _autolag(mod, endog, exog, startlag, maxlag, method, modargs=(),
             fitargs=(), regresults=False):
    """
    Returns the results for the lag length that maximimizes the info criterion.

    Parameters
    ----------
    mod : Model class
        Model estimator class.
    modargs : tuple
        args to pass to model.  See notes.
    fitargs : tuple
        args to pass to fit.  See notes.
    lagstart : int
        The first zero-indexed column to hold a lag.  See Notes.
    maxlag : int
        The highest lag order for lag length selection.
    method : str {"aic","bic","t-stat"}
        aic - Akaike Information Criterion
        bic - Bayes Information Criterion
        t-stat - Based on last lag

    Returns
    -------
    icbest : float
        Best information criteria.
    bestlag : int
        The lag length that minimizes the information criterion.


    Notes
    -----
    Does estimation like mod(endog, exog[:,:i], *modargs).fit(*fitargs)
    where i goes from lagstart to lagstart+maxlag+1.  Therefore, lags are
    assumed to be in contiguous columns from low to high lag length with
    the highest lag in the last column.
    """
    #TODO: can tcol be replaced by maxlag + 2?
    #TODO: This could be changed to laggedRHS and exog keyword arguments if
    #    this will be more general.

    results = {}
    method = method.lower()
    for lag in range(startlag, startlag + maxlag + 1):
        mod_instance = mod(endog, exog[:, :lag], *modargs)
        results[lag] = mod_instance.fit()

    if method == "aic":
        icbest, bestlag = min((v.aic, k) for k, v in iteritems(results))
    elif method == "bic":
        icbest, bestlag = min((v.bic, k) for k, v in iteritems(results))
    elif method == "t-stat":
        #stop = stats.norm.ppf(.95)
        stop = 1.6448536269514722
        for lag in range(startlag + maxlag, startlag - 1, -1):
            icbest = np.abs(results[lag].tvalues[-1])
            if np.abs(icbest) >= stop:
                bestlag = lag
                icbest = icbest
                break
    else:
        raise ValueError("Information Criterion %s not understood.") % method

    if not regresults:
        return icbest, bestlag
    else:
        return icbest, bestlag, results


def acovf(x, unbiased=False, demean=True, fft=False):
    '''
    Autocovariance for 1D

    Parameters
    ----------
    x : array
        Time series data. Must be 1d.
    unbiased : bool
        If True, then denominators is n-k, otherwise n
    demean : bool
        If True, then subtract the mean x from each element of x
    fft : bool
        If True, use FFT convolution.  This method should be preferred
        for long time series.

    Returns
    -------
    acovf : array
        autocovariance function
    '''
    x = np.squeeze(np.asarray(x))
    if x.ndim > 1:
        raise ValueError("x must be 1d. Got %d dims." % x.ndim)
    n = len(x)

    if demean:
        xo = x - x.mean()
    else:
        xo = x
    if unbiased:
        xi = np.arange(1, n + 1)
        d = np.hstack((xi, xi[:-1][::-1]))
    else:
        d = n * np.ones(2 * n - 1)
    if fft:
        nobs = len(xo)
        Frf = np.fft.fft(xo, n=nobs * 2)
        acov = np.fft.ifft(Frf * np.conjugate(Frf))[:nobs] / d[n - 1:]
        return acov.real
    else:
        return (np.correlate(xo, xo, 'full') / d)[n - 1:]


def q_stat(x, nobs, type="ljungbox"):
    """
    Return's Ljung-Box Q Statistic

    x : array-like
        Array of autocorrelation coefficients.  Can be obtained from acf.
    nobs : int
        Number of observations in the entire sample (ie., not just the length
        of the autocorrelation function results.

    Returns
    -------
    q-stat : array
        Ljung-Box Q-statistic for autocorrelation parameters
    p-value : array
        P-value of the Q statistic

    Notes
    ------
    Written to be used with acf.
    """
    x = np.asarray(x)
    if type == "ljungbox":
        ret = (nobs * (nobs + 2) *
               np.cumsum((1. / (nobs - np.arange(1, len(x) + 1))) * x**2))
    chi2 = stats.chi2.sf(ret, np.arange(1, len(x) + 1))
    return ret, chi2


#NOTE: Changed unbiased to False
#see for example
# http://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm
def acf(x, unbiased=False, nlags=40, qstat=False, fft=False, alpha=None):
    '''
    Autocorrelation function for 1d arrays.

    Parameters
    ----------
    x : array
       Time series data
    unbiased : bool
       If True, then denominators for autocovariance are n-k, otherwise n
    nlags: int, optional
        Number of lags to return autocorrelation for.
    qstat : bool, optional
        If True, returns the Ljung-Box q statistic for each autocorrelation
        coefficient.  See q_stat for more information.
    fft : bool, optional
        If True, computes the ACF via FFT.
    alpha : scalar, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        Bartlett\'s formula.

    Returns
    -------
    acf : array
        autocorrelation function
    confint : array, optional
        Confidence intervals for the ACF. Returned if confint is not None.
    qstat : array, optional
        The Ljung-Box Q-Statistic.  Returned if q_stat is True.
    pvalues : array, optional
        The p-values associated with the Q-statistics.  Returned if q_stat is
        True.

    Notes
    -----
    The acf at lag 0 (ie., 1) is returned.

    This is based np.correlate which does full convolution. For very long time
    series it is recommended to use fft convolution instead.

    If unbiased is true, the denominator for the autocovariance is adjusted
    but the autocorrelation is not an unbiased estimtor.
    '''
    nobs = len(x)
    d = nobs  # changes if unbiased
    if not fft:
        avf = acovf(x, unbiased=unbiased, demean=True)
        #acf = np.take(avf/avf[0], range(1,nlags+1))
        acf = avf[:nlags + 1] / avf[0]
    else:
        x = np.squeeze(np.asarray(x))
        #JP: move to acovf
        x0 = x - x.mean()
        # ensure that we always use a power of 2 or 3 for zero-padding,
        # this way we'll ensure O(n log n) runtime of the fft.
        n = _next_regular(2 * nobs + 1)
        Frf = np.fft.fft(x0, n=n)  # zero-pad for separability
        if unbiased:
            d = nobs - np.arange(nobs)
        acf = np.fft.ifft(Frf * np.conjugate(Frf))[:nobs] / d
        acf /= acf[0]
        #acf = np.take(np.real(acf), range(1,nlags+1))
        acf = np.real(acf[:nlags + 1])   # keep lag 0
    if not (qstat or alpha):
        return acf
    if alpha is not None:
        varacf = np.ones(nlags + 1) / nobs
        varacf[0] = 0
        varacf[1] = 1. / nobs
        varacf[2:] *= 1 + 2 * np.cumsum(acf[1:-1]**2)
        interval = stats.norm.ppf(1 - alpha / 2.) * np.sqrt(varacf)
        confint = np.array(lzip(acf - interval, acf + interval))
        if not qstat:
            return acf, confint
    if qstat:
        qstat, pvalue = q_stat(acf[1:], nobs=nobs)  # drop lag 0
        if alpha is not None:
            return acf, confint, qstat, pvalue
        else:
            return acf, qstat, pvalue


def pacf_yw(x, nlags=40, method='unbiased'):
    '''Partial autocorrelation estimated with non-recursive yule_walker

    Parameters
    ----------
    x : 1d array
        observations of time series for which pacf is calculated
    nlags : int
        largest lag for which pacf is returned
    method : 'unbiased' (default) or 'mle'
        method for the autocovariance calculations in yule walker

    Returns
    -------
    pacf : 1d array
        partial autocorrelations, maxlag+1 elements

    Notes
    -----
    This solves yule_walker for each desired lag and contains
    currently duplicate calculations.
    '''
    pacf = [1.]
    for k in range(1, nlags + 1):
        pacf.append(yule_walker(x, k, method=method)[0][-1])
    return np.array(pacf)


#NOTE: this is incorrect.
def pacf_ols(x, nlags=40):
    '''Calculate partial autocorrelations

    Parameters
    ----------
    x : 1d array
        observations of time series for which pacf is calculated
    nlags : int
        Number of lags for which pacf is returned.  Lag 0 is not returned.

    Returns
    -------
    pacf : 1d array
        partial autocorrelations, maxlag+1 elements

    Notes
    -----
    This solves a separate OLS estimation for each desired lag.
    '''
    #TODO: add warnings for Yule-Walker
    #NOTE: demeaning and not using a constant gave incorrect answers?
    #JP: demeaning should have a better estimate of the constant
    #maybe we can compare small sample properties with a MonteCarlo
    xlags, x0 = lagmat(x, nlags, original='sep')
    #xlags = sm.add_constant(lagmat(x, nlags), prepend=True)
    xlags = add_constant(xlags)
    pacf = [1.]
    for k in range(1, nlags+1):
        res = OLS(x0[k:], xlags[k:, :k+1]).fit()
         #np.take(xlags[k:], range(1,k+1)+[-1],

        pacf.append(res.params[-1])
    return np.array(pacf)


def pacf(x, nlags=40, method='ywunbiased', alpha=None):
    '''Partial autocorrelation estimated

    Parameters
    ----------
    x : 1d array
        observations of time series for which pacf is calculated
    nlags : int
        largest lag for which pacf is returned
    method : 'ywunbiased' (default) or 'ywmle' or 'ols'
        specifies which method for the calculations to use:

        - yw or ywunbiased : yule walker with bias correction in denominator
          for acovf
        - ywm or ywmle : yule walker without bias correction
        - ols - regression of time series on lags of it and on constant
        - ld or ldunbiased : Levinson-Durbin recursion with bias correction
        - ldb or ldbiased : Levinson-Durbin recursion without bias correction

    alpha : scalar, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        1/sqrt(len(x))

    Returns
    -------
    pacf : 1d array
        partial autocorrelations, nlags elements, including lag zero
    confint : array, optional
        Confidence intervals for the PACF. Returned if confint is not None.

    Notes
    -----
    This solves yule_walker equations or ols for each desired lag
    and contains currently duplicate calculations.
    '''

    if method == 'ols':
        ret = pacf_ols(x, nlags=nlags)
    elif method in ['yw', 'ywu', 'ywunbiased', 'yw_unbiased']:
        ret = pacf_yw(x, nlags=nlags, method='unbiased')
    elif method in ['ywm', 'ywmle', 'yw_mle']:
        ret = pacf_yw(x, nlags=nlags, method='mle')
    elif method in ['ld', 'ldu', 'ldunbiase', 'ld_unbiased']:
        acv = acovf(x, unbiased=True)
        ld_ = levinson_durbin(acv, nlags=nlags, isacov=True)
        #print 'ld', ld_
        ret = ld_[2]
    # inconsistent naming with ywmle
    elif method in ['ldb', 'ldbiased', 'ld_biased']:
        acv = acovf(x, unbiased=False)
        ld_ = levinson_durbin(acv, nlags=nlags, isacov=True)
        ret = ld_[2]
    else:
        raise ValueError('method not available')
    if alpha is not None:
        varacf = 1. / len(x) # for all lags >=1
        interval = stats.norm.ppf(1. - alpha / 2.) * np.sqrt(varacf)
        confint = np.array(lzip(ret - interval, ret + interval))
        confint[0] = ret[0]  # fix confidence interval for lag 0 to varpacf=0
        return ret, confint
    else:
        return ret


def ccovf(x, y, unbiased=True, demean=True):
    ''' crosscovariance for 1D

    Parameters
    ----------
    x, y : arrays
       time series data
    unbiased : boolean
       if True, then denominators is n-k, otherwise n

    Returns
    -------
    ccovf : array
        autocovariance function

    Notes
    -----
    This uses np.correlate which does full convolution. For very long time
    series it is recommended to use fft convolution instead.
    '''
    n = len(x)
    if demean:
        xo = x - x.mean()
        yo = y - y.mean()
    else:
        xo = x
        yo = y
    if unbiased:
        xi = np.ones(n)
        d = np.correlate(xi, xi, 'full')
    else:
        d = n
    return (np.correlate(xo, yo, 'full') / d)[n - 1:]


def ccf(x, y, unbiased=True):
    '''cross-correlation function for 1d

    Parameters
    ----------
    x, y : arrays
       time series data
    unbiased : boolean
       if True, then denominators for autocovariance is n-k, otherwise n

    Returns
    -------
    ccf : array
        cross-correlation function of x and y

    Notes
    -----
    This is based np.correlate which does full convolution. For very long time
    series it is recommended to use fft convolution instead.

    If unbiased is true, the denominator for the autocovariance is adjusted
    but the autocorrelation is not an unbiased estimtor.

    '''
    cvf = ccovf(x, y, unbiased=unbiased, demean=True)
    return cvf / (np.std(x) * np.std(y))


def periodogram(X):
    """
    Returns the periodogram for the natural frequency of X

    Parameters
    ----------
    X : array-like
        Array for which the periodogram is desired.

    Returns
    -------
    pgram : array
        1./len(X) * np.abs(np.fft.fft(X))**2


    References
    ----------
    Brockwell and Davis.
    """
    X = np.asarray(X)
    #if kernel == "bartlett":
    #    w = 1 - np.arange(M+1.)/M   #JP removed integer division

    pergr = 1. / len(X) * np.abs(np.fft.fft(X))**2
    pergr[0] = 0.  # what are the implications of this?
    return pergr


#copied from nitime and statsmodels\sandbox\tsa\examples\try_ld_nitime.py
#TODO: check what to return, for testing and trying out returns everything
def levinson_durbin(s, nlags=10, isacov=False):
    '''Levinson-Durbin recursion for autoregressive processes

    Parameters
    ----------
    s : array_like
        If isacov is False, then this is the time series. If iasacov is true
        then this is interpreted as autocovariance starting with lag 0
    nlags : integer
        largest lag to include in recursion or order of the autoregressive
        process
    isacov : boolean
        flag to indicate whether the first argument, s, contains the
        autocovariances or the data series.

    Returns
    -------
    sigma_v : float
        estimate of the error variance ?
    arcoefs : ndarray
        estimate of the autoregressive coefficients
    pacf : ndarray
        partial autocorrelation function
    sigma : ndarray
        entire sigma array from intermediate result, last value is sigma_v
    phi : ndarray
        entire phi array from intermediate result, last column contains
        autoregressive coefficients for AR(nlags) with a leading 1

    Notes
    -----
    This function returns currently all results, but maybe we drop sigma and
    phi from the returns.

    If this function is called with the time series (isacov=False), then the
    sample autocovariance function is calculated with the default options
    (biased, no fft).
    '''
    s = np.asarray(s)
    order = nlags  # rename compared to nitime
    #from nitime

    ##if sxx is not None and type(sxx) == np.ndarray:
    ##    sxx_m = sxx[:order+1]
    ##else:
    ##    sxx_m = ut.autocov(s)[:order+1]
    if isacov:
        sxx_m = s
    else:
        sxx_m = acovf(s)[:order + 1]  # not tested

    phi = np.zeros((order + 1, order + 1), 'd')
    sig = np.zeros(order + 1)
    # initial points for the recursion
    phi[1, 1] = sxx_m[1] / sxx_m[0]
    sig[1] = sxx_m[0] - phi[1, 1] * sxx_m[1]
    for k in range(2, order + 1):
        phi[k, k] = (sxx_m[k] - np.dot(phi[1:k, k-1],
                                       sxx_m[1:k][::-1])) / sig[k-1]
        for j in range(1, k):
            phi[j, k] = phi[j, k-1] - phi[k, k] * phi[k-j, k-1]
        sig[k] = sig[k-1] * (1 - phi[k, k]**2)

    sigma_v = sig[-1]
    arcoefs = phi[1:, -1]
    pacf_ = np.diag(phi).copy()
    pacf_[0] = 1.
    return sigma_v, arcoefs, pacf_, sig, phi  # return everything


def grangercausalitytests(x, maxlag, addconst=True, verbose=True):
    """four tests for granger non causality of 2 timeseries

    all four tests give similar results
    `params_ftest` and `ssr_ftest` are equivalent based on F test which is
    identical to lmtest:grangertest in R

    Parameters
    ----------
    x : array, 2d, (nobs,2)
        data for test whether the time series in the second column Granger
        causes the time series in the first column
    maxlag : integer
        the Granger causality test results are calculated for all lags up to
        maxlag
    verbose : bool
        print results if true

    Returns
    -------
    results : dictionary
        all test results, dictionary keys are the number of lags. For each
        lag the values are a tuple, with the first element a dictionary with
        teststatistic, pvalues, degrees of freedom, the second element are
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

    'params_ftest', 'ssr_ftest' are based on F distribution

    'ssr_chi2test', 'lrtest' are based on chi-square distribution

    References
    ----------
    http://en.wikipedia.org/wiki/Granger_causality
    Greene: Econometric Analysis

    """
    from scipy import stats

    x = np.asarray(x)

    if x.shape[0] <= 3 * maxlag + int(addconst):
        raise ValueError("Insufficient observations. Maximum allowable "
                         "lag is {0}".format(int((x.shape[0] - int(addconst)) /
                                                 3) - 1))

    resli = {}

    for mlg in range(1, maxlag + 1):
        result = {}
        if verbose:
            print('\nGranger Causality')
            print('number of lags (no zero)', mlg)
        mxlg = mlg

        # create lagmat of both time series
        dta = lagmat2ds(x, mxlg, trim='both', dropex=1)

        #add constant
        if addconst:
            dtaown = add_constant(dta[:, 1:(mxlg + 1)], prepend=False)
            dtajoint = add_constant(dta[:, 1:], prepend=False)
        else:
            raise NotImplementedError('Not Implemented')
            #dtaown = dta[:, 1:mxlg]
            #dtajoint = dta[:, 1:]

        # Run ols on both models without and with lags of second variable
        res2down = OLS(dta[:, 0], dtaown).fit()
        res2djoint = OLS(dta[:, 0], dtajoint).fit()

        #print results
        #for ssr based tests see:
        #http://support.sas.com/rnd/app/examples/ets/granger/index.htm
        #the other tests are made-up

        # Granger Causality test using ssr (F statistic)
        fgc1 = ((res2down.ssr - res2djoint.ssr) /
                res2djoint.ssr / mxlg * res2djoint.df_resid)
        if verbose:
            print('ssr based F test:         F=%-8.4f, p=%-8.4f, df_denom=%d,'
                   ' df_num=%d' % (fgc1,
                                    stats.f.sf(fgc1, mxlg,
                                               res2djoint.df_resid),
                                    res2djoint.df_resid, mxlg))
        result['ssr_ftest'] = (fgc1,
                               stats.f.sf(fgc1, mxlg, res2djoint.df_resid),
                               res2djoint.df_resid, mxlg)

        # Granger Causality test using ssr (ch2 statistic)
        fgc2 = res2down.nobs * (res2down.ssr - res2djoint.ssr) / res2djoint.ssr
        if verbose:
            print('ssr based chi2 test:   chi2=%-8.4f, p=%-8.4f, '
                   'df=%d' % (fgc2, stats.chi2.sf(fgc2, mxlg), mxlg))
        result['ssr_chi2test'] = (fgc2, stats.chi2.sf(fgc2, mxlg), mxlg)

        #likelihood ratio test pvalue:
        lr = -2 * (res2down.llf - res2djoint.llf)
        if verbose:
            print('likelihood ratio test: chi2=%-8.4f, p=%-8.4f, df=%d' %
                   (lr, stats.chi2.sf(lr, mxlg), mxlg))
        result['lrtest'] = (lr, stats.chi2.sf(lr, mxlg), mxlg)

        # F test that all lag coefficients of exog are zero
        rconstr = np.column_stack((np.zeros((mxlg, mxlg)),
                                   np.eye(mxlg, mxlg),
                                   np.zeros((mxlg, 1))))
        ftres = res2djoint.f_test(rconstr)
        if verbose:
            print('parameter F test:         F=%-8.4f, p=%-8.4f, df_denom=%d,'
                   ' df_num=%d' % (ftres.fvalue, ftres.pvalue, ftres.df_denom,
                                    ftres.df_num))
        result['params_ftest'] = (np.squeeze(ftres.fvalue)[()],
                                  np.squeeze(ftres.pvalue)[()],
                                  ftres.df_denom, ftres.df_num)

        resli[mxlg] = (result, [res2down, res2djoint, rconstr])

    return resli


def _safe_arma_fit(y, order, model_kw, trend, fit_kw, start_params=None):
    try:
        return ARMA(y, order=order, **model_kw).fit(disp=0, trend=trend,
                                                    start_params=start_params,
                                                    **fit_kw)
    except LinAlgError:
        # SVD convergence failure on badly misspecified models
        return

    except ValueError as error:
        if start_params is not None:  # don't recurse again
            # user supplied start_params only get one chance
            return
        # try a little harder, should be handled in fit really
        elif ((hasattr(error, 'message') and 'initial' not in error.message)
              or 'initial' in str(error)):  # py2 and py3
            start_params = [.1] * sum(order)
            if trend == 'c':
                start_params = [.1] + start_params
            return _safe_arma_fit(y, order, model_kw, trend, fit_kw,
                                  start_params)
        else:
            return
    except:  # no idea what happened
        return


def arma_order_select_ic(y, max_ar=4, max_ma=2, ic='bic', trend='c',
                         model_kw={}, fit_kw={}):
    """
    Returns information criteria for many ARMA models

    Parameters
    ----------
    y : array-like
        Time-series data
    max_ar : int
        Maximum number of AR lags to use. Default 4.
    max_ma : int
        Maximum number of MA lags to use. Default 2.
    ic : str, list
        Information criteria to report. Either a single string or a list
        of different criteria is possible.
    trend : str
        The trend to use when fitting the ARMA models.
    model_kw : dict
        Keyword arguments to be passed to the ``ARMA`` model
    fit_kw : dict
        Keyword arguments to be passed to ``ARMA.fit``.

    Returns
    -------
    obj : Results object
        Each ic is an attribute with a DataFrame for the results. The AR order
        used is the row index. The ma order used is the column index. The
        minimum orders are available as ``ic_min_order``.

    Examples
    --------

    >>> from statsmodels.tsa.arima_process import arma_generate_sample
    >>> import statsmodels.api as sm
    >>> import numpy as np

    >>> arparams = np.array([.75, -.25])
    >>> maparams = np.array([.65, .35])
    >>> arparams = np.r_[1, -arparams]
    >>> maparam = np.r_[1, maparams]
    >>> nobs = 250
    >>> np.random.seed(2014)
    >>> y = arma_generate_sample(arparams, maparams, nobs)
    >>> res = sm.tsa.arma_order_select_ic(y, ic=['aic', 'bic'], trend='nc')
    >>> res.aic_min_order
    >>> res.bic_min_order

    Notes
    -----
    This method can be used to tentatively identify the order of an ARMA
    process, provided that the time series is stationary and invertible. This
    function computes the full exact MLE estimate of each model and can be,
    therefore a little slow. An implementation using approximate estimates
    will be provided in the future. In the meantime, consider passing
    {method : 'css'} to fit_kw.
    """
    from pandas import DataFrame

    ar_range = lrange(0, max_ar + 1)
    ma_range = lrange(0, max_ma + 1)
    if isinstance(ic, string_types):
        ic = [ic]
    elif not isinstance(ic, (list, tuple)):
        raise ValueError("Need a list or a tuple for ic if not a string.")

    results = np.zeros((len(ic), max_ar + 1, max_ma + 1))

    for ar in ar_range:
        for ma in ma_range:
            if ar == 0 and ma == 0 and trend == 'nc':
                results[:, ar, ma] = np.nan
                continue

            mod = _safe_arma_fit(y, (ar, ma), model_kw, trend, fit_kw)
            if mod is None:
                results[:, ar, ma] = np.nan
                continue

            for i, criteria in enumerate(ic):
                results[i, ar, ma] = getattr(mod, criteria)

    dfs = [DataFrame(res, columns=ma_range, index=ar_range) for res in results]

    res = dict(zip(ic, dfs))

    # add the minimums to the results dict
    min_res = {}
    for i, result in iteritems(res):
        mins = np.where(result.min().min() == result)
        min_res.update({i + '_min_order' : (mins[0][0], mins[1][0])})
    res.update(min_res)

    return Bunch(**res)


if __name__ == "__main__":
    import statsmodels.api as sm
    data = sm.datasets.macrodata.load().data
    x = data['realgdp']
# adf is tested now.
    adf = adfuller(x, 4, autolag=None)
    adfbic = adfuller(x, autolag="bic")
    adfaic = adfuller(x, autolag="aic")
    adftstat = adfuller(x, autolag="t-stat")

# acf is tested now
    acf1, ci1, Q, pvalue = acf(x, nlags=40, confint=95, qstat=True)
    acf2, ci2, Q2, pvalue2 = acf(x, nlags=40, confint=95, fft=True, qstat=True)
    acf3, ci3, Q3, pvalue3 = acf(x, nlags=40, confint=95, qstat=True,
                                 unbiased=True)
    acf4, ci4, Q4, pvalue4 = acf(x, nlags=40, confint=95, fft=True, qstat=True,
                                 unbiased=True)

# pacf is tested now
#    pacf1 = pacorr(x)
#    pacfols = pacf_ols(x, nlags=40)
#    pacfyw = pacf_yw(x, nlags=40, method="mle")
    y = np.random.normal(size=(100, 2))
    grangercausalitytests(y, 2)
