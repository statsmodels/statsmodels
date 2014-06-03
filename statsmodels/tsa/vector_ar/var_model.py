"""
Vector Autoregression (VAR) processes

References
----------
Lutkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.

Hamilton, J. D. 1994. *Time Series Analysis*. Princeton Press.
"""

from __future__ import division, print_function
from statsmodels.compat.python import (range, lrange, string_types, StringIO, iteritems,
                                cStringIO)

from collections import defaultdict

import numpy as np
import numpy.linalg as npl
from numpy.linalg import cholesky as chol, solve
import scipy.stats as stats
import scipy.linalg as L

from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import chain_dot
from statsmodels.tsa.tsatools import vec, unvec
from statsmodels.tools import data as data_util

from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.output import VARSummary

import statsmodels.tsa.tsatools as tsa
import statsmodels.tsa.vector_ar.output as output
import statsmodels.tsa.vector_ar.plotting as plotting
import statsmodels.tsa.vector_ar.util as util
import statsmodels.tsa.base.tsa_model as tsbase
import statsmodels.base.wrapper as wrap

mat = np.array

#-------------------------------------------------------------------------------
# VAR process routines

def ma_rep(coefs, maxn=10):
    r"""
    MA(\infty) representation of VAR(p) process

    Parameters
    ----------
    coefs : ndarray (p x k x k)
    maxn : int
        Number of MA matrices to compute

    Notes
    -----
    VAR(p) process as

    .. math:: y_t = A_1 y_{t-1} + \ldots + A_p y_{t-p} + u_t

    can be equivalently represented as

    .. math:: y_t = \mu + \sum_{i=0}^\infty \Phi_i u_{t-i}

    e.g. can recursively compute the \Phi_i matrices with \Phi_0 = I_k

    Returns
    -------
    phis : ndarray (maxn + 1 x k x k)
    """
    p, k, k = coefs.shape
    phis = np.zeros((maxn+1, k, k))
    phis[0] = np.eye(k)

    # recursively compute Phi matrices
    for i in range(1, maxn + 1):
        for j in range(1, i+1):
            if j > p:
                break

            phis[i] += np.dot(phis[i-j], coefs[j-1])

    return phis

def is_stable(coefs, verbose=False):
    """
    Determine stability of VAR(p) system by examining the eigenvalues of the
    VAR(1) representation

    Parameters
    ----------
    coefs : ndarray (p x k x k)

    Returns
    -------
    is_stable : bool
    """
    A_var1 = util.comp_matrix(coefs)
    eigs = np.linalg.eigvals(A_var1)

    if verbose:
        print('Eigenvalues of VAR(1) rep')
        for val in np.abs(eigs):
            print(val)

    return (np.abs(eigs) <= 1).all()

def var_acf(coefs, sig_u, nlags=None):
    """
    Compute autocovariance function ACF_y(h) up to nlags of stable VAR(p)
    process

    Parameters
    ----------
    coefs : ndarray (p x k x k)
        Coefficient matrices A_i
    sig_u : ndarray (k x k)
        Covariance of white noise process u_t
    nlags : int, optional
        Defaults to order p of system

    Notes
    -----
    Ref: Lutkepohl p.28-29

    Returns
    -------
    acf : ndarray, (p, k, k)
    """
    p, k, _ = coefs.shape
    if nlags is None:
        nlags = p

    # p x k x k, ACF for lags 0, ..., p-1
    result = np.zeros((nlags + 1, k, k))
    result[:p] = _var_acf(coefs, sig_u)

    # yule-walker equations
    for h in range(p, nlags + 1):
        # compute ACF for lag=h
        # G(h) = A_1 G(h-1) + ... + A_p G(h-p)

        for j in range(p):
            result[h] += np.dot(coefs[j], result[h-j-1])

    return result

def _var_acf(coefs, sig_u):
    """
    Compute autocovariance function ACF_y(h) for h=1,...,p

    Notes
    -----
    Lutkepohl (2005) p.29
    """
    p, k, k2 = coefs.shape
    assert(k == k2)

    A = util.comp_matrix(coefs)
    # construct VAR(1) noise covariance
    SigU = np.zeros((k*p, k*p))
    SigU[:k,:k] = sig_u

    # vec(ACF) = (I_(kp)^2 - kron(A, A))^-1 vec(cov_resid)
    vecACF = L.solve(np.eye((k*p)**2) - np.kron(A, A), vec(SigU))

    acf = unvec(vecACF)
    acf = acf[:k].T.reshape((p, k, k))

    return acf

def forecast(y, coefs, trend_coefs, steps):
    """
    Produce linear MSE forecast

    Parameters
    ----------
    y : ndarray (k_ar x neqs)
        The initial values to use for the forecasts.
    coefs : ndarray (k_ar x neqs x neqs)
        Each of the k_ar matrices are for lag 1, ... , lag k_ar. Where the
        columns are the variable and the rows are the equations.
        Ie., coefs[i-1] is the estimated A_i matrix. See VARResults Notes.
    trend_coefs : ndarray
        1d or 2d array. If 1d, should be of length neqs and is assumed to be
        a vector of constants. If 2d should be of shape k_trend x neqs.
    steps : int
        Number of steps ahead to forecast

    Returns
    -------
    forecasts : ndarray (steps x neqs)

    Notes
    -----
    Also used by DynamicVAR class
    """
    #TODO: give math not a page in a book
    # Lutkepohl p. 37
    y = np.asarray(y) # handle pandas but not structured arrays, oh well.
    p = len(coefs)
    k = len(coefs[0])
    # initial value
    #TODO: This is now wrong for trend_coefs != intercept
    forcs = np.zeros((steps, k)) + trend_coefs

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

def forecast_cov(ma_coefs, sig_u, steps):
    """
    Compute theoretical forecast error variance matrices

    Parameters
    ----------

    Returns
    -------
    forc_covs : ndarray (steps x neqs x neqs)
    """
    k = len(sig_u)
    forc_covs = np.zeros((steps, k, k))

    prior = np.zeros((k, k))
    for h in range(steps):
        # Sigma(h) = Sigma(h-1) + Phi Sig_u Phi'
        phi = ma_coefs[h]
        var = chain_dot(phi, sig_u, phi.T)
        forc_covs[h] = prior = prior + var

    return forc_covs

def var_loglike(resid, cov_resid_mle, nobs):
    r"""
    Returns the value of the VAR(p) log-likelihood.

    Parameters
    ----------
    resid : ndarray (nobs x neqs)
        The residuals of each equation.
    cov_resid_mle : ndarray
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
        The value of the loglikelihood function for a VAR(p) model

    Notes
    -----
    The loglikelihood function for the VAR(p) is

    .. math::

       -\left(\frac{nobs}{2}\right)
       \left(\ln\left|cov_sigma_mle\right|-neqs\ln\left(2\pi\right)-neqs\right)
    """
    logdet = util.get_logdet(np.asarray(cov_resid_mle))
    neqs = len(cov_resid_mle)
    part1 = - (nobs * neqs / 2) * np.log(2 * np.pi)
    part2 = - (nobs / 2) * (logdet + neqs)
    return part1 + part2

def _reordered(self, order):
    #Create new arrays to hold rearranged results from .fit()
    Y = self.Y
    # X are the lagged endogenous variables
    X = self.X
    params = self.params
    cov_resid = self.cov_resid
    names = self.model.endog_names
    k_ar = self.k_ar
    k_trend = self.k_trend
    neqs = self.neqs
    Y_reordered = np.zeros([np.size(Y,0),np.size(Y,1)])
    X_reordered = np.zeros([np.size(X, 0),
                                 np.size(X, 1)])
    params_reordered_inc, params_reordered = [np.zeros([np.size(params,0),
                                        np.size(params,1)]) for i in range(2)]
    cov_resid_reordered_inc, cov_resid_reordered = [np.zeros([np.size(cov_resid,0),
                                    np.size(cov_resid,1)]) for i in range(2)]

    names_new = []

    #Rearrange elements and fill in new arrays
    for i, c in enumerate(order):
        Y_reordered[:,i] = Y[:,c]
        if k_trend > 0:
            params_reordered_inc[0,i] = params[0,i]
            X_reordered[:,0] = X[:,0]
        for j in range(k_ar):
            params_reordered_inc[i + j*neqs + k_trend, :] = params[c + j * neqs +
                                                                    k_trend, :]
            X_reordered[:,i + j*neqs + k_trend] = X[:, c + j * neqs + k_trend]
        cov_resid_reordered_inc[i,:] = cov_resid[c,:]
        names_new.append(names[c])
    for i, c in enumerate(order):
        params_reordered[:,i] = params_reordered_inc[:,c]
        cov_resid_reordered[:,i] = cov_resid_reordered_inc[:,c]

    #TODO: reset cached names and do this at data level?
    #What's the best way to achieve this? We can't just return self.model
    # right?
    self.endog_names = names_new
    names_new = util.make_lag_names(names_new, k_ar, k_trend)
    return VARResults(self.model, Y_reordered, X_reordered, params_reordered,
                      cov_resid_reordered, k_ar, names_new)

_var_math_doc = """
    .. math::

       y_t = A_1 y_{t-1} + \\ldots + A_p y_{t-p} + u_t

    where

    .. math::

       u_t \sim {\sf Normal}(0, \Sigma_u)"""

_var_model_doc = """
    Fit VAR(p) process and do lag order selection

    """ + _var_math_doc

_var_params_doc = """Y : array-like
        2-d endogenous response variable. The independent variable.
    nlags : int
        The number of lags to use in the model fitting.
    names : array-like
        Names is deprecated. Use a DataFrame or a structured array to give
        names to VAR. Must match number of columns of endog"""

_var_reference_doc = """
    References
    ----------
    Lutkepohl (2005) New Introduction to Multiple Time Series Analysis"""

def _get_info_criteria(cov_resid_mle, nobs, neqs, k_ar, k_trend):
    free_params = k_ar * neqs ** 2 + neqs * k_trend
    df_resid = nobs - (neqs * k_ar + k_trend)
    df_model = neqs * k_ar + k_trend
    ld = util.get_logdet(cov_resid_mle)

    # See Lutkepohl pp. 146-150

    aic = ld + (2. / nobs) * free_params
    bic = ld + (np.log(nobs) / nobs) * free_params
    hqic = ld + (2. * np.log(np.log(nobs)) / nobs) * free_params
    fpe = ((nobs + df_model) / df_resid) ** neqs * np.exp(ld)
    return aic, bic, hqic, fpe

def _estimate_var_ic(Y, k_ar, k_trend, trim=0, trend='c'):
    """
    Helper function to fit VAR that aids in lag order selection

    Y is expected to be an array.
    trim is number of variables to trim from the beginning of Y. It's used
    so that the ICs are comparable across fits with different lags.
    """
    Y = Y[trim:]
    X = util.get_lagged_y(Y, k_ar, trend=trend)
    Y_sample = Y[k_ar:]

    #TODO check -QR is O(k**3) and SVD is O(nobs*k**2)
    params = np.linalg.lstsq(X, Y_sample)[0]
    resid = Y_sample - np.dot(X, params)

    nobs, neqs = Y_sample.shape

    # MLE estimate of covariance matrix of residuals
    cov_resid_mle = np.dot(resid.T, resid) / nobs
    return _get_info_criteria(cov_resid_mle, nobs, neqs, k_ar, k_trend)

def select_order_fit(Y, maxlags=None, ic="aic", trend="c", verbose=False):
    """
    Return results from the best lag order according to information criterion

    Parameters
    ----------
    Y : array-like
        The endogenous variables
    maxlags : int
        Maximum number of lags to check for order selection, defaults to
        12 * (nobs/100.)**(1./4), see select_order function
    ic : {'aic', 'fpe', 'hqic', 'bic', None}
        Information criterion to use for VAR order selection:

        * aic : Akaike
        * fpe : Final prediction error
        * hqic : Hannan-Quinn
        * bic : Bayesian a.k.a. Schwarz
    trend : str {"c", "ct", "ctt", "nc"}
        Available options are:

        * "c" - add constant
        * "ct" - constant and trend
        * "ctt" - constant, linear and quadratic trend
        * "nc" - co constant, no trend

        Note that these are prepended to the columns of X.
    verbose : bool
        Print order selection output to the screen. Default is False.

    Returns
    -------
    fitted : VARResults
        A VARResults instance using the best lag order indicated by `ic`.
    """
    Y_array = np.asarray(Y)
    if data_util._is_structured_ndarray(Y_array):
        Y_array = Y_array.view((float, len(Y_array.dtype)))

    if maxlags is None:
        maxlags = int(round(12*(len(Y)/100.)**(1/4.)))

    k_trend = util.get_trendorder(trend)
    ics = defaultdict(list)
    for k_ar in range(maxlags + 1):
        # exclude some periods so same nobs used for each lag order
        res = _estimate_var_ic(Y_array, k_ar, k_trend, trim=maxlags - k_ar,
                               trend=trend)
        ics["aic"].append(res[0])
        ics["bic"].append(res[1])
        ics["hqic"].append(res[2])
        ics["fpe"].append(res[3])

    selected_orders = dict((k, mat(v).argmin())
                           for k, v in ics.iteritems())
    try:
        nlags = selected_orders[ic]
    except:
        KeyError("ic %s is not understood" % ic)
    return VAR(Y, nlags).fit(trend=trend)

class VAR(tsbase.TimeSeriesModel):
    __doc__ = tsbase._tsa_doc % {"model" : _var_model_doc,
                                 "params" : _var_params_doc,
                                 "extra_params" : tsbase._missing_param_doc,
                                 "extra_sections" : _var_reference_doc}
    def __init__(self, Y, nlags=None, dates=None, names=None, freq=None,
                 missing='none'):
        if nlags is None:
            warn("In the 0.6.0 release nlags will not be optional in the "
                 "model constructor.", FutureWarning)
        else:
            self.k_ar = nlags
        super(VAR, self).__init__(Y, None, dates, freq, missing=missing)
        if self.endog.ndim == 1:
            raise ValueError("Only one variable given to VAR. Expects > 1.")
        if names is not None:
            import warnings
            warnings.warn("The names argument is deprecated and will be "
                          "removed in 0.6.0. Use a pandas object or structured "
                          "array for names.", FutureWarning)
            self.names = names
        else:
            self.names = self.endog_names
        self.Y = self.endog #keep alias for now
        self.neqs = self.Y.shape[1]

    def _get_predict_start(self, start, k_ar):
        if start is None:
            start = k_ar
        return super(VAR, self)._get_predict_start(start)

    def loglike(self, params, k_ar=None):
        r"""
        Returns the value of the VAR(p) log-likelihood.

        Parameters
        ----------
        params : ndarray
            The parameters in the order given in VARResults.
        k_ar : int, optional
            The number of lags if the equation has not yet been fit.

        Returns
        -------
        llf : float
            The value of the loglikelihood function for a VAR(p) model

        Notes
        -----
        The loglikelihood function for the VAR(p) is

        .. math::

           -\left(\frac{nobs}{2}\right)
           \left(\ln\left|cov_sigma_mle\right|-neqs\ln\left(2\pi\right)-neqs\right)
        """
        if k_ar is None:
            try:
                k_ar = self.k_ar
            except:
                raise ValueError("Must give k_ar, if the model has not been "
                                 "fit")

        Y_sample = self.Y[k_ar:]
        nobs = len(Y_sample)
        X = util.get_lagged_y(Y, k_ar, trend=trend)
        resid = Y_sample - np.dot(X, params)
        sse = np.dot(resid.T, resid)
        cov_resid_mle = sse / nobs
        return var_loglike(resid, cov_resid_mle, nobs)

    def predict(self, params, start=None, end=None, lags=1, trend='c'):
        """
        Returns in-sample predictions or forecasts
        """
        start = self._get_predict_start(start, lags)
        end, out_of_sample = self._get_predict_end(end)

        if end < start:
            raise ValueError("end is before start")
        if end == start + out_of_sample:
            return np.array([])

        k_trend = util.get_trendorder(trend)
        k = self.neqs
        k_ar = lags

        predictedvalues = np.zeros((end + 1 - start + out_of_sample, k))
        if k_trend != 0:
            trend_coefs = params[:k_trend]
            #TODO: this is now wrong for trend_coefs != intercept
            predictedvalues += trend_coefs

        Y = self.Y
        # lagged Y, includes trend variables in first column(s)
        X = util.get_lagged_y(Y, lags, trend=trend)
        fittedvalues = np.dot(X, params)

        fv_start = start - k_ar
        pv_end = min(len(predictedvalues), len(fittedvalues) - fv_start)
        fv_end = min(len(fittedvalues), end-k_ar+1)
        predictedvalues[:pv_end] = fittedvalues[fv_start:fv_end]

        if not out_of_sample:
            return predictedvalues

        # fit out of sample
        Y = Y[-k_ar:]
        coefs = params[k_trend:].reshape((k_ar, k, k)).swapaxes(1,2)
        predictedvalues[pv_end:] = forecast(Y, coefs, trend_coefs,
                                            out_of_sample)
        return predictedvalues

    def fit(self, maxlags=None, method='ols', ic=None, trend='c',
            verbose=False):
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
            Information criterion to use for VAR order selection:

            * aic : Akaike
            * fpe : Final prediction error
            * hqic : Hannan-Quinn
            * bic : Bayesian a.k.a. Schwarz

        verbose : bool, default False
            Print order selection output to the screen
        trend : str {"c", "ct", "ctt", "nc"}
            Available options are:

            * "c" - add constant
            * "ct" - constant and trend
            * "ctt" - constant, linear and quadratic trend
            * "nc" - co constant, no trend

            Note that these are prepended to the columns of X.

        Returns
        -------
        est : VARResults
            The results instance.

        Notes
        -----
        Fits the VAR model using OLS.
        """
        if maxlags is not None:
            warn("The maxlags argument to fit is deprecated. Please use the "
                 "model constructor argument nlags. maxlags overwrites nlags"
                 " given in the model constructor.", FutureWarning)

            k_ar = maxlags
        else:
            try:
                assert hasattr(self, "k_ar")
            except:
                raise ValueError("Please give nlags to the model constructor "
                        "before calling fit.")
            k_ar = self.k_ar

        if ic is not None:
            warn("The ic argument to fit is deprecated. Please use the "
                 "select_order classmethod or select_order_fit instead.",
                 FutureWarning)
            selections = self.select_order(self.Y, maxlags=maxlags,
                                           verbose=verbose)
            if ic not in selections:
                raise Exception("%s not recognized, must be among %s"
                                % (ic, sorted(selections)))
            k_ar = selections[ic]
            if verbose:
                print('Using %d based on %s criterion' %  (k_ar, ic))
        else:
            if k_ar is None:
                k_ar = 1

        k_trend = util.get_trendorder(trend)
        #TODO: anything we attach here might be stale on refit. Test.
        self.nobs = len(self.Y) - k_ar
        self.trend = trend

        return self._estimate_var(k_ar, trend=trend)

    #NOTE: maybe we should have a select_sample method in TSAModel to do what
    #      offset is doing?
    def _estimate_var(self, k_ar, offset=0, trend='c'):
        """
        The offset is an int. For the number of periods to drop from
        beginning so that the sample size is comparable when doing order
        selection.
        """
        # have to do this again because select_order doesn't call fit
        self.k_trend = k_trend = util.get_trendorder(trend)

        if offset < 0: # pragma: no cover
            raise ValueError('offset must be >= 0')

        Y = self.Y[offset:]

        # lagged Y, includes trend variable(s) in first column(s)
        X = util.get_lagged_y(Y, k_ar, trend=trend)
        Y_sample = Y[k_ar:]

        # Lutkepohl p75, about 5x faster than stated formula
        #TODO: Check - QR is O(k^3) and SVD is O(nobs*k^2)
        params = np.linalg.lstsq(X, Y_sample)[0]
        resid = Y_sample - np.dot(X, params)


        available_obs = len(Y_sample)

        df_resid = available_obs - (self.neqs * k_ar + k_trend)

        # Unbiased estimate of covariance matrix of residuals
        sse = np.dot(resid.T, resid)
        cov_resid = sse / df_resid

        varfit = VARResults(self, Y, X, params, cov_resid, k_ar,
                            self.endog_names)
        #TODO: remove names - need them right now for _reorder
        return VARResultsWrapper(varfit)

    @classmethod
    def select_order(self, Y=None, maxlags=None, verbose=True, trend='c'):
        """
        Compute lag order selections based on each information criteria

        Parameters
        ----------
        Y : array-like
            The endogenous variables.
        maxlags : int
            if None, defaults to 12 * (nobs/100.)**(1./4)
        verbose : bool, default True
            If True, print table of info criteria and selected orders
        trend : str {"c", "ct", "ctt", "nc"}
            Available options are:

            * "c" - add constant
            * "ct" - constant and trend
            * "ctt" - constant, linear and quadratic trend
            * "nc" - co constant, no trend

            Note that these are prepended to the columns of X.

        Returns
        -------
        selections : dict {info_crit -> selected_order}
        """
        if Y is None:
            warn("The Y argument will not be optional in 0.6.0",
                 FutureWarning)
            Y_array = self.Y
        else: # deal with Y as pandas
            Y_array = np.asarray(Y)
            if data_util._is_structured_ndarray(Y_array):
                Y_array = Y_array.view((float, len(Y_array.dtype)))

        if hasattr(self, "k_ar"):
            warn("In the 0.6.0 release you will not be able to use "
                 "this method on a model instance.", FutureWarning)

        if maxlags is None:
            maxlags = int(round(12*(len(Y)/100.)**(1/4.)))

        k_trend = util.get_trendorder(trend)

        ics = defaultdict(list)
        for k_ar in range(maxlags + 1):
            # exclude some periods so same nobs used for each lag order
            res = _estimate_var_ic(Y_array, k_ar, k_trend,
                                   trim=maxlags - k_ar, trend=trend)
            ics["aic"].append(res[0])
            ics["bic"].append(res[1])
            ics["hqic"].append(res[2])
            ics["fpe"].append(res[3])

        selected_orders = dict((k, mat(v).argmin())
                               for k, v in iteritems(ics))

        if verbose:
            output.print_ic_table(ics, selected_orders)

        return selected_orders

select_order = VAR.select_order # convenience function

#-----------------------------------------------------------------------------
# VARProcess class: for known or unknown VAR process

class VARProcess(object):
    """
    Class represents a known VAR(p) process

    Parameters
    ----------
    coefs : ndarray (p x neqs x neqs)
        Each of the p matrices are for lag 1, ... , lag p. Where the columns
        are the variable and the rows are the equations.
        Ie., coefs[i-1] is the estimated A_i matrix. See Notes.
    trend_coefs : ndarray
        1d or 2d array. If 1d, should be of length neqs and is assumed to be
        a vector of constants. If 2d should be of shape k_trend x neqs
    cov_resid : ndarray (neqs x neqs)
        The covariance matrix of the residuals. :math:`\Sigma_u` in the Notes.
    names : sequence (length neqs)
        The names of the endogenous variables.

    Returns
    -------
    **Attributes**:

    Notes
    -----
    The VAR(p) process is assumed to be

    """ + _var_math_doc

    def __init__(self, coefs, trend_coefs, cov_resid, names=None):
        self.k_ar = len(coefs)
        self.neqs = coefs.shape[1]
        self.coefs = coefs
        self.trend_coefs = trend_coefs
        self.cov_resid = cov_resid
        self.names = names

    #NOTE: This should go up a level in a systems of equations base class
    def get_eq_index(self, name):
        "Return integer position of requested equation name"
        return util.get_index(self.model.endog_names, name)

    #TODO: Do we have this for other models? What are we doing?
    def __str__(self):
        output = ('VAR(%d) process for %d-dimensional response y_t'
                  % (self.k_ar, self.neqs))
        output += '\nstable: %s' % self.is_stable()
        output += '\nmean: %s' % self.mean()

        return output

    def is_stable(self, verbose=False):
        """Determine stability based on model coefficients

        Parameters
        ----------
        verbose : bool
            Print eigenvalues of the VAR(1) companion

        Notes
        -----
        Checks if det(I - Az) = 0 for any mod(z) <= 1, so all the eigenvalues
        of the companion matrix must lie outside the unit circle
        """
        #TODO: define companion matrix in documentation
        return is_stable(self.coefs, verbose=verbose)

    def generate_sample(self, size=100):
        """
        Generate VAR samples.

        Parameters
        ----------
        size : int
            The number of observations to return.
        """
        return util.varsim(self.coefs, self.trend_coefs, self.cov_resid,
                           steps=size)

    def plotsim(self, steps=1000, ax=None):
        """
        Plot a simulation from the VAR(p) process.

        Parameters
        ----------
        steps : int
            The number of observations to simulate.
        ax : matplotlib.axes, optional
            An existing `matplotlib.axes`
        """
        Y = self.generate_sample(size=steps)
        return plotting.plot_timeseries(Y)

    def mean(self):
        r"""Mean of stable process

        Lutkepohl eq. 2.1.23

        .. math:: \mu = (I - A_1 - \dots - A_p)^{-1} \alpha
        """
        #TODO: Define A_1, A_p, and \alpha
        return solve(self._char_mat, self.trend_coefs)

    def ma_rep(self, maxn=10):
        r"""Compute MA representation of the VAR model

        Parameters
        ----------
        maxn : int
            Number of coefficient matrices to compute

        Returns
        -------
        coefs : ndarray (maxn x k x k)

        Notes
        -----
        These are the MA(:math:`\infty`) coefficient matrices.
        """
        return ma_rep(self.coefs, maxn=maxn)

    def orth_ma_rep(self, maxn=10, P=None):
        r"""Compute Orthogonalized MA coefficient matrices using P matrix such
        that :math:`cov_resid = PP^\prime`. P defaults to the Cholesky
        decomposition of :math:`cov_resid`

        Parameters
        ----------
        maxn : int
            Number of coefficient matrices to compute
        P : ndarray (k x k), optional
            Matrix such that cov_resid = PP', defaults to Cholesky
            decomposition.

        Returns
        -------
        coefs : ndarray (maxn x k x k)
        """
        if P is None:
            P = self._chol_cov_resid

        ma_mats = self.ma_rep(maxn=maxn)
        return mat([np.dot(coefs, P) for coefs in ma_mats])

    def long_run_effects(self):
        """Compute long-run effect of unit impulse

        .. math::

            \Psi_\infty = \sum_{i=0}^\infty \Phi_i
        """
        return L.inv(self._char_mat)

    @cache_readonly
    def _chol_cov_resid(self):
        return chol(self.cov_resid)

    @cache_readonly
    def _char_mat(self):
        return np.eye(self.neqs) - self.coefs.sum(0)

    def acf(self, nlags=None):
        """Compute theoretical autocovariance function

        Parameters
        ----------
        nlags : int or None
            The number of lags to include in the autocovariance function. The
            default is the number of lags included in the model.

        Returns
        -------
        acf : ndarray (p x k x k)
        """
        return var_acf(self.coefs, self.cov_resid, nlags=nlags)

    def acorr(self, nlags=None):
        """Compute theoretical autocorrelation function

        Parameters
        ----------
        nlags : int or None
            The number of lags to include in the autocorrelation function. The
            default is the number of lags included in the model.

        Returns
        -------
        acorr : ndarray (p x k x k)
        """
        return util.acf_to_acorr(self.acf(nlags=nlags))

    def plot_acorr(self, nlags=10, linewidth=8):
        "Plot theoretical autocorrelation function"
        plotting.plot_full_acorr(self.acorr(nlags=nlags), linewidth=linewidth,
                                 names=self.model.endog_names)

    def forecast(self, y=None, steps=1):
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

        Returns
        -------
        forecasts : ndarray (steps x neqs)
            The forecasted values.
        """
        #TODO: give the math instead the reference in Notes
        # Lutkepohl pp 37-38

        if y is not None:
            warn("The use of y is deprecated and will be removed in 0.6.0. "
                 "If you want to give initial values use "
                 "statsmodels.tsa.var.forecast.")
        else:
            y = self.model.Y[-self.k_ar:]
        return forecast(y, self.coefs, self.trend_coefs, steps)

    def mse(self, steps):
        """
        Compute theoretical forecast error variance matrices

        Parameters
        ----------
        steps : int
            Number of steps ahead

        Notes
        -----
        .. math:: \mathrm{MSE}(h) = \sum_{i=0}^{h-1} \Phi cov_resid \Phi^T

        Returns
        -------
        forc_covs : ndarray (steps x neqs x neqs)
        """
        ma_coefs = self.ma_rep(steps)

        k = len(self.cov_resid)
        forc_covs = np.zeros((steps, k, k))

        prior = np.zeros((k, k))
        for h in range(steps):
            # Sigma(h) = Sigma(h-1) + Phi Sig_u Phi'
            phi = ma_coefs[h]
            var = chain_dot(phi, self.cov_resid, phi.T)
            forc_covs[h] = prior = prior + var

        return forc_covs

    forecast_cov = mse

    def _forecast_vars(self, steps):
        covs = self.forecast_cov(steps)

        # Take diagonal for each cov
        inds = np.arange(self.neqs)
        return covs[:, inds, inds]

    def forecast_interval(self, y=None, steps=1, alpha=0.05):
        """Construct forecast interval estimates assuming the y are Gaussian

        Parameters
        ----------
        y : ndarray (k_ar x neqs)
            Deprecated. The initial values to use for the forecasts. If None,
            the last k_ar values of the original endogenous variables are
            used.
        steps : int
            The number of steps ahead to forecast.

        Returns
        -------
        (lower, mid, upper) : (ndarray, ndarray, ndarray)
        """
        #NOTE: use math not a page number
        #Lutkepohl pp. 39-40
        assert(0 < alpha < 1)
        q = util.norm_signif_level(alpha)

        point_forecast = self.forecast(y, steps)
        sigma = np.sqrt(self._forecast_vars(steps))

        forc_lower = point_forecast - q * sigma
        forc_upper = point_forecast + q * sigma

        return point_forecast, forc_lower, forc_upper


#-------------------------------------------------------------------------------
# VARResults class


class VARResults(VARProcess, tsbase.TimeSeriesModelResults):
    """The results instances of a fitted VAR(p) model.

    Parameters
    ----------
    model : VAR model instance
        A fitted VAR model
    Y : array
        2d array that contains the endogenous variables
    X : array
        2d array that contains the lagged endogenous variables.
    params : array
        The VAR(p) coefficients. The shape is ((k_trend + k_ar*neqs) x neqs)
        Each equation is expected to be in a column. The order of the
        parameters are trend variables first, then first lag of the first
        equation variable, the second lag of the second equation variables,
        etc.
    cov_resid : array
        The covariance matrix of the residuals. This is the equation by
        equation covariance.
    lag_order : int
        The number of lags used.

    Returns
    -------
    **Attributes**
    coefs : ndarray (k_ar x neqs x neqs)
        Each of the k_ar arrays are for lag 1, ... , lag k_ar. Where the
        columns are the variable and the rows are the equations.
        Ie., coefs[i-1] is the estimated A_i matrix. See Notes.
    cov_params : ndarray
        Variance covariance of the model coefficients.
    det_cov_resid : float
        The determinant of cov_resid
    df_model : int
        Model degress of freedom.
    df_resid : int
        Residual degrees of freedom. Equals `nobs - (neqs * k_ar + k_trend)`
    Y : array
        The endogenous variables.
    X : array
        The lagged endogenous variables, preceeded by the trend variable(s),
        if requested during `fit`. They are in the order of the equations then
        the lags. Ie., after the trend variables in increasing polynomial
        order, y_1{t-1}, ..., y_{neqs}{t-1}, ..., y_1{t-k_ar}, ...
        y_{neqs}{t-1}
    fittedvalues : array
        `nobs x neqs` array of in-sample predicted values.
    info_criteria : dict
        A dictionary containing the information criteria values.
    k_ar : int
        The number of lags in the model.
    k_trend : int
        The number of trend variables included. Equals the polynomial order
        of a trend variable + 1.
    llf : float
        The log-likelihood of the fitted model.
    model : `VARModel`
        A reference to the `statsmodels.tsa.vector_ar.var_model.VARModel`
        instance.
    names : list
        The names of the
    neqs : int
        Number of variables (equations)
    nobs : int
        The number of observations available for estimation. This excludes
        the first k_ar "pre-sample" observations of the original Y.
    n_totobs : int
        The total number of observations. `k_ar` + `nobs`
    k_ar : int
        Order of VAR process
    params : ndarray (k_trend + neqs*k_ar) x neqs
        Trend_coefficients and A_i matrices in stacked form.
    names : list
        variables names
    resid : ndarray
        `nobs` x `neqs` array of residuals.
    cov_resid : ndarray (neqs x neqs)
        Estimate of white noise process variance Var(resid_t). This is the
        equation by equation covariance matrix. It is the same as
        `cov_resid_mle` * `nobs` / (`nobs` - `neqs`*`k_ar` - `k_trend`)
    cov_resid_mle : ndarray (neqs x neqs)
        cov_resid without a degrees of freedom adjustment. Asymptotically,
        `cov_resid_mle` is equivalent to `cov_resid`
    trendorder : int
        The polynomial order of the trend. Ie., nobs**trendorder. If the
        model is fit without a constant, then trendorder = None.
    Y : array
        Endogenous variable.
    X : array
        Lagged endogenous variables. Trend variable(s) are prepended in
        columns. Ie., the first `k_trend` columns are the trend.

    Notes
    -----
    """ + _var_math_doc
    _model_type = 'VAR'

    def __init__(self, model, Y, X, params, cov_resid, k_ar, names):

        self.model = model
        self.Y = self.endog = Y  #keep alias for now
        self.X = X

        self.n_totobs, neqs = self.Y.shape
        self.nobs = self.n_totobs - k_ar
        k_trend = model.k_trend
        if k_trend > 0: # make this the polynomial trend order
            trendorder = k_trend - 1
        else:
            trendorder = None
        self.k_trend = k_trend
        self.trendorder = trendorder
        if model.exog_names is not None:
            warn("This model instance has previously been fit. The "
                 "model.exog_names will be changed, but the previous result "
                 "instance's will still be correct. This will be fixed with "
                 "changes for 0.6.0.", FutureWarning)
        self.exog_names = self.model.exog_names = util.make_lag_names(names,
                                                                      k_ar,
                                                                      k_trend)
        #TODO: do we need to attach this to model?
        # what is data.exog_names is going to be none, but maybe that's ok
        # what are the expectations of the wrappers?
        #model.exog_names = self.exog_names
        self.params = params

        # Initialize VARProcess parent class
        # construct coefficient matrices
        # Each matrix needs to be transposed
        reshaped = self.params[self.k_trend:]
        reshaped = reshaped.reshape((k_ar, neqs, neqs))

        # Need to transpose each coefficient matrix
        trend_coefs = self.params[:self.k_trend]
        coefs = reshaped.swapaxes(1, 2).copy()

        super(VARResults, self).__init__(coefs, trend_coefs, cov_resid,
                                         names=names)

    @cache_readonly
    def normalized_cov_params(self):
        return self.cov_params()

    @cache_readonly
    def coef_names(self):
        "Coefficient names (deprecated)"
        warn("coef_names is deprecated and will be removed in 0.6.0."
             "Use exog_names", FutureWarning)
        return self.exog_names

    def plot(self, ax=None):
        """
        Plot input time series

        Parameters
        ----------
        ax : matplotlib.axes, optional
            An existing matplotlib.axes instance. Should be a list of axes of
            length `neqs`

        Returns
        -------
        fig : `matplotlib.figure`
            The figure that contains the axes
        """
        return plotting.plot_timeseries(self.Y, names=self.model.endog_names,
                                 index=self.model.data.dates, ax=ax)

    @property
    def df_model(self):
        "Number of parameters for each equation `neqs`*`k_ar` + `k_trend`"
        return self.neqs * self.k_ar + self.k_trend

    @property
    def df_resid(self):
        "Number of observations minus number of estimated parameters"
        return self.nobs - self.df_model

    @cache_readonly
    def fittedvalues(self):
        "The predicted in-sample values of the response variables of the model"
        return np.dot(self.X, self.params)

    @cache_readonly
    def resid(self):
        "Residuals of response variable resulting from estimated coefficients"
        return self.Y[self.k_ar:] - self.fittedvalues

    def sample_acov(self, nlags=1):
        ""
        return _compute_acov(self.Y[self.k_ar:], nlags=nlags)

    def sample_acorr(self, nlags=1):
        """
        Parameters
        ----------
        nlags : int
            The number of lags to include. Does not count the zero lag, which
            will be returned.

        Returns
        -------
        acorr : ndarray
            The autocorrelation including the zero lag. Shape is (`nlags` + 1
            x `neqs` x `neqs`)
        """
        acovs = self.sample_acov(nlags=nlags)
        return _acovs_to_acorrs(acovs)

    def plot_sample_acorr(self, nlags=10, linewidth=8, **plot_kwargs):
        """
        Plot theoretical autocorrelation function

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
        return plotting.plot_full_acorr(self.sample_acorr(nlags=nlags),
                                        linewidth=linewidth,
                                        names=self.model.endog_names,
                                        **plot_kwargs)

    def resid_acov(self, nlags=1):
        """
        Compute centered sample autocovariance (including lag 0)

        Parameters
        ----------
        nlags : int
            The number of lags excluding the zero lag to include.

        Returns
        -------
        acov : ndarray
            The autocovariance for the residuals. The shape is (nlags + 1 x
            neqs x neqs).
        """
        return _compute_acov(self.resid, nlags=nlags)

    def resid_acorr(self, nlags=1):
        """
        Compute sample autocorrelation (including lag 0)

        Parameters
        ----------
        nlags : int

        Returns
        -------
        acorr : ndarray
            The autocorrelation for the residuals. The shape is (nlags + 1 x
            neqs x neqs).
        """
        acovs = self.resid_acov(nlags=nlags)
        return _acovs_to_acorrs(acovs)

    @cache_readonly
    def resid_corr(self):
        "Centered residual correlation matrix"
        return self.resid_acorr(0)[0]

    @cache_readonly
    def cov_resid_mle(self):
        """(Biased) maximum likelihood estimate of noise process covariance
        """
        return self.cov_resid * self.df_resid / self.nobs

    def cov_params(self, r_matrix=None, *args, **kwargs):
        """
        Estimated variance-covariance of model coefficients

        Parameters
        ----------
        r_matrix : array-like
            Can be 1d, or 2d. If `r_matrix` is not None the covariance is
            r_matrix kron((X.T X)^(-1), cov_resid) r_matrix.T

        Returns
        -------
        kron((X'X)^{-1}, cov_resid)

        Notes
        -----
        Covariance of vec(B), where B is the matrix
        [trend coefficient(s), A_1, ..., A_p] (neqs x (neqs*k_ar + k_trend))
        Adjusted to be an unbiased estimator


        See VARModel or the VAR documentation for the definition of the A
        matrices.
        """
        #TODO: This should inherit from a systems of equations model
        #      and this should be in the super class
        X = self.X
        cov = np.kron(L.inv(np.dot(X.T, X)), self.cov_resid)
        if r_matrix is not None:
            return np.dot(np.dot(r_matrix, cov), r_matrix)
        else:
            return cov

    def cov_ybar(self):
        r"""
        Asymptotically consistent estimate of covariance of the sample mean

        .. math::

            \sqrt(T)(\bar{y} - \mu)\rightarrow{\cal N}(0, \Sigma_{\bar{y}})\\

            \Sigma_{\bar{y}}=B cov_resid B^\prime, \text{where } B = (I_K - A_1
            - \cdots - A_p)^{-1}
        """
        #Lutkepohl Proposition 3.3

        Ainv = L.inv(np.eye(self.neqs) - self.coefs.sum(0))
        return chain_dot(Ainv, self.cov_resid, Ainv.T)

#------------------------------------------------------------
# Estimation-related things

    @cache_readonly
    def _XTX(self):
        # X'X
        return np.dot(self.X.T, self.X)

    @property
    def _cov_params_ex_trend(self):
        """
        Estimated covariance matrix of model coefficients ex intercept
        """
        i = self.neqs*self.k_trend
        # drop trend variables
        return self.cov_params()[i:,i:]

    @cache_readonly
    def _cov_cov_resid(self):
        """
        Estimated covariance matrix of vech(cov_resid)
        """
        D_K = tsa.duplication_matrix(self.neqs)
        D_Kinv = npl.pinv(D_K)

        sigxsig = np.kron(self.cov_resid, self.cov_resid)
        return 2 * chain_dot(D_Kinv, sigxsig, D_Kinv.T)

    @cache_readonly
    def llf(self):
        "VAR(p) log-likelihood"
        return var_loglike(self.resid, self.cov_resid_mle, self.nobs)

    @cache_readonly
    def stderr(self):
        "Standard errors of coefficients"
        stderr = np.sqrt(np.diag(self.cov_params()))
        return stderr.reshape((self.df_model, self.neqs), order='C')

    bse = stderr  # statsmodels interface?

    def conf_int(self, alpha=.05, cols=None):
        confint = super(VARResults, self).conf_int(alpha=alpha,
                                                            cols=cols)
        return confint.transpose(2,0,1)

    @cache_readonly
    def tvalues(self):
        "t statistics of coefficients"
        return self.params / self.stderr

    @cache_readonly
    def pvalues(self):
        """Two-sided p-values for model coefficients

        Assumed to be distributed t(df_resid)
        """
        return stats.t.sf(np.abs(self.tvalues), self.df_resid)*2

    def plot_forecast(self, steps, alpha=0.05, plot_stderr=True, ax=None):
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
        ax : matplotlib.axes, optional
            An existing matplotlib.axes instance. Should be a list of axes of
            length `neqs`

        Returns
        -------
        fig : `matplotlib.figure`
            The figure that contains the axes
        """
        mid, lower, upper = self.forecast_interval(None, steps, alpha=alpha)
        plotting.plot_var_forc(self.Y, mid, lower, upper,
                               names=self.model.endog_names,
                               plot_stderr=plot_stderr)

    # Forecast error covariance functions

    def forecast_cov(self, steps=1):
        r"""Compute forecast covariance matrices for desired number of steps

        Parameters
        ----------
        steps : int

        Notes
        -----
        .. math:: \Sigma_{\hat y}(h) = \Sigma_y(h) + \Omega(h) / T

        Ref: Lutkepohl pp. 96-97

        Returns
        -------
        covs : ndarray (steps x k x k)
        """
        mse = self.mse(steps)
        omegas = self._omega_forc_cov(steps)
        return mse + omegas / self.nobs

    #Monte Carlo irf standard errors
    def irf_errband_mc(self, orth=False, repl=1000, T=10,
                       signif=0.05, seed=None, burn=100, cum=False):
        """
        Compute Monte Carlo integrated error bands assuming normally
        distributed for impulse response functions

        Parameters
        ----------
        orth: bool, default False
            Compute orthoganalized impulse response error bands
        repl: int
            number of Monte Carlo replications to perform
        T: int, default 10
            number of impulse response periods
        signif: float (0 < signif <1)
            Significance level for error bars, defaults to 95% CI
        seed: int
            np.random.seed for replications
        burn: int
            number of initial observations to discard for simulation
        cum: bool, default False
            produce cumulative irf error bands

        Notes
        -----
        Lutkepohl (2005) Appendix D

        Returns
        -------
        Tuple of lower and upper arrays of ma_rep monte carlo standard errors

        """
        neqs = self.neqs
        mean = self.mean()
        k_ar = self.k_ar
        coefs = self.coefs
        cov_resid = self.cov_resid
        trend_coefs = self.trend_coefs
        df_model = self.df_model
        nobs = self.nobs

        ma_coll = np.zeros((repl, T+1, neqs, neqs))

        if (orth == True and cum == True):
            fill_coll = lambda sim : VAR(sim).fit(maxlags=k_ar).\
                              orth_ma_rep(maxn=T).cumsum(axis=0)
        elif (orth == True and cum == False):
            fill_coll = lambda sim : VAR(sim).fit(maxlags=k_ar).\
                              orth_ma_rep(maxn=T)
        elif (orth == False and cum == True):
            fill_coll = lambda sim : VAR(sim).fit(maxlags=k_ar).\
                              ma_rep(maxn=T).cumsum(axis=0)
        elif (orth == False and cum == False):
            fill_coll = lambda sim : VAR(sim).fit(maxlags=k_ar).\
                              ma_rep(maxn=T)

        for i in range(repl):
            #discard first hundred to eliminate correct for starting bias
            sim = util.varsim(coefs, trend_coefs, cov_resid,
                              steps=nobs+burn)
            sim = sim[burn:]
            ma_coll[i,:,:,:] = fill_coll(sim)

        ma_sort = np.sort(ma_coll, axis=0) #sort to get quantiles
        index = round(signif/2*repl)-1,round((1-signif/2)*repl)-1
        lower = ma_sort[index[0],:, :, :]
        upper = ma_sort[index[1],:, :, :]
        return lower, upper

    def irf_resim(self, orth=False, repl=1000, T=10,
                      seed=None, burn=100, cum=False):

        """
        Simulates impulse response function, returning an array of simulations.
        Used for Sims-Zha error band calculation.

        Parameters
        ----------
        orth: bool, default False
            Compute orthoganalized impulse response error bands
        repl: int
            number of Monte Carlo replications to perform
        T: int, default 10
            number of impulse response periods
        signif: float (0 < signif <1)
            Significance level for error bars, defaults to 95% CI
        seed: int
            np.random.seed for replications
        burn: int
            number of initial observations to discard for simulation
        cum: bool, default False
            produce cumulative irf error bands

        Notes
        -----
        Sims, Christoper A., and Tao Zha. 1999. "Error Bands for Impulse Response." Econometrica 67: 1113-1155.

        Returns
        -------
        Array of simulated impulse response functions

        """
        #TODO: notes -> references
        neqs = self.neqs
        mean = self.mean()
        k_ar = self.k_ar
        coefs = self.coefs
        cov_resid = self.cov_resid
        trend_coefs = self.trend_coefs
        df_model = self.df_model
        nobs = self.nobs
        if seed is not None:
            np.random.seed(seed=seed)

        ma_coll = np.zeros((repl, T+1, neqs, neqs))

        if (orth == True and cum == True):
            fill_coll = lambda sim : VAR(sim).fit(maxlags=k_ar).\
                              orth_ma_rep(maxn=T).cumsum(axis=0)
        elif (orth == True and cum == False):
            fill_coll = lambda sim : VAR(sim).fit(maxlags=k_ar).\
                              orth_ma_rep(maxn=T)
        elif (orth == False and cum == True):
            fill_coll = lambda sim : VAR(sim).fit(maxlags=k_ar).\
                              ma_rep(maxn=T).cumsum(axis=0)
        elif (orth == False and cum == False):
            fill_coll = lambda sim : VAR(sim).fit(maxlags=k_ar).\
                              ma_rep(maxn=T)

        for i in range(repl):
            #discard first hundred to eliminate correct for starting bias
            sim = util.varsim(coefs, trend_coefs, cov_resid,
                              steps=nobs+burn)
            sim = sim[burn:]
            ma_coll[i,:,:,:] = fill_coll(sim)

        return ma_coll


    def _omega_forc_cov(self, steps):
        # Approximate MSE matrix \Omega(h) as defined in Lut p97
        G = self._XTX
        Ginv = L.inv(G)

        # memoize powers of B for speedup
        # TODO: see if can memoize better
        B = self._bmat_forc_cov()
        _B = {}
        def bpow(i):
            if i not in _B:
                _B[i] = np.linalg.matrix_power(B, i)

            return _B[i]

        phis = self.ma_rep(steps)
        sig_u = self.cov_resid

        omegas = np.zeros((steps, self.neqs, self.neqs))
        for h in range(1, steps + 1):
            if h == 1:
                omegas[h-1] = self.df_model * self.cov_resid
                continue

            om = omegas[h-1]
            for i in range(h):
                for j in range(h):
                    Bi = bpow(h - 1 - i)
                    Bj = bpow(h - 1 - j)
                    mult = np.trace(chain_dot(Bi.T, Ginv, Bj, G))
                    om += mult * chain_dot(phis[i], sig_u, phis[j].T)
            omegas[h-1] = om

        return omegas

    def _bmat_forc_cov(self):
        # B as defined on p. 96 of Lut
        upper = np.zeros((1, self.df_model))
        upper[0,0] = 1

        lower_dim = self.neqs * (self.k_ar - 1)
        I = np.eye(lower_dim)
        lower = np.column_stack((np.zeros((lower_dim, 1)), I,
                                 np.zeros((lower_dim, self.neqs))))

        return np.vstack((upper, self.params.T, lower))

    def summary(self):
        """Compute console output summary of estimates

        Returns
        -------
        summary : VARSummary
            A `statsmodels.tsa.vector_ar.output.VARSummary` class.
        """
        return VARSummary(self)

    #TODO: deprecate and rename to get_irf since it returns a class?
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
            #TODO: I think we can do this now but refactor _reorder first
            raise NotImplementedError("alternate variable order not "
                                      "implemented (yet)")

        return IRAnalysis(self, P=var_decomp, periods=periods)

    #TODO: deprecate and rename to get_fevd since it returns a class?
    def fevd(self, periods=10, var_decomp=None):
        """
        Compute forecast error variance decomposition

        Parameters
        ----------
        periods : int
            The number of periods for which to give the FEVD.
        var_decomp : ndarray (neqs x neqs), lower triangular
            Must satisfy `cov_resid` = P P', where P is the passed matrix.
            If P is None, defaults to Cholesky decomposition of `cov_resid`.

        Returns
        -------
        fevd : FEVD instance
            A `statsmodels.tsa.vector_ar.var_model.FEVD` instance.
        """
        return FEVD(self, P=var_decomp, periods=periods)

    def reorder(self, order):
        """
        Reorder variables for structural specification

        Parameters
        ----------
        order : list
            A list of integers or names. A fitted VAR model with the order
            given is returned. This is more efficient than re-fitting the
            model.
        """
        if len(order) != len(self.params[0,:]):
            raise ValueError("Reorder specification length should match "
                             "number of endogenous variables")
       #This convert order to list of integers if given as strings
        if isinstance(order[0], string_types):
            order_new = []
            for i, nam in enumerate(order):
                order_new.append(self.model.endog_names.index(order[i]))
            order = order_new
            self.model.data.xnames = None #TODO: remove when deprecated fixed
        return _reordered(self, order)

#-------------------------------------------------------------------------------
# VAR Diagnostics: Granger-causality, whiteness of residuals, normality, etc.

    def test_causality_all(self, kind='F', signif=0.05):
        """
        Returns a DataFrame with tests for all equations and variables.

        Parameters
        ----------
        kind : str {'F', 'Wald'}
            Perform F-test or Wald (Chi-sq) test
        signif : float, default 5%
            Significance level for computing critical values for test,
            defaulting to standard 0.95 level

        Returns
        -------
        tbl : DataFrame
            A hierarchical index DataFrame with tests for each equation
            for each variable.

        Notes
        -----
        If an F-test is requested, then the degrees of freedom given in the
        results table will be the denominator degrees of freedom. The
        """
        kind = kind.lower()
        if kind == 'f':
            columns = ['F', 'df1', 'df2', 'prob(>F)']
        elif kind == 'wald':
            columns = ['chi2', 'df', 'prob(>chi2)']
        else:
            raise ValueError("kind %s not understood" % kind)
        from pandas import DataFrame, MultiIndex
        table = DataFrame(np.zeros((9,len(columns))), columns=columns)
        index = []
        variables = self.model.endog_names
        i = 0
        for vari in variables:
            others = []
            for j, ex_vari in enumerate(variables):
                if vari == ex_vari: # don't want to test this
                    continue
                others.append(ex_vari)
                res = self.test_causality(vari, ex_vari, kind=kind,
                                          verbose=False)
                if kind == 'f':
                    row = (res['statistic'],) + res['df'] + (res['pvalue'],)
                else:
                    row = (res['statistic'], res['df'], res['pvalue'])
                table.ix[[i], columns] = row
                i += 1
                index.append([vari, ex_vari])
            res = self.test_causality(vari, others, kind=kind, verbose=False)
            if kind == 'f':
                row = (res['statistic'],) + res['df'] + (res['pvalue'],)
            else:
                row = (res['statistic'], res['df'], res['pvalue'])
            table.ix[[i], columns] = row
            index.append([vari, 'ALL'])
            i += 1
        table.index = MultiIndex.from_tuples(index, names=['Equation',
                                                           'Excluded'])

        return table

    def test_causality(self, equation, variables, kind='F', signif=0.05,
                       verbose=True):
        """Test for Granger causality

        Parameters
        ----------
        equation : string or int
            Equation to test for causality
        variables : sequence (of strings or ints)
            List, tuple, etc. of variables to test for Granger-causality
        kind : str {'F', 'wald'}
            Perform F-test or Wald (chi-sq) test
        signif : float, default 5%
            Significance level for computing critical values for test,
            defaulting to standard 0.95 level

        Notes
        -----
        Null hypothesis is that there is no Granger-causality for the indicated
        variables. The degrees of freedom in the F-test are based on the
        number of variables in the VAR system, that is, degrees of freedom
        are equal to the number of equations in the VAR times degree of freedom
        of a single equation.

        Returns
        -------
        results : dict
            The results of the Granger Causality test

        See Also
        --------
        `statsmodels.tsa.vector_ar.var_model.VARResults.test_causality_all`
        """
        if isinstance(variables, (string_types, int, np.integer)):
            variables = [variables]

        k, p = self.neqs, self.k_ar

        # number of restrictions
        N = len(variables) * self.k_ar

        # Make restriction matrix
        C = np.zeros((N, k ** 2 * p + k), dtype=float)

        eq_index = self.get_eq_index(equation)
        vinds = mat([self.get_eq_index(v) for v in variables])

        # remember, vec is column order!
        offsets = np.concatenate([k + k ** 2 * j + k * vinds + eq_index
                                  for j in range(p)])
        C[np.arange(N), offsets] = 1

        # Lutkepohl 3.6.5
        Cb = np.dot(C, vec(self.params.T))
        middle = L.inv(chain_dot(C, self.cov_params(), C.T))

        # wald statistic
        lam_wald = statistic = chain_dot(Cb, middle, Cb)

        if kind.lower() == 'wald':
            df = N
            dist = stats.chi2(df)
        elif kind.lower() == 'f':
            statistic = lam_wald / N
            df = (N, k * self.df_resid)
            dist = stats.f(*df)
        else:
            raise Exception('kind %s not recognized' % kind)

        pvalue = dist.sf(statistic)
        crit_value = dist.ppf(1 - signif)

        conclusion = 'fail to reject' if statistic < crit_value else 'reject'
        results = {
            'statistic' : statistic,
            'crit_value' : crit_value,
            'pvalue' : pvalue,
            'df' : df,
            'conclusion' : conclusion,
            'signif' :  signif
        }

        if verbose:
            summ = output.causality_summary(results, variables, equation, kind)

            print(summ)

        return results

    def test_whiteness(self, nlags=10, plot=True, linewidth=8):
        """
        Test white noise assumption. Sample (Y) autocorrelations are compared
        with the standard :math:`2 / \sqrt(T)` bounds.

        Parameters
        ----------
        nlags : int
            The number of lags for the autocorrelations.
        plot : boolean, default True
            Plot autocorrelations with 2 / sqrt(T) bounds
        linewidth : int
            The linewidth for the plots.
        """
        acorrs = self.sample_acorr(nlags)
        bound = 2 / np.sqrt(self.nobs)

        # TODO: this probably needs some UI work

        if (np.abs(acorrs) > bound).any():
            print('FAIL: Some autocorrelations exceed %.4f bound. '
                   'See plot' % bound)
        else:
            print('PASS: No autocorrelations exceed %.4f bound' % bound)

        if plot:
            fig = plotting.plot_full_acorr(acorrs[1:],
                                           xlabel=np.arange(1, nlags+1),
                                           err_bound=bound,
                                           linewidth=linewidth)
            fig.suptitle(r"ACF plots with $2 / \sqrt{T}$ bounds "
                         "for testing whiteness assumption")

    def test_normality(self, signif=0.05, verbose=True):
        """
        Test assumption of normal-distributed errors using Jarque-Bera-style
        omnibus Chi^2 test

        Parameters
        ----------
        signif : float
            Test significance threshold
        verbose : bool
            If True, prints the summary.

        Returns
        -------
        results : dict
            A dictionary that holds the test results.

        Notes
        -----
        H0 (null) : data are generated by a Gaussian-distributed process
        """
        #TODO: remove print statement?
        Pinv = npl.inv(self._chol_cov_resid)

        w = np.array([np.dot(Pinv, u) for u in self.resid])

        b1 = (w ** 3).sum(0) / self.nobs
        lam_skew = self.nobs * np.dot(b1, b1) / 6

        b2 = (w ** 4).sum(0) / self.nobs - 3
        lam_kurt = self.nobs * np.dot(b2, b2) / 24

        lam_omni = lam_skew + lam_kurt

        omni_dist = stats.chi2(self.neqs * 2)
        omni_pvalue = omni_dist.sf(lam_omni)
        crit_omni = omni_dist.ppf(1 - signif)

        conclusion = 'fail to reject' if lam_omni < crit_omni else 'reject'

        results = {
            'statistic' : lam_omni,
            'crit_value' : crit_omni,
            'pvalue' : omni_pvalue,
            'df' : self.neqs * 2,
            'conclusion' : conclusion,
            'signif' :  signif,
            'null' : "Data are generated by a Gaussian-distributed process."
        }

        if verbose:
            summ = output.normality_summary(results)
            print(summ)

        return results

    @cache_readonly
    def detomega(self):
        """
        detomega is deprecated. Use `det_cov_resid`
        """
        warn("detomega is deprecated and will be removed in 0.6.0. Use "
             "det_cov_resid.", FutureWarning)
        return self.det_cov_resid

    @cache_readonly
    def det_cov_resid(self):
        r"""
        Returns determinant of the cov_resid with degrees of freedom correction

        .. math::

           \hat \Omega = \frac{T}{T - Kp - 1} \hat \Omega_{\mathrm{MLE}}
        """
        #TODO: Fix the math
        return L.det(self.cov_resid)

    @cache_readonly
    def info_criteria(self):
        "Dictionary containing information criteria of model"
        nobs = self.nobs
        neqs = self.neqs
        k_ar = self.k_ar
        free_params = k_ar * neqs ** 2 + neqs * self.k_trend

        ld = util.get_logdet(self.cov_resid_mle)

        # See Lutkepohl pp. 146-150

        aic = ld + (2. / nobs) * free_params
        bic = ld + (np.log(nobs) / nobs) * free_params
        hqic = ld + (2. * np.log(np.log(nobs)) / nobs) * free_params
        fpe = ((nobs + self.df_model) / self.df_resid) ** neqs * np.exp(ld)

        return {
            'aic' : aic,
            'bic' : bic,
            'hqic' : hqic,
            'fpe' : fpe
            }

    @property
    def aic(self):
        """
        Akaike information criterion

        ln(det(cov_resid_mle)) + (2 / nobs) * k_ar * k_trend * neqs ** 2

        Notes
        -----
        Uses the definition from Lutkepohl.
        """
        return self.info_criteria['aic']

    @property
    def fpe(self):
        """Final Prediction Error (FPE)

        det(cov_resid_mle) * ((nobs + df_model)/df_resid)**neqs

        Notes
        -----
        Uses the definition from Lutkepohl.
        """
        return self.info_criteria['fpe']

    @property
    def hqic(self):
        """
        Hannan-Quinn criterion

        ln(det(cov_resid_mle)) + 2*ln(ln(nobs))/nobs * k_ar * k_trend * neqs**2

        Notes
        -----
        Uses the definition from Lutkepohl.
        """
        return self.info_criteria['hqic']

    @property
    def bic(self):
        """
        Bayesian a.k.a. Schwarz information criterion

        ln(det(cov_resid_mle)) + ln(nobs)/nobs * k_ar * k_trend * neqs ** 2

        Notes
        -----
        Uses the definition from Lutkepohl.
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
        arr = np.zeros((p,p))
        arr[:neqs,:] = np.column_stack(self.coefs)
        arr[neqs:,:-neqs] = np.eye(p-neqs)
        roots = np.linalg.eig(arr)[0]**-1
        idx = np.argsort(np.abs(roots))[::-1] # sort by reverse modulus
        return roots[idx]

    def t_test(*args, **kwargs):
        raise NotImplementedError

    def f_test(*args, **kwargs):
        raise NotImplementedError

class VARResultsWrapper(wrap.ResultsWrapper):
    _attrs = {'bse' : 'columns_eq', 'params' : 'columns_eq',
              'pvalues' : 'columns_eq', 'tvalues' : 'columns_eq',
              'cov_resid' : 'cov_eq', 'cov_resid_mle' : 'cov_eq',
              'stderr' : 'columns_eq'}
    _wrap_attrs = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._wrap_attrs,
                                    _attrs)
    _methods = {'conf_int' : 'columns_eq', 'cov_params' : 'cov2d'}
    _wrap_methods = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._wrap_methods,
                                     _methods)
wrap.populate_wrapper(VARResultsWrapper, VARResults)

class FEVD(object):
    """
    Compute and plot Forecast error variance decomposition and asymptotic
    standard errors
    """
    def __init__(self, results, P=None, periods=None):
        self.periods = periods

        self.results = results
        self.neqs = results.neqs
        self.endog_names = results.model.endog_names

        self.irfobj = results.irf(var_decomp=P, periods=periods)
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

    #TODO: Docs
    def plot(self, periods=None, figsize=(10,10), **plot_kwds):
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
        fig : matplotlib.figure
            The matplotlib figure instance that contains the axes.
        """
        k = self.neqs
        fig, axes = plotting._create_mpl_subplots(None, k, 1, figsize=figsize)

        periods = periods or self.periods

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
                                color=colors[j],
                                label=self.endog_names[j],
                                **plot_kwds)

                handles.append(handle)

            ax.set_title(self.endog_names[i])

        # just use the last axis to get handles for plotting
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        plotting.adjust_subplots(fig, right=0.85)
        return fig

#-------------------------------------------------------------------------------

#TODO: These should be made available everywhere.
def _compute_acov(x, nlags=1):
    x = x - x.mean(0)

    result = []
    for lag in range(nlags + 1):
        if lag > 0:
            r = np.dot(x[lag:].T, x[:-lag])
        else:
            r = np.dot(x.T, x)

        result.append(r)

    return np.array(result) / len(x)

def _acovs_to_acorrs(acovs):
    sd = np.sqrt(np.diag(acovs[0]))
    return acovs / np.outer(sd, sd)

if __name__ == '__main__':
    import statsmodels.api as sm
    from statsmodels.tsa.vector_ar.util import parse_lutkepohl_data

    np.set_printoptions(linewidth=140, precision=5)

    sdata, dates = parse_lutkepohl_data('data/%s.dat' % 'e1')

    names = sdata.dtype.names
    data = data_util.struct_to_ndarray(sdata)
    adj_data = np.diff(np.log(data), axis=0)
    # est = VAR(adj_data, p=2, dates=dates[1:], names=names)
    model = VAR(adj_data[:-16], dates=dates[1:-16], names=names)
    # model = VAR(adj_data[:-16], dates=dates[1:-16], names=names)

    est = model.fit(maxlags=2)
    irf = est.irf()

    y = est.y[-2:]
    """
    # irf.plot_irf()

    # i = 2; j = 1
    # cv = irf.cum_effect_cov(orth=True)
    # print np.sqrt(cv[:, j * 3 + i, j * 3 + i]) / 1e-2

    # data = np.genfromtxt('Canada.csv', delimiter=',', names=True)
    # data = data.view((float, 4))
    """

    '''
    mdata = sm.datasets.macrodata.load().data
    mdata2 = mdata[['realgdp','realcons','realinv']]
    names = mdata2.dtype.names
    data = mdata2.view((float,3))
    data = np.diff(np.log(data), axis=0)

    import pandas as pn
    df = pn.DataFrame.fromRecords(mdata)
    df = np.log(df.reindex(columns=names))
    df = (df - df.shift(1)).dropna()

    model = VAR(df)
    est = model.fit(maxlags=2)
    irf = est.irf()
    '''

