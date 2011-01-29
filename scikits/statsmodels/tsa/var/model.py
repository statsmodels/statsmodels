"""
Vector Autoregression (VAR) processes

References
----------
Lutkepohl (2005) New Introduction to Multiple Time Series Analysis
"""

from __future__ import division

from cStringIO import StringIO

import numpy as np
import numpy.linalg as npl
from numpy.linalg import cholesky as chol, solve
import scipy.stats as stats
import scipy.linalg as L

import matplotlib.pyplot as plt
import matplotlib as mpl

from scikits.statsmodels.decorators import cache_readonly
from scikits.statsmodels.tools import chain_dot
from scikits.statsmodels.iolib import SimpleTable
from scikits.statsmodels.tsa.tsatools import vec, unvec
import scikits.statsmodels.tsa.tsatools as tsa

from .util import pprint_matrix

mat = np.array

try:
    import pandas.util.testing as test
    import IPython.core.debugger as _
    st = test.set_trace
except ImportError:
    import pdb
    st = pdb.set_trace

class MPLConfigurator(object):

    def __init__(self):
        self._inverse_actions = []

    def revert(self):
        for action in self._inverse_actions:
            action()

    def set_fontsize(self, size):
        old_size = mpl.rcParams['font.size']
        mpl.rcParams['font.size'] = size

        def revert():
            mpl.rcParams['font.size'] = old_size

        self._inverse_actions.append(revert)

#-------------------------------------------------------------------------------
# Utilities

def norm_signif_level(alpha=0.05):
    return stats.norm.ppf(1 - alpha / 2)

def get_logdet(m):
    from scikits.statsmodels.compatibility import np_slogdet
    logdet = np_slogdet(m)

    if logdet[0] == -1:
        raise ValueError("Omega matrix is not positive definite")
    elif logdet[0] == 0:
        raise ValueError("Omega matrix is singluar")
    else:
        logdet = logdet[1]

    return logdet

def _interpret_data(data, names):
    """
    Convert passed data structure to form required by VAR estimation classes

    Parameters
    ----------

    Returns
    -------

    """
    if isinstance(data, np.ndarray):
        provided_names = data.dtype.names

        # structured array type
        if provided_names:
            if names is None:
                names = provided_names
            else:
                assert(len(names) == len(provided_names))

            Y = _struct_to_ndarray(data)
        else:
            Y = data
    else:
        raise Exception('cannot handle other input types at the moment')

    return Y, names

#-------------------------------------------------------------------------------
# VAR process routines

def varsim(coefs, intercept, sig_u, steps=100, initvalues=None):
    """
    Simulate simple VAR(p) process with known coefficients, intercept, white
    noise covariance, etc.
    """
    from numpy.random import multivariate_normal as rmvnorm
    p, k, k = coefs.shape
    ugen = rmvnorm(np.zeros(len(sig_u)), sig_u, steps)
    result = np.zeros((steps, k))
    result[p:] = intercept + ugen[p:]

    # add in AR terms
    for t in xrange(p, steps):
        ygen = result[t]
        for j in xrange(p):
            ygen += np.dot(coefs[j], result[t-j-1])

    return result

def ma_rep(coefs, intercept, maxn=10):
    """
    MA(\infty) representation of VAR(p) process

    Parameters
    ----------
    coefs : ndarray (p x k x k)
    intercept : ndarry length-k
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
    """
    p, k, k = coefs.shape
    phis = np.zeros((maxn+1, k, k))
    phis[0] = np.eye(k)

    # recursively compute Phi matrices
    for i in xrange(1, maxn + 1):
        phi = phis[i]
        for j in xrange(1, i+1):
            if j > p:
                break

            phi += np.dot(phis[i-j], coefs[j-1])

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
    A_var1 = var1_rep(coefs)
    eigs = np.linalg.eigvals(A_var1)

    if verbose:
        print 'Eigenvalues of VAR(1) rep'
        for val in np.abs(eigs):
            print val

    return (np.abs(eigs) <= 1).all()

def var1_rep(coefs):
    """
    Return coefficient matrix for the VAR(1) representation for a VAR(p) process
    (companion form)

    A = [A_1 A_2 ... A_p-1 A_p
         I_K 0       0     0
         0   I_K ... 0     0
         0 ...       I_K   0]
    """
    p, k, k2 = coefs.shape
    assert(k == k2)

    kp = k * p

    result = np.zeros((kp, kp))
    result[:k] = np.concatenate(coefs, axis=1)

    # Set I_K matrices
    if p > 1:
        result[np.arange(k, kp), np.arange(kp-k)] = 1

    return result

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
    p, k, k2 = coefs.shape
    if nlags is None:
        nlags = p

    # p x k x k, ACF for lags 0, ..., p-1
    result = np.zeros((nlags + 1, k, k))
    result[:p] = _var_acf(coefs, sig_u)

    # yule-walker equations
    for h in xrange(p, nlags + 1):
        # compute ACF for lag=h
        # G(h) = A_1 G(h-1) + ... + A_p G(h-p)

        res = result[h]
        for j in xrange(p):
            res += np.dot(coefs[j], result[h-j-1])

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

    A = var1_rep(coefs)
    # construct VAR(1) noise covariance
    SigU = np.zeros((k*p, k*p))
    SigU[:k,:k] = sig_u

    # vec(ACF) = (I_(kp)^2 - kron(A, A))^-1 vec(Sigma_U)
    vecACF = L.solve(np.eye((k*p)**2) - np.kron(A, A), vec(SigU))

    acf = unvec(vecACF)
    acf = acf[:k].T.reshape((p, k, k))

    return acf

def _acf_to_acorr(acf):
    diag = np.diag(acf[0])
    # numpy broadcasting sufficient
    return acf / np.sqrt(np.outer(diag, diag))

def forecast(y, coefs, intercept, steps):
    """
    Produce linear MSE forecast

    Parameters
    ----------


    Notes
    -----
    Lutkepohl p. 37
    """
    p = len(coefs)
    k = len(coefs[0])
    # initial value
    forcs = np.zeros((steps, k)) + intercept

    # h=0 forecast should be latest observation
    # forcs[0] = y[-1]

    # make indices easier to think about
    for h in xrange(1, steps + 1):
        f = forcs[h - 1]

        # y_t(h) = intercept + sum_1^p A_i y_t_(h-i)

        for i in xrange(1, p + 1):
            # slightly hackish
            if h - i <= 0:
                # e.g. when h=1, h-1 = 0, which is y[-1]
                prior_y = y[h - i - 1]
            else:
                # e.g. when h=2, h-1=1, which is forcs[0]
                prior_y = forcs[h - i - 1]

            # i=1 is coefs[0]
            f += np.dot(coefs[i - 1], prior_y)

    return forcs

def forecast_cov(ma_coefs, sig_u, steps):
    """
    Compute forecast error variance matrices
    """
    k = len(sig_u)
    forc_covs = np.zeros((steps, k, k))

    prior = np.zeros((k, k))
    for h in xrange(steps):
        # Sigma(h) = Sigma(h-1) + Phi Sig_u Phi'
        phi = ma_coefs[h]
        var = chain_dot(phi, sig_u, phi.T)
        forc_covs[h] = prior = prior + var

    return forc_covs

def granger_causes(coefs):
    pass

#-------------------------------------------------------------------------------
# Plotting functions

def plot_mts(Y, names=None, index=None):
    """
    Plot multiple time series
    """

    k = Y.shape[1]
    rows, cols = k, 1

    plt.figure(figsize=(10, 10))

    for j in range(k):
        ts = Y[:, j]

        ax = plt.subplot(rows, cols, j+1)
        if index is not None:
            ax.plot(index, ts)
        else:
            ax.plot(ts)

        if names is not None:
            ax.set_title(names[j])

def plot_var_forc(prior, forc, err_upper, err_lower,
                  index=None, names=None):
    n, k = prior.shape
    rows, cols = k, 1

    fig = plt.figure(figsize=(10, 10))

    prange = np.arange(n)
    rng_f = np.arange(n - 1, n + len(forc))
    rng_err = np.arange(n, n + len(forc))

    for j in range(k):
        ax = plt.subplot(rows, cols, j+1)

        p1 = ax.plot(prange, prior[:, j], 'k')
        p2 = ax.plot(rng_f, np.r_[prior[-1:, j], forc[:, j]], 'k--')
        p3 = ax.plot(rng_err, err_upper[:, j], 'k-.')
        ax.plot(rng_err, err_lower[:, j], 'k-.')

        if names is not None:
            ax.set_title(names[j])

    fig.legend((p1, p2, p3), ('Observed', 'Forecast', 'Forc 2 STD err'),
               'upper right')

def plot_with_error(y, error, x=None, axes=None, value_fmt='k',
                    error_fmt='k--', alpha=0.05):
    """
    Make plot with optional error bars

    Parameters
    ----------
    y :
    error : array or None

    """
    if axes is None:
        axes = plt.gca()

    if x is not None:
        plot_action = lambda y, fmt: axes.plot(x, y, fmt)
    else:
        plot_action = lambda y, fmt: axes.plot(y, fmt)

    plot_action(y, value_fmt)

    if error is not None:
        q = norm_signif_level(alpha)
        plot_action(y - q * error, error_fmt)
        plot_action(y + q * error, error_fmt)

def plot_acorr(acf, fontsize=8, linewidth=8):
    """

    Parameters
    ----------



    """
    config = MPLConfigurator()
    config.set_fontsize(fontsize)

    lags, k, k = acf.shape
    acorr = _acf_to_acorr(acf)
    plt.figure(figsize=(10, 10))
    xs = np.arange(lags)

    for i in range(k):
        for j in range(k):
            ax = plt.subplot(k, k, i * k + j + 1)
            ax.vlines(xs, [0], acorr[:, i, j], lw=linewidth)

            ax.axhline(0, color='k')
            ax.set_ylim([-1, 1])

            # hack?
            ax.set_xlim([-1, xs[-1] + 1])

    _adjust_subplots()
    config.revert()

def plot_acorr_with_error():
    pass

def _adjust_subplots(**kwds):
    passed_kwds = dict(bottom=0.05, top=0.925,
                       left=0.05, right=0.95,
                       hspace=0.2)
    passed_kwds.update(kwds)
    plt.subplots_adjust(**passed_kwds)

#-------------------------------------------------------------------------------
# VARProcess class: for known or unknown VAR process

def _struct_to_ndarray(arr):
    return arr.view((float, len(arr.dtype.names)))

def lag_select():
    pass

import unittest
class TestVAR(unittest.TestCase):
    pass

class VAR(object):
    """
    Fit VAR(p) process and do lag order selection

    .. math:: y_t = A_1 y_{t-1} + \ldots + A_p y_{t-p} + u_t

    Notes
    -----
    **References**
    Lutkepohl (2005) New Introduction to Multiple Time Series Analysis

    Returns
    -------
    .fit() method returns VAREstimator object
    """

    def __init__(self, endog, names=None, dates=None):
        self.y, self.names = _interpret_data(endog, names)
        self.nobs, self.neqs = self.y.shape

        if dates is not None:
            assert(self.nobs == len(dates))
        self.dates = dates

    def loglike(self, params, omega):
        pass

    def fit(self, maxlags=None, method='ols', ic=None, verbose=False):
        """
        Fit the VAR model

        Parameters
        ----------
        maxlags : int
            Maximum number of lags to check for order selection, defaults to
            12 * (nobs/100.)**(1./4), see select_order function
        method : {'ols'}
            Estimation method to use
        ic : {'aic', 'fpe', 'hq', 'sic', None}
            Information criterion to use for VAR order selection.
            aic : Akaike
            fpe : Final prediction error
            hq : Hannan-Quinn
            sic : Schwarz
        verbose : bool, default False
            Print order selection output to the screen

        Notes
        -----
        Lutkepohl pp. 146-153

        Returns
        -------
        est : VAREstimator
        """
        lags = maxlags

        if ic is not None:
            selections = self.select_order(maxlags=maxlags, verbose=verbose)
            if ic not in selections:
                raise Exception("%s not recognized, must be among %s"
                                % (ic, sorted(selections)))
            lags = selections[ic]
            if verbose:
                print 'Using %d based on %s criterion' %  (lags, ic)

        return VAREstimator(self.y, p=lags, names=self.names, dates=self.dates)

    def select_order(self, maxlags=None, verbose=True):
        """

        Parameters
        ----------

        Returns
        -------
        """
        from collections import defaultdict

        if maxlags is None:
            maxlags = int(round(12*(self.nobs/100.)**(1/4.)))

        ics = defaultdict(list)
        for p in range(maxlags + 1):
            est = VAREstimator(self.y, p=p)
            for k, v in est.info_criteria.iteritems():
                ics[k].append(v)

        selected_orders = dict((k, mat(v).argmin())
                               for k, v in ics.iteritems())

        if verbose:
            self._print_ic_table(ics, selected_orders)

        return selected_orders

    @staticmethod
    def _print_ic_table(ics, selected_orders):
        """

        """
        # Can factor this out into a utility method if so desired

        cols = sorted(ics)

        data = mat([["%#10.4F" % v for v in ics[c]] for c in cols],
                   dtype=object).T

        # start minimums
        for i, col in enumerate(cols):
            idx = int(selected_orders[col]), i
            data[idx] = data[idx] + '*'
            # data[idx] = data[idx][:-1] + '*' # super hack, ugh

        fmt = dict(_default_table_fmt,
                   data_fmts=("%s",) * len(cols))

        buf = StringIO()
        table = SimpleTable(data, cols, range(len(data)),
                            title='VAR Order Selection', txt_fmt=fmt)
        print >> buf, table
        print >> buf, '* Minimum'

        print buf.getvalue()

class VARProcess(object):
    """

    Parameters
    ----------

    Returns
    -------

    **Attributes**:

    """
    def __init__(self):
        pass

    def get_eq_index(self, name):
        try:
            result = self.names.index(name)
        except Exception:
            if not isinstance(name, int):
                raise
            result = name

        return result

    def __str__(self):
        output = ('VAR(%d) process for %d-dimensional response y_t'
                  % (self.p, self.k))
        output += '\nstable: %s' % self.is_stable()
        output += '\nmean: %s' % self.mean()

        return output

    def is_stable(self, verbose=False):
        """
        Determine stability based on model coefficients

        Parameters
        ----------
        verbose : bool
            Print eigenvalues of VAR(1) rep matrix

        Notes
        -----

        """
        return is_stable(self.coefs, verbose=verbose)

    def plotsim(self, steps=1000):
        Y = varsim(self.coefs, self.intercept, self.sigma_u, steps=steps)
        plot_mts(Y)

    def plotforc(self, y, steps, alpha=0.05):
        """

        """
        mid, lower, upper = self.forecast_interval(y, steps, alpha=alpha)
        plot_var_forc(y, mid, lower, upper,
                      index=self.dates,
                      names=self.names)

    def mean(self):
        """
        Mean of stable process

        .. math:: \mu = (I - A_1 - \dots - A_p)^{-1} \alpha
        """
        return solve(self._char_mat, self.intercept)

    def ma_rep(self, maxn=10):
        """
        Compute MA(\infty) coefficient matrices (also are impulse response
        matrices))

        Parameters
        ----------
        maxn : int
            Number of coefficient matrices to compute

        Returns
        -------
        coefs : ndarray (maxn x k x k)
        """
        return ma_rep(self.coefs, self.intercept, maxn=maxn)

    def orth_ma_rep(self, maxn=10, P=None):
        """
        Compute Orthogonalized MA coefficient matrices using P matrix such that
        \Sigma_u = PP'. P defaults to the Cholesky decomposition of \Sigma_u

        Parameters
        ----------
        maxn : int
            Number of coefficient matrices to compute
        P : ndarray (k x k), optional
            Matrix such that Sigma_u = PP', defaults to Cholesky descomp

        Returns
        -------
        coefs : ndarray (maxn x k x k)
        """
        if P is None:
            P = self._chol_sigma_u

        ma_mats = self.ma_rep(maxn=maxn)
        return mat([np.dot(coefs, P) for coefs in ma_mats])

    def long_run_effects(self):
        return L.inv(self._char_mat)

    @cache_readonly
    def _chol_sigma_u(self):
        return chol(self.sigma_u)

    @cache_readonly
    def _char_mat(self):
        return np.eye(self.k) - self.coefs.sum(0)

    def acf(self, nlags=None):
        return var_acf(self.coefs, self.sigma_u, nlags=nlags)

    def acorr(self, nlags=None):
        return _acf_to_acorr(var_acf(self.coefs, self.sigma_u, nlags=nlags))

    def plot_acorr(self, nlags=10, linewidth=8):
        plot_acorr(self.acorr(nlags=nlags), linewidth=linewidth)

    def forecast(self, y, steps):
        """
        Produce linear minimum MSE forecasts for desired number of steps ahead,
        using prior values y

        Parameters
        ----------
        y : ndarray (p x k)
        steps : int

        Returns
        -------

        Notes
        -----
        Lutkepohl pp 37-38
        """
        return forecast(y, self.coefs, self.intercept, steps)

    def forecast_cov(self, steps):
        """
        Compute forecast error covariance matrices

        Parameters
        ----------


        Returns
        -------
        covs : (steps, k, k)
        """
        return self.mse(steps)

    def mse(self, steps):
        return forecast_cov(self.ma_rep(steps), self.sigma_u, steps)

    def _forecast_vars(self, steps):
        covs = self.forecast_cov(steps)

        # Take diagonal for each cov
        inds = np.arange(self.k)
        return covs[:, inds, inds]

    def forecast_interval(self, y, steps, alpha=0.05):
        """
        Construct forecast interval estimates assuming the y are Gaussian

        Parameters
        ----------

        Notes
        -----
        Lutkepohl pp. 39-40

        Returns
        -------
        (lower, mid, upper) : (ndarray, ndarray, ndarray)
        """
        assert(0 < alpha < 1)
        q = norm_signif_level(alpha)

        point_forecast = self.forecast(y, steps)
        sigma = np.sqrt(self._forecast_vars(steps))

        forc_lower = point_forecast - q * sigma
        forc_upper = point_forecast + q * sigma

        return point_forecast, forc_lower, forc_upper

    def forecast_dyn(self):
        """
        "Forward filter": compute forecasts for each time step
        """

        pass

    def irf(self, nperiods=50):
        """
        Compute impulse responses to shocks in system
        """
        pass

    def fevd(self):
        """
        Compute forecast error variance decomposition ("fevd")
        """
        pass


#-------------------------------------------------------------------------------
# Known VAR process and Estimator classes

class KnownVARProcess(VARProcess):
    """
    Class for analyzing VAR(p) process with known coefficient matrices and white
    noise covariance matrix

    Parameters
    ----------

    """
    def __init__(self, intercept, coefs, sigma_u):
        self.p = len(coefs)
        self.k = len(coefs[0])

        self.intercept = intercept
        self.coefs = coefs

        self.sigma_u = sigma_u

class VAREstimator(VARProcess):
    """
    Estimate VAR(p) process with fixed number of lags

    Returns
    -------
    **Attributes**
    k : int
        Number of variables (equations)
    p : int
        Order of VAR process
    T : Number of model observations (len(data) - p)
    y : ndarray (K x T)
        Observed data
    names : ndarray (K)
        variables names

    df_model : int
    df_resid : int

    coefs : ndarray (p x K x K)
        Estimated A_i matrices, A_i = coefs[i-1]
    intercept : ndarray (K)
    beta : ndarray (Kp + 1) x K
        A_i matrices and intercept in stacked form [int A_1 ... A_p]

    sigma_u : ndarray (K x K)
        Estimate of white noise process variance Var[u_t]
    """
    _model_type = 'VAR'

    def __init__(self, data, p=1, names=None, dates=None):
        self.p = p
        self.y, self.names = _interpret_data(data, names)
        self.nobs, self.k = self.y.shape

        # This will be the dimension of the Z matrix
        self.T = self.nobs  - self.p

        self.dates = dates

        if dates is not None:
            assert(self.nobs == len(dates))

        self.names = names
        self.dates = dates

    def data_summary(self):
        pass

    def plot(self):
        plot_mts(self.y, names=self.names, index=self.dates)

    @property
    def df_model(self):
        """
        Number of estimated parameters, including the intercept
        """
        return self.k * self.p + 1

    @property
    def df_resid(self):
        return self.T - self.df_model

    def __str__(self):
        output = ('VAR(%d) process for %d-dimensional response y_t'
                  % (self.p, self.k))
        output += '\nstable: %s' % self.is_stable()
        output += '\nmean: %s' % self.mean()

        return output

    def forecast_dyn(self):
        """
        "Forward filter": compute forecasts for each time step
        """

        pass

#-------------------------------------------------------------------------------
# VARProcess interface

    @cache_readonly
    def coefs(self):
        # Each matrix needs to be transposed
        reshaped = self.beta[1:].reshape((self.p, self.k, self.k))

        # Need to transpose each coefficient matrix
        return reshaped.swapaxes(1, 2).copy()

    @property
    def intercept(self):
        return self.beta[0]

    @cache_readonly
    def beta(self):
        return self._est_beta()

    @cache_readonly
    def sigma_u(self):
        return self._est_sigma_u()

    @cache_readonly
    def resid(self):
        # Lutkepohl p75, this is slower
        # middle = np.eye(self.T) - chain_dot(z, self._zzinv, z.T)
        # return chain_dot(y.T, middle, y) / self.df_resid

        # about 5x faster
        return self._y_sample - np.dot(self.Z, self.beta)

#-------------------------------------------------------------------------------
# Auxiliary variables for estimation

    @cache_readonly
    def Z(self):
        """
        Predictor matrix for VAR(p) process

        Z := (Z_0, ..., Z_T).T (T x Kp)
        Z_t = [1 y_t y_{t-1} ... y_{t - p + 1}] (Kp x 1)

        Ref: Lutkepohl p.70 (transposed)
        """
        y = self.y
        p = self.p

        # Ravel C order, need to put in descending order
        Z = mat([y[t-p : t][::-1].ravel() for t in xrange(p, self.nobs)])

        # Add intercept
        return np.concatenate((np.ones((self.T, 1)), Z), axis=1)

    @cache_readonly
    def _zz(self):
        return np.dot(self.Z.T, self.Z)

    @cache_readonly
    def _zzinv(self):
        try:
            zzinv = L.inv(self._zz)
        except np.linalg.LinAlgError:
            zzinv = L.pinv(self._zz)

        return zzinv

    @cache_readonly
    def _y_sample(self):
        # drop presample observations
        return self.y[self.p:]

    #------------------------------------------------------------
    # Coefficient estimation

    def _est_beta(self):
        res = np.linalg.lstsq(self.Z, self._y_sample)
        # coefs = chain_dot(self._zzinv, self.Z.T, self._y_sample)
        return res[0]

    def _est_sigma_u(self):
        """
        Unbiased estimate of covariance matrix $\Sigma_u$ of the white noise
        process $u$

        equivalent definition

        .. math:: \frac{1}{T - Kp - 1} Y^\prime (I_T - Z (Z^\prime Z)^{-1}
        Z^\prime) Y

        Ref: Lutkepohl p.75
        """
        # df_resid right now is T - Kp - 1, which is a suggested correction
        return np.dot(self.resid.T, self.resid) / self.df_resid

    def _est_var_ybar(self):
        Ainv = L.inv(np.eye(self.k) - self.coefs.sum(0))
        return chain_dot(Ainv, self.sigma_u, Ainv.T)

    def _t_mean(self):
        self.mean() / np.sqrt(np.diag(self._est_var_ybar()))

    @property
    def _cov_alpha(self):
        """
        Estimated covariance matrix of model coefficients ex intercept
        """
        # drop intercept
        return self.cov_beta[self.k:, self.k:]

    @cache_readonly
    def _cov_sigma(self):
        """
        Estimated covariance matrix of vech(sigma_u)
        """
        D_K = tsa.duplication_matrix(self.k)
        D_Kinv = npl.pinv(D_K)

        sigxsig = np.kron(self.sigma_u, self.sigma_u)
        return 2 * chain_dot(D_Kinv, sigxsig, D_Kinv.T)

    @cache_readonly
    def cov_beta(self):
        """
        Covariance of vec(B), where B is the matrix

        [intercept, A_1, ..., A_p] (K x (Kp + 1))

        Notes
        -----
        Adjusted to be an unbiased estimator

        Ref: Lutkepohl p.74-75

        Returns
        -------
        cov_beta : ndarray (K^2p + K x K^2p + K) (# parameters)
        """
        return np.kron(L.inv(self._zz), self.sigma_u)

    @cache_readonly
    def stderr(self):
        """
        Standard errors of coefficients, reshaped to match in size
        """
        stderr = np.sqrt(np.diag(self.cov_beta))
        return stderr.reshape((self.df_model, self.k), order='C')

    def t(self):
        """
        Compute t-statistics. Use Student-t(T - Kp - 1) = t(df_resid) to test
        significance.
        """
        return self.beta / self.stderr

    @cache_readonly
    def pvalues(self):
        return stats.t.sf(np.abs(self.t()), self.df_resid)*2

    def forecast_interval(self, steps, alpha=0.05):
        """
        Construct forecast interval estimates assuming the y are Gaussian

        Parameters
        ----------

        Notes
        -----
        Lutkepohl pp. 39-40

        Returns
        -------
        (lower, mid, upper) : (ndarray, ndarray, ndarray)
        """
        return VARProcess.forecast_interval(
            self, self.y[-self.p:], steps, alpha=alpha)

    def plotforc(self, steps, alpha=0.05):
        """

        """
        mid, lower, upper = self.forecast_interval(steps, alpha=alpha)
        plot_var_forc(self.y, mid, lower, upper, names=self.names)

    # Forecast error covariance functions

    def forecast_cov(self, steps=1):
        """
        Compute forecast covariance matrices for desired number of steps

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
        return mse + omegas / self.T

    def _omega_forc_cov(self, steps):
        # Approximate MSE matrix \Omega(h) as defined in Lut p97
        G = self._zz
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
        sig_u = self.sigma_u

        omegas = np.zeros((steps, self.k, self.k))
        for h in range(1, steps + 1):
            if h == 1:
                omegas[h-1] = self.df_model * self.sigma_u
                continue

            om = omegas[h-1]
            for i in range(h):
                for j in range(h):
                    Bi = bpow(h - 1 - i)
                    Bj = bpow(h - 1 - j)
                    mult = np.trace(chain_dot(Bi.T, Ginv, Bj, G))
                    om += mult * chain_dot(phis[i], sig_u, phis[j].T)

        return omegas

    def _bmat_forc_cov(self):
        # B as defined on p. 96 of Lut
        upper = np.zeros((1, self.df_model))
        upper[0,0] = 1

        lower_dim = self.k * (self.p - 1)
        I = np.eye(lower_dim)
        lower = np.column_stack((np.zeros((lower_dim, 1)), I,
                                 np.zeros((lower_dim, self.k))))

        return np.vstack((upper, self.beta.T, lower))

    def summary(self):
        return VARSummary(self)

    def irf(self, periods=10, var_decomp=None):
        """
        Analyze impulse responses to shocks in system

        Parameters
        ----------
        periods : int
        var_decomp : ndarray (k x k), lower triangular
            Must satisfy Omega = P P', where P is the passed matrix. Defaults to
            Cholesky decomposition of Omega

        Returns
        -------

        """
        return IRAnalysis(self, P=var_decomp, periods=periods)

    def fevd(self, periods=10, var_decomp=None):
        """
        Compute forecast error variance decomposition ("fevd")

        Returns
        -------
        fevd : FEVD instance
        """
        return FEVD(self, P=var_decomp, periods=periods)

#-------------------------------------------------------------------------------
# VAR Diagnostics: Granger-causality, whiteness of residuals, normality, etc.

    def test_causality(self, equation, variables, kind='f', signif=0.05,
                       verbose=True):
        """
        Compute test statistic for null hypothesis of Granger-noncausality,
        general function to test joint Granger-causality of multiple variables

        Parameters
        ----------
        equation : string or int
            Equation to test for causality
        variables : sequence (of strings or ints)
            List, tuple, etc. of variables to test for Granger-causality
        kind : {'f', 'wald'}
            Perform F-test or Wald (chi-sq) test
        signif : float, default 5%
            Significance level for computing critical values for test,
            defaulting to standard 0.95 level

        Notes
        -----
        Null hypothesis is that there is no Granger-causality for the indicated
        variables

        Returns
        -------
        results : dict
        """
        k, p = self.k, self.p

        # number of restrictions
        N = len(variables) * self.p

        # Make restriction matrix
        C = np.zeros((N, k ** 2 * self.p + k), dtype=float)

        eq_index = self.get_eq_index(equation)
        vinds = mat([self.get_eq_index(v) for v in variables])

        # remember, vec is column order!
        offsets = np.concatenate([k + k ** 2 * j + k * vinds + eq_index
                                  for j in range(p)])
        C[np.arange(N), offsets] = 1

        # Lutkepohl 3.6.5
        Cb = np.dot(C, vec(self.beta.T))
        middle = L.inv(chain_dot(C, self.cov_beta, C.T))

        # wald statistic
        lam_wald = chain_dot(Cb, middle, Cb)

        if kind.lower() == 'wald':
            test_stat = lam_wald
            df = N
            dist = stats.chi2(df)
            pvalue = stats.chi2.sf(test_stat, N)
            crit_value = stats.chi2.ppf(1 - signif, N)
        elif kind.lower() == 'f':
            test_stat = lam_wald / N
            df = (N, self.df_resid)
            dist = stats.f(*df)
        else:
            raise Exception('kind %s not recognized' % kind)

        pvalue = dist.sf(test_stat)
        crit_value = dist.ppf(1 - signif)

        conclusion = 'fail to reject' if test_stat < crit_value else 'reject'
        results = {
            'test_stat' : test_stat,
            'crit_value' : crit_value,
            'pvalue' : pvalue,
            'df' : df,
            'conclusion' : conclusion,
            'signif' :  signif
        }

        if verbose:
            self._causality_summary(results, variables, equation, signif, kind)

        return results

    @staticmethod
    def _causality_summary(results, variables, equation, signif, kind):
        fmt = dict(_default_table_fmt,
                   data_fmts=["%#15.6F","%#15.6F","%#15.3F", "%s"])

        buf = StringIO()
        table = SimpleTable([[results['test_stat'],
                              results['crit_value'],
                              results['pvalue'],
                              str(results['df'])]],
                            ['Test statistic', 'Critical Value', 'p-value',
                             'df'], [''], title=None, txt_fmt=fmt)

        print >> buf, "Granger causality %s-test" % kind
        print >> buf, table

        print >> buf, 'H_0: %s do not Granger-cause %s' % (variables, equation)

        buf.write("Conclusion: %s H_0" % results['conclusion'])
        buf.write(" at %.2f%% significance level" % (results['signif'] * 100))

        print buf.getvalue()

    def test_whiteness(self):
        pass

    def test_normality(self):
        pass

    @cache_readonly
    def info_criteria(self):
        # information criteria for order selection
        return {
            'aic' : self._aic(),
            'hq' : self._hqic(),
            'sic' : self._sic(),
            'fpe' : self._log_fpe()
            }

    def _aic(self):
        # Akaike information criterion
        logdet = np.log(self._logdet_sigma_mle())
        return logdet + (2. * self.p / self.T) * (self.k ** 2)

    def _log_fpe(self):
        """
        Lutkepohl p. 147
        """
        # Log Final Prediction Error (FPE)
        factor = ((self.T + self.df_model) / self.df_resid) ** self.k
        return np.log(factor * self._logdet_sigma_mle())

    def _hqic(self):
        # Hannan-Quinn criterion
        logdet = np.log(self._logdet_sigma_mle())
        T = self.T

        return logdet + 2 * np.log(np.log(T)) * self.p * self.k ** 2 / T

    def _sic(self):
        # Schwarz criterion
        logdet = np.log(self._logdet_sigma_mle())
        return logdet + (np.log(self.T) / self.T) * self.p * self.k ** 2

    def _logdet_sigma_mle(self):
        sigma_mle = self.sigma_u * self.df_resid / self.T
        detsigma = L.det(sigma_mle)
        return detsigma

class BaseIRAnalysis(object):
    """
    Base class for plotting and computing IRF-related statistics, want to be
    able to handle known and estimated processes
    """

    def __init__(self, model, P=None, periods=10):
        self.model = model
        self.periods = periods
        self.k, self.lags, self.T  = model.k, model.p, model.T

        if P is None:
            P = model._chol_sigma_u
        self.P = P

        self.irfs = model.ma_rep(periods)
        self.orth_irfs = model.orth_ma_rep(periods)

        self.cum_effects = self.irfs.cumsum(axis=0)
        self.orth_cum_effects = self.orth_irfs.cumsum(axis=0)

        self.lr_effects = model.long_run_effects()
        self.orth_lr_effects = np.dot(model.long_run_effects(), P)

        # auxiliary stuff
        self._A = var1_rep(model.coefs)

    def cov(self, *args, **kwargs):
        raise NotImplementedError

    def cum_effect_cov(self, *args, **kwargs):
        raise NotImplementedError

    def plot_irf(self, orth=False, impcol=None, rescol=None, signif=0.05,
                 plot_params=None, subplot_params=None):
        """
        Plot impulse responses

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse responses
        impcol : string or int
            variable providing the impulse
        rescol : string or int
            variable affected by the impulse
        signif : float (0 < signif < 1)
            Significance level for error bars, defaults to 95% CI
        subplot_params : dict
            To pass to subplot plotting funcions. Example: if fonts are too big,
            pass {'fontsize' : 8} or some number to your taste.
        plot_params : dict
        """
        if orth:
            title = 'Impulse responses (orthogonalized)'
            irfs = self.orth_irfs
        else:
            title = 'Impulse responses'
            irfs = self.irfs

        try:
            stderr = self.cov(orth=orth)
        except NotImplementedError:
            stderr = None

        self._grid_plot(irfs, stderr, impcol, rescol, title,
                        signif=signif, subplot_params=subplot_params,
                        plot_params=plot_params)

    def plot_cum_effects(self, orth=False, impcol=None, rescol=None,
                         signif=0.05, plot_params=None,
                         subplot_params=None):
        if orth:
            title = 'Cumulative responses responses (orthogonalized)'
            cum_effects = self.orth_cum_effects
            lr_effects = self.orth_lr_effects
        else:
            title = 'Cumulative responses'
            cum_effects = self.cum_effects
            lr_effects = self.lr_effects

        try:
            stderr = self.cum_effect_cov(orth=orth)
        except NotImplementedError:
            stderr = None

        self._grid_plot(cum_effects, stderr, impcol, rescol, title,
                        signif=signif, hlines=lr_effects,
                        subplot_params=subplot_params,
                        plot_params=plot_params)

    def _grid_plot(self, values, stderr, impcol, rescol, title,
                   signif=0.05, hlines=None, subplot_params=None,
                   plot_params=None, figsize=(10,10)):
        """
        Reusable function to make flexible grid plots of impulse responses and
        comulative effects

        values : (T + 1) x k x k
        stderr : T x k x k
        hlines : k x k
        """
        if subplot_params is None:
            subplot_params = {}
        if plot_params is None:
            plot_params = {}

        nrows, ncols, to_plot = self._get_plot_config(impcol, rescol)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,
                                 squeeze=False, figsize=figsize)

        # fill out space
        _adjust_subplots()

        fig.suptitle(title, fontsize=14)

        subtitle_temp = r'%s$\rightarrow$%s'

        names = self.model.names
        for (j, i, ai, aj) in to_plot:
            ax = axes[ai][aj]

            # hm, how hackish is this?
            if stderr is not None:
                sig = np.sqrt(stderr[:, j * self.k + i, j * self.k + i])
                plot_with_error(values[:, i, j], sig, axes=ax, alpha=signif,
                                value_fmt='b')
            else:
                plot_with_error(values[:, i, j], None, axes=ax, value_fmt='b')

            ax.axhline(0, color='k')

            if hlines is not None:
                ax.axhline(hlines[i,j], color='k')

            sz = subplot_params.get('fontsize', 12)
            ax.set_title(subtitle_temp % (names[j], names[i]), fontsize=sz)

    def _get_plot_config(self, impcol, rescol):
        nrows = ncols = k = self.k
        if impcol is not None and rescol is not None:
            # plot one impulse-response pair
            nrows = ncols = 1
            j = self.model.get_eq_index(impcol)
            i = self.model.get_eq_index(rescol)
            to_plot = [(j, i, 0, 0)]
        elif impcol is not None:
            # plot impacts of impulse in one variable
            ncols = 1
            j = self.model.get_eq_index(impcol)
            to_plot = [(j, i, i, 0) for i in range(k)]
        elif rescol is not None:
            # plot only things having impact on particular variable
            ncols = 1
            i = self.model.get_eq_index(rescol)
            to_plot = [(j, i, j, 0) for j in range(k)]
        else:
            # plot everything
            to_plot = [(j, i, i, j) for i in range(k) for j in range(k)]

        return nrows, ncols, to_plot


class IRAnalysis(BaseIRAnalysis):
    """
    Impulse response analysis class. Computes impulse responses, asymptotic
    standard errors, and produces relevant plots

    Parameters
    ----------
    model : VAR instance

    Notes
    -----
    Using Lutkepohl (2005) notation
    """
    def __init__(self, model, P=None, periods=10):
        BaseIRAnalysis.__init__(self, model, P=P, periods=periods)

        self.cov_a = model._cov_alpha
        self.cov_sig = model._cov_sigma

        # memoize dict for G matrix function
        self._g_memo = {}

    def cov(self, orth=False):
        """

        Notes
        -----
        Lutkepohl eq 3.7.5

        Returns
        -------
        """
        if orth:
            return self._orth_cov()

        covs = self._empty_covm(self.periods + 1)
        covs[0] = np.zeros((self.k ** 2, self.k ** 2))
        for i in range(1, self.periods + 1):
            Gi = self.G[i - 1]
            covs[i] = chain_dot(Gi, self.cov_a, Gi.T)

        return covs

    def _orth_cov(self):
        """

        Notes
        -----
        Lutkepohl 3.7.8

        Returns
        -------

        """
        Ik = np.eye(self.k)
        PIk = np.kron(self.P.T, Ik)
        H = self.H

        covs = self._empty_covm(self.periods + 1)
        for i in range(self.periods + 1):
            if i == 0:
                apiece = 0
            else:
                Ci = np.dot(PIk, self.G[i-1])
                apiece = chain_dot(Ci, self.cov_a, Ci.T)

            Cibar = np.dot(np.kron(Ik, self.irfs[i]), H)
            bpiece = chain_dot(Cibar, self.cov_sig, Cibar.T) / self.T

            # Lutkepohl typo, cov_sig correct
            covs[i] = apiece + bpiece

        return covs

    def cum_effect_cov(self, orth=False):
        """

        Parameters
        ----------
        orth : boolean

        Notes
        -----
        eq. 3.7.7 (non-orth), 3.7.10 (orth)

        Returns
        -------

        """
        Ik = np.eye(self.k)
        PIk = np.kron(self.P.T, Ik)

        F = 0.
        covs = self._empty_covm(self.periods + 1)
        for i in range(self.periods + 1):
            if i > 0:
                F = F + self.G[i - 1]

            if orth:
                if i == 0:
                    apiece = 0
                else:
                    Bn = np.dot(PIk, F)
                    apiece = chain_dot(Bn, self.cov_a, Bn.T)

                Bnbar = np.dot(np.kron(Ik, self.cum_effects[i]), self.H)
                bpiece = chain_dot(Bnbar, self.cov_sig, Bnbar.T) / self.T

                covs[i] = apiece + bpiece
            else:
                if i == 0:
                    covs[i] = np.zeros((self.k**2, self.k**2))
                    continue

                covs[i] = chain_dot(F, self.cov_a, F.T)

        return covs

    def lr_effect_cov(self, orth=False):
        """

        Returns
        -------

        """
        lre = self.lr_effects
        Finfty = np.kron(np.tile(lre.T, self.lags), lre)

        if orth:
            Binf = np.dot(np.kron(self.P.T, np.eye(self.k)), Finfty)
            Binfbar = np.dot(np.kron(np.eye(self.k), self.lr_effects), self.H)

            return (chain_dot(Binf, self.cov_a, Binf.T) +
                    chain_dot(Binfbar, self.cov_a, Binfbar.T))
        else:
            return chain_dot(Finfty, self.cov_a, Finfty.T)

    def _empty_covm(self, periods):
        return np.zeros((periods, self.k ** 2, self.k ** 2),
                        dtype=float)

    @cache_readonly
    def G(self):
        def _make_g(i):
            # p. 111 Lutkepohl
            G = 0.
            for m in range(i):
                # be a bit cute to go faster
                idx = i - 1 - m
                if idx in self._g_memo:
                    apow = self._g_memo[idx]
                else:
                    apow = npl.matrix_power(self._A.T, idx)[:self.k]

                    self._g_memo[idx] = apow

                # take first K rows
                piece = np.kron(apow, self.irfs[m])
                G = G + piece

            return G

        return [_make_g(i) for i in range(1, self.periods + 1)]

    @cache_readonly
    def H(self):
        k = self.k
        Lk = tsa.elimination_matrix(k)
        Kkk = tsa.commutation_matrix(k, k)
        Ik = np.eye(k)

        # B = chain_dot(Lk, np.eye(k**2) + commutation_matrix(k, k),
        #               np.kron(self.P, np.eye(k)), Lk.T)

        # return np.dot(Lk.T, L.inv(B))

        B = chain_dot(Lk,
                      np.dot(np.kron(Ik, self.P), Kkk) + np.kron(self.P, Ik),
                      Lk.T)

        return np.dot(Lk.T, L.inv(B))

    def fevd_table(self):
        pass

class FEVD(object):
    """
    Compute and plot Forecast error variance decomposition and asymptotic
    standard errors
    """
    def __init__(self, model, P=None, periods=None):
        self.periods = periods

        self.model = model
        self.k = model.k
        self.names = model.names

        self.irfobj = model.irf(var_decomp=P, periods=periods)
        self.orth_irfs = self.irfobj.orth_irfs

        # cumulative impulse responses
        irfs = (self.orth_irfs[:periods] ** 2).cumsum(axis=0)

        rng = range(self.k)
        mse = self.model.mse(periods)[:, rng, rng]

        # lag x equation x component
        fevd = np.empty_like(irfs)

        for i in range(periods):
            fevd[i] = (irfs[i].T / mse[i]).T

        # switch to equation x lag x component
        self.decomp = fevd.swapaxes(0, 1)

    def summary(self):
        buf = StringIO()

        rng = range(self.periods)
        for i in range(self.k):
            ppm = pprint_matrix(self.decomp[i], rng, self.names)

            print >> buf, 'FEVD for %s' % self.names[i]
            print >> buf, ppm

        print buf.getvalue()

    def cov(self, lag):
        """
        Compute asymptotic standard errors

        Returns
        -------
        """
        pass

    def plot(self, periods=None, figsize=(10,10), **plot_kwds):
        """
        Plot graphical display of FEVD

        Parameters
        ----------
        periods : int, default None
            Defaults to number originally specified. Can be at most that number
        """
        k = self.k
        periods = periods or self.periods

        fig, axes = plt.subplots(nrows=k, figsize=(10,10))

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
                                color=colors[j], label=self.names[j],
                                **plot_kwds)

                handles.append(handle)

            ax.set_title(self.names[i])

        # just use the last axis to get handles for plotting
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        _adjust_subplots(right=0.85)

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    from scikits.statsmodels.tsa.var.util import parse_data

    path = 'scikits/statsmodels/sandbox/tsa/data/%s.dat'
    sdata, dates = parse_data(path % 'e1')

    names = sdata.dtype.names
    data = _struct_to_ndarray(sdata)

    adj_data = np.diff(np.log(data), axis=0)
    # est = VAR(adj_data, p=2, dates=dates[1:], names=names)
    model = VAR(adj_data[:-16], dates=dates[1:-16], names=names)
    est = model.fit(maxlags=2)
    irf = est.irf()

    y = est.y[-2:]

    from scikits.statsmodels.tsa.var.alt import VAR2

    est2 = VAR2(adj_data[:-16])
    res2 = est2.fit(maxlag=2)

    # irf.plot_irf()

    # i = 2; j = 1
    # cv = irf.cum_effect_cov(orth=True)
    # print np.sqrt(cv[:, j * 3 + i, j * 3 + i]) / 1e-2

    data = np.genfromtxt('Canada.csv', delimiter=',', names=True)

    model = VAR(data)
    est = model.fit(maxlags=2)
    irf = est.irf()
