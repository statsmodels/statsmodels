"""
Vector Autoregression (VAR) processes

References
----------
Lutkepohl (2005) New Introduction to Multiple Time Series Analysis
"""

from __future__ import division

import numpy as np
import numpy.linalg as npl
from numpy.linalg import cholesky as chol, solve
from numpy.random import multivariate_normal as rmvnormn
import scipy.stats as stats
import scipy.linalg as L

import matplotlib.pyplot as plt
import matplotlib as mpl

from scikits.statsmodels.decorators import cache_readonly
from scikits.statsmodels.tools import chain_dot
from scikits.statsmodels.iolib import SimpleTable

mat = np.array

try:
    import pandas.util.testing as test
    st = test.set_trace
except ImportError:
    pass

class MPLConfigurator(object):

    def revert(self):
        pass

    def set_fontsize(self, size):
        pass

def parse_data(path):
    """
    Parse data files from Lutkepohl (2005) book

    Source for data files: www.jmulti.de
    """

    from collections import deque
    from datetime import datetime
    import pandas
    import pandas.core.datetools as dt
    import re

    regex = re.compile('<(.*) (\w)([\d]+)>.*')
    lines = deque(open(path))

    to_skip = 0
    while '*/' not in lines.popleft():
        to_skip += 1

    while True:
        to_skip += 1
        line = lines.popleft()
        m = regex.match(line)
        if m:
            year, freq, start_point = m.groups()
            break

    data = np.genfromtxt(path, names=True, skip_header=to_skip+1)

    n = len(data)

    # generate the corresponding date range (using pandas for now)
    start_point = int(start_point)
    year = int(year)

    offsets = {
        'Q' : dt.BQuarterEnd(),
        'M' : dt.BMonthEnd(),
        'A' : dt.BYearEnd()
    }

    # create an instance
    offset = offsets[freq]

    inc = offset * (start_point - 1)
    start_date = offset.rollforward(datetime(year, 1, 1)) + inc

    offset = offsets[freq]
    date_range = pandas.DateRange(start_date, offset=offset, periods=n)

    return data, date_range

#-------------------------------------------------------------------------------
# Utilities

def vec(mat):
    return mat.ravel('F')

def vech(mat):
    # Gets Fortran-order
    return mat.T.take(_triu_indices(len(mat)))

# tril/triu/diag, suitable for ndarray.take

def _tril_indices(n):
    rows, cols = np.tril_indices(n)
    return rows * n + cols

def _triu_indices(n):
    rows, cols = np.triu_indices(n)
    return rows * n + cols

def _diag_indices(n):
    rows, cols = np.diag_indices(n)
    return rows * n + cols

def unvec(v):
    k = int(np.sqrt(len(v)))
    assert(k * k == len(v))
    return v.reshape((k, k), order='F')

def unvech(v):
    # quadratic formula, correct fp error
    rows = .5 * (-1 + np.sqrt(1 + 8 * len(v)))
    rows = int(np.round(rows))

    result = np.zeros((rows, rows))
    result[np.triu_indices(rows)] = v
    result = result + result.T

    # divide diagonal elements by 2
    result[np.diag_indices(rows)] /= 2

    return result

def duplication_matrix_old(k):
    """
    Create duplication matrix for converting vech to vec

    Notes
    -----
    For symmetric matrix S we have

    vec(S) = D_k vech(S)
    """
    result = np.empty((k * (k + 1) / 2, k * k), dtype=float)

    # most efficient way?
    for i, row in enumerate(np.eye(k * (k + 1) / 2)):
        result[i] = unvech(row).ravel()

    # leaves it in fortran order... oh well, for now
    return result.T

def duplication_matrix(n):
    """
    Create duplication matrix D_n which satisfies vec(S) = D_n vech(S) for
    symmetric matrix S

    Returns
    -------

    """
    onesmat = np.ones((n, n))
    vech_mask = vec(np.tril(onesmat)) == 1
    subdiag_mask = vec(np.tril(onesmat, k=-1)) != 0

    D = np.eye(n * n)
    D[subdiag_mask] = D[subdiag_mask] + D[-vech_mask]
    return D[vech_mask].T

def elimination_matrix(n):
    """
    Create the elimination matrix L_n which satisfies vech(M) = L_n vec(M) for
    any matrix M

    Parameters
    ----------

    Returns
    -------

    """
    vech_indices = vec(np.tril(np.ones((n, n))))
    return np.eye(n * n)[vech_indices != 0]

def commutation_matrix(p, q):
    """
    Create the commutation matrix K_{p,q} satisfying vec(A') = K_{p,q} vec(A)

    Parameters
    ----------
    p : int
    q : int

    Returns
    -------
    K : ndarray (pq x pq)
    """
    K = np.eye(p * q)
    indices = np.arange(p * q).reshape((p, q), order='F')
    return K.take(indices.ravel(), axis=0)

def test_duplication_matrix():
    k = 10
    m = unvech(np.random.randn(k * (k + 1) / 2))
    D3 = duplication_matrix(3)
    assert(np.array_equal(vec(m), np.dot(D3, vech(m))))

def test_elimination_matrix():
    m = np.random.randn(3, 3)
    L3 = elimination_matrix(3)
    assert(np.array_equal(vech(m), np.dot(L3, vec(m))))

def test_commutation_matrix():
    m = np.random.randn(4, 3)
    K = commutation_matrix(4, 3)
    assert(np.array_equal(vec(m.T), np.dot(K, vec(m))))

def test_vec():
    pass

def test_vech():
    pass

#-------------------------------------------------------------------------------
# VAR process routines

def varsim(coefs, intercept, sig_u, steps=100, initvalues=None):
    """
    Simulate simple VAR(p) process with known coefficients, intercept, white
    noise covariance, etc.
    """
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
    vecACF = solve(np.eye((k*p)**2) - np.kron(A, A), vec(SigU))

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

def plot_acorr(acf):
    """

    """
    # hack
    old_size = mpl.rcParams['font.size']
    mpl.rcParams['font.size'] = 8

    lags, k, k = acf.shape
    acorr = _acf_to_acorr(acf)
    plt.figure(figsize=(10, 10))
    xs = np.arange(lags)

    for i in range(k):
        for j in range(k):
            ax = plt.subplot(k, k, i * k + j + 1)
            ax.vlines(xs, [0], acorr[:, i, j])

            ax.axhline(0, color='k')
            ax.set_ylim([-1, 1])

            # hack?
            ax.set_xlim([-1, xs[-1] + 1])

    mpl.rcParams['font.size'] = old_size

#-------------------------------------------------------------------------------
# VARProcess class: for known or unknown VAR process

def _struct_to_ndarray(arr):
    return arr.view((float, len(arr.dtype.names)))

def lag_select():
    pass

import unittest
class TestVAR(unittest.TestCase):
    pass

class VARProcess(object):

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

    def plot_acorr(self, nlags=10):
        plot_acorr(self.acf(nlags=nlags))

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
        q = stats.norm.ppf(1 - alpha / 2)

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
# Estimator and Known VAR process classes

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

class VAR(VARProcess):
    """
    Estimate VAR(p) process

    .. math:: y_t = A_1 y_{t-1} + \ldots + A_p y_{t-p} + u_t

    Notes
    -----
    **References**
    Lutkepohl (2005) New Introduction to Multiple Time Series Analysis

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

    def __init__(self, data, p=1, names=None, dates=None):
        self.p = p
        self.y, self.names = self._interpret_data(data, names)
        self.nobs, self.k = self.y.shape

        # This will be the dimension of the Z matrix
        self.T = self.nobs  - self.p

        self.dates = dates

        if dates is not None:
            assert(self.nobs == len(dates))

        self.names = names
        self.dates = dates

    @staticmethod
    def _interpret_data(data, names):
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

#-------------------------------------------------------------------------------
# Forecast error covariance

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

    def forecast_cov(self, steps=1):
        """
        Compute forecast covariance matrices for desired number of steps

        Parameters
        ----------

        Notes
        -----
        .. math:: \Sigma_{\hat y}(h) = \Sigma_y(h) + \Omega(h) / T

        Ref: Lutkepohl pp. 96-97

        Returns
        -------

        """
        forc_covs = VARProcess.forecast_cov(self, steps)
        omegas = self._omega_forc_cov(steps)
        return forc_covs + omegas / self.T

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

    def _approx_mse(self, h):
        """

        """
        pass

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

    def summary(self):
        return VARSummary(self)

#-------------------------------------------------------------------------------
# Impulse responses, long run effects, standard errors

    def irf(self, periods=10, P=None):
        """
        Compute impulse responses to shocks in system

        Returns
        -------

        """
        return IRAnalysis(self, P=P, periods=periods)

    @property
    def _cov_alpha(self):
        # drop intercept
        return self.cov_beta[self.k:, self.k:]

    @cache_readonly
    def _cov_sigma(self):
        D_K = duplication_matrix(self.k)
        D_Kinv = npl.pinv(D_K)

        sigxsig = np.kron(self.sigma_u, self.sigma_u)
        return 2 * chain_dot(D_Kinv, sigxsig, D_Kinv.T)

class IRAnalysis(object):
    """
    Impulse response analysis class

    Parameters
    ----------
    model : VAR instance

    Notes
    -----
    Using Lutkepohl (2005) notation
    """

    def __init__(self, model, P=None, periods=10):
        self.model = model
        self.periods = periods
        self.k, self.lags, self.T = model.k, model.p, model.T

        if P is None:
            P = model._chol_sigma_u
        self.P = P

        self.irfs = model.ma_rep(periods)
        self.orth_irfs = model.orth_ma_rep(periods)

        self.lr_effects = model.long_run_effects()
        self.cum_effects = self.irfs.cumsum(axis=0)

        self.cov_a = model._cov_alpha
        self.cov_sig = model._cov_sigma

        # auxiliary stuff
        self._A = var1_rep(model.coefs)
        self._g_memo = {}

    def cov(self, orth=False):
        """

        Returns
        -------
        """
        if orth:
            return self._orth_cov()

        covs = self._empty_covm()
        for i in range(self.periods):
            Gi = self.G[i]
            covs[i] = chain_dot(Gi, self.cov_a, Gi.T)

        return covs

    def _orth_cov(self):
        """

        Returns
        -------

        """
        covs = self._empty_covm()

        Ik = np.eye(self.k)
        PIk = np.kron(self.P.T, Ik)
        H = self.H

        for i in range(self.periods):
            Ci = np.dot(PIk, self.G[i])
            Cibar = np.dot(np.kron(Ik, self.irfs[i + 1]), H)

            # is this right? I think Lutkepohl has a typo!
            covs[i] = (chain_dot(Ci, self.cov_a, Ci.T) +
                       chain_dot(Cibar, self.cov_sig, Cibar.T)) / self.T

        return covs

    def cum_effect_cov(self, orth=False):
        """

        Parameters
        ----------
        orth : boolean

        Returns
        -------

        """
        Ik = np.eye(self.k)
        PIk = np.kron(self.P.T, Ik)

        F = 0.
        covs = self._empty_covm()
        for i in range(self.periods):
            F = F + self.G[i]
            if orth:
                Bn = np.dot(PIk, F)
                Bnbar = np.dot(np.kron(Ik, self.cum_effects[i + 1]), self.H)

                covs[i] = (chain_dot(Bn, self.cov_a, Bn.T) +
                           chain_dot(Bnbar, self.cov_sig, Bnbar.T)) / self.T

            else:
                covs[i] = chain_dot(F, self.cov_a, F.T) / self.T

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
                    chain_dot(Binfbar, self.cov_a, Binfbar.T)) / self.T
        else:
            return chain_dot(Finfty, self.cov_a, Finfty.T) / self.T

    def fevd(self, steps=1):
        """

        Parameters
        ----------

        Returns
        -------
        """
        # cumulative impulse responses
        irfs = (self.orth_irfs[:steps] ** 2).cumsum(axis=0)
        mse = self.model.forecast_cov(steps)
        return irfs / mse

    def fevd_cov(self, lag):
        """

        Returns
        -------
        """
        pass

    def _empty_covm(self):
        return np.zeros((self.periods, self.k ** 2, self.k ** 2), dtype=float)

    @cache_readonly
    def G(self):
        _memo = {}
        def _make_g(i):
            # p. 111 Lutkepohl
            G = 0.
            for m in range(i):
                # be a bit cute to go faster
                idx = i - 1 - m
                if idx in _memo:
                    apow = _memo[idx]
                else:
                    apow = npl.matrix_power(self._A.T, idx)[:self.k]

                    _memo[idx] = apow

                # take first K rows
                piece = np.kron(apow, self.irfs[m])
                G = G + piece

            return G

        return [_make_g(i) for i in range(1, self.periods + 1)]

    @cache_readonly
    def H(self):
        k = self.k
        Lk = elimination_matrix(k)

        B = chain_dot(Lk, np.eye(k**2) + commutation_matrix(k, k),
                      np.kron(self.P, np.eye(k)), Lk.T)

        return np.dot(Lk.T, L.inv(B))

    def plot_irf(self, orth=True):
        irfs = self.irfs
        stderr = self.cov()

        k = self.k

        plt.figure(figsize=(10, 10))

        i = 2
        j=1

        ax = plt.gca()

        est = irfs[:, i, j]
        sig = np.sqrt(np.r_[0, stderr[:, j * k + i, j * k + i]])
        ax.plot(est, 'k')
        ax.plot(est - 1.96 * sig, 'k--')
        ax.plot(est + 1.96 * sig, 'k--')
        ax.axhline(0)

        # for j in range(k):
        #     ts = Y[:, j]

        #     ax = plt.subplot(rows, cols, j+1)
        #     if index is not None:
        #         ax.plot(index, ts)
        #     else:
        #         ax.plot(ts)

        #     if names is not None:
        #         ax.set_title(names[j])


    def plot_cum_effects(self, orth=True):
        pass

    def fevd_table(self):
        pass

class FEVD(object):
    """
    Forecast error variance decomposition
    """
    def __init__(self, model):
        self.model = model

    def summary(self):
        pass

    def plot(self, *args, **kwargs):
        pass

class VARSummary(object):
    default_fmt = dict(
        #data_fmts = ["%#12.6g","%#12.6g","%#10.4g","%#5.4g"],
        #data_fmts = ["%#10.4g","%#10.4g","%#10.4g","%#6.4g"],
        data_fmts = ["%#15.6F","%#15.6F","%#15.3F","%#14.3F"],
        empty_cell = '',
        #colwidths = 10,
        colsep='  ',
        row_pre = '',
        row_post = '',
        table_dec_above='=',
        table_dec_below='=',
        header_dec_below='-',
        header_fmt = '%s',
        stub_fmt = '%s',
        title_align='c',
        header_align = 'r',
        data_aligns = 'r',
        stubs_align = 'l',
        fmt = 'txt'
    )

    part1_fmt = dict(default_fmt,
        data_fmts = ["%s"],
        colwidths = 15,
        colsep=' ',
        table_dec_below='',
        header_dec_below=None,
    )
    part2_fmt = dict(default_fmt,
        data_fmts = ["%#12.6g","%#12.6g","%#10.4g","%#5.4g"],
        colwidths = None,
        colsep='    ',
        table_dec_above='-',
        table_dec_below='-',
        header_dec_below=None,
    )

    def __init__(self, estimator):
        self.model = estimator
        self.summary = self.make()

    def __repr__(self):
        return self.summary

    def _lag_names(self):
        lag_names = []

        # take care of lagged endogenous names
        for i in range(1, self.model.p+1):
            for name in self.model.names:
                lag_names.append('L'+str(i)+'.'+name)

        # put them together
        Xnames = lag_names

        # handle the constant name
        trendorder = 1 # self.trendorder
        if trendorder != 0:
            Xnames.insert(0, 'const')
        if trendorder > 1:
            Xnames.insert(0, 'trend')
        if trendorder > 2:
            Xnames.insert(0, 'trend**2')

        return Xnames

    def make(self, endog_names=None, exog_names=None):
        """
        Summary of VAR model
        """
        return self._coef_table()
        # return '\n'.join((self._header_table(),
        #                   self._stats_table(),
        #                   self._coef_table()))

    def _header_table(self):
        import time

        model = self.model

        modeltype = type(model).__name__
        t = time.localtime()

        # ncoefs = len(model.beta) #TODO: change when we allow coef restrictions

        # Header information
        part1title = "Summary of Regression Results"
        part1data = [[modeltype],
                     ["OLS"], #TODO: change when fit methods change
                     [time.strftime("%a, %d, %b, %Y", t)],
                     [time.strftime("%H:%M:%S", t)]]
        part1header = None
        part1stubs = ('Model:',
                     'Method:',
                     'Date:',
                     'Time:')
        part1 = SimpleTable(part1data, part1header, part1stubs,
                            title=part1title, txt_fmt=self.part1_fmt)

        return str(part1)

    def _stats_table(self):
        #TODO: do we want individual statistics or should users just
        # use results if wanted?
        # Handle overall fit statistics
        part2Lstubs = ('No. of Equations:',
                       'Nobs:',
                       'Log likelihood:',
                       'AIC:')
        part2Rstubs = ('BIC:',
                       'HQIC:',
                       'FPE:',
                       'Det(Omega_mle):')
        part2Ldata = [[self.neqs],[self.nobs],[self.llf],[self.aic]]
        part2Rdata = [[self.bic],[self.hqic],[self.fpe],[self.detomega]]
        part2Lheader = None
        part2L = SimpleTable(part2Ldata, part2Lheader, part2Lstubs,
                             txt_fmt = self.part2_fmt)
        part2R = SimpleTable(part2Rdata, part2Lheader, part2Rstubs,
                             txt_fmt = self.part2_fmt)
        part2L.extend_right(part2R)

        return str(part2L)

    def _coef_table(self):
        Xnames = self._lag_names() * self.model.k

        model = self.model

        data = zip(model.beta.ravel(),
                   model.stderr.ravel(),
                   model.t().ravel(),
                   model.pvalues.ravel())

        header = ('coefficient','std. error','z-stat','prob')
        table = SimpleTable(data, header, Xnames, title=None,
                            txt_fmt = self.default_fmt)

        return str(table)


#-------------------------------------------------------------------------------

def foo():

    a1 = [[.7, .1, 0],
          [0, .4, .1],
          [.9, 0, .8]]
    a2 = [[-.2, 0, 0],
          [0, .1, .1],
          [0, 0, 0]]

    a3 = [[-.2, 0, 0],
          [0, .1, .1],
          [0, 0, 0]]

    coefs = mat([a1, a2])

    nu = mat([2, 1, 0])

    P = mat([[.5, .1, 0],
             [0, .3, 0],
             [0, 0, .9]])
    sig_u = np.dot(P, P.T)

    m = VAR(nu, coefs, sig_u)

    y_2000 = mat([.7, 1.0, 1.5])
    y_1999 = mat([1.0, 1.5, 3.0])
    yprior = mat([y_1999, y_2000])

    forc = m.forecast(yprior, 10)

    m.plotforc(yprior, 10)

    coefs2 = mat([[[.5, 0, 0],
               [.1, .1, .3],
               [0, .2, .3]]])

    sig_u2 = mat([[2.25, 0, 0],
                  [0, 1, .5],
                  [0, .5, .74]])

    m2 = VAR(nu, coefs2, sig_u2)

    coefs3 = mat([[[.5, .1],
                [.4, .5]],
               [[0, 0],
                [.25, 0]]])

    sig_u3 = mat([[0.09, 0],
                  [0, 0.04]])

    m3 = VAR(mat([1, 1]), coefs3, sig_u3)

    acf = var_acf(coefs, sig_u, nlags=10)
    acf3 = var_acf(coefs3, sig_u3, nlags=10)

    plt.show()

if __name__ == '__main__':
    path = 'scikits/statsmodels/sandbox/tsa/data/%s.dat'
    sdata, dates = parse_data(path % 'e1')

    names = sdata.dtype.names
    data = _struct_to_ndarray(sdata)

    adj_data = np.diff(np.log(data), axis=0)
    # est = VAR(adj_data, p=2, dates=dates[1:], names=names)
    est = VAR(adj_data[:-16], p=2, dates=dates[1:-16], names=names)

    irf = est.irf()

    y = est.y[-2:]

    from scikits.statsmodels.tsa.var import VAR2

    est2 = VAR2(adj_data[:-16])
    res2 = est2.fit(maxlag=2)
