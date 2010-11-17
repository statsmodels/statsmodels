"""
References
----------
Lutkepohl (2005) New Introduction to Multiple Time Series Analysis
"""

from __future__ import division

import numpy as np
from numpy.linalg import cholesky as chol, solve
from numpy.random import multivariate_normal as mvn
import scipy.stats as stats
import scipy.linalg as L

import matplotlib.pyplot as plt
import matplotlib as mpl

import pandas.util.testing as test

from scikits.statsmodels.tools import chain_dot
from scikits.statsmodels.decorators import cache_readonly

mat = np.array

st = test.set_trace

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
    # would kind of like to use ndarray.take here
    indices = np.triu_indices(len(mat))

    # Gets Fortran-order
    return mat.T[indices]

def unvec(v):
    k = int(np.sqrt(len(v)))
    assert(k * k == len(v))
    return v.reshape((k, k), order='F')

def unvech(v):
    pass

def test_vec():
    pass

def test_vech():
    pass

#-------------------------------------------------------------------------------
# VAR process routines

def varsim(coefs, nu, sig_u, steps=100, initvalues=None):
    """

    """
    p, k, k = coefs.shape

    ugen = mvn(np.zeros(len(sig_u)), sig_u, steps)
    result = np.zeros((steps, k))
    result[p:] = nu + ugen[p:]

    # add in AR terms
    for t in xrange(p, steps):
        ygen = result[t]
        for j in xrange(p):
            ygen += np.dot(coefs[j], result[t-j-1])

    return result

def ma_rep(coefs, nu, maxn=10):
    """
    MA(\infty) representation
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

    Returns
    -------
    bool
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
    D = np.sqrt(np.outer(diag, diag))

    acorr = np.empty_like(acf)
    for i in xrange(len(acf)):
        acorr[i] = acf[i] / D

    return acorr

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

        # y_t(h) = nu + sum_1^p A_i y_t_(h-i)

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
        var = np.dot(phi, np.dot(sig_u, phi.T))
        forc_covs[h] = prior = prior + var

    return forc_covs

def granger_causes(coefs):
    pass

#-------------------------------------------------------------------------------
# Plotting functions

def plot_mts(Y, names=None):
    k = Y.shape[1]
    rows, cols = k, 1

    plt.figure(figsize=(10, 10))

    for j in range(k):
        ts = Y[:, j]

        ax = plt.subplot(rows, cols, j+1)
        ax.plot(ts)
        if names is not None:
            ax.set_title(names[j])

def plot_var_forc(prior, forc, err_upper, err_lower, names=None):
    n, k = prior.shape
    rows, cols = k, 1

    plt.figure(figsize=(10, 10))

    prange = np.arange(n)
    rng_f = np.arange(n - 1, n + len(forc))
    rng_err = np.arange(n, n + len(forc))

    for j in range(k):
        ax = plt.subplot(rows, cols, j+1)

        ax.plot(prange, prior[:, j], 'k')
        ax.plot(rng_f, np.r_[prior[-1:, j], forc[:, j]], 'k--')
        ax.plot(rng_err, err_upper[:, j], 'k-.')
        ax.plot(rng_err, err_lower[:, j], 'k-.')

        if names is not None:
            ax.set_title(names[j])

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
        plot_var_forc(y, mid, lower, upper, names=self.names)

    def mean(self):
        """
        Mean of stable process

        ::

            \mu = (I - A_1 - \dots - A_p)^{-1} intercept
        """
        return solve(self._char_mat, self.intercept)

    def long_run_effects(self):
        return L.inv(self._char_mat)

    def ma_rep(self, maxn=10):
        """

        """
        return ma_rep(self.coefs, self.intercept, maxn=maxn)

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

    def irf(self):
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
# Analysis classes

class IRF(object):
    """
    Impulse response function
    """

    def plot(self, *args, **kwargs):
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

#-------------------------------------------------------------------------------
# Estimator and Known VAR process classes

class KnownVARProcess(VARProcess):
    """
    Class for analyzing VAR(p) process with known coefficient matrices and white
    noise covariance matrix
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

    Notes
    -----
    **References**
    Lutkepohl (2005) New Introduction to Multiple Time Series Analysis

    Returns
    -------
    **Attributes**
    Y : ndarray (K x T)
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
        plot_mts(self.y, names=self.names)

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
        Y = self.y
        p = self.p

        # Ravel C order, need to put in descending order
        Z = mat([Y[t-p : t][::-1].ravel() for t in xrange(p, self.nobs)])

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
        coefs = chain_dot(self._zzinv, self.Z.T, self._y_sample)
        return coefs

    def _est_sigma_u(self):
        """
        Unbiased estimate of covariance matrix $\Sgma_u$ of the white noise
        process $u$

        equivalent definition

        .. math:: \frac{1}{T - Kp - 1} Y^\prime (I_T - Z (Z^\prime Z)^{-1}
        Z^\prime) Y

        Ref: Lutkepohl p.75
        """
        y = self._y_sample
        z = self.Z
        b = self.beta

        # Lutkepohl p75, this is slower
        # middle = np.eye(self.T) - chain_dot(z, self._zzinv, z.T)
        # return chain_dot(y.T, middle, y) / self.df_resid

        # about 5x faster
        resid = y - np.dot(z, b)
        # df_resid right now is T - Kp - 1, which is a suggested correction
        return np.dot(resid.T, resid) / self.df_resid

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
        # Approximate MSE matrix \Omega(h) as defined in Lut
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
        lower = np.c_[np.zeros((lower_dim, 1)), I,
                      np.zeros((lower_dim, self.k))]

        return np.r_[upper, self.beta.T, lower]

    def _approx_mse(self, h):
        """

        """

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



path = 'scikits/statsmodels/sandbox/tsa/data/%s.dat'
sdata, dates = parse_data(path % 'e1')

names = sdata.dtype.names
data = _struct_to_ndarray(sdata)

adj_data = np.diff(np.log(data), axis=0)
# est = VAR(adj_data, p=2, dates=dates[1:], names=names)
est = VAR(adj_data[:-16], p=2, dates=dates[1:-16], names=names)

y = est.y[-2:]
