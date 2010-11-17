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

import matplotlib.pyplot as plt
import matplotlib as mpl

import pandas.util.testing as test

mat = np.array

st = test.set_trace

class MPLConfigurator(object):

    def revert(self):
        pass

    def set_fontsize(self, size):
        pass

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

def var_mean(coefs, intercept):
    p, k, k = coefs.shape
    return solve(np.eye(k) - coefs.sum(0), nu)

def ma_rep(coefs, nu, maxn=10):
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

def is_stable(coefs):
    """

    Returns
    -------
    bool
    """
    A_var1 = var1_rep(coefs)
    eigs = np.linalg.eigvals(A_var1)
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
    Compute autocovariance function ACF_y(h) up to nlags

    Parameters
    ----------


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

#-------------------------------------------------------------------------------
# Plotting functions

def plot_mts(Y):
    k = Y.shape[1]
    rows, cols = k, 1

    plt.figure(figsize=(10, 10))

    for j in range(k):
        ax = plt.subplot(rows, cols, j+1)
        ax.plot(Y[:, j])

def plot_var_forc(prior, forc, err_upper, err_lower):
    n, k = prior.shape
    rows, cols = k, 1

    plt.figure(figsize=(10, 10))

    prange = np.arange(n)
    frange = np.arange(n - 1, n + len(forc) - 1)

    for j in range(k):
        ax = plt.subplot(rows, cols, j+1)

        ax.plot(prange, prior[:, j], 'g')
        ax.plot(frange, forc[:, j], 'k')
        ax.plot(frange, err_upper[:, j], 'k--')
        ax.plot(frange, err_lower[:, j], 'k--')

def plot_acorr(acf):
    """

    """
    # hack
    old_size = mpl.rcParams['font.size']
    mpl.rcParams['font.size'] = 8

    lags, k, k = acf.shapepp
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

class VAR(object):
    """
    Represents a known vector autoregression system

    y_t = \nu + \sum_{i=1}^p A_i y_{t-i} + u_t
    u_t \sim \mathcal{N}(0, \Sigma_u)
    u_t are iid
    """

    def __init__(self, nu, coefs, sig_u):
        self.p = len(coefs)
        self.k = len(coefs[0])

        self.nu = nu
        self.coefs = coefs

        self.sig_u = sig_u

    # def __repr__(self):
    #     pass

    def __str__(self):
        output = ('VAR(%d) process for %d-dimensional response y_t'
                  % (self.p, self.k))
        output += '\nstable: %s' % self.is_stable()
        output += '\nmean: %s' % self.mean()

        return output

    def is_stable(self):
        return is_stable(self.coefs)

    def plotsim(self, steps=1000):
        Y = varsim(self.coefs, self.nu, self.sig_u, steps=steps)
        plot_mts(Y)

    def plotforc(self, y, steps, alpha=0.05):
        """

        """
        lower, mid, upper = self.forecast_interval(y, alpha, steps)
        plot_var_forc(y, mid, lower, upper)

    def mean(self):
        """
        Mean of stable process

        ::

            \mu = (I - A_1 - \dots - A_p)^{-1} \nu
        """
        return solve(np.eye(self.k) - self.coefs.sum(0), self.nu)

    def ma_rep(self, maxn=10):
        """

        """
        return ma_rep(self.coefs, self.nu, maxn=maxn)

    def acf(self, nlags=None):
        return var_acf(self.coefs, self.sig_u, nlags=nlags)

    def acorr(self, nlags=None):
        return _acf_to_acorr(var_acf(self.coefs, self.sig_u, nlags=nlags))

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
        return forecast(y, self.coefs, self.nu, steps)

    def forecast_cov(self, steps):
        """
        Compute forecast error covariance matrices

        Parameters
        ----------


        Returns
        -------
        covs : (steps, k, k)
        """
        return forecast_cov(self.ma_rep(steps), self.sig_u, steps)

    def _forecast_vars(self, steps):
        covs = self.forecast_cov(steps)

        # Take diagonal for each cov
        inds = np.arange(self.k)
        return covs[:, inds, inds]

    def forecast_interval(self, y, alpha, steps):
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
        sigma = self._forecast_vars(steps)

        forc_lower = point_forecast - q * sigma
        forc_upper = point_forecast + q * sigma

        return forc_lower, point_forecast, forc_upper

    def forecast_dyn(self):
        """
        "Forward filter": compute forecasts for each time step
        """

        pass

    def plot_forecast(self, steps, alpha=None):
        """

        """
        pass

    def fevd(self):
        pass

#-------------------------------------------------------------------------------


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
