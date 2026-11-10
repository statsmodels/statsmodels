# -*- coding: utf-8 -*-
"""
symmetric matrix helper functions for use with covariance and correlation
matrices

follows Neudecker and Wesselman 1990
more details in Magnus and Neudecker 1996
some correlation functions based on Staiger and Hakstian 1982

Warning: These are a reference implementation and the matrices can become
vary large. The objective is to replace these functions with efficient
operators or sparse matrices and use the matrix functions as reference and
test cases and for prototyping.



Created on Thu Nov 23 13:39:47 2017

Author: Josef Perktold
"""

import numpy as np

def vec(x):
    """ravel matrix in fortran order (stacking columns)
    """
    return np.ravel(x, order='F')

def vech(x):
    """ravel lower triangular part of matrix in fortran order (stacking columns)

    behavior for arrays with more than 2 dimensions not checked yet
    """
    if x.ndim == 2:
        idx = np.triu_indices_from(x.T)
        return x.T[idx[0], idx[1]] #, x[idx[1], idx[0]]
    elif x.ndim > 2:
        #try ravel last two indices
        #idx = np.triu_indices(x.shape[-2::-1])
        n_rows, n_cols = x.shape[-2:]
        idr, idc = np.array([[i, j] for j in range(n_cols)
                                    for i in range(j, n_rows)]).T
        return x[..., idr, idc]


def veclow(x):
    """ravel lower triangular part of matrix excluding diagonal

    This is the same as vech after dropping diagonal elements

    """
    if x.ndim == 2:
        idx = np.triu_indices_from(x.T, k=1)
        return x.T[idx[0], idx[1]] #, x[idx[1], idx[0]]
    else:
        raise ValueError('x needs to be 2-dimensional')


def vech_cross_product(x0, x1):
    """vectorized cross product with lower triangel

    TODO: this should require symmetry, and maybe x1 = x0, otherwise dropping
    above diagonal might not make sense

    TODO: we also want resorted diagonal, off-diagonal for use with correlation
    i.e. std first and then correlation coefficients.
    """
    n_rows, n_cols = x0.shape[-1], x1.shape[-1]
    idr, idc = np.array([[i, j] for j in range(n_cols)
                                for i in range(j, n_rows)]).T
    return x0[..., idr] * x1[..., idc]


def unvec(x, n_rows, n_cols=None):
    """create matrix from fortran raveled 1-d array
    """
    if n_cols is None:
        n_cols = n_rows

    return x.reshape(n_rows, n_cols, order='F')


def unvech(x, n_rows, n_cols=None):
    if n_cols is None:
        n_cols = n_rows

    #  we use triu but transpose to get fortran ordered tril
    n_rows, n_cols = n_cols, n_cols
    idx = np.triu_indices(n_rows, m=n_cols)
    x_new = np.zeros((n_rows, n_cols), dtype=x.dtype)
    x_new[idx[0], idx[1]] = x
    return x_new.T


def dg(x):
    """create matrix with off-diagonal elements set to zero
    """
    return np.diag(x.diagonal())


def E(i, j, nr, nc):
    """create unit matrix with 1 in (i,j)th element and zero otherwise
    """
    x = np.zeros((nr, nc), np.int64)
    x[i, j] = 1
    return x


def K(n):
    """selection matrix

    symmetric case only
    """
    k = sum(np.kron(E(i, j, n, n), E(i, j, n, n).T)
            for i in range(n) for j in range(n))
    return k

def Ms(n):
    k = K(n)
    return (np.eye(*k.shape) + k) / 2.

def u(i, n):
    """unit vector
    """
    u_ = np.zeros(n, np.int64)
    u_[i] = 1
    return u_


def L(n):
    """elimination matrix
    symmetric case
    """
    # they use 1-based indexing
    # k = sum(u(int(round((j - 1)*n + i - 0.5* j*(j - 1) -1)), n*(n+1)//2)[:, None].dot(vec(E(i, j, n, n))[None, :])
    k = sum(u(int(np.trunc((j)*n + i - 0.5* (j + 1)*(j))), n*(n+1)//2)[:, None].dot(vec(E(i, j, n, n))[None, :])
            for i in range(n) for j in range(i+1))
    return k


def Md_0(n):
    l = L(n)
    ltl = l.T.dot(l)
    k = K(n)
    md = ltl.dot(k).dot(ltl)
    return md


def Md(n):
    """symmetric case
    """
    md = sum(np.kron(E(i, i, n, n), E(i, i, n, n).T)
            for i in range(n))
    return md


def Dup(n):
    """duplication matrix
    """
    l = L(n)
    ltl = l.T.dot(l)
    k = K(n)
    d = l.T + k.dot(l.T) - ltl.dot(k).dot(l.T)
    return d


def ravel_indices(n_rows, n_cols):
    """indices for ravel in fortran order
    """
    ii, jj = np.meshgrid(np.arange(n_rows), np.arange(n_cols))
    return ii.ravel(order='C'), jj.ravel(order='C')


def _cov_cov(cov, nobs, assume='elliptical', kurt=0, data=None):
    """covariance of sample covariance

    TODO: API needs to change because for the general case we need the data
    or the empirical forth moments.

    This returns and estimate of the covariance of the sample covariance
    for one of three possible cases, general without distribution assumption.
    for normal distributed data and for elliptically symmetrically distributed
    data

    Parameters
    ----------
    cov : array_like
        sample covariance matrix
    nobs : int
        number of observations for which sample covariance was computed
    assume : 'elliptical' or 'general'
        Distribution assumption used for the forth moment of the data. If
        assume is 'general', then the sample forth moments are used, if
        assume is 'elliptical', then the forth moments implied by a
        an elliptically symmetric distribution is assumed. The relevant
        parameter in this case is the kurtosis `kurt`. If kurt is zero, than
        a normal distribution of the data is assumed.
    kurt : float
        (excess) kurtosis for the elliptical case. see assume.
    data : array_like
        Only needed if assume is 'general'. The data is used to compute the
        4th moments.

    Returns
    -------
    V : ndarray
       estimate of the covariance of the sample covariance matrix

    """
    cov = np.asarray(cov)
    mom2 = cov # alias
    k_vars = cov.shape[0]
    if assume in ['general', 'g']:
        # distribution free
        data = data - data.mean(0)
        mom4 = 0
        for row in data:
            c = np.outer(row, row)  # outer_product
            mom4 += np.kron(c, c)
        mom4 /= nobs

        V_g = mom4 - np.outer(vec(mom2), vec(mom2))
        V_g = V_g / nobs
        return V_g

    elif assume in ['e', 'elliptical'] and kurt == 0:
        # assuming normality, if this is correct, Neudecker 1996 equ 6
        # for 2 variables

        V_n = (np.eye(k_vars**2) + K(k_vars)).dot(np.kron(mom2, mom2)) / nobs
        #cc1 = 2*(Ms(2)).dot(np.kron(mom2, mom2)) / nobs
        return V_n

    elif assume in ['e', 'elliptical'] and kurt != 0:
        # The normal result can be extended to elliptically symmetric case with
        # dependence on the kurtosis parameter

        k = K(k_vars)
        V_e = (1 + kurt) * ((np.eye(*k.shape) + k)).dot(np.kron(mom2, mom2))
        V_e += kurt * np.outer(vec(mom2), vec(mom2))
        return V_e
    else:
        raise ValueError('`assume` or `kurt` not available')


def cov_cov_data(data, assume='elliptical', kurt=0):
    """
    covariance of sample covariance

    TODO: add estimator for kurt instead of `kurt` if kurt=None or option
    `kurt='mardia' or kurt='univariate'. Does the second work?

        Parameters
    ----------
    cov : array_like
        sample covariance matrix
    assume : 'elliptical' or 'general'
        Distribution assumption used for the forth moment of the data. If
        assume is 'general', then the sample forth moments are used, if
        assume is 'elliptical', then the forth moments implied by a
        an elliptically symmetric distribution is assumed. The relevant
        parameter in this case is the kurtosis `kurt`. If kurt is zero, than
        a normal distribution of the data is assumed.
    kurt : float
        (excess) kurtosis for the elliptical case. see assume.

    Returns
    -------
    V : ndarray
       estimate of the covariance of the sample covariance matrix

    """
    data = np.asarray(data)
    nobs = data.shape[0]

    cov = np.cov(data, rowvar=0, ddof=1)
    v = _cov_cov(cov, nobs, assume=assume, kurt=kurt, data=data)
    return v


def _cov_corr(cov, cov_cov, nobs):
    """covariance of the correlation matrix, general case

    """
    n = cov.shape[0]   # k_vars, n is used in reference
    # Neudecker and Wesselman eq (4.8)
    std = np.sqrt(np.diag(cov))
    std_inv_mat = np.diag(1 / std)
    corr = cov / std[:,None] / std[None, :]
    outer = np.eye(n**2) - Ms(n).dot(np.kron(np.eye(n), corr).dot(Md(n)))
    outer = outer.dot(np.kron(std_inv_mat, std_inv_mat))

    cov_corr_ = outer.dot(cov_cov).dot(outer.T) / nobs
    return cov_corr_


def _cov_corr_elliptical_vech(corr, nobs, kurt=0, drop=False):
    """covariance of the correlation matrix for elliptical distribution

    This returns the covariance matrix of vech.
    In article it is veclow, excluding diagonal
    If drop is True, then we artificially drop down to veclow. This needs
    Duplow.

    """
    n = corr.shape[0]   # k_vars, n is used in reference
    # Neudecker 1996 eq (4), notation Kd is Md

    # Note dropping the outer Dup doesn't work
    outer = np.eye(n**2) - (np.kron(np.eye(n), corr).dot(Md(n)))
    cov_corr_ = outer.dot(np.kron(corr, corr)).dot(outer.T)
    dd = Dup(n)
    if drop:
        dd = dd[:, dd.sum(0) == 2]
    cov_corr_ = (dd.T.dot(cov_corr_).dot(dd)) / 2
    if kurt != 0:
        cov_corr_ *= (1 + kurt)
    return cov_corr_ / nobs


def _cov_cov_vech(cov, nobs):
    """covariance of vech of sample covariance assuming normality

    This is the inverse of the information matrix for the lower triangular
    elements for estimating cov.

    This is a function just to store a formula. It will be better to reuse
    intermediate results.

    """
    k_vars = cov.shape[0]
    d2 = Dup(k_vars)
    d2_pinv = np.linalg.pinv(d2)  # this is D^+
    # from Magnus Neudecker:
    cc_vech = 2 * d2_pinv.dot(np.kron(cov, cov)).dot(d2_pinv.T) / nobs
    return cc_vech


def _gls_cov_vech(data, cov0=None, ddof=0):
    """gls using weight matrix assuming normal distributed data

    Another function just to store some pieces.
    This needs to be extended to linear functions of vech(cov).
    """
    data = np.asarray(data)
    nobs, k_vars = data.shape
    data = data - data.mean(0)
    if cov0 is None:
        cov0 = data.T.dot(data) / (nobs - ddof)

    weights = _cov_cov_vech(cov0, nobs)
    # the following should impose restrictions and use residuals
    mom = vech_cross_product(data, data)
    momcond = mom.mean(0)   # I think this is currently be just vech(cov_sample)
    gls_qf = momcond.dot(weights).dot(momcond)
    return gls_qf


def mom4(data, method='einsum', ddof=0):
    """empirical 4th moment of data, not centered

    To compute the centered moment, subtract the mean from data.

    Parameters
    ----------
    data : array_like
        2-D data with observations in rows and variables in columns
    method : str
        Computational method, mainly used as verification and for testing.
        If method is 'einsum', then numpy.einsum is used. Otherwise, a Python
        loop over all observations is used. Results are identical up to
        floating point precision in the computation.
    ddof : float
        degrees of freedom correction in denominator. Default is ddof=0 and
        the denominator is nobs

    Returns
    -------
    mom4 : ndarray
        2-D array raveled according to vec(cov).

    """
    y = np.asarray(data)
    nobs = y.shape[0]
    if method == 'einsum':
        m4 = np.einsum('ti,tj,tk,th->ijkh', y, y, y, y)
        k_vec = m4.shape[0] ** 2
        m4 = m4.reshape((k_vec, k_vec))
    else:
        m4 = 0
        for row in y:
            c = np.outer(row, row)  # outer_product
            m4 += np.kron(c, c)

    return m4 / (nobs - ddof)


def _mom4_normal(i, j, k, h, cov):
    """Forth moment for normal distribution

    explicit formula Steiger, Hakstian eq (4.1)

    """
    P = cov # shortcut
    Mijkh = P[i, j] * P[k, h] + P[i, k] * P[j, h] + P[i, h] * P[j, k]
    return Mijkh


def _cov_cov_mom4(cov, mom4, nobs):
    """covariance of sample covariance matrix, general distribution case

    Steiger, Hakstian eq (2.12)
    """
    c = mom4 - np.outer(vec(cov), vec(cov))
    return c


def cov_cov_fisherz(corr, cov_cov):
    """covariance of Fisher's Z-transformed correlation matrix

    Steiger, Hakstian eq (4.6)
    """
    r = vec(corr**2)
    c = cov_cov - np.outer(3, r)
    return c


def cov_corr_coef_normal(i, j, k, h, corr):
    """covariance of correlation coefficients for normal distributed data

    explicit formula
    Steiger
    coded from Steiger and Hakstian 1982 eq (4.2)
    """
    P = corr # shorthand
    c = 0.5 * P[i, j] * P[k, h] * (P[i, k]**2 + P[i,h]**2 + P[j, k]**2 +
                                   P[j,h]**2)
    c += P[i, k] * P[j, h] + P[i, h] * P[j, k]
    c -= P[i, j] * (P[j, k] * P[j, h] + P[i, k] * P[i, h])
    c -= P[k, h] * (P[j, k] * P[i, k] + P[j, h] * P[i, h])
    return c


def cov_corr_coef(i, j, k, h, corr, mom4):
    """covariance of correlation coefficients for normal distributed data

    assumes forth is 4-dimensional for indexing
    e.g. mom4 = np.einsum('ti,tj,tk,th->ijkh', y, y, y, y) / nobs

    explicit formula
    Steiger
    coded from Steiger and Hakstian 1982 eq (3.4)

    needs to be divided by nobs or df
    """
    P = corr # shorthand
    M = mom4

    c = M[i, j, k, h] + 0.25 * P[i, j] * P[k, h] * (
            M[i, i, k, k] + M[j, j, k, k,] + M[i,i,h,h] + M[j, j, h, h] )
    c -= 0.5 * P[i, j] * (M[i, i, k, h] + M[j, j, k, h])
    c -= 0.5 * P[k, h] * (M[i, j, k, k] + M[i, j, h, h])
    return c
