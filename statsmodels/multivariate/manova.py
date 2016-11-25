# -*- coding: utf-8 -*-

"""Multivariate analysis of variance

author: Yichuan Liu
"""
from __future__ import print_function, division

from statsmodels.base.model import Model
import numpy as np
from numpy.linalg import eigvals, inv, pinv, matrix_rank
from scipy import stats
import pandas as pd


def fit_manova(X, Y):
    """
    MANOVA fitting Y = X * B
    where Y is dependent variables, X is independent variables

    Parameters
    ----------
    Y : array-like, each column is a dependent variable
    X : array-like, each column is a independent variable

    Returns
    -------
    a tuple of matrices or values necessary for hypothesis testing

    """
    n_sample, n_y_vars = Y.shape
    n_sample_x, n_x_vars = X.shape
    if n_sample != n_sample_x:
        raise ValueError('X(n=%d) and Y(n=%d) should have the same number of '
                         'rows!' % (n_sample_x, n_sample))

    # Calculate the matrices necessary for hypothesis testing
    df_resid = X.shape[0] - X.shape[1]

    # Regression coefficients
    B = pinv(X).dot(Y)

    # inverse of X'X
    inv_cov = inv(X.T.dot(X))

    # Y'Y - (XB)'XB
    t = X.dot(B)
    YYBXXB = np.subtract(Y.T.dot(Y), t.T.dot(t))
    return (B, df_resid, inv_cov, YYBXXB)


def test_manova(fit_output, contrast_L, transform_M=None):
    """
    MANOVA hypothesis testing

    For Y = X * B, where Y is dependent variables, X is independent variables
    testing L * B * M = 0 where L is the contast matrix for hypothesis testing
    and M is the transformation matrix for transforming the dependent variables
    in Y.

    Testing is based on forming the following matrices:
        H = M'(LB)'(L(X'X)^L')^(LB)M   (`^` denotes inverse)
        E = M'(Y'Y - B'(X'X)B)M
    And then solving the eigenvalues of (E + H)^ * H

    .. [1] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/
default/viewer.htm#statug_introreg_sect012.htm

    Parameters
    ----------
    fit_output : tuple
        Output of ``fit_manova``

    contrast_L : array-like, at least 1 row (1 by n_x_vars+1 or 1 by n_x_vars)
        Hypothesis to be tested. IV in columns, sub-hypothesis in rows.
        First column is intercept if .
    transform_M : array-like
        Transform matrix. Default to be n_y_vars by n_y_vars identity
        matrix (i.e. do not transform Y matrix).
    contrast_L: array

    transform_M: 2-D array of size n_y_vars by n_y_vars
        th

    Returns
    -------
    results : DataFrame
        MANOVA table

    """
    # H = M'(LB)'(L(X'X)^L')^(LB)M   (`^` denotes inverse)
    # E = M'(Y'Y - B'(X'X)B)M
    B, df_resid, inv_cov, YYBXXB = fit_output
    M = transform_M
    if M is None:
        M = np.eye(B.shape[1])
    L = contrast_L
    v = df_resid
    # t1 = (LB)M
    t1 = L.dot(B).dot(M)

    # H = t1'L(X'X)^L't1
    t2 = L.dot(inv_cov).dot(L.T)
    q = matrix_rank(t2)
    H = t1.T.dot(inv(t2)).dot(t1)

    # E = M'(Y'Y - B'(X'X)B)M
    E = M.T.dot(YYBXXB).dot(M)

    EH = np.add(E, H)
    p = matrix_rank(EH)

    # eigenvalues of (E + H)^H
    eigv2 = np.sort(eigvals(inv(EH).dot(H)))

    # eigenvalues of (E+H)^H
    eigv1 = np.array([i / (1 - i) for i in eigv2])

    s = np.min([p, q])
    m = (np.abs(p - q) - 1) / 2
    n = (v - p - 1) / 2

    results = pd.DataFrame({'value': [], 'F Value': [], 'Num DF': [],
                            'Den DF': [], 'Pr > F': []})

    def fn(x):
        return np.real([x])[0]

    results.loc["Wilks’ lambda", 'value'] = fn(np.prod(1 - eigv2))

    results.loc["Pillai’s trace", 'value'] = fn(eigv2.sum())

    results.loc["Hotelling-Lawley trace", 'value'] = fn(eigv1.sum())

    results.loc["Roy’s greatest root", 'value'] = fn(eigv1.max())

    r = v - (p - q + 1)/2
    u = (p*q - 2) / 4
    df1 = p * q
    if p*p + q*q - 5 > 0:
        t = np.sqrt((p*p*q*q - 4) / (p*p + q*q - 5))
    else:
        t = 1
    df2 = r*t - 2*u
    lmd = results.loc["Wilks’ lambda", 'value']
    lmd = np.power(lmd, 1 / t)
    F = (1 - lmd) / lmd * df2 / df1
    results.loc["Wilks’ lambda", 'Num DF'] = df1
    results.loc["Wilks’ lambda", 'Den DF'] = df2
    results.loc["Wilks’ lambda", 'F Value'] = F
    pval = stats.f.sf(F, df1, df2)
    results.loc["Wilks’ lambda", 'Pr > F'] = pval

    V = results.loc["Pillai’s trace", 'value']
    df1 = s * (2*m + s + 1)
    df2 = s * (2*n + s + 1)
    F = df2 / df1 * V / (s - V)
    results.loc["Pillai’s trace", 'Num DF'] = df1
    results.loc["Pillai’s trace", 'Den DF'] = df2
    results.loc["Pillai’s trace", 'F Value'] = F
    pval = stats.f.sf(F, df1, df2)
    results.loc["Pillai’s trace", 'Pr > F'] = pval

    U = results.loc["Hotelling-Lawley trace", 'value']
    if n > 0:
        b = (p + 2*n) * (q + 2*n) / 2 / (2*n + 1) / (n - 1)
        df1 = p * q
        df2 = 4 + (p*q + 2) / (b - 1)
        c = (df2 - 2) / 2 / n
        F = df2 / df1 * U / c
    else:
        df1 = s * (2*m + s + 1)
        df2 = s * (s*n + 1)
        F = df2 / df1 / s * U
    results.loc["Hotelling-Lawley trace", 'Num DF'] = df1
    results.loc["Hotelling-Lawley trace", 'Den DF'] = df2
    results.loc["Hotelling-Lawley trace", 'F Value'] = F
    pval = stats.f.sf(F, df1, df2)
    results.loc["Hotelling-Lawley trace", 'Pr > F'] = pval

    sigma = results.loc["Roy’s greatest root", 'value']
    r = np.max([p, q])
    df1 = r
    df2 = v - r + q
    F = df2 / df1 * sigma
    results.loc["Roy’s greatest root", 'Num DF'] = df1
    results.loc["Roy’s greatest root", 'Den DF'] = df2
    results.loc["Roy’s greatest root", 'F Value'] = F
    pval = stats.f.sf(F, df1, df2)
    results.loc["Roy’s greatest root", 'Pr > F'] = pval
    return results.iloc[:, [4, 2, 0, 1, 3]]


class MANOVA(Model):
    """
    Multivariate analysis of variance


    Parameters
    ----------
    endog : array-like
        Dependent variables (DV). A n_sample x n_y_var array where n_sample is
        the number of observations and n_y_var is the number of DV.

    exog : array-like
        Independent variables (IV). A n_sample x n_x_var array where n is the
        number of observations and n_x_var is the number of IV. An intercept is
        not included by default and should be added by the user (models
        specified using a formula include an intercept by default)

    Attributes
    -----------
    df_resid : float
        The number of observation `n` minus the number of IV `q`.

    endog : array
        See Parameters.

    exog : array
        See Parameters.

    design_info : patsy.DesignInfo
        Contain design info for the independent variables if model is
        constructed using `from_formula`

    """
    def __init__(self, endog, exog, design_info=None, **kwargs):
        Y, X = endog, exog
        self.design_info = design_info
        out = fit_manova(X, Y)
        self.reg_coeffs, self.df_resid, self.inv_cov_, self.YYBXXB_ = out
        super(MANOVA, self).__init__(endog, exog)

    @classmethod
    def from_formula(cls, formula, data, subset=None, drop_cols=None,
                     tests=None, *args, **kwargs):
        mod = super(MANOVA, cls).from_formula(formula, data,
                                              subset=subset,
                                              drop_cols=drop_cols,
                                              tests=tests,
                                              *args, **kwargs)
        terms = mod.design_info.term_name_slices
        hypothesis = []
        for key in terms:
            L_contrast = np.eye(mod.exog.shape[1])[terms[key], :]
            hypothesis.append((key, L_contrast, None))
        return mod, hypothesis

    def test(self, H=None):
        """
        Testing the genernal hypothesis
            L * B * M = 0
        for each tuple (name, L, M) in `H` where B is the coefficient matrix
        for the linear model Y = X * B

        Parameters
        ----------
        H: A list of tuples
           Hypothesis to be tested. Each element is a tuple (name, L, M)
           containing a string `name`, the contrast matrix L and the transform
           matrix M for transforming dependent variables, respectively. If M is
           `None`, it would be set to an identity matrix (i.e. no dependent
           variable transformation).
           If `H` is None, the effect of each independent variable on the
           dependent variables will be tested.

        Returns
        -------
        results: a list of tuples (name, manova_table) for each tuple in `H`
            MANOVA results

        """
        if H is None:
            H = []
            for i in range(self.exog.shape[1]):
                name = 'x%d' % (i)
                L = np.zeros([1, self.exog.shape[1]])
                L[i] = 1
                H.append([name, L, None])

        results = []
        for name, L, M in H:
            fit_output = (self.reg_coeffs, self.df_resid, self.inv_cov_,
                          self.YYBXXB_)
            manova_table = test_manova(fit_output, L, M)
            results.append((name, manova_table))
        return results
