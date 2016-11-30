# -*- coding: utf-8 -*-

"""Multivariate analysis of variance

author: Yichuan Liu
"""
from __future__ import print_function, division

from statsmodels.base.model import Model
import numpy as np
from numpy.linalg import eigvals, inv, solve, matrix_rank, qr
from scipy import stats
import pandas as pd
from statsmodels.iolib import summary2


def fit_manova(x, y):
    """
    For a MANOVA problem y = x * params
    where y is dependent variables, x is independent variables
    Perform QR decomposition of x to calculate the sums of squares
    and cross-products (SSCP) of residuals
    These are necessary for performing MANOVA hypothesis tests

    No acture

    Parameters
    ----------
    x : array-like, each column is a independent variable
    y : array-like, each column is a dependent variable

    Returns
    -------
    a tuple of matrices or values necessary for hypothesis testing

    .. [1] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introreg_sect012.htm
    .. [2] ftp://public.dhe.ibm.com/software/analytics/spss/documentation/statistics/20.0/en/client/Manuals/IBM_SPSS_Statistics_Algorithms.pdf

    """
    nobs, k_endog = y.shape
    nobs1, k_exog= x.shape
    if nobs != nobs1:
        raise ValueError('x(n=%d) and y(n=%d) should have the same number of '
                         'rows!' % (nobs1, nobs))

    # Calculate the matrices necessary for hypothesis testing
    df_resid = nobs - k_exog

    q, r = qr(x)
    u = q.T.dot(y)
    sscpr = np.subtract(y.T.dot(y), u.T.dot(u))
    return (df_resid, u, sscpr)


def multivariate_stats(eigenvals, p, q, df_resid):
    """
    Testing MANOVA statistics

    Parameters
    ----------
    eigenvals : array
        The eigenvalues of (E + H)^H matrix where `^` denote inverse
    p : int
        Rank of E + H
    q : int
        Rank of X
    df_resid
        Residual degree of freedom (n_samples minus n_variables of X)

    Returns
    -------

    .. [1] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introreg_sect012.htm
    .. [2] ftp://public.dhe.ibm.com/software/analytics/spss/documentation/statistics/20.0/en/client/Manuals/IBM_SPSS_Statistics_Algorithms.pdf
    """
    eigv2 = eigenvals
    eigv1 = np.array([i / (1 - i) for i in eigv2])
    v = df_resid

    s = np.min([p, q])
    m = (np.abs(p - q) - 1) / 2
    n = (v - p - 1) / 2

    results = pd.DataFrame({'Value': [], 'F Value': [], 'Num DF': [],
                            'Den DF': [], 'Pr > F': []})

    def fn(x):
        return np.real([x])[0]

    results.loc["Wilks’ lambda", 'Value'] = fn(np.prod(1 - eigv2))

    results.loc["Pillai’s trace", 'Value'] = fn(eigv2.sum())

    results.loc["Hotelling-Lawley trace", 'Value'] = fn(eigv1.sum())

    results.loc["Roy’s greatest root", 'Value'] = fn(eigv1.max())

    r = v - (p - q + 1)/2
    u = (p*q - 2) / 4
    df1 = p * q
    if p*p + q*q - 5 > 0:
        t = np.sqrt((p*p*q*q - 4) / (p*p + q*q - 5))
    else:
        t = 1
    df2 = r*t - 2*u
    lmd = results.loc["Wilks’ lambda", 'Value']
    lmd = np.power(lmd, 1 / t)
    F = (1 - lmd) / lmd * df2 / df1
    results.loc["Wilks’ lambda", 'Num DF'] = df1
    results.loc["Wilks’ lambda", 'Den DF'] = df2
    results.loc["Wilks’ lambda", 'F Value'] = F
    pval = stats.f.sf(F, df1, df2)
    results.loc["Wilks’ lambda", 'Pr > F'] = pval

    V = results.loc["Pillai’s trace", 'Value']
    df1 = s * (2*m + s + 1)
    df2 = s * (2*n + s + 1)
    F = df2 / df1 * V / (s - V)
    results.loc["Pillai’s trace", 'Num DF'] = df1
    results.loc["Pillai’s trace", 'Den DF'] = df2
    results.loc["Pillai’s trace", 'F Value'] = F
    pval = stats.f.sf(F, df1, df2)
    results.loc["Pillai’s trace", 'Pr > F'] = pval

    U = results.loc["Hotelling-Lawley trace", 'Value']
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

    sigma = results.loc["Roy’s greatest root", 'Value']
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


def test_manova(results, contrast_L, transform_M=None):
    """
    MANOVA hypothesis testing

    For y = x * params, where y is dependent variables and x is independent
    variables, testing L * params * M = 0 where L is the contast matrix for
    hypothesis testing and M is the transformation matrix for transforming the
    dependent variables in y.

    Algorithm:
        H = (L * u * M)' * (L * u * M)
        E = M' * sscpr * M
        solve (H + E) * T = H
    And then solving the eigenvalues of T * H

    .. [1] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introreg_sect012.htm
    .. [2] ftp://public.dhe.ibm.com/software/analytics/spss/documentation/statistics/20.0/en/client/Manuals/IBM_SPSS_Statistics_Algorithms.pdf

    Parameters
    ----------
    results : tuple
        Output of ``fit_manova``
    contrast_L : array-like
        Contrast matrix for hypothesis testing. Each row is an hypothesis and
        each column is an independent variable.
        At least 1 row (1 by k_exog, the number of independent variables)
    transform_M : array-like
        Transform matrix. Default to be k_endog by k_endog identity
        matrix (i.e. do not transform y matrix).

    Returns
    -------
    results : MANOVAResults

    """
    df_resid, u, sscpr = results
    M = transform_M
    if M is None:
        M = np.eye(u.shape[1])
    L = contrast_L
    t1 = L .dot(u)
    q = matrix_rank(t1)
    t1 = t1.dot(M)
    H = t1.T.dot(t1)

    # E = M'(Y'Y - B'(X'X)B)M
    E = M.T.dot(sscpr).dot(M)

    EH = np.add(E, H)
    p = matrix_rank(EH)

    # eigenvalues of (E + H)^H
    eigv2 = np.sort(eigvals(solve(EH, H)))
    return multivariate_stats(eigv2, p, q, df_resid)


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
    sscpr : array
        Sums of squares and cross-products of residuals
    endog : array
        See Parameters.
    exog : array
        See Parameters.
    design_info : patsy.DesignInfo
        Contain design info for the independent variables if model is
        constructed using `from_formula`

    """
    def __init__(self, endog, exog, design_info=None, **kwargs):
        self.design_info = design_info
        out = fit_manova(exog, endog)
        self.df_resid, self.u_, self.sscpr = out
        super(MANOVA, self).__init__(endog, exog)

    @classmethod
    def from_formula(cls, formula, data, subset=None, drop_cols=None,
                     *args, **kwargs):
        mod = super(MANOVA, cls).from_formula(formula, data,
                                              subset=subset,
                                              drop_cols=drop_cols,
                                              *args, **kwargs)
        return mod

    def test(self, hypothesis=None):
        """
        Testing the genernal hypothesis
            L * params * M = 0
        where `params` is the regression coefficient matrix for the
        linear model y = x * params

        Parameters
        ----------
        hypothesis: A list of array-like
           Hypothesis to be tested. Each element is an array-like [name, L, M]
           containing a string `name`, the contrast matrix L and the transform
           matrix M for transforming dependent variables, respectively. If M is
           `None`, it is set to an identity matrix (i.e. no dependent
           variable transformation).
           If `hypothesis` is None: 1) the effect of each independent variable
           on the dependent variables will be tested. Or 2) if model is created
           using a formula,  `hypothesis` will be created according to
           `design_info`. 1) and 2) is equivalent if no additional variables
           are created by the formula (e.g. dummy variables for categorical
           variables and interaction terms)

        Returns
        -------
        results: MANOVAResults

        """
        if hypothesis is None:
            if self.design_info is not None:
                terms = self.design_info.term_name_slices
                hypothesis = []
                for key in terms:
                    L_contrast = np.eye(self.exog.shape[1])[terms[key], :]
                    hypothesis.append([key, L_contrast, None])
            else:
                hypothesis = []
                for i in range(self.exog.shape[1]):
                    name = 'x%d' % (i)
                    L = np.zeros([1, self.exog.shape[1]])
                    L[i] = 1
                    hypothesis.append([name, L, None])

        results = []
        self.hypothesis = hypothesis
        for name, L, M in hypothesis:
            if len(L.shape) != 2:
                raise ValueError('Contrast matrix L must be a 2-d array!')
            if L.shape[1] != self.exog.shape[1]:
                raise ValueError('Contrast matrix L should have the same '
                                 'number of columns as exog! %d != %d' %
                                 (L.shape[1], self.exog.shape[1]))
            if M is not None:
                if len(M.shape) != 2:
                    raise ValueError('Transform matrix M must be a 2-d array!')
                if M.shape[0] != self.endog.shape[1]:
                    raise ValueError('Transform matrix M should have the same '
                                     'number of rows as the number of columns '
                                     'of endog! %d != %d' %
                                     (M.shape[0], self.exog.shape[1]))
            fit_output = (self.df_resid, self.u_, self.sscpr)
            manova_table = test_manova(fit_output, L, M)
            results.append((name, manova_table))
        return MANOVAResults(results)


class MANOVAResults(object):
    """
    MANOVA results class

    Can be accessed as a list, each element containing a tuple (name, df) where
    `name` is the effect (i.e. term in model) name and `df` is a DataFrame
    containing the MANOVA test statistics

    """
    def __init__(self, results):
        self.results = results

    def __str__(self):
        return self.summary().__str__()

    def __getitem__(self, item):
        return self.results[item]

    def summary(self):
        summ = summary2.Summary()
        summ.add_title('MANOVA results')
        for h in self.results:
            summ.add_dict({'Effect':h[0]})
            summ.add_df(h[1])
        return summ


