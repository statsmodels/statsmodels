# -*- coding: utf-8 -*-

"""General linear model

author: Yichuan Liu
"""
from __future__ import print_function, division

import numpy as np
from numpy.linalg import eigvals, inv, solve, matrix_rank, pinv, svd
from scipy import stats
import pandas as pd
from patsy import DesignInfo
from statsmodels.compat import string_types

from statsmodels.base.model import Model
from statsmodels.iolib import summary2


def _multivariate_ols_fit(x, y, method='svd', tolerance=1e-8):
    """
    Solve multivariate linear model y = x * params
    where y is dependent variables, x is independent variables

    No acture

    Parameters
    ----------
    x : array-like
        each column is a independent variable
    y : array-like
        each column is a dependent variable
    method : string
        'svd' - Singular value decomposition
        'pinv' - Moore-Penrose pseudoinverse
    tolerance : float, a small positive number
        Tolerance for eigenvalue. Values smaller than tolerance is considered
        zero.
    Returns
    -------
    a tuple of matrices or values necessary for hypotheses testing

    .. [1] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introreg_sect012.htm

    """
    nobs, k_endog = y.shape
    nobs1, k_exog= x.shape
    if nobs != nobs1:
        raise ValueError('x(n=%d) and y(n=%d) should have the same number of '
                         'rows!' % (nobs1, nobs))

    # Calculate the matrices necessary for hypotheses testing
    df_resid = nobs - k_exog
    if method == 'pinv':
        # Regression coefficients matrix
        pinv_x = pinv(x)
        params = pinv_x.dot(y)

        # inverse of x'x
        inv_cov = pinv_x.dot(pinv_x.T)
        if matrix_rank(inv_cov,tol=tolerance) < k_exog:
            raise ValueError('Covariance of x singular!')

        # Sums of squares and cross-products of residuals
        # Y'Y - (X * params)'B * params
        t = x.dot(params)
        sscpr = np.subtract(y.T.dot(y), t.T.dot(t))
        return (params, df_resid, inv_cov, sscpr)
    elif method == 'svd':
        u, s, v = svd(x, 0)
        if (s > tolerance).sum() < len(s):
            raise ValueError('Covariance of x singular!')
        invs = 1. / s

        params = v.T.dot(np.diag(invs)).dot(u.T).dot(y)
        inv_cov = v.T.dot(np.diag(np.power(invs, 2))).dot(v)
        t = np.diag(s).dot(v).dot(params)
        sscpr = np.subtract(y.T.dot(y), t.T.dot(t))
        return (params, df_resid, inv_cov, sscpr)
    else:
        raise ValueError('%s is not a supported method!' % method)


def multivariate_stats(eigenvals,
                       r_err_sscp,
                       r_contrast, df_resid, tolerance=1e-8):
    """
    For multivariate linear model Y = X * B
    Testing hypotheses
        L*B*M = 0
    where L is contrast matrix, B is the parameters of the
    multivariate linear model and M is dependent variable transform matrix.
        T = L*inv(X'X)*L'
        H = M'B'L'*inv(T)*LBM
        E =  M'(Y'Y - B'X'XB)M

    Parameters
    ----------
    eigenvals : array
        The eigenvalues of inv(E + H)*H
    r_err_sscp : int
        Rank of E + H
    r_contrast : int
        Rank of T matrix
    df_resid : int
        Residual degree of freedom (n_samples minus n_variables of X)
    tolerance : float
        smaller than which eigenvalue is considered 0

    Returns
    -------

    .. [1] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introreg_sect012.htm
    """
    v = df_resid
    p = r_err_sscp
    q = r_contrast
    s = np.min([p, q])
    ind = eigenvals > tolerance
    n_e = ind.sum()
    eigv2 = eigenvals[ind]
    eigv1 = np.array([i / (1 - i) for i in eigv2])
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


def _multivariate_ols_test(hypotheses, fit_results, exog_names,
                            endog_names):
    def fn(L, M):
        # .. [1] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introreg_sect012.htm
        params, df_resid, inv_cov, sscpr = fit_results
        # t1 = (L * params)M
        t1 = L.dot(params).dot(M)
        # H = t1'L(X'X)^L't1
        t2 = L.dot(inv_cov).dot(L.T)
        q = matrix_rank(t2)
        H = t1.T.dot(inv(t2)).dot(t1)

        # E = M'(Y'Y - B'(X'X)B)M
        E = M.T.dot(sscpr).dot(M)
        return E, H, q, df_resid

    return _multivariate_test(hypotheses, exog_names, endog_names, fn)


def _multivariate_test(hypotheses, exog_names, endog_names, fn):
    """
    Multivariate linear model hypotheses testing

    For y = x * params, where y are the dependent variables and x are the
    independent variables, testing L * params * M = 0 where L is the contrast
    matrix for hypotheses testing and M is the transformation matrix for
    transforming the dependent variables in y.

    Algorithm:
        T = L*inv(X'X)*L'
        H = M'B'L'*inv(T)*LBM
        E =  M'(Y'Y - B'X'XB)M
    And then finding the eigenvalues of inv(H + E)*H

    .. [1] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introreg_sect012.htm

    Parameters
    ----------
    hypotheses: A list of tuples
        Hypothesis to be tested. Each element is an array-like
                            (name, contrast_L, transform_M)
        containing a string `name`, the contrast matrix L and the transform
        matrix M (for transforming dependent variables), respectively.
        contrast_L : array-like or an array of strings
           Contrast matrix for hypotheses testing.
           If array-like, each row is an hypotheses and each column is an
           independent variable. At least 1 row
           (1 by k_exog, the number of independent variables) is required.
           If an array of strings, it will be passed to
           patsy.DesignInfo().linear_constraint.
        transform_M : array-like or an array of strings
            Transform matrix.
            Default to be k_endog by k_endog identity matrix
            (i.e. do not transform y matrix).
           If an array of strings, it will be passed to
           patsy.DesignInfo().linear_constraint.
        If `hypotheses` is None: 1) the effect of each independent variable
        on the dependent variables will be tested. Or 2) if model is created
        using a formula,  `hypotheses` will be created according to
        `design_info`. 1) and 2) is equivalent if no additional variables
        are created by the formula (e.g. dummy variables for categorical
        variables and interaction terms)
    k_xvar : int
        The number of independent variables
    k_yvar : int
        The number of dependent variables
    fn : function
        a function fn(contrast_L, transform_M) that returns E, H, q, df_resid
        where q is the rank of T matrix


    Returns
    -------
    results : MANOVAResults

    """
    k_xvar = len(exog_names)
    k_yvar = len(endog_names)
    results = {}
    for name, L, M in hypotheses:
        if any(isinstance(i, string_types) for i in L):
            L = DesignInfo(exog_names).linear_constraint(L).coefs
        else:
            if len(L.shape) != 2:
                raise ValueError('Contrast matrix L must be a 2-d array!')
            if L.shape[1] != k_xvar:
                raise ValueError('Contrast matrix L should have the same '
                                 'number of columns as exog! %d != %d' %
                                 (L.shape[1], k_xvar))
        if M is None:
            M = np.eye(k_yvar)
        elif any(isinstance(i, string_types) for i in M):
            M = DesignInfo(endog_names).linear_constraint(M).coefs.T
        else:
            if M is not None:
                if len(M.shape) != 2:
                    raise ValueError('Transform matrix M must be a 2-d array!')
                if M.shape[0] != k_yvar:
                    raise ValueError('Transform matrix M should have the same '
                                     'number of rows as the number of columns '
                                     'of endog! %d != %d' %
                                     (M.shape[0], k_yvar))
        E, H, q, df_resid = fn(L, M)
        EH = np.add(E, H)
        p = matrix_rank(EH)

        # eigenvalues of (E + H)^H
        eigv2 = np.sort(eigvals(solve(EH, H)))
        stat_table = multivariate_stats(eigv2, p, q, df_resid)

        results[name] = {'stat':stat_table, 'contrast_L':L,
                         'transform_M':M}
    return results


class _MultivariateOLS(Model):
    """
    Multivariate linear model via least squares


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
    def __init__(self, endog, exog, missing='none', hasconst=None, **kwargs):
        super(_MultivariateOLS, self).__init__(endog, exog, **kwargs)

    def fit(self, method='svd'):
        self.fittedmod = _multivariate_ols_fit(
            self.exog, self.endog, method=method)
        return _MultivariateOLSResults(self)


class _MultivariateOLSResults(object):
    """
    _MultivariateOLS results class

    Can be accessed as a list, each element containing a tuple (name, df) where
    `name` is the effect (i.e. term in model) name and `df` is a DataFrame
    containing the test statistics

    """
    def __init__(self, fitted_mv_ols):
        self.design_info = fitted_mv_ols.data.design_info
        self.exog_names = fitted_mv_ols.exog_names
        self.endog_names = fitted_mv_ols.endog_names
        self.fittedmod = fitted_mv_ols.fittedmod

    def __str__(self):
        return self.summary().__str__()

    def __getitem__(self, item):
        return self.results[item]

    def mv_test(self, hypotheses=None):
        """
        Testing the linear hypotheses
            L * params * M = 0
        where `params` is the regression coefficient matrix for the
        linear model y = x * params, `M` is a dependent variable transform
        matrix.

        Parameters
        ----------
        hypotheses: A list of array-like
            Hypotheses to be tested. Each element is an array-like
                                [name, contrast_L, transform_M]
            containing a string `name`, the contrast matrix L and the transform
            matrix M (for transforming dependent variables), respectively.
            contrast_L : array-like or an array of strings
               Contrast matrix for hypotheses testing.
               If array-like, each row is an hypotheses and each column is an
               independent variable. At least 1 row
               (1 by k_exog, the number of independent variables) is required.
               If an array of strings, it will be passed to
               patsy.DesignInfo().linear_constraint.
            transform_M : array-like or an array of strings
                Transform matrix.
                Default to be k_endog by k_endog identity matrix
                (i.e. do not transform y matrix).
               If an array of strings, it will be passed to
               patsy.DesignInfo().linear_constraint.
            If `hypotheses` is None: 1) the effect of each independent variable
            on the dependent variables will be tested. Or 2) if model is
            created using a formula,  `hypotheses` will be created according to
            `design_info`. 1) and 2) is equivalent if no additional variables
            are created by the formula (e.g. dummy variables for categorical
            variables and interaction terms)

        Returns
        -------
        results: _MultivariateOLSResults

        """
        k_xvar = len(self.exog_names)
        if hypotheses is None:
            if self.design_info is not None:
                terms = self.design_info.term_name_slices
                hypotheses = []
                for key in terms:
                    L_contrast = np.eye(k_xvar)[terms[key], :]
                    hypotheses.append([key, L_contrast, None])
            else:
                hypotheses = []
                for i in range(k_xvar):
                    name = 'x%d' % (i)
                    L = np.zeros([1, k_xvar])
                    L[i] = 1
                    hypotheses.append([name, L, None])

        results = _multivariate_ols_test(hypotheses, self.fittedmod,
                                          self.exog_names, self.endog_names)

        return _MultivariateTestResults(results)

    def summary(self):
        raise NotImplementedError


class _MultivariateTestResults(object):
    def __init__(self, mv_test_df):
        self.results = mv_test_df

    def __str__(self):
        return self.summary().__str__()

    def __getitem__(self, item):
        return self.results[item]

    def summary(self, contrast_L=False, transform_M=False):
        summ = summary2.Summary()
        summ.add_title('Multivariate linear model')
        for key in self.results:
            summ.add_dict({'':''})
            df = self.results[key]['stat'].copy()
            df = df.reset_index()
            c = df.columns.values
            c[0] = key
            df.columns = c
            df.index = ['', '', '', '']
            summ.add_df(df)
            if contrast_L:
                summ.add_dict({key:' contrast L='})
                df = pd.DataFrame(self.results[key]['contrast_L'],
                                  columns=self.exog_names)
                summ.add_df(df)
            if transform_M:
                summ.add_dict({key:' transform M='})
                df = pd.DataFrame(self.results[key]['transform_M'],
                                  index=self.endog_names)
                print(df)
                summ.add_df(df)
        return summ
