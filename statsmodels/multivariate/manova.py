# -*- coding: utf-8 -*-

"""Multivariate analysis of variance

author: Yichuan Liu
"""
from __future__ import print_function, division

from statsmodels.base.model import Model
import numpy as np
from numpy.linalg import matrix_rank, qr
from statsmodels.iolib import summary2
from .multivariate_ols import _multivariate_htest


def manova_fit(x, y):
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

    .. [1] ftp://public.dhe.ibm.com/software/analytics/spss/documentation/statistics/20.0/en/client/Manuals/IBM_SPSS_Statistics_Algorithms.pdf

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
    fittedvalues = u
    return (df_resid, fittedvalues, sscpr)


def manova_htest(hypothesis, fittedvalues, exog_names,
                            endog_names):
    """
    MANOVA hypothesis testing

    For y = x * params, where y is dependent variables and x is independent
    variables, testing L * params * M = 0 where L is the contast matrix for
    hypothesis testing and M is the transformation matrix for transforming the
    dependent variables in y.

    Algorithm:
        H = (L * u * M)' * (L * u * M)
        E = M' * sscpr * M

    .. [1] ftp://public.dhe.ibm.com/software/analytics/spss/documentation/statistics/20.0/en/client/Manuals/IBM_SPSS_Statistics_Algorithms.pdf

    Parameters
    ----------
    hypothesis: A list of array-like
        Hypothesis to be tested. Each element is an array-like
                            [name, contrast_L, transform_M]
        containing a string `name`, the contrast matrix L and the transform
        matrix M (for transforming dependent variables), respectively.
        contrast_L : array-like or an array of strings
           Contrast matrix for hypothesis testing.
           If array-like, each row is an hypothesis and each column is an
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
        If `hypothesis` is None: 1) the effect of each independent variable
        on the dependent variables will be tested. Or 2) if model is
        created using a formula,  `hypothesis` will be created according to
        `design_info`. 1) and 2) is equivalent if no additional variables
        are created by the formula (e.g. dummy variables for categorical
        variables and interaction terms)
    fittedvalues : tuple
        Output of ``fit_manova``
    exog_names : a list of string
    endog_names : a list of string

    Returns
    -------
    a dict of manova tables
    """
    def fn(L, M):
        df_resid, u, sscpr = fittedvalues
        t1 = L.dot(u)
        q = matrix_rank(t1)
        t1 = t1.dot(M)
        H = t1.T.dot(t1)

        # E = M'(Y'Y - B'(X'X)B)M
        E = M.T.dot(sscpr).dot(M)
        return E, H, q, df_resid

    return _multivariate_htest(hypothesis, exog_names, endog_names, fn)


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

    .. [1] ftp://public.dhe.ibm.com/software/analytics/spss/documentation/statistics/20.0/en/client/Manuals/IBM_SPSS_Statistics_Algorithms.pdf

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
        super(MANOVA, self).__init__(endog, exog)
        self.fittedmod = manova_fit(self.exog, self.endog)

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

        results = manova_htest(hypothesis, self.fittedmod,self.exog_names,
                               self.endog_names)

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
        return self.results[item]['stat']

    def summary(self):
        summ = summary2.Summary()
        summ.add_title('MANOVA results')
        for key in self.results:
            summ.add_dict({'Effect':key})
            summ.add_df(self.results[key]['stat'])
        return summ

