# -*- coding: utf-8 -*-

"""Multivariate analysis of variance

author: Yichuan Liu
"""
from __future__ import print_function, division

from statsmodels.base.model import Model
import numpy as np
from numpy.linalg import matrix_rank, qr
from statsmodels.iolib import summary2
from .multivariate_ols import _multivariate_test, _hypotheses_doc
__docformat__ = 'restructuredtext en'


def _manova_fit(x, y):
    """
    For a MANOVA problem y = x * params
    where y is dependent variables, x is independent variables
    Perform QR decomposition of x to calculate the sums of squares
    and cross-products (SSCP) of residuals
    These are necessary for performing MANOVA hypotheses tests

    No acture

    Parameters
    ----------
    x : array-like, each column is a independent variable
    y : array-like, each column is a dependent variable

    Returns
    -------
    a tuple of matrices or values necessary for hypotheses testing

    .. [1] ftp://public.dhe.ibm.com/software/analytics/spss/documentation/statistics/20.0/en/client/Manuals/IBM_SPSS_Statistics_Algorithms.pdf

    """
    nobs, k_endog = y.shape
    nobs1, k_exog= x.shape
    if nobs != nobs1:
        raise ValueError('x(n=%d) and y(n=%d) should have the same number of '
                         'rows!' % (nobs1, nobs))

    # Calculate the matrices necessary for hypotheses testing
    df_resid = nobs - k_exog

    q, r = qr(x)
    u = q.T.dot(y)
    sscpr = np.subtract(y.T.dot(y), u.T.dot(u))
    fittedvalues = u
    return (df_resid, fittedvalues, sscpr)


def _manova_test(hypotheses, fit_results, exog_names, endog_names):
    def fn(L, M):
        df_resid, u, sscpr = fit_results
        t1 = L.dot(u)
        q = matrix_rank(t1)
        t1 = t1.dot(M)
        H = t1.T.dot(t1)

        # E = M'(Y'Y - B'(X'X)B)M
        E = M.T.dot(sscpr).dot(M)
        return E, H, q, df_resid

    return _multivariate_test(hypotheses, exog_names, endog_names, fn)

_manova_test.__doc__ = (
    """
    MANOVA hypotheses testing

    For y = x * params, where y is dependent variables and x is independent
    variables, testing L * params * M = 0 where L is the contast matrix for
    hypotheses testing and M is the transformation matrix for transforming the
    dependent variables in y.

    Algorithm:
        H = (L * u * M)' * (L * u * M)
        E = M' * sscpr * M

    .. [1] ftp://public.dhe.ibm.com/software/analytics/spss/documentation/statistics/20.0/en/client/Manuals/IBM_SPSS_Statistics_Algorithms.pdf

    Parameters
    ----------
    """ + _hypotheses_doc +
    """
    fittedvalues : tuple
        Output of ``fit_manova``
    exog_names : a list of string
    endog_names : a list of string

    Returns
    -------
    a dict of manova tables
    """)


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
    def __init__(self, endog, exog, missing='none', hasconst=None, **kwargs):
        super(MANOVA, self).__init__(endog, exog, **kwargs)
        self.fittedmod = _manova_fit(self.exog, self.endog)

    def mv_test(self):
        """
        MANOVA statistical testing

        Returns
        -------
        results: MANOVAResults

        """
        if (hasattr(self, 'data') and self.data is not None and
                    self.data.design_info is not None):
            terms = self.data.design_info.term_name_slices
            hypotheses = []
            for key in terms:
                L_contrast = np.eye(self.exog.shape[1])[terms[key], :]
                hypotheses.append([key, L_contrast, None])
        else:
            hypotheses = []
            for i in range(self.exog.shape[1]):
                name = 'x%d' % (i)
                L = np.zeros([1, self.exog.shape[1]])
                L[i] = 1
                hypotheses.append([name, L, None])
        results = _manova_test(hypotheses, self.fittedmod,self.exog_names,
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

