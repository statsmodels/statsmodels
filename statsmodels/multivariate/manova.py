# -*- coding: utf-8 -*-

"""Multivariate analysis of variance

author: Yichuan Liu
"""
from __future__ import print_function, division

from statsmodels.base.model import Model
import numpy as np
from numpy.linalg import matrix_rank, qr
from statsmodels.iolib import summary2
from .multivariate_ols import _multivariate_ols_test, _hypotheses_doc
from .multivariate_ols import MultivariateTestResults
from .multivariate_ols import _multivariate_ols_fit
__docformat__ = 'restructuredtext en'


class MANOVA(Model):
    """
    Multivariate analysis of variance


    Parameters
    ----------
    endog : array-like
        Dependent variables. A nobs x k_endog array where nobs is
        the number of observations and k_endog is the number of dependent
         variables.

    exog : array-like
        Independent variables. A nobs x k_exog array where nobs is the
        number of observations and k_exog is the number of independent
        variables. An intercept is not included by default and should be added
        by the user (models specified using a formula include an intercept by
        default)

    .. [1] ftp://public.dhe.ibm.com/software/analytics/spss/documentation/statistics/20.0/en/client/Manuals/IBM_SPSS_Statistics_Algorithms.pdf

    Attributes
    -----------
    endog : array
        See Parameters.
    exog : array
        See Parameters.

    """
    def __init__(self, endog, exog, missing='none', hasconst=None, **kwargs):
        super(MANOVA, self).__init__(endog, exog, **kwargs)
        self._fittedmod = _multivariate_ols_fit(self.endog, self.exog)

    def mv_test(self, hypotheses=None):
        if hypotheses is None:
            if (hasattr(self, 'data') and self.data is not None and
                        hasattr(self.data, 'design_info')):
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

        results = _multivariate_ols_test(hypotheses, self._fittedmod,
                                         self.exog_names, self.endog_names)

        return MultivariateTestResults(results)
    mv_test.__doc__ = (
        """
        Testing the linear hypotheses
            L * params * M = 0
        where `params` is the regression coefficient matrix for the
        linear model y = x * params

        Parameters
        ----------
        """ + _hypotheses_doc +
        """

        Returns
        -------
        results: MultivariateTestResults

        """
    )
