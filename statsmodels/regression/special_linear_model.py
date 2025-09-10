# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:24:12 2018

Author: Josef Perktold
"""

import numpy as np
from statsmodels.tools.decorators import (cache_readonly,
                                          cache_writable)

from statsmodels.regression.linear_model import OLS, RegressionResults

from numpy.testing import assert_allclose


class OLSVectorizedResults(RegressionResults):
    r"""results class for vectorized OLS
    """

    @cache_writable()
    def scale(self):
        wresid = self.wresid
        return (wresid * wresid).sum(0) / self.df_resid


    @cache_readonly
    def bse(self):
        return np.sqrt(self.scale *
                       np.diag(self.normalized_cov_params)[:, None])

    @cache_readonly
    def ssr(self):
        wresid = self.wresid
        return (wresid * wresid).sum(0)

    @cache_readonly
    def uncentered_tss(self):
        wendog = self.model.wendog
        return (wendog * wendog).sum(0)

    def conf_int(self, alpha=.05, cols=None):
        #print('using OLSVectorizedResults.conf_int')
        ci =  super(OLSVectorizedResults, self).conf_int(alpha=alpha,
                     cols=cols)

        return np.rollaxis(ci, -1)

    def t_test(self, r_matrix, cov_p=None, scale=None, use_t=None):
        """
        Compute a t-test for a single linear hypothesis of the form Rb = q

        This is an adjusted version of the corresponding LikelihoodModelResults
        method adjusted to be vectorized across endog.

        Parameters
        ----------
        r_matrix : array-like, str, tuple
            - array : If an array is given, a p x k 2d array or length k 1d
              array specifying the linear restrictions. It is assumed
              that the linear combination is equal to zero.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q). If q is given,
              can be either a scalar or a length p row vector.
        cov_p : array-like, optional
            An alternative estimate for the parameter covariance matrix.
            If None is given, self.normalized_cov_params is used.
        scale : float, optional
            An optional `scale` to use.  Default is the scale specified
            by the model fit.
        use_t : bool, optional
            If use_t is None, then the default of the model is used.
            If use_t is True, then the p-values are based on the t
            distribution.
            If use_t is False, then the p-values are based on the normal
            distribution.

        Returns
        -------
        res : ContrastResults instance
            The results for the test are attributes of this results instance.
            The available results have the same elements as the parameter table
            in `summary()`.

        Examples
        --------
        >>> import numpy as np
        >>> import statsmodels.api as sm

        See Also
        ---------
        tvalues : individual t statistics
        f_test : for F tests
        patsy.DesignInfo.linear_constraint
        """
        from statsmodels.tools.tools import recipr
        from statsmodels.stats.contrast import ContrastResults
        from patsy import DesignInfo
        names = self.model.data.param_names
        LC = DesignInfo(names).linear_constraint(r_matrix)
        r_matrix, q_matrix = LC.coefs, LC.constants
        num_ttests = r_matrix.shape[0]
        num_params = r_matrix.shape[1]

        if (cov_p is None and self.normalized_cov_params is None and
                not hasattr(self, 'cov_params_default')):
            raise ValueError('Need covariance of parameters for computing '
                             'T statistics')
        if num_params != self.params.shape[0]:
            raise ValueError('r_matrix and params are not aligned')
        if q_matrix is None:
            q_matrix = np.zeros(num_ttests)
        else:
            q_matrix = np.asarray(q_matrix)
            q_matrix = q_matrix.squeeze()
        if q_matrix.size > 1:
            if q_matrix.shape[0] != num_ttests:
                raise ValueError("r_matrix and q_matrix must have the same "
                                 "number of rows")

        if use_t is None:
            # switch to use_t false if undefined
            use_t = (hasattr(self, 'use_t') and self.use_t)

        _t = _sd = None

        _effect = np.dot(r_matrix, self.params)

        # vectorized OLS: we use initially scale=1 and then add the scale
        cov_p = self.normalized_cov_params
        # Perform the test
        if num_ttests > 1:
            _sd = np.sqrt(np.diag(self.cov_params(
                r_matrix=r_matrix, cov_p=cov_p)))
        else:
            _sd = np.sqrt(self.cov_params(r_matrix=r_matrix, cov_p=cov_p))

        _sd = np.squeeze(_sd) * np.sqrt(self.scale)
        _effect = np.squeeze(_effect)
        _t = (_effect - q_matrix) * recipr(_sd)

        df_resid = getattr(self, 'df_resid_inference', self.df_resid)

        if use_t:
            return ContrastResults(effect=_effect, t=_t, sd=_sd,
                                   df_denom=df_resid)
        else:
            return ContrastResults(effect=_effect, statistic=_t, sd=_sd,
                                   df_denom=df_resid,
                                   distribution='norm')

    def summary(self):
        from statsmodels.iolib.summary import summary_params_2dflat
        summ = summary_params_2dflat(self,
                                     endog_names=self.model.endog_names)[1]
        return summ


class OLSVectorized(OLS):
    _results_class = OLSVectorizedResults
