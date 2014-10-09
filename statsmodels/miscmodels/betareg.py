# -*- coding: utf-8 -*-

u"""
Beta regression for modeling rates and proportions.

References
----------
Grün, Bettina, Ioannis Kosmidis, and Achim Zeileis. Extended beta regression
in R: Shaken, stirred, mixed, and partitioned. No. 2011-22. Working Papers in
Economics and Statistics, 2011.

Smithson, Michael, and Jay Verkuilen. "A better lemon squeezer?
Maximum-likelihood regression with beta-distributed dependent variables."
Psychological methods 11.1 (2006): 54.
"""
from __future__ import print_function
import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.special import gammaln as lgamma

from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.genmod.families import Binomial

# this is only need while #2024 is open.
class Logit(sm.families.links.Logit):

    """Logit tranform that won't overflow with large numbers."""

    def inverse(self, z):
        return 1 / (1. + np.exp(-z))

_init_example = """

    Beta regression with default of logit-link for exog and log-link
    for precision.

    >>> mod = Beta(endog, exog)
    >>> rslt = mod.fit()
    >>> print(rslt.summary())

    We can also specify a formula and a specific structure and use the
    identity-link for precision.

    >>> from sm.families.links import identity
    >>> Z = patsy.dmatrix('~ temp', dat, return_type='dataframe')
    >>> mod = Beta.from_formula('iyield ~ C(batch, Treatment(10)) + temp',
    ...                         dat, exog_precision=Z, link_precision=identity())

    In the case of proportion-data, we may think that the precision depends on
    the number of measurements. E.g for sequence data, on the number of
    sequence reads covering a site:

    >>> Z = patsy.dmatrix('~ coverage', df)
    >>> mod = Beta.from_formula('methylation ~ disease + age + gender + coverage', df, Z)
    >>> rslt = mod.fit()

"""

class Beta(GenericLikelihoodModel):

    """Beta Regression.

    This implementation uses a `precision` parameter
    """

    def __init__(self, endog, exog, exog_precision=None, link=Logit(),
            link_precision=sm.families.links.Log(), **kwds):
        """
        Parameters
        ----------
        endog : array-like
            1d array of endogenous values (i.e. responses, outcomes,
            dependent variables, or 'Y' values).
        exog : array-like
            2d array of exogeneous values (i.e. covariates, predictors,
            independent variables, regressors, or 'X' values). A nobs x k
            array where `nobs` is the number of observations and `k` is
            the number of regressors. An intercept is not included by
            default and should be added by the user. See
            `statsmodels.tools.add_constant`.
        exog_precision : array-like
            2d array of variables for the precision.
        link : link
            Any link in sm.families.links for `exog`
        link_precision : link
            Any link in sm.families.links for `exog_precision`

        Examples
        --------
        {example}

        See Also
        --------
        :ref:`links`

        """.format(example=_init_example)
        etmp = np.array(endog)
        assert np.all((0 < etmp) & (etmp < 1))
        if exog_precision is None:
            extra_names = ['precision']
            exog_precision = np.ones((len(endog), 1), dtype='f')
        else:
            extra_names = ['precision-%s' % zc for zc in \
                    (exog_precision.columns \
                    if hasattr(exog_precision, 'columns') 
                    else range(1, exog_precision.shape[1] + 1))]

        kwds['extra_params_names'] = extra_names

        super(Beta, self).__init__(endog, exog, **kwds)
        self.link = link
        self.link_precision = link_precision

        self.exog_precision = exog_precision
        assert len(self.exog_precision) == len(self.endog)

    def nloglikeobs(self, params):
        """
        Negative log-likelihood.

        Parameters
        ----------

        params : np.ndarray
            Parameter estimates
        """
        return -self._ll_br(self.endog, self.exog, self.exog_precision, params)

    def fit(self, start_params=None, maxiter=100000, maxfun=5000, disp=False,
            method='bfgs', **kwds):
        """
        Fit the model.

        Parameters
        ----------
        start_params : array-like
            A vector of starting values for the regression
            coefficients.  If None, a default is chosen.
        maxiter : integer
            The maximum number of iterations
        disp : bool
            Show convergence stats.
        method : str
            The optimization method to use.
        """

        if start_params is None:
            start_params = sm.GLM(self.endog, self.exog, family=Binomial()
                                 ).fit(disp=False).params
            start_params = np.append(start_params, [0.5] \
                    * self.exog_precision.shape[1])

        return super(Beta, self).fit(start_params=start_params,
                                        maxiter=maxiter, maxfun=maxfun,
                                        method=method, disp=disp, **kwds)

    def _ll_br(self, y, X, Z, params):
        nz = Z.shape[1]

        Xparams = params[:-nz]
        Zparams = params[-nz:]

        mu = self.link.inverse(np.dot(X, Xparams))
        phi = self.link_precision.inverse(np.dot(Z, Zparams))

        alpha = mu * phi
        beta = (1 - mu) * phi

        if np.any(alpha <= np.finfo(float).eps): return np.array(-np.inf)
        if np.any(beta <= np.finfo(float).eps): return np.array(-np.inf)

        ll = lgamma(phi) - lgamma(mu * phi) - lgamma((1 - mu) * phi) \
                + (mu * phi - 1) * np.log(y) + (((1 - mu) * phi) - 1) \
                * np.log(1 - y)

        return ll

if __name__ == "__main__":

    import patsy

    fex = pd.read_csv('tests/foodexpenditure.csv')
    m = Beta.from_formula(' I(food/income) ~ income + persons', fex)
    print(m.fit().summary())
    #print GLM.from_formula('iyield ~ C(batch) + temp', dat, family=Binomial()).fit().summary()

    dev = pd.read_csv('tests/methylation-test.csv')
    Z = patsy.dmatrix('~ age', dev, return_type='dataframe')
    m = Beta.from_formula('methylation ~ gender + CpG', dev,
            exog_precision=Z,
            link_precision=sm.families.links.identity())
    print(m.fit().summary())
