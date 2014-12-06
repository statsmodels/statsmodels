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
from statsmodels.tools.decorators import cache_readonly
import patsy

from scipy.special import gammaln as lgamma

from statsmodels.base.model import (GenericLikelihoodModel,
                                    GenericLikelihoodModelResults)
from statsmodels.genmod.families import Binomial

Logit = sm.families.links.logit

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

    @classmethod
    def from_formula(cls, formula, data, exog_precision_formula=None,
                     *args, **kwargs):
        if exog_precision_formula is not None:
            if 'subset' in kwargs:
                d = data.ix[kwargs['subset']]
                Z = patsy.dmatrix(exog_precision_formula, d)
            else:
                Z = patsy.dmatrix(exog_precision_formula, data)
            kwargs['exog_precision'] = Z

        return super(Beta, cls).from_formula(formula, data, *args,
                                      **kwargs)


    def predict(self, params, exog=None):
        """predict values for mean, conditional expectation E(endog | exog)

        """
        if exog is None:
            exog = self.exog
        k_mean = self.exog.shape[1]

        params_mean = params[:k_mean]
        Zparams = params[k_mean:]
        mu = self.link.inverse(np.dot(exog, params_mean))
        return mu


    def predict_precision(self, params, exog_precision=None):
        """predict values for precision parameter for given exog_precision

        """
        if exog_precision is None:
            exog_precision = self.exog_precision

        k_mean = self.exog.shape[1]
        params_precision = params[k_mean:]
        linpred = np.dot(exog_precision, params_precision)
        phi = self.link_precision.inverse(linpred)

        return phi


    def predict_var(self, params, exog=None, exog_precision=None):
        """predict values for conditional variance V(endog | exog)

        """
        mean = self.predict(params, exog=exog)
        precision = self.predict_precision(params,
                                           exog_precision=exog_precision)

        var_endog = mean * (1 - mean) / (1 + precision)
        return var_endog


    def nloglikeobs(self, params):
        """
        Negative log-likelihood.

        Parameters
        ----------

        params : np.ndarray
            Parameter estimates
        """
        return -self._ll_br(self.endog, self.exog, self.exog_precision, params)

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


    def score(self, params):
        """
        Returns the score vector of the profile log-likelihood.

        http://www.tandfonline.com/doi/pdf/10.1080/00949650903389993
        """
        sf = self.score_factor(params)

        d1 = np.dot(sf[:, 0], self.exog)
        d2 = np.dot(sf[:, 1], self.exog_precision)
        return np.concatenate((d1, d2))


    def score_check(self, params):
        """inherited score with finite differences
        """
        return super(Beta, self).score(params)


    def score_factor(self, params):
        """derivative of loglikelihood function without the exog

        This needs to be multiplied with the exog to obtain the score_obs
        """
        from scipy import special
        digamma = special.psi

        y, X, Z = self.endog, self.exog, self.exog_precision
        nz = Z.shape[1]
        Xparams = params[:-nz]
        Zparams = params[-nz:]

        # NO LINKS
        mu = self.link.inverse(np.dot(X, Xparams))
        phi = self.link_precision.inverse(np.dot(Z, Zparams))

        ystar = np.log( y / (1. - y))
        mustar = digamma(mu * phi) - digamma((1 - mu) * phi)
        yt = np.log(1 - y)
        mut = digamma((1 - mu) * phi) - digamma(phi)

        t = 1. / self.link.deriv(mu)
        h = 1. / self.link_precision.deriv(phi)
        #
        sf1 = phi * t * (ystar - mustar)
        sf2 = h * ( mu * (ystar - mustar) + yt - mut)

        return np.column_stack((sf1, sf2))


    def score_hessian_factor(self, params, return_hessian=False, observed=True):
        """derivatives of loglikelihood function without the exog

        This needs to be multiplied with the exog to obtain the score_obs

        This calculates score and hessian factors at the same time, since there
        is a large overlap in calculations
        """
        from scipy import special
        digamma = special.psi

        y, X, Z = self.endog, self.exog, self.exog_precision
        nz = Z.shape[1]
        Xparams = params[:-nz]
        Zparams = params[-nz:]

        # NO LINKS
        mu = self.link.inverse(np.dot(X, Xparams))
        phi = self.link_precision.inverse(np.dot(Z, Zparams))

        ystar = np.log( y / (1. - y))
        mustar = digamma(mu * phi) - digamma((1 - mu) * phi)
        yt = np.log(1 - y)
        mut = digamma((1 - mu) * phi) - digamma(phi)

        t = 1. / self.link.deriv(mu)
        h = 1. / self.link_precision.deriv(phi)

        ymu_star = (ystar - mustar)
        sf1 = phi * t * ymu_star
        sf2 = h * ( mu * ymu_star + yt - mut)

        if return_hessian:
            trigamma = lambda x: special.polygamma(1, x)
            var_star = trigamma(mu * phi) + trigamma((1 - mu) * phi)
            var_t = trigamma((1 - mu) * phi) - trigamma(phi)

            c = - trigamma((1 - mu) * phi)
            s = self.link.deriv2(mu)
            q = self.link_precision.deriv2(phi)


            jbb = (phi * t) * var_star
            if observed:
                jbb += s * t**2 * ymu_star

            jbb *= t * phi

            jbg = phi * t * h * (mu * var_star + c)
            if observed:
                jbg -= ymu_star * t * h

            jgg = h**2 * (mu**2 * var_star + 2 * mu * c + var_t)
            if observed:
                jgg += (mu * ymu_star + yt - mut) * q * h**3    # **3 ?

        return np.column_stack((sf1, sf2)), (-jbb, -jbg, -jgg)


    def score_obs(self, params):
        sf = self.score_factor(params)

        # elementwise product for each row (observation)
        d1 = sf[:, :1] * self.exog
        d2 = sf[:, 1:2] * self.exog_precision
        return np.column_stack((d1, d2))


    def hessian_1(self, params, observed=True):
        _, hf = self.score_hessian_factor(params, return_hessian=True,
                                          observed=observed)

        hf11, hf12, hf22 = hf

        # elementwise product for each row (observation)
        d11 = (self.exog.T * hf11).dot(self.exog)
        d12 = (self.exog.T * hf12).dot(self.exog_precision)
        d22 = (self.exog_precision.T * hf22).dot(self.exog_precision)
        return np.bmat([[d11, d12], [d12.T, d22]]).A


    def _start_params(self, niter=2, return_intermediate=False):
        """find starting values

        Returns
        -------
        sp : ndarray
            start parameters for the optimization

        Notes
        -----
        This calculates a few iteration of weighted least squares. This is not
        a full scoring algorithm.

        """
        # WLS of the mean equation uses the implied weights (inverse variance),
        # WLS for the precision equations uses weights that only take
        # account of the link transformation of the precision endog.
        from statsmodels.regression.linear_model import OLS, WLS
        res_m = OLS(self.link(self.endog), self.exog).fit()
        fitted = self.link.inverse(res_m.fittedvalues)
        resid = self.endog - fitted

        prec_i = fitted * (1 - fitted) / np.maximum(np.abs(resid), 1e-2)**2 - 1
        res_p = OLS(self.link_precision(prec_i), self.exog_precision).fit()
        prec_fitted = self.link_precision.inverse(res_p.fittedvalues)
        #sp = np.concatenate((res_m.params, res_p.params))

        for _ in range(niter):
            y_var_inv = (1 + prec_fitted) / (fitted * (1 - fitted))
            #y_var = fitted * (1 - fitted) / (1 + prec_fitted)

            ylink_var_inv = y_var_inv / self.link.deriv(fitted)**2
            res_m2 = WLS(self.link(self.endog), self.exog,
                         weights=ylink_var_inv).fit()
            fitted = self.link.inverse(res_m2.fittedvalues)
            resid2 = self.endog - fitted

            prec_i2 = (fitted * (1 - fitted) /
                       np.maximum(np.abs(resid2), 1e-2)**2 - 1)
            w_p = 1. / self.link_precision.deriv(prec_fitted)**2
            res_p2 = WLS(self.link_precision(prec_i2), self.exog_precision,
                         weights=w_p).fit()
            prec_fitted = self.link_precision.inverse(res_p2.fittedvalues)
            sp2 = np.concatenate((res_m2.params, res_p2.params))

        if return_intermediate:
            return sp2, res_m2, res_p2

        return sp2


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
            start_params = self._start_params()
#           # http://www.ime.usp.br/~sferrari/beta.pdf suggests starting phi
#           # on page 8

        self.results_class = BetaRegressionResults
        return super(Beta, self).fit(start_params=start_params,
                                        maxiter=maxiter, maxfun=maxfun,
                                        method=method, disp=disp, **kwds)


class BetaRegressionResults(GenericLikelihoodModelResults):

    # GenericLikeihoodmodel doesn't define fittedvalues, residuals and similar
    @cache_readonly
    def fittedvalues(self):
        return self.model.predict(self.params)

    @cache_readonly
    def fitted_precision(self):
        return self.model.predict_precision(self.params)


    @cache_readonly
    def resid(self):
        return self.model.endog - self.fittedvalues


    @cache_readonly
    def resid_pearson(self):
        return self.resid / np.sqrt(self.model.predict_var(self.params))


    def get_distribution_params(self):
        mean = self.fittedvalues
        precision = self.fitted_precision
        return precision * mean, precision * (1 - mean)


    def get_distribution(self):
        from scipy import stats
        distr = stats.beta(*self.get_distribution_params())
        return distr


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
