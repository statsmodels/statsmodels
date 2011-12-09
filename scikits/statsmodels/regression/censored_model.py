import numpy as np
import scikits.statsmodels.base.model as base
import scikits.statsmodels.base.wrapper as wrap
from scikits.statsmodels.tools.decorators import cache_readonly
from scikits.statsmodels.sandbox.regression.numdiff import (approx_hess_cs,
    approx_fprime_cs, approx_hess, approx_fprime)
from scipy.stats import norm

#TODO: clean-up estimation, delete webuse, write specific results class,
#      figure out good separation for params and scale, can't concentrate
#      likelihood, generalize margins from discrete for use here as well

class Tobit(base.LikelihoodModel):
    def __init__(self, endog, exog, left=True, right=True):
        super(Tobit, self).__init__(endog, exog)
        # set up censoring
        self._transparams = False # need to exp(sigma) to keep positive in fit
        self._init_censored(left, right)

    def _init_censored(self, left, right):
        if left is False and right is False:
            raise ValueError("Must give a censoring level on the right or "
                             "left.")
        endog = self.endog
        exog = self.exog
        if left is True:
            left = self.endog.min()
        elif left is False:
            left = -np.inf
        if right is True:
            right = self.endog.max()
        elif right is False:
            right = np.inf
        self.left, self.right = left, right

        # do this all here, so that we only have to do the indexing once
        left_idx = endog <= left # do we need to attach these?
        right_idx = endog >= right
        self._left_endog = endog[left_idx]
        self.n_lcens = left_idx.sum()
        self.n_rcens = right_idx.sum()
        self._right_endog = endog[right_idx]
        self._left_exog = exog[left_idx]
        self._right_exog = exog[right_idx]
        center_idx = ~np.logical_or(left_idx, right_idx)
        self._center_endog = endog[center_idx]
        self._center_exog = exog[center_idx]

    def score(self, params):
        #NOTE: it might be easier to just use the numerical differentiation
        #they're almost exactly the same here
        sigma = params[-1]
        if self._transparams:
            sigma = np.exp(sigma)
        params = params[:-1]

        left_exog = self._left_exog
        if left_exog.size > 0:
            scaled_left_exog = (self.left - np.dot(left_exog, params)) / sigma
            part1 = norm.pdf(scaled_left_exog)/norm.cdf(scaled_left_exog)
            dLdB_left = -np.sum(left_exog/sigma * part1[:,None], axis=0)
            dLdSigma_left = -np.sum(part1 * scaled_left_exog, axis=0)
        else:
            dLdB_left = dLdSigma_left = 0

        right_exog = self._right_exog
        if right_exog.size > 0:
            scaled_right_exog = (np.dot(right_exog,
                                            params) - self.right) / sigma
            part1 = norm.pdf(scaled_right_exog)/norm.cdf(scaled_right_exog)
            dLdB_right = np.sum(right_exog/sigma * part1[:,None], axis=0)
            dLdSigma_right = -np.sum(scaled_right_exog*part1, axis=0)
        else:
            dLdB_right = dLdSigma_right = 0

        endog = self._center_endog
        exog = self._center_exog
        scaled_center_exog = (endog - np.dot(exog, params))/sigma
        dLdB_center = np.sum(exog/sigma * scaled_center_exog[:,None], axis=0)
        dLdSigma_center = np.sum(scaled_center_exog**2 - 1, axis=0)
        dLdparams = dLdB_left + dLdB_right + dLdB_center
        dLdsigma = dLdSigma_left + dLdSigma_right + dLdSigma_center
        return np.r_[dLdparams, dLdsigma]
        #loglike = self.loglike
        #return approx_fprime(params, loglike)

    def hessian(self, params):
        loglike = self.loglike
        return approx_hess(params, loglike, epsilon=1e-4)[0]

    def loglike(self, params): #NOTE: these needs sigma as last parameter
        sigma = params[-1]
        if self._transparams:
            sigma = np.exp(sigma)
        params = params[:-1]

        left_exog = self._left_exog
        if left_exog.size > 0:
            left_exog = (self.left - np.dot(left_exog, params)) / sigma
            left_like = np.sum(norm.logcdf(left_exog))
            # can get overflow from the above if cdf is very small
            # seems to mean bad starting values, but might need a check
            #if np.isinf(left_like) and left_like < 0:
            #    left_like = -1e4
        else:
            left_like = 0
        right_exog = self._right_exog
        if right_exog.size > 0:
            right_exog = (np.dot(right_exog, params) - self.right) / sigma
            right_like = np.sum(norm.logcdf(right_exog))
            #if np.isinf(right_like) and left_like < 0:
            #    left_like = -1e4
        else:
            right_like = 0
        center_endog = self._center_endog
        center_exog = self._center_exog
        center_like = np.sum(norm.logpdf((center_endog - np.dot(center_exog,
                                   params)) / sigma) - np.log(sigma))
        llf = left_like + right_like + center_like
        return llf

    def _get_start_params(self):
        """
        Get starting parameters from an OLS estimation
        """
        ols_res = sm.OLS(self._center_endog, self._center_exog).fit()
        params = ols_res.params
        sigma = np.log(ols_res.scale ** .5)
        start_params = np.r_[params, sigma]
        return start_params

    def fit(self, start_params = None, **kwargs):
        self._transparams = True
        if start_params is None:
            start_params = self._get_start_params()
        else:
            start_params[-1] = np.log(start_params[-1])
        mlefit = super(Tobit, self).fit(start_params=start_params, **kwargs)
        self._transparams = False
        return TobitResults(self, mlefit)

class TobitResults(base.LikelihoodModelResults):
    def __init__(self, model, mlefit):
        self.model = model
        # re-parameterize from exponential
        sigma = np.exp(mlefit.params[-1])
        mlefit.params[-1] = sigma
        self.model_params = mlefit.params[:-1]
        self.__dict__.update(mlefit.__dict__)
        self.scale = sigma # overwrite from mlefit
        self._model_stats() # attach model statistics

    def _model_stats(self):
        model = self.model
        exog = model.exog
        self.nobs = nobs = exog.shape[0]
        self.n_lcens = model.n_lcens
        self.n_rcens = model.n_rcens
        self.n_ucens = nobs - self.n_lcens - self.n_rcens
        self.k_constant = k_constant = int(np.any(exog.var(0) == 0))
        self.k_params = k_params = exog.shape[1] - k_constant
        self.k_ancillary = 1 # number of ancillary parameters, sigma here
        self.df_model = k_params - self.k_constant
        self.df_resid = nobs - self.df_model

    def cov_params(self):
        hess = self.model.hessian(self.params)
        return np.linalg.inv(-hess)

    @cache_readonly
    def chi2(self):
        return 2 * (self.llf-self.llnull)

    @cache_readonly
    def chi2_pvalue(self):
        from scipy.stats import chi2
        #NOTE: df == number of restrictions
        return chi2.sf(self.chi2, self.df_model)

    @cache_readonly
    def llnull(self):
        endog = self.model.endog
        left = self.model.left
        right = self.model.right
        res = Tobit(endog, np.ones_like(endog), left=left,
                     right=right).fit(method='bfgs', disp=0)
        try:
            assert not res.mle_retvals['warnflag']
        except:
            raise AssertionError("Null Likelihood did not converge. Please "
                    "report.")
        return res.llf

def webuse(data, baseurl='http://www.stata-press.com/data/r11/'):
    """
    Parameters
    ----------
    data : str
    Name of dataset to fetch.

    Examples
    --------
    >>> dta = webuse('auto')

    Notes
    -----
    Make sure baseurl has trailing forward slash. Doesn't do any
    error checking in response URLs.
    """
    # lazy imports
    import pandas
    from scikits.statsmodels.iolib import genfromdta
    from urllib2 import urlopen
    from urlparse import urljoin
    from StringIO import StringIO

    url = urljoin(baseurl, data+'.dta')
    dta = urlopen(url)
    dta = StringIO(dta.read()) # make it truly file-like
    return genfromdta(dta)

if __name__ == "__main__":
    import pandas
    import scikits.statsmodels.api as sm
    dta = webuse('auto')
    df = pandas.DataFrame.from_records(dta)
    df['rep78'] = df['rep78'].astype(float)
    df.ix[df['rep78'] == -999, 'rep78'] = np.nan
    df['weight'] = df['weight'].astype(float) #TODO: fix StataReader for ints
    exog = sm.add_constant(df['weight'] / 1000, prepend=True)
    endog = df['mpg']
    #mod = Tobit(endog, exog, left=17, right=False).fit(method='bfgs')

    #bse = np.sqrt(np.diag(np.linalg.inv(-approx_hess(np.r_[mod.params,
    #                np.log(mod.scale)], mod.model.loglike)[0])))
    # now since we take the exponent in the likelihood of tobit, we have
    # f(x) = exp(log(x)) == exp(x)*1/x, so
    #bse[-1] = bse[-1] * mod.scale

    #mod2 = Tobit(endog, exog, left=False, right=24).fit(method='bfgs')

    #mod3 = Tobit(endog, exog, left=17, right=24).fit(method='bfgs')

    # compare to Stata results in the manual, parameters and bse look good

    # replicate Fair's results
    data = sm.datasets.fair.load()
    endog = data.endog
    exog = sm.add_constant(data.exog, prepend=False)
    mod = Tobit(endog, exog, left=0, right=False).fit(method='bfgs')
    #NOTE: newton's method doesn't work well, I think this is due to the
    #Hessian approximation
    #fair_mod = Tobit(endog, exog, left=0, right=False).fit(method='newton')
    #cov = np.linalg.inv(-approx_hess(mod.params, mod.model.loglike)[0])
    #bse = np.sqrt(np.diag(np.linalg.inv(-approx_hess(mod.params,
    #                    mod.model.loglike)[0])))
    # now since we take the exponent in the likelihood of tobit, we have
    # f(x) = exp(log(x)) == exp(x)*1/x, so
    #bse[-1] = bse[-1] * mod.scale
