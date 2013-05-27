#!/usr/bin/env python

'''
Quantile regression model

Model parameters are estimated using iterated reweighted least squares. The
asymptotic covariance matrix estimated using kernel density estimation.

Author: Vincent Arel-Bundock
License: BSD-3
Created: 2013-03-19

The original IRLS function was written for Matlab by Shapour Mohammadi,
University of Tehran, 2008 (shmohammadi@gmail.com), with some lines based on
code written by James P. Lesage in Applied Econometrics Using MATLAB(1999).PP.
73-4.  Translated to python with permission from original author by Christian
Prinoth (christian at prinoth dot name).
'''

import numpy as np
import warnings
import scipy.stats as stats
from scipy.linalg import pinv
from scipy.stats import norm
from statsmodels.tools.tools import chain_dot
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import (RegressionModel,
                                                 RegressionResults,
                                                 RegressionResultsWrapper)


class QuantReg(RegressionModel):
    '''Quantile Regression

    Estimate a quantile regression model using iterative reweighted least
    squares.

    Parameters
    ----------
    endog : array or dataframe
        endogenous/response variable
    exog : array or dataframe
        exogenous/explanatory variable(s)

    Notes
    -----
    The Least Absolute Deviation (LAD) estimator is a special case where
    quantile is set to 0.5 (q argument of the fit method).

    The asymptotic covariance matrix is estimated following the procedure in
    Greene (2008, p.407-408), using either the logistic or gaussian kernels
    (kernel argument of the fit method).

    References
    ----------
    General:

    * Birkes, D. and Y. Dodge(1993). Alternative Methods of Regression, John Wiley and Sons.
    * Green,W. H. (2008). Econometric Analysis. Sixth Edition. International Student Edition.
    * Koenker, R. (2005). Quantile Regression. New York: Cambridge University Press.
    * LeSage, J. P.(1999). Applied Econometrics Using MATLAB,

    Kernels (used by the fit method):

    * Green (2008) Table 14.2

    Bandwidth selection (used by the fit method):

    * Bofinger, E. (1975). Estimation of a density function using order statistics. Australian Journal of Statistics 17: 1-17.
    * Chamberlain, G. (1994). Quantile regression, censoring, and the structure of wages. In Advances in Econometrics, Vol. 1: Sixth World Congress, ed. C. A. Sims, 171-209. Cambridge: Cambridge University Press.
    * Hall, P., and S. Sheather. (1988). On the distribution of the Studentized quantile. Journal of the Royal Statistical Society, Series B 50: 381-391.

    Keywords
    --------
    Keywords: Least Absolute Deviation(LAD) Regression, Quantile Regression,
    Regression, Robust Estimation.
    '''

    def __init__(self, endog, exog):
        super(QuantReg, self).__init__(endog, exog)

    def whiten(self, data):
        """
        QuantReg model whitener does nothing: returns data.
        """
        return data

    def fit(self, q=.5, vcov='robust', kernel='epa', bandwidth='hsheather',
            **kwargs):
        '''Solve by Iterative Weighted Least Squares

        Parameters
        ----------
        q : float
            Quantile must be between 0 and 1
        vcov : string, method used to calculate the variance-covariance matrix
               of the parameters. Default is ``robust``

               robust : heteroskedasticity robust standard errors (as suggested
                        in Greene 6th edition)
               iid : iid errors (as in Stata 12)
        kernel : string, kernel to use in the kernel density estimation for the
                 asymptotic covariance matrix

                 epa: Epanechnikov
                 cos: Cosine
                 gau: Gaussian
                 par: Parzene
        bandwidth: string, Bandwidth selection method in kernel density
                   estimation for asymptotic covariance estimate (full
                   references in QuantReg docstring)

                   hsheather: Hall-Sheather (1988)
                   bofinger: Bofinger (1975)
                   chamberlain: Chamberlain (1994)
        '''

        if q < 0 or q > 1:
            raise Exception('p must be between 0 and 1')

        kern_names = ['biw', 'cos', 'epa', 'gau', 'par']
        if kernel not in kern_names:
            raise Exception("kernel must be one of " + ', '.join(kern_names))
        else:
            kernel = kernels[kernel]

        if bandwidth == 'hsheather':
            bandwidth = hall_sheather
        elif bandwidth == 'bofinger':
            bandwidth = bofinger
        elif bandwidth == 'chamberlain':
            bandwidth = chamberlain
        else:
            raise Exception("bandwidth must be in 'hsheather', 'bofinger', 'chamberlain'")

        endog = self.endog
        exog = self.exog
        nobs = self.nobs
        rank = self.rank
        n_iter = 0
        xstar = exog
        beta = np.ones(rank)  # TODO: better start
        diff = 10

        max_iter = 1000
        while n_iter < max_iter and diff > 1e-6:
            n_iter += 1
            beta0 = beta
            xtx = np.dot(xstar.T, exog)
            xty = np.dot(xstar.T, endog)
            beta = np.dot(pinv(xtx), xty)
            resid = endog - np.dot(exog, beta)
            #JP: bound resid away from zero,
            #    why not symmetric: * np.sign(resid), shouldn't matter
            resid[np.abs(resid) < .000001] = .000001
            resid = np.where(resid < 0, q * resid, (1-q) * resid)
            resid = np.abs(resid)
            xstar = exog / resid[:, np.newaxis]
            diff = np.max(np.abs(beta - beta0))

        if n_iter == max_iter:
            warnings.warn("Maximum number of iterations (1000) reached.")

        e = endog - np.dot(exog, beta)
        # Greene (2008, p.407) writes that Stata 6 uses this bandwidth:
        # h = 0.9 * np.std(e) / (nobs**0.2)
        # Instead, we calculate bandwidth as in Stata 12
        iqre = stats.scoreatpercentile(e, 75) - stats.scoreatpercentile(e, 25)
        h = bandwidth(nobs, q)
        h = min(np.std(endog),
                iqre / 1.34) * (norm.ppf(q + h) - norm.ppf(q - h))

        fhat0 = 1. / (nobs * h) * np.sum(kernel(e / h))

        if vcov == 'robust':
            d = np.where(e > 0, (q/fhat0)**2, ((1-q)/fhat0)**2)
            xtxi = pinv(np.dot(exog.T, exog))
            xtdx = np.dot(exog.T * d[np.newaxis, :], exog)
            vcov = chain_dot(xtxi, xtdx, xtxi)
        elif vcov == 'iid':
            vcov = (1. / fhat0)**2 * q * (1 - q) * pinv(np.dot(exog.T, exog))
        else:
            raise Exception("vcov must be 'robust' or 'iid'")

        lfit = QuantRegResults(self, beta, normalized_cov_params=vcov)

        lfit.q = q
        lfit.iterations = n_iter
        lfit.sparsity = 1. / fhat0
        lfit.bandwidth = h

        return RegressionResultsWrapper(lfit)


def _parzen(u):
    z = np.where(np.abs(u) <= .5, 4./3 - 8. * u**2 + 8. * np.abs(u)**3,
                 8. * (1 - np.abs(u))**3 / 3.)
    z[np.abs(u) > 1] = 0
    return z


kernels = {}
kernels['biw'] = lambda u: 15. / 16 * (1 - u**2)**2 * np.where(np.abs(u) <= 1, 1, 0)
kernels['cos'] = lambda u: np.where(np.abs(u) <= .5, 1 + np.cos(2 * np.pi * u), 0)
kernels['epa'] = lambda u: 3. / 4 * (1-u**2) * np.where(np.abs(u) <= 1, 1, 0)
kernels['gau'] = lambda u: norm.pdf(u)
kernels['par'] = _parzen
#kernels['bet'] = lambda u: np.where(np.abs(u) <= 1, .75 * (1 - u) * (1 + u), 0)
#kernels['log'] = lambda u: logistic.pdf(u) * (1 - logistic.pdf(u))
#kernels['tri'] = lambda u: np.where(np.abs(u) <= 1, 1 - np.abs(u), 0)
#kernels['trw'] = lambda u: 35. / 32 * (1 - u**2)**3 * np.where(np.abs(u) <= 1, 1, 0)
#kernels['uni'] = lambda u: 1. / 2 * np.where(np.abs(u) <= 1, 1, 0)


def hall_sheather(n, q, alpha=.05):
    z = norm.ppf(q)
    num = 1.5 * norm.pdf(z)**2.
    den = 2. * z**2. + 1.
    h = n**(-1. / 3) * norm.ppf(1. - alpha / 2.)**(2./3) * (num / den)**(1./3)
    return h


def bofinger(n, q):
    num = 9. / 2 * norm.pdf(2 * norm.ppf(q))**4
    den = (2 * norm.ppf(q)**2 + 1)**2
    h = n**(-1. / 5) * (num / den)**(1. / 5)
    return h


def chamberlain(n, q, alpha=.05):
    return norm.ppf(1 - alpha / 2) * np.sqrt(q*(1 - q) / n)


class QuantRegResults(RegressionResults):
    '''Results instance for the QuantReg model'''

    @cache_readonly
    def prsquared(self):
        q = self.q
        endog = self.model.endog
        e = self.resid
        e = np.where(e < 0, (1 - q) * e, q * e)
        e = np.abs(e)
        ered = endog - stats.scoreatpercentile(endog, q * 100)
        ered = np.where(ered < 0, (1 - q) * ered, q * ered)
        ered = np.abs(ered)
        return 1 - np.sum(e) / np.sum(ered)

    @cache_readonly
    def scale(self):
        return 1.
    #@cache_readonly
    #def aic(self):
        #return np.nan

    @cache_readonly
    def bic(self):
        return np.nan

    @cache_readonly
    def aic(self):
        return np.nan

    @cache_readonly
    def llf(self):
        return np.nan

    @cache_readonly
    def rsquared(self):
        return np.nan

    @cache_readonly
    def HC0_se(self):
        raise NotImplementedError

    @cache_readonly
    def HC1_se(self):
        raise NotImplementedError

    @cache_readonly
    def HC2_se(self):
        raise NotImplementedError

    @cache_readonly
    def HC3_se(self):
        raise NotImplementedError
