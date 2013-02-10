#!/usr/bin/env python
import numpy as np
import scipy.stats as stats
from scipy.linalg import pinv
from scipy.stats import logistic, norm
from statsmodels.tools.tools import chain_dot as dot
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

    General References
    ------------------

    * Birkes, D. and Y. Dodge(1993). Alternative Methods of Regression, John Wiley and Sons.
    * Green,W. H. (2008). Econometric Analysis. Sixth Edition. International Student Edition.
    * LeSage, J. P.(1999),Applied Econometrics Using MATLAB,

    License
    -------

    Original Matlab code by Shapour Mohammadi, University of Tehran, 2008
    (shmohammadi@gmail.com). First translation to python with permission from
    original author by Christian Prinoth (christian at prinoth dot name).
    Further adapted and modified by the statsmodels team.

    Keywords
    --------

    Keywords: Least Absolute Deviation(LAD) Regression, Quantile Regression,
    Regression, Robust Estimation.
    '''

    def __init__(self, endog, exog):
        super(QuantReg, self).__init__(endog, exog)

    def whiten(self, Y):
        """
        QuantReg model whitener does nothing: returns Y.
        """
        return Y

    def fit(self, q=.5, kernel='epa', bandwidth='hsheather', **kwargs):
        '''Solve by Iterative Weighted Least Squares

        Parameters
        ----------

        q : float
            quantile must be between 0 and 1
        kernel : string ('logistic' or 'gaussian')
            kernel to use for computation of estimate asymptotic covariance
            matrix 
            
        Notes
        -----

        Some lines of this section is based on a code written by
        James P. Lesage in Applied Econometrics Using MATLAB(1999).PP. 73-4.
        '''

        self.q = q

        if q < 0 or q > 1:
            raise Exception('p must be between 0 and 1')

        kern_names = ['bet', 'biw', 'cos', 'epa', 'gau', 'log', 'tri', 'trw',
                      'uni']  
        if kernel not in kern_names:
            raise Exception("kernel must be in " + ', '.join(kern_names))
        else:
            kernel = kernels[kernel]

        bwidth_names = ['hsheather', 'bofinger', 'chamberlain']
        if bandwidth not in bwidth_names:
            raise Exception("kernel must be in " + ', '.join(bwidth_names))
        elif bandwidth == 'hsheather':
            bandwidth = hall_sheather
        elif bandwidth == 'bofinger':
            bandwidth = bofinger
        else:
            bandwidth = chamberlain

        endog = self.endog
        exog = self.exog
        nobs = self.nobs
        rank = self.rank
        itrat = 0
        xstar = exog
        beta = np.ones(rank)  # TODO: better start
        diff = 10

        while itrat < 1000 and diff > 1e-6:
            itrat += 1
            beta0 = beta
            beta = dot(pinv(dot(xstar.T, exog)), xstar.T, endog)
            resid = endog - dot(exog, beta)
            #JP: bound resid away from zero,
            #    why not symmetric: * np.sign(resid), shouldn't matter
            resid[np.abs(resid) < .000001] = .000001
            resid = np.where(resid < 0, q * resid, (1-q) * resid)
            resid = np.abs(resid)
            xstar = exog / resid[:, np.newaxis]
            diff = np.max(np.abs(beta - beta0))

        e = endog - dot(exog, beta)
        # Greene (2008, p.407) writes that Stata 6 uses this bandwidth:
        #h = 0.9 * np.std(e) / (nobs**0.2)
        # Instead, we calculate bandwidth as in Stata 12
        iqre = np.percentile(e, 75) - np.percentile(e, 25)
        h = bandwidth(nobs, q)
        h = min(np.std(endog), iqre / 1.34) * (norm.ppf(q + h) - norm.ppf(q - h)) 

        fhat0 = 1. / (nobs * h) * np.sum(kernel(e / h))

        D = np.where(e > 0, (q/fhat0)**2, ((1-q)/fhat0)**2)
        D = np.diag(D)
        vcov = dot(pinv(dot(exog.T, exog)), dot(exog.T, D, exog),
                   pinv(dot(exog.T, exog)))

        lfit = QuantRegResults(self, beta, normalized_cov_params=vcov)
        return RegressionResultsWrapper(lfit)

kernels = {}
kernels['bet'] = lambda u: np.where(np.abs(u) <= 1, .75 * (1 - u) * (1 + u), 0) 
kernels['biw'] = lambda u: 15. / 16 * (1 - u**2)**2 * np.where(np.abs(u) <= 1, 1, 0)
kernels['cos'] = lambda u: np.where(np.abs(u) <= .5, 1 + np.cos(2 * np.pi * u), 0)
kernels['epa'] = lambda u: 3. / 4 * (1-u**2) * np.where(np.abs(u) <= 1, 1, 0)
kernels['gau'] = lambda u: norm.pdf(u)
kernels['log'] = lambda u: logistic.pdf(u) * (1 - logistic.pdf(u))
kernels['tri'] = lambda u: np.where(np.abs(u) <= 1, 1 - np.abs(u), 0)
kernels['trw'] = lambda u: 35. / 32 * (1 - u**2)**3 * np.where(np.abs(u) <= 1, 1, 0)
kernels['uni'] = lambda u: 1. / 2 * np.where(np.abs(u) <= 1, 1, 0)

def hall_sheather(n, q, alpha=.05):
    num = 3 * norm.pdf(norm.ppf(q))**4
    den = 2 * (2 * norm.ppf(q)**2 + 1)
    h = n**(-1./3) * norm.ppf(1 - alpha / 2)**(2./3) * (num / den)**(1./3)
    return h

def bofinger(n, q):
    num = 9./2 * norm.pdf(2 * norm.ppf(q))**4
    den = (2 * norm.ppf(q)**2 + 1)**2
    h = n**(-1./5) * (num/den)**(1./5)
    return h

def chamberlain(n, q, alpha=.05):
    return norm.ppf(1-alpha/2) * np.sqrt(q*(1-q)/n)

class QuantRegResults(RegressionResults):
    '''Results instance for the QuantReg model'''

    @cache_readonly
    def prsquared(self):
        q = self.model.q
        endog = self.model.endog
        e = self.resid
        e = np.where(e < 0, (1-q) * e, q * e)
        e = np.abs(e)
        ered = endog - stats.scoreatpercentile(endog, q * 100)
        ered = np.where(ered < 0, (1-q) * ered, q * ered)
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
