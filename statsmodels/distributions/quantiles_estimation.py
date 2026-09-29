# -*- coding: utf-8 -*-
"""Using empirical quantiles to estimate loc and scale

Created on Sun Aug 19 23:10:25 2012
Author: Josef Perktold

random idea for future while looking at order statistics:

for qqplot, probability plot
like the line, it would be nice to give an option to add confidence intervals
as visual aid
(asymptotically it doesn't look difficult, something
like http://en.wikipedia.org/wiki/Order_statistic#Large_sample_sizes and maybe in the SAS help based on estimated distribution I guess.
It might not be very accurate in small samples.)

also with outliers providing a line estimated by RLM is better



using qqplot, quantiles to estimate loc and scale for fixed shape parameters

It is (or is similar to) GMM on percentiles with an identity weight matrix.
(where's my GMM version?)

References
----------

Hassanein Biometrika 1971, Percentile Estimators for the Parameters of the
Weibull Distribution.
    typo in formula for V^{-1}, original in Hammersley and Morton 1964

Hammersley, J. M. and K. W. Morton (1964):
The Estimation of Location and Scale Parameters from Grouped Data,
Biometrika, Vol. 41, No. 3/4 (Dec., 1954), pp. 296-301
Stable URL: http://www.jstor.org/stable/2332710 .


"""

import numpy as np
from scipy import stats
import statsmodels.api as sm


class QuantilesDistUniform(object):
    '''distribution of quantiles of the uniform distribution

    Warning : no error checking for valid arguments.
        p in interval [0,1] or (0,1)
        kth in range(1,nobs)

    What I need or want:
        * theoretical distribution, expected for p
        * distribution at all data points, kth = range(1,nobs)
        * distribution only at certain orders for estimation
        * estimation of loc and scale, given shape parameters
        * compare with GMM code
        * use for Probability Plot confidence intervals

    INCOMPLETE

    '''

    def __init__(self, data=None, sorted=False, nobs=None):

        #maybe data shouldn't be optional
        if not data is None:
            data = np.asarray(data)
            self.nobs = data.shape[0]
            if not sorted:
                self.data = np.sort(data)
            else:
                self.data = data
        else:
            self.nobs = nobs
            if nobs is None:
                raise ValueError('Either data or nobs needs to be given' )

        self.beta_marginal = stats.beta(data, )

    def beta_marginal(self, kth):
        '''kth order statistic
        '''
        return stats.beta(kth, self.nobs + 1 - kth)

    def normal_marginal_mean(self, kth):
        p = kth * 1.0 / self.nobs   #conversion to int is reversed
        return p

    def normal_marginal_var(self, kth):
        #add check for k in range(1, nobs), what about bounds, zero and nobs ?
        p = kth * 1.0 / self.nobs   #conversion to int is reversed
        return p * (1-p)

    def mvn_joint_mean(self, p):
        p = kth * 1.0 / self.nobs   #conversion to int is reversed
        return p

    def mnv_joint_cov_f(self, p1, p2):
        #unfinished
        #currentl implementation in LocScaleQEst
        #for i <= j lower triangle, i.e. p1 <= p2
        denom = pdf(ppf(p1)) * pdf(ppf(p2))
        c = p1 * (1-p2) / denom / self.nobs
        return c

def locscale(params):
    return dict(loc=params[-2], scale=params[-1])

def band2diag(x, symm=True):
    '''

    x : list of list

    requires x to be in lower diagonal form
    symm or upper triangular band
    '''
    #x = np.asarray(x)
    #d, n = x.shape
    #works for ndarray and list of lists
    n = len(x[0])
    d = len(x)
    y = np.zeros((n,n), dtype=np.asarray(x[0][0]).dtype)
    idxn = np.arange(n)

    y.flat[idxn * (n+1)] = x[0]
    for k in range(1,d):
        #use trick from np.diagflat
        #upper
        i = idxn[:n-k]
        idx = i+k+i*n
        y.flat[idx] = x[k]  #need to shorten if array
        if symm:
            #lower
            idx = i+(i+k)*n
            y.flat[idx] = x[k]

    return y


class LocScaleQEst(object):
    '''Estimate location and scale based on Quantiles

    similar to fitting a straight line to a qq-plot, but with
    more options

    This does not take into account that we only have two parameter, constant
    and slope in the regression, no shortcuts in OLS are used.

    TODO: add cached attributes to avoid recalculation if fit is used several
    times with different options

    '''
    def __init__(self, data, dist=stats.norm, distargs=()):
        self.data = np.array(data)  #need copy
        if data.ndim !=1:
            raise ValueError('need 1-D data')
        self.nobs = len(self.data)

        self.dist = dist
        self.distargs = distargs

        self.endog = np.sort(self.data) #get sorted copy
        self._initialize()

    def _initialize(self):
        dist = self.dist
        distargs = self.distargs
        nobs = self.nobs

        kth = np.arange(1, nobs+1, dtype=float)
        self.probs = probs = kth / (nobs + 1.)    #lambda
        #probs =   # check unequal spacing of probs
        self.ppf_x = ppf_x = dist.ppf(probs, *distargs) #U
        self.pdf_x = pdf_x = dist.pdf(ppf_x, *distargs) #f
        #d_probs = np.diff(probs)

        #get covariance matrix of moment conditions
        fact_low = probs / pdf_x
        fact_high = (1. - probs) / pdf_x
        v = fact_low[:,None] * fact_high  #upper triangle is correct
        larger = kth[:,None] > kth
        #smaller = kth[:,None] < kth
        v[larger] = v.T[larger]
        self.v_moms = v / nobs

        #for regression

        self.exog = sm.add_constant(ppf_x, prepend=True) #just 2 columns

    def fit(self, method='GLS', rlm_options=None):
        '''estimate loc and scale from empirical quantiles

        Parameters
        ----------
        method : string in {'GLS', 'OLS', 'RLM', 'G-RLM', 'GLS2, 'G-RLM2'}
            'GLS' : is the default and uses the theoretical covariance matrix of
                the quantiles.
            'OLS', 'RLM' : use the corresponding estimation class without taking
                the covariance of quantiles into account.
            'G-RLM' : whitens the data before calling RLM
            'GLS2', 'G-RLM2' : produce the same outcome as 'GLS' and 'G-RLS' but
                use explicit inversion and cholesky decomposition of the
                (nobs, nobs) covariance matrix of quantiles.
                not recommended, for testing only
            'sp_mom' : uses scipy ``fit_loc_scale`` method of the distribution
                Note: returns directly loc and scale
                for comparison, it uses first two sample moments not quantiles
        rlm_options : None or dict
            options used without changes in the creation of the ``RLM``
            instance.

        Returns
        -------
        result : Result instance
            The unchanged instance of the result of the model fit.
            ``params`` contains location in the first and scale in the second
            index.
            Note, if method is 'sp_mom', then loc and scale are directly
            returned.

        '''

        if rlm_options is None:
            rlm_options = {}

        endog, exog = self.endog, self.exog
        if method == 'GLS2':
            return sm.GLS(endog, exog, sigma=self.v_moms).fit()
        elif method == 'OLS':
            return sm.OLS(endog, exog).fit()
        elif method == 'RLM':
            return sm.RLM(endog, exog).fit()
        elif method == 'G-RLM2':
            #use GLS class to do the whitening
            #this part should get explicit whitening
            mod_gls = sm.GLS(endog, exog, sigma=self.v_moms)
            wendog, wexog = mod_gls.wendog, mod_gls.wexog
            return sm.RLM(wendog, wexog, **rlm_options).fit()
        elif method == 'GLS':
            wendog = self.whiten(self.endog)
            wexog = self.whiten(self.exog)
            return sm.GLS(wendog, wexog).fit()
        elif method == 'G-RLM':
            wendog = self.whiten(self.endog)
            wexog = self.whiten(self.exog)
            return sm.RLM(wendog, wexog, **rlm_options).fit()
        elif method == 'sp_mom':
            res = self.dist.fit_loc_scale(endog, *self.distargs)
            return res
        else:
            raise ValueError('invalid method argument')



    def vinv_diags(self, equal_spaced='True'):
        '''calculate diagonals of the inverse covariance matrix of quantiles

        Parameters
        ----------
        equal_spaced : bool
            'True' (default),  Is for debugging, see notes.

        Returns
        -------
        diags : list
            the list contains two ndarrays, the first is the main diagonal,
            the second is the diagonal above and below the main diagonal.
            The corresponding covariance matrix is banded and symmetric.

        Notes
        -----
        In the estimation we only need equal spaced probabilities. The case
        for unequal spaced probabilities has not been tested yet.

        '''
        probs = self.probs
        pdf_x = self.pdf_x
        nobs = self.nobs

        probs_e = np.concatenate(([0], probs, [1])) #extended
        d_probs_e = np.diff(probs_e)
        if np.all(d_probs_e == 1. / self.nobs + 1):
            fac = (nobs + 1) * nobs
            vii = pdf_x**2 * 2 * fac
            vij = - pdf_x[1:] * pdf_x[:-1] * fac
        else:
            ds2_probs_e = probs_e[2:] - probs_e[:-2]
            vii = pdf_x**2 * ds2_probs_e / d_probs_e[1:] / d_probs_e[:-1] * nobs
            vij = - pdf_x[1:] * pdf_x[:-1] / d_probs_e[2:] * nobs

        return [vii, vij]

    def chol_vinv_diags(self, diags):
        '''diagonals of the cholesky decomposition of the covariance matrix

        Parameters
        ----------
        diags : list
            list of main diagonal and one off-diagonal of symmetric, banded
            matrix

        Returns
        -------
        diags_chol : list
            list of two diagonals of cholesky decomposition of matrix given
            by ``diags``

        Notes
        -----
        doesn't use any attributes
        '''
        #what's minimum scipy version ?
        from scipy.linalg import cholesky_banded
#        n = len(diags[0])
#        d = len(diags)
#        band_low = np.zeros((d,x))
#        band_low[i] = diags
        #use that we only have one off-diagonal
        band_low = np.concatenate((diags[0], diags[1], [0])).reshape(2,-1)
        result = cholesky_banded(band_low, lower=True)
        return result[0], result[1][:-1]

    def whiten(self, x):
        '''linear transformation, whitening of x with "cholsigmainv"

        Parameters
        ----------
        x : ndarray, 1-D or 2-D
            array to be whitened

        Results
        wx : ndarray
            transformed array with same shape as x

        special case: diags is banded symmetric with 1 off-diagonal

        '''
        diags = self.vinv_diags()
        chdiags = self.chol_vinv_diags(diags)
        if x.ndim == 2:
            chdiags = [chdiags[0][:,None], chdiags[1][:,None]]

        #cholesky has only diagonal and one lower off-diagonal
        res = x * chdiags[0]
        #res[1:] += x[1:] * chdiags[1]
        #res[:-1] += x[:-1] * chdiags[1]
        res[:-1] += x[1:] * chdiags[1]
        return res
