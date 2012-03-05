"""Linear Model with Student-t distributed errors

Because the t distribution has fatter tails than the normal distribution, it
can be used to model observations with heavier tails and observations that have
some outliers. For the latter case, the t-distribution provides more robust
estimators for mean or mean parameters (what about var?).



References
----------
Kenneth L. Lange, Roderick J. A. Little, Jeremy M. G. Taylor (1989)
Robust Statistical Modeling Using the t Distribution
Journal of the American Statistical Association
Vol. 84, No. 408 (Dec., 1989), pp. 881-896
Published by: American Statistical Association
Stable URL: http://www.jstor.org/stable/2290063

not read yet


Created on 2010-09-24
Author: josef-pktd
License: BSD

TODO
----
* add starting values based on OLS
* bugs: store_params doesn't seem to be defined, I think this was a module
        global for debugging - commented out
* parameter restriction: check whether version with some fixed parameters works


"""
#mostly copied from the examples directory written for trying out generic mle.

import numpy as np
from scipy import special #, stats
#redefine some shortcuts
np_log = np.log
np_pi = np.pi
sps_gamln = special.gammaln


from statsmodels.base.model import GenericLikelihoodModel

class TLinearModel(GenericLikelihoodModel):
    '''Maximum Likelihood Estimation of Linear Model with t-distributed errors

    This is an example for generic MLE.

    Except for defining the negative log-likelihood method, all
    methods and results are generic. Gradients and Hessian
    and all resulting statistics are based on numerical
    differentiation.

    '''


    def loglike(self, params):
        return -self.nloglikeobs(params).sum(0)

    def nloglikeobs(self, params):
        """
        Loglikelihood of linear model with t distributed errors.

        Parameters
        ----------
        params : array
            The parameters of the model. The last 2 parameters are degrees of
            freedom and scale.

        Returns
        -------
        loglike : array, (nobs,)
            The log likelihood of the model evaluated at `params` for each
            observation defined by self.endog and self.exog.

        Notes
        -----
        .. math :: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]

        The t distribution is the standard t distribution and not a standardized
        t distribution, which means that the scale parameter is not equal to the
        standard deviation.

        self.fixed_params and self.expandparams can be used to fix some
        parameters. (I doubt this has been tested in this model.)

        """
        #print len(params),
        #store_params.append(params)
        if not self.fixed_params is None:
            #print 'using fixed'
            params = self.expandparams(params)

        beta = params[:-2]
        df = params[-2]
        scale = np.abs(params[-1])  #TODO check behavior around zero
        loc = np.dot(self.exog, beta)
        endog = self.endog
        x = (endog - loc)/scale
        #next part is stats.t._logpdf
        lPx = sps_gamln((df+1)/2) - sps_gamln(df/2.)
        lPx -= 0.5*np_log(df*np_pi) + (df+1)/2.*np_log(1+(x**2)/df)
        lPx -= np_log(scale)  # correction for scale
        return -lPx


from scipy import stats
from statsmodels.tsa.arma_mle import Arma

class TArma(Arma):
    '''Univariate Arma Model with t-distributed errors

    This inherit all methods except loglike from tsa.arma_mle.Arma

    This uses the standard t-distribution, the implied variance of
    the error is not equal to scale, but ::

        error_variance = df/(df-2)*scale**2

    Notes
    -----
    This might be replaced by a standardized t-distribution with scale**2
    equal to variance

    '''

    def loglike(self, params):
        return -self.nloglikeobs(params).sum(0)


    #add for Jacobian calculation  bsejac in GenericMLE, copied from loglike
    def nloglikeobs(self, params):
        """
        Loglikelihood for arma model for each observation, t-distribute

        Notes
        -----
        The ancillary parameter is assumed to be the last element of
        the params vector
        """

        errorsest = self.geterrors(params)
        #sigma2 = np.maximum(params[-1]**2, 1e-6)  #do I need this
        #axis = 0
        #nobs = len(errorsest)

        df = params[-2]
        scale = np.abs(params[-1])
        llike  = - stats.t._logpdf(errorsest/scale, df) + np_log(scale)
        return llike

