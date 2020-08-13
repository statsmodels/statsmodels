# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 20:24:42 2015

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel, GenericLikelihoodModelResults


class OrderedModel(GenericLikelihoodModel):
    """Ordinal Model based on logistic or normal distribution

    The parameterization corresponds to the proportional odds model.

    The mode assumes that the endogenous variable is ordered but that the
    labels have no numeric interpretation besides the ordering.

    The model is based on a latent linear variable, where we observe only

    y_latent = X beta + u

    The observed variable is defined by the interval

    y = {0 if y_latent <= cut_0
         1 of cut_0 < y_latent <= cut_1
         ...
         K if cut_K < y_latent

    The probability of observing y=k conditional on the explanatory variables
    X is given by

    prob(y = k | x) = Prob(cut_k < y_latent <= cut_k+1)
                    = Prob(cut_k - x beta < u <= cut_k+1 - x beta
                    = F(cut_k+1 - x beta) - F(cut_k - x beta)

    Where F is the cumulative distribution of u which is either the normal
    or the logistic distribution, but can be set to any other continuous
    distribution. We use standardized distributions to avoid identifiability
    problems.


    Parameters
    ----------
    endog : array_like
        endogenous or dependent ordered categorical variable with k levels.
        Labels or values of endog will internally transformed to consecutive
        integers, 0, 1, 2, ...
    exog : array_like
        exogenous explanatory variables. This should not include an intercept.
        (TODO: verify)
    distr : string 'probit' or 'logit', or a distribution instance
        The default is currently 'probit' which uses the normal distribution
        and corresponds to an ordered Probit model. The distribution is
        assumed to have the main methods of scipy.stats distributions, mainly
        cdf, pdf and ppf. The inverse cdf, ppf, is only use to calculate
        starting values.

    Status: initial version, subclasses `GenericLikelihoodModel`

    """

    def __init__(self, endog, exog, distr='probit', **kwds):
        super(OrderedModel, self).__init__(endog, exog, **kwds)
        unique, index = np.unique(self.endog, return_inverse=True)
        self.k_levels = len(unique)
        self.endog = index
        self.labels = unique
        if distr == 'probit':
            self.distr = stats.norm
        elif distr == 'logit':
            self.distr = stats.logistic
        else:
            self.distr = distr

        self.k_vars = self.exog.shape[1]
        self.results_class = OrderedResults  #TODO: doesn't work


    def cdf(self, x):
        """cdf evaluated at x
        """
        return self.distr.cdf(x)

    def prob(self, low, upp):
        """interval probability
        """
        return np.maximum(self.cdf(upp) - self.cdf(low), 0)

    def transform_threshold_params(self, params):
        """transformation of the parameters in the optimization

        Parameters
        ----------
        params : nd_array
            contains (exog_coef, transformed_thresholds) where exog_coef are
            the coefficient for the explanatory variables in the linear term,
            transformed threshold or cutoff points. The first, lowest threshold
            is unchanged, all other thresholds are in terms of exponentiated
            increments

        Returns
        -------
        thresh : nd_array
            thresh are the thresholds or cutoff constants for the intervals.

        """
        th_params = params[-(self.k_levels - 1):]
        thresh = np.concatenate((th_params[:1], np.exp(th_params[1:]))).cumsum()
        thresh = np.concatenate(([-np.inf], thresh, [np.inf]))
        return thresh

    def transform_reverse_threshold_params(self, params):
        """obtain transformed thresholds from original thresholds, cutoff
        constants.

        """
        start_ppf = params
        thresh_params = np.concatenate((start_ppf[:1], np.log(np.diff(start_ppf[:-1]))))
        return thresh_params


    def predict(self, params, exog=None):
        """predicted probabilities for each level of the ordinal endog.


        """
        #structure of params = [beta, constants_or_thresholds]

        # explicit in several steps to avoid bugs
        th_params = params[-(self.k_levels - 1):]
        thresh = np.concatenate((th_params[:1], np.exp(th_params[1:]))).cumsum()
        thresh = np.concatenate(([-np.inf], thresh, [np.inf]))
        xb = self.exog.dot(params[:-(self.k_levels - 1)])[:,None]
        low = thresh[:-1] - xb
        upp = thresh[1:] - xb
        prob = self.prob(low, upp)
        return prob


    def loglike(self, params):

        #structure of params = [beta, constants_or_thresholds]

        thresh = np.concatenate(([-np.inf], params[-(self.k_levels - 1):], [np.inf]))

        # explicit in several steps to avoid bugs
        th_params = params[-(self.k_levels - 1):]
        thresh = np.concatenate((th_params[:1], np.exp(th_params[1:]))).cumsum()
        thresh = np.concatenate(([-np.inf], thresh, [np.inf]))
        thresh_i_low = thresh[self.endog]
        thresh_i_upp = thresh[self.endog + 1]
        xb = self.exog.dot(params[:-(self.k_levels - 1)])
        low = thresh_i_low - xb
        upp = thresh_i_upp - xb
        prob = self.prob(low, upp)
        return np.log(prob + 1e-20).sum()

    @property
    def start_params(self):
        # start params based on model without exog
        freq = np.bincount(self.endog) / len(self.endog)
        start_ppf = self.distr.ppf(np.clip(freq.cumsum(), 0, 1))
        start_threshold = self.transform_reverse_threshold_params(start_ppf)
        start_params = np.concatenate((np.zeros(self.k_vars), start_threshold))
        return start_params


class OrderedResults(GenericLikelihoodModelResults):

    pass
