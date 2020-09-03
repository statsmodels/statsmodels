# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 20:24:42 2015

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel, \
    GenericLikelihoodModelResults
from statsmodels.compat.pandas import Appender


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
        pd.Series with Categorical as dtype should be preferred as it gives
        the order relation between the levels.
    exog : array_like
        exogenous explanatory variables. This should not include an intercept.
        (TODO: verify)
        pd.DataFrame are also accepted.
    distr : string 'probit' or 'logit', or a distribution instance
        The default is currently 'probit' which uses the normal distribution
        and corresponds to an ordered Probit model. The distribution is
        assumed to have the main methods of scipy.stats distributions, mainly
        cdf, pdf and ppf. The inverse cdf, ppf, is only use to calculate
        starting values.

    Status: initial version, subclasses `GenericLikelihoodModel`

    """

    def __init__(self, endog, exog, offset=None, distr='probit', **kwds):

        if distr == 'probit':
            self.distr = stats.norm
        elif distr == 'logit':
            self.distr = stats.logistic
        else:
            self.distr = distr

        if offset is not None:
            offset = np.asarray(offset)

        # TODO: check if super can handle offset
        self.offset = offset

        self.names, endog, exog = self._check_inputs(endog, exog)

        super(OrderedModel, self).__init__(endog, exog, **kwds)

        unique, index = np.unique(self.endog, return_inverse=True)
        self.k_levels = len(unique)
        self.endog = index
        self.labels = unique

        if self.exog is not None:
            self.nobs, self.k_vars = self.exog.shape
        else:  # no exog in model
            self.nobs, self.k_vars = self.endog.shape[0], 0
        self.results_class = OrderedResults

    def _check_inputs(self, endog, exog):
        """
        checks if self.distrib is legal and does the Pandas
        support for endog and exog. Also retrieves columns & categories
        names for .summary() of the results class.
        """
        names = {}
        if not isinstance(self.distr, stats.rv_continuous):
            msg = (
                f"{self.distr.name} must be a scipy.stats distribution."
            )
            raise ValueError(msg)

        # Pandas' support
        if (isinstance(exog, pd.DataFrame)) or (isinstance(exog, pd.Series)):
            exog_name = ([exog.name] if isinstance(exog, pd.Series)
                         else exog.columns.tolist())
            names['xname'] = exog_name
            exog = np.asarray(exog)

        if isinstance(endog, pd.Series):
            if isinstance(endog.dtypes, CategoricalDtype):
                if not endog.dtype.ordered:
                    import warnings
                    warnings.warn("the endog has ordered == False, "
                                  "risk of capturing a wrong order for the "
                                  "categories. ordered == True preferred.",
                                  Warning)
                endog_name = endog.name
                threshold_name = [str(x) + '/' + str(y)
                                  for x, y in zip(endog.values.categories[:-1],
                                                  endog.values.categories[1:])]
                names['yname'] = endog_name
                names['xname'] = names['xname'] + threshold_name
                endog = np.asarray(endog.values.codes)
            else:
                msg = (
                    "If the endog is a pandas.Serie "
                    "it must be of categoricalDtype."
                )
                raise ValueError(msg)

        return names, endog, exog

    def cdf(self, x):
        """cdf evaluated at x
        """
        return self.distr.cdf(x)

    def pdf(self, x):
        """pdf evaluated at x
        """
        return self.distr.pdf(x)

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
        thresh = np.concatenate((th_params[:1],
                                 np.exp(th_params[1:]))).cumsum()
        thresh = np.concatenate(([-np.inf], thresh, [np.inf]))
        return thresh

    def transform_reverse_threshold_params(self, params):
        """obtain transformed thresholds from original thresholds, cutoff
        constants.

        """
        start_ppf = params
        thresh_params = np.concatenate((start_ppf[:1],
                                        np.log(np.diff(start_ppf[:-1]))))
        return thresh_params

    def predict(self, params, exog=None):
        """predicted probabilities for each level of the ordinal endog.


        """
        if exog is None:
            exog = self.exog
        # structure of params = [beta, constants_or_thresholds]

        # explicit in several steps to avoid bugs
        th_params = params[-(self.k_levels - 1):]
        thresh = np.concatenate((th_params[:1],
                                 np.exp(th_params[1:]))).cumsum()
        thresh = np.concatenate(([-np.inf], thresh, [np.inf]))
        xb = exog.dot(params[:-(self.k_levels - 1)])[:, None]
        low = thresh[:-1] - xb
        upp = thresh[1:] - xb
        prob = self.prob(low, upp)
        return prob

    def _linpred(self, params, exog=None, offset=None):
        """linear prediction of latent variable `x b`

        currently only for exog from estimation sample (in-sample)
        """
        if exog is None:
            exog = self.exog
        if offset is None:
            offset = self.offset
        if exog is not None:
            linpred = self.exog.dot(params[:-(self.k_levels - 1)])
        else:  # means self.exog is also None
            linpred = np.zeros(self.nobs)
        if offset is not None:
            linpred += offset
        return linpred

    def _bounds(self, params):
        thresh = self.transform_threshold_params(params)

        thresh_i_low = thresh[self.endog]
        thresh_i_upp = thresh[self.endog + 1]
        xb = self._linpred(params)
        low = thresh_i_low - xb
        upp = thresh_i_upp - xb
        return low, upp

    def loglike(self, params):

        thresh = self.transform_threshold_params(params)

        thresh_i_low = thresh[self.endog]
        thresh_i_upp = thresh[self.endog + 1]
        xb = self._linpred(params)
        low = thresh_i_low - xb
        upp = thresh_i_upp - xb
        prob = self.prob(low, upp)
        return np.log(prob + 1e-20).sum()

    def loglikeobs(self, params):

        low, upp = self._bounds(params)
        prob = self.prob(low, upp)
        return np.log(prob + 1e-20)

    def score_obs_(self, params):
        """score, first derivative of loglike for each observations

        This currently only implements the derivative with respect to the
        exog parameters, but not with respect to threshold-constant parameters.

        """
        low, upp = self._bounds(params)

        prob = self.prob(low, upp)
        pdf_upp = self.pdf(upp)
        pdf_low = self.pdf(low)

        # TODO the following doesn't work yet because of the incremental exp
        # parameterization. The following was written base on Greene for the
        # simple non-incremental parameterization.
        # k = self.k_levels - 1
        # idx = self.endog
        # score_factor = np.zeros((self.nobs, k + 1 + 2)) #+2 avoids idx bounds
        #
        # rows = np.arange(self.nobs)
        # shift = 1
        # score_factor[rows, shift + idx-1] = -pdf_low
        # score_factor[rows, shift + idx] = pdf_upp
        # score_factor[:, 0] = pdf_upp - pdf_low
        score_factor = (pdf_upp - pdf_low)[:, None]
        score_factor /= prob[:, None]

        so = np.column_stack((-score_factor[:, :1] * self.exog,
                              score_factor[:, 1:]))
        return so

    @property
    def start_params(self):
        # start params based on model without exog
        freq = np.bincount(self.endog) / len(self.endog)
        start_ppf = self.distr.ppf(np.clip(freq.cumsum(), 0, 1))
        start_threshold = self.transform_reverse_threshold_params(start_ppf)
        start_params = np.concatenate((np.zeros(self.k_vars), start_threshold))
        return start_params

    @Appender(GenericLikelihoodModel.fit.__doc__)
    def fit(self, start_params=None, method='nm', maxiter=500, full_output=1,
            disp=1, callback=None, retall=0, **kwargs):

        fit_method = super(OrderedModel, self).fit
        mlefit = fit_method(start_params=start_params,
                            method=method, maxiter=maxiter,
                            full_output=full_output,
                            disp=disp, callback=callback, **kwargs)
        # use the proper result class
        ordmlefit = OrderedResults(self, mlefit)

        return ordmlefit


class OrderedResults(GenericLikelihoodModelResults):
    @Appender(GenericLikelihoodModelResults.summary.__doc__)
    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        names = self.model.names
        return super(OrderedResults, self).summary(**names)
