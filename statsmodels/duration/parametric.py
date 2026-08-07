# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 22:19:01 2021

Author: Josef Perktod
License: BSD-3
"""

import numpy as np
from scipy import stats

from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.genmod.families import links


def weibull_min_logsf(x, c, scale=1):
    x = x / scale
    return -np.power(x, c)


class WeibullModel(GenericLikelihoodModel):
    """Weibull model with link for scale

    """

    def __init__(self, endog, exog, censored=None,
                 link=links.Log(), **kwds):
        self.link = link
        super(WeibullModel, self).__init__(endog, exog, **kwds)
        if censored is not None:
            self.censored = censored.astype(int)
        else:
            self.censored = np.zeros(len(self.endog), np.int)

        self.k_params = self.exog.shape[1] + 1
        self.nobs = self.endog.shape[0]
        self.k_extra = 1
        self.k_constant = self.df_null = 2  # two parameters in null model
        self.df_resid = self.nobs - self.k_params

    def loglike(self, params):
        params_ex = params[:-1]
        params_shape = params[-1]
        linpred = self.exog.dot(params_ex)
        m = self.link.inverse(linpred)
        llf = (1 - self.censored) * stats.weibull_min.logpdf(
            self.endog, params_shape, scale=m)
        # stats.weibull_min.logsf overflows in my older version of scipy,
        # changed in newer versions
        # llf2 = self.censored * stats.weibull_min.logsf(
        #     self.endog, params_shape, scale=m)
        llf += self.censored * weibull_min_logsf(self.endog,
                                                 params_shape, scale=m)
        return llf.sum()

    def get_distribution(self, params, exog=None):
        """similar to a predict method
        """
        if exog is None:
            exog = self.exog
        params_ex = params[:-1]
        params_shape = params[-1]
        linpred = exog.dot(params_ex)
        m = self.link.inverse(linpred)
        return stats.weibull_min(params_shape, scale=m)


class ExtremeValueModel(GenericLikelihoodModel):
    """Gumbel extreme value Model

    Linear link for location parameter, and constant scale

    """

    def __init__(self, endog, exog, censored=None, **kwds):
        super(ExtremeValueModel, self).__init__(endog, exog, **kwds)
        if censored is not None:
            self.censored = censored.astype(int)
        else:
            self.censored = np.zeros(len(self.endog), np.int)

        self.k_params = self.exog.shape[1] + 1
        self.nobs = self.endog.shape[0]
        self.k_extra = 1
        self.k_constant = self.df_null = 2  # two parameters in null model
        self.df_resid = self.nobs - self.k_params

    def loglike(self, params):
        params_ex = params[:-1]
        params_shape = params[-1]
        m = self.exog.dot(params_ex)
        llf = (1 - self.censored) * stats.gumbel_l.logpdf(self.endog, loc=m,
                                                          scale=params_shape)

        llf += self.censored * stats.gumbel_l.logsf(self.endog,
                                                    loc=m, scale=params_shape)
        return llf.sum()

    def get_distribution(self, params, exog=None, distr="extreme-value"):
        """Create frozen distribution based on predicted values.

        similar to a predict method

        Returns
        -------
        Frozen scipy distribution instance, either ``wibull_min`` or
        ``gumbel_l``.
        """
        if exog is None:
            exog = self.exog
        params_ex = params[:-1]
        params_shape = params[-1]
        m = np.exp(exog.dot(params_ex))
        if distr == "weibull":
            d = stats.weibull_min(1 / params_shape, scale=m)
        elif distr in ["ev", "extreme-value"]:
            d = stats.gumbel_l(loc=m, scale=params_shape)
        else:
            raise ValueError('distr should be "extreme-value" or "weibull"')

        return d
