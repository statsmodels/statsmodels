# -*- coding: utf-8 -*-
"""
Distribution Families on a restricted domain, R+ or unit interval

Created on Mon Oct 11 16:08:28 2021

Author: Josef Perktod
License: BSD-3
"""

import numpy as np
from scipy import stats, special

from .base import DFamily


# define some shortcuts
lngamma = special.gammaln


class BetaMP(DFamily):
    """Beta distribution family with mean and precision parameterization
    """
    k_args = 2
    names_arg = ["mean", "precision"]
    distribution = stats.beta
    domain = "ui"

    def _convert_dargs_sp(self, mean, precision):
        mu = mean
        phi = precision

        eps_lb = 1e-200
        alpha = np.clip(mu * phi, eps_lb, np.inf)
        beta = np.clip((1 - mu) * phi, eps_lb, np.inf)
        return alpha, beta

    def loglike_obs(self, endog, mean, precision):
        """
        Loglikelihood for observations with data arguments.

        Parameters
        ----------
        endog : ndarray
            Observed values of the response variable, endog.
            ``endog`` is currently a required argument.
        mean : ndarray
            Predicted values for first parameter, mean, of the distribution.
        precision : ndarray
            Predicted values for second parameter, precision, of the
            distribution.

        Returns
        -------
        loglike : ndarray
            The log likelihood for each observation of the model evaluated
            at `params`.
        """
        y = endog
        mu = mean
        phi = precision

        eps_lb = 1e-200
        alpha = np.clip(mu * phi, eps_lb, np.inf)
        beta = np.clip((1 - mu) * phi, eps_lb, np.inf)

        ll = (lngamma(phi) - lngamma(alpha)
              - lngamma(beta)
              + (mu * phi - 1) * np.log(y)
              + (((1 - mu) * phi) - 1) * np.log(1 - y))

        return ll


class WeibullMin(DFamily):

    k_args = 2
    names_arg = ["scale", "shape"]
    distribution = stats.weibull_min
    domain = "realpp"

    def _convert_dargs_sp(self, scale, shape):
        # Note: fixed loc = 0
        return shape, 0, np.sqrt(scale)

    def loglike_obs(self, endog, scale, shape):

        llf = stats.weibull_min.logpdf(
            endog, shape, scale=np.sqrt(scale))
        return llf
