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


FLOAT_EPS = np.finfo(float).eps
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


class Gamma(DFamily):
    """Gamma distribution family.
    """
    k_args = 2
    names_arg = ["mean", "scale"]
    distribution = stats.gamma
    domain = "realpp"

    def _convert_dargs_sp(self, mean, scale):
        shape = 1 / scale
        scale_g = mean * scale
        return shape, 0, scale_g

    def _clean(self, x):
        """
        Helper function to trim the data so that it is in (0,inf)
        Notes
        -----
        The need for this function was discovered through usage and its
        possible that other families might need a check for validity of the
        domain.
        """
        return np.clip(x, FLOAT_EPS, np.inf)

    def loglike_obs(self, endog, mean, scale):
        endog_mu = self._clean(endog / mean)
        ll_obs = (np.log(endog_mu / scale) - endog_mu) / scale
        ll_obs -= special.gammaln(1 / scale) + np.log(endog)

        return ll_obs


class WeibullMin(DFamily):
    """Weibull (Min) distribution family.

    Note, `scale` parameter is the variance parameter. For rescaling or
    standardizing the random variables, square root of ``scale`` needs to be
    used.
    """
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
