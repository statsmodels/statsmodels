# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 16:09:01 2021

Author: Josef Perktod
License: BSD-3
"""

import numpy as np
from scipy import stats, special

from .base import DFamily


class BetaBinomialStd(DFamily):
    """BetaBinomial family with standard parameterization
    """
    k_args = 2
    names_arg = ["a", "b"]
    distribution = stats.betabinom
    domain = "intp"

    def _convert_dargs_sp(self, a, b, n_trials=None):
        return n_trials, a, b

    def loglike_obs(self, endog, a, b, n_trials=None):
        k = endog
        n = n_trials
        t1 = - (special.betaln(k + 1, n - k + 1) + np.log(n + 1))
        t2 = special.betaln(k + a, n - k + b) - special.betaln(a, b)
        return t1 + t2


class BetaBinomialScale(BetaBinomialStd):
    """BetaBinomial family with scale parameterization
    """
    k_args = 2
    names_arg = ["mean", "scale"]

    def _convert_dargs_sp(self, mean, scale, n_trials=None):
        n = n_trials
        a = mean / scale
        b = (1 - mean) / scale
        return n, a, b

    def loglike_obs(self, endog, mean, scale, n_trials=None):
        k = endog
        n = n_trials
        a = mean / scale
        b = (1 - mean) / scale
        t1 = - (special.betaln(k + 1, n - k + 1) + np.log(n + 1))
        t2 = special.betaln(k + a, n - k + b) - special.betaln(a, b)
        return t1 + t2


class BetaBinomialPrecision(BetaBinomialStd):
    """BetaBinomial family with precision parameterization
    """
    k_args = 2
    names_arg = ["mean", "precision"]

    def _convert_dargs_sp(self, mean, precision, n_trials=None):
        n = n_trials
        a = mean * precision
        b = (1 - mean) * precision
        return n, a, b

    def loglike_obs(self, endog, mean, precision, n_trials=None):
        k = endog
        n = n_trials
        a = mean * precision
        b = (1 - mean) * precision
        t1 = - (special.betaln(k + 1, n - k + 1) + np.log(n + 1))
        t2 = special.betaln(k + a, n - k + b) - special.betaln(a, b)
        return t1 + t2


class BetaBinomialDispersion(BetaBinomialStd):
    """BetaBinomial family with dispersion parameterization
    """
    k_args = 2
    names_arg = ["mean", "dispersion"]

    def _convert_dargs_sp(self, mean, dispersion, n_trials=None):
        n = n_trials
        precision = 1 / dispersion - 1
        a = mean * precision
        b = (1 - mean) * precision
        return n, a, b

    def loglike_obs(self, endog, mean, dispersion, n_trials=None):
        k = endog
        n = n_trials
        precision = 1 / dispersion - 1  # Note dispersion parameter in [0, 1]
        a = mean * precision
        b = (1 - mean) * precision
        t1 = - (special.betaln(k + 1, n - k + 1) + np.log(n + 1))
        t2 = special.betaln(k + a, n - k + b) - special.betaln(a, b)
        return t1 + t2
