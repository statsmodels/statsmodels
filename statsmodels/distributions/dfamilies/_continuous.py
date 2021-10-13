# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 16:08:28 2021

Author: Josef Perktod
License: BSD-3
"""

import numpy as np
from scipy import stats, special

from .base import DFamily

# define some shortcuts
np_log = np.log
np_pi = np.pi
sps_gamln = special.gammaln


class Gaussian(DFamily):
    """Gaussian distribution family

    Warning: derivatives not verified yet.
    """

    k_args = 2
    names_arg = ["mean", "scale"]
    distribution = stats.norm
    domain = "real"

    def _convert_dargs_sp(self, mean, scale):
        return mean, np.sqrt(scale)

    def loglike_obs(self, endog, mean, scale):
        ll_obs = -(endog - mean) ** 2 / scale
        ll_obs += -np.log(scale) - np.log(2 * np.pi)
        ll_obs /= 2
        return ll_obs

    def score_obs(self, endog, mu, scale):
        dlldmu = - (endog - mu) / scale
        dllds = 0.5 * (- 1 / scale + (endog - mu) ** 2 / scale**2)
        return dlldmu, dllds

    def hessian_factor(self, endog, mu, scale):
        dlldmu2 = mu
        dlldms = (endog - mu) / scale**2
        dllds2 = 0.5 / scale**2 - (endog - mu) ** 2 / scale**3
        return dlldmu2, dlldms, dllds2

    def cdf(self, endog, mu, scale):
        return stats.norm.cdf(endog, mu, scale)

    def deriv_pdf(self, endog, mu, scale):
        return - endog * self.pdf(endog, mu, scale)


class StudentT(DFamily):
    """Student-t distribution family
    """

    k_args = 3
    names_arg = ["mean", "scale", "df"]
    distribution = stats.t
    domain = "real"

    def _convert_dargs_sp(self, mean, scale, df):
        return df, mean, np.sqrt(scale)

    def loglike_obs(self, endog, mean, scale, df):
        scale_sqrt = np.sqrt(scale)
        x = (endog - mean) / scale_sqrt

        llf = (-0.5 * np_log(df * np_pi) + sps_gamln((df + 1) / 2)
               - sps_gamln(df / 2.) - 0.5 * np_log(scale)  # correction scale
               - (df + 1) / 2. * np_log(1 + x**2 / df))
        return llf


class JohnsonSU(DFamily):
    """JohnsonSU distribution family
    """

    k_args = 4
    names_arg = ["mean", "scale", "shape1", "shape2"]
    distribution = stats.johnsonsu
    domain = "real"

    def _convert_dargs_sp(self, mu, scale, shape1, shape2):
        return shape1, shape2, mu, np.sqrt(scale)

    def loglike_obs(self, endog, mu, scale, shape1, shape2):
        scale_sqrt = np.sqrt(scale)
        ll_obs = stats.johnsonsu.logpdf(endog, shape1, shape2, mu, scale_sqrt)
        return ll_obs
