# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:19:45 2021

Author: Josef Perktold
License: BSD-3

"""
import sys

import numpy as np

import scipy.optimize as optimize
import scipy.stats as stats
import scipy.integrate as integrate
from scipy._lib._util import check_random_state  # noqa

from . import transforms
from .copulas import Copula


class ArchimedeanCopula(Copula):

    def __init__(self, transform, args=None):
        super().__init__(d=2, args=args)
        self.transform = transform

    def cdf(self, u, args=None):
        """Evaluate CDF of multivariate Archimedean copula."""
        args = self._validate_args(args)

        axis = -1
        phi = self.transform.evaluate
        phi_inv = self.transform.inverse
        cdfv = phi_inv(phi(u, *args).sum(axis), *args)
        # clip numerical noise
        out = cdfv if isinstance(cdfv, np.ndarray) else None
        cdfv = np.clip(cdfv, 0., 1., out=out)  # inplace if possible
        return cdfv

    def pdf(self, u, args=None):
        """Evaluate PDF of multivariate Archimedean copula."""
        args = self._validate_args(args)
        axis = -1
        u = np.asarray(u)
        if u.shape[-1] > 2:
            msg = "pdf is currently only available for bivariate copula"
            raise ValueError(msg)
        # phi = self.transform.evaluate
        # phi_inv = self.transform.inverse
        phi_d1 = self.transform.deriv
        phi_d2 = self.transform.deriv2

        cdfv = self.cdf(u, args=args)

        pdfv = - np.product(phi_d1(u, *args), axis)
        pdfv *= phi_d2(cdfv, *args)
        pdfv /= phi_d1(cdfv, *args)**3

        return pdfv

    def logpdf(self, u, args=None):
        """Evaluate log PDF of multivariate Archimedean copula."""
        # TODO: replace by formulas, and exp in pdf
        args = self._validate_args(args)
        axis = -1
        u = np.asarray(u)
        if u.shape[-1] > 2:
            msg = "pdf is currently only available for bivariate copula"
            raise ValueError(msg)

        phi_d1 = self.transform.deriv
        phi_d2 = self.transform.deriv2

        cdfv = self.cdf(u, args=args)

        # I need np.abs because derivatives are negative,
        # is this correct for mv?
        logpdfv = np.sum(np.log(np.abs(phi_d1(u, *args))), axis)
        logpdfv += np.log(np.abs(phi_d2(cdfv, *args) / phi_d1(cdfv, *args)**3))

        return logpdfv


class ClaytonCopula(ArchimedeanCopula):
    r"""Clayton copula.

    Dependence is greater in the negative tail than in the positive.

    .. math::

        C_\theta(u,v) = \left[ \max\left\{ u^{-\theta} + v^{-\theta} -1 ;
        0 \right\} \right]^{-1/\theta}

    with :math:`\theta\in[-1,\infty)\backslash\{0\}`.

    """

    def __init__(self, args=None):
        if args <= -1 or args == 0:
            raise ValueError('Theta must be > -1 and !=0')

        super().__init__(transforms.TransfClayton(), args=args)

    def random(self, n=1, random_state=None, args=None):
        theta = self._validate_args(args)[0]
        rng = check_random_state(random_state)
        x = rng.random((n, 2))
        v = stats.gamma(1. / theta).rvs(size=(n, 1), random_state=rng)
        return (1 - np.log(x) / v) ** (-1. / theta)

    def pdf(self, u, args=None):
        theta = self._validate_args(args)[0]
        a = (theta + 1) * np.prod(u, axis=1) ** -(theta + 1)
        b = np.sum(u ** -theta, axis=1) - 1
        c = -(2 * theta + 1) / theta
        return a * b ** c

    def cdf(self, u, args=None):
        theta = self._validate_args(args)[0]
        return (np.sum(u ** (-theta), axis=1) - 1) ** (-1.0 / theta)

    def _theta_from_tau(self, tau):
        self.args = 2 * tau / (1 - tau)
        return self.args


class FrankCopula(ArchimedeanCopula):
    r"""Frank copula.

    Dependence is symmetric.

    .. math::

        C_\theta(\mathbf{u}) = -\frac{1}{\theta} \log \left[ 1-
        \frac{ \prod_j (1-\exp(- \theta u_j)) }{ (1 - \exp(-\theta)-1)^{d -
        1} } \right]

    with :math:`\theta\in \mathbb{R}\backslash\{0\}, \mathbf{u} \in [0, 1]^d`.

    """

    def __init__(self, args=None):
        if args == 0:
            raise ValueError('Theta must be !=0')

        super().__init__(transforms.TransfFrank(), args=args)

    def random(self, n=1, random_state=None, args=None):
        args = self._validate_args(args)[0]
        rng = check_random_state(random_state)
        x = rng.random((n, 2))
        v = stats.logser.rvs(1. - np.exp(-args),
                             size=(n, 1), random_state=rng)

        return -1. / args * np.log(1.
                                   + np.exp(-(-np.log(x) / v))
                                   * (np.exp(-args) - 1.))

    # explicit BV formulas copied from Joe 1997 p. 141
    # todo: check expm1 and log1p for improved numerical precision

    def pdf(self, u, args=None):
        theta = self._validate_args(args)[0]
        u = np.asarray(u)
        if u.shape[1] != 2:
            return super().pdf(u)

        g_ = np.exp(-theta * np.sum(u, axis=1)) - 1
        g1 = np.exp(-theta) - 1

        num = -theta * g1 * (1 + g_)
        aux = np.prod(np.exp(-theta * u) - 1, axis=1) + g1
        den = aux ** 2
        return num / den

    def cdf(self, u, args=None):
        theta = self._validate_args(args)[0]
        u = np.asarray(u)
        dim = u.shape[1]
        if dim != 2:
            return super().cdf(u)

        num = np.prod(1 - np.exp(-theta * u), axis=1)
        den = (1 - np.exp(-theta)) ** (dim - 1)

        return -1.0 / theta * np.log(1 - num / den)

    def logpdf(self, u, args=None):
        theta = self._validate_args(args)[0]
        u = np.asarray(u)
        if u.shape[-1] == 2:
            # bivariate case
            u1, u2 = u[..., 0], u[..., 1]
            b = 1 - np.exp(-theta)
            pdf = np.log(theta * b) - theta * (u1 + u2)
            pdf -= 2 * np.log(b - (1 - np.exp(- theta * u1)) *
                              (1 - np.exp(- theta * u2)))
            return pdf
        else:
            super().logpdf(u)

    def cdfcond_2g1(self, u, args=None):
        theta = self._validate_args(args)[0]
        u = np.asarray(u)
        if u.shape[-1] == 2:
            # bivariate case
            u1, u2 = u[..., 0], u[..., 1]
            cdfc = np.exp(-theta * u1)
            cdfc /= np.expm1(-theta) / np.expm1(-theta * u2) + np.expm1(-theta
                                                                        * u1)
            return cdfc
        else:
            raise NotImplementedError

    def ppfcond_2g1(self, q, u1, args=None):
        theta = self._validate_args(args)[0]
        u1 = np.asarray(u1)
        if u1.shape[-1] == 1:
            # bivariate case, conditional on value of first variable
            ppfc = - np.log(1 + np.expm1(-theta) /
                            ((1 / q - 1) * np.exp(-theta * u1) + 1)) / theta

            return ppfc
        else:
            raise NotImplementedError


    def _theta_from_tau(self, tau):
        MIN_FLOAT_LOG = np.log(sys.float_info.min)
        MAX_FLOAT_LOG = np.log(sys.float_info.max)
        EPSILON = np.finfo(np.float32).eps

        def _theta_from_tau(alpha):
            def debye(t):
                return t / (np.exp(t) - 1)

            debye_value = integrate.quad(debye, EPSILON, alpha)[0] / alpha
            return 4 * (debye_value - 1) / alpha + 1 - tau

        result = optimize.least_squares(_theta_from_tau, 1, bounds=(
            MIN_FLOAT_LOG,
                                                           MAX_FLOAT_LOG))
        self.args = result.x[0]
        return self.args


class GumbelCopula(ArchimedeanCopula):
    r"""Gumbel copula.

    Dependence is greater in the positive tail than in the negative.

    .. math::

        C_\theta(u,v) = \exp\!\left[ -\left( (-\log(u))^\theta +
        (-\log(v))^\theta \right)^{1/\theta} \right]

    with :math:`\theta\in[1,\infty)`.

    """

    def __init__(self, args=None):
        if args <= 1:
            raise ValueError('Theta must be > 1')

        super().__init__(transforms.TransfGumbel(), args=args)

    def random(self, n=1, random_state=None, args=None):
        theta = self._validate_args(args)[0]
        rng = check_random_state(random_state)
        x = rng.random((n, 2))
        v = stats.levy_stable.rvs(
            1. / theta, 1., 0,
            np.cos(np.pi / (2 * theta)) ** theta,
            size=(n, 1), random_state=rng
        )
        return np.exp(-(-np.log(x) / v) ** (1. / theta))

    def pdf(self, u, args=None):
        theta = self._validate_args(args)[0]
        xy = -np.log(u)
        xy_theta = xy ** theta

        sum_xy_theta = np.sum(xy_theta, axis=1)
        sum_xy_theta_theta = sum_xy_theta ** (1.0 / theta)

        a = np.exp(-sum_xy_theta_theta)
        b = sum_xy_theta_theta + theta - 1.0
        c = sum_xy_theta ** (1.0 / theta - 2)
        d = np.prod(xy, axis=1) ** (theta - 1.0)
        e = np.prod(u, axis=1) ** (- 1.0)

        return a * b * c * d * e

    def cdf(self, u, args=None):
        theta = self._validate_args(args)[0]
        h = np.sum((-np.log(u)) ** theta, axis=1)
        cdf = np.exp(-h ** (1.0 / theta))
        return cdf

    def _theta_from_tau(self, tau):
        self.args = 1 / (1 - tau)
        return self.args
