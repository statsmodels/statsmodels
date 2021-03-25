# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:19:45 2021

Author: Josef Perktold
License: BSD-3

"""
import sys

import numpy as np
from scipy import stats, integrate as integrate

from . import transforms
from .copulas import Copula


class ArchimedeanCopula(Copula):

    def __init__(self, transform):
        super().__init__(d=2)
        self.transform = transform

    def cdf(self, u, args=()):
        '''evaluate cdf of multivariate Archimedean copula
        '''
        axis = -1
        phi = self.transform.evaluate
        phi_inv = self.transform.inverse
        cdfv = phi_inv(phi(u, *args).sum(axis), *args)
        # clip numerical noise
        out = cdfv if isinstance(cdfv, np.ndarray) else None
        cdfv = np.clip(cdfv, 0., 1., out=out)  # inplace if possible
        return cdfv

    def pdf(self, u, args=()):
        '''evaluate cdf of multivariate Archimedean copula
        '''
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

    def logpdf(self, u, args=()):
        '''evaluate cdf of multivariate Archimedean copula
        '''
        # TODO: replace by formulas, and exp in pdf
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


class FrankCopulaGeneratorBased(ArchimedeanCopula):

    # explicit BV formulas copied from Joe 1997 p. 141
    # todo: check expm1 and log1p for improved numerical precision
    def __init__(self):
        super().__init__(transforms.TransfFrank())

    def cdf(self, u, args=()):
        u = np.asarray(u)
        th = args[0]
        if u.shape[-1] == 2:
            # bivariate case
            u1, u2 = u[..., 0], u[..., 1]
            b = 1 - np.exp(-th)
            cdf = - np.log(1 - (1 - np.exp(- th * u1)) *
                           (1 - np.exp(- th * u2)) / b) / th
            return cdf
        else:
            super(FrankCopula, self).pdf(u, args=args)

    def pdf(self, u, args=()):
        u = np.asarray(u)
        th = args[0]
        if u.shape[-1] == 2:
            # bivariate case
            u1, u2 = u[..., 0], u[..., 1]
            b = 1 - np.exp(-th)
            pdf = th * b * np.exp(- th * (u1 + u2))
            pdf /= (b - (1 - np.exp(- th * u1)) * (1 - np.exp(- th * u2)))**2
            return pdf
        else:
            super(FrankCopula, self).pdf(u, args=args)

    def logpdf(self, u, args=()):
        u = np.asarray(u)
        th = args[0]
        if u.shape[-1] == 2:
            # bivariate case
            u1, u2 = u[..., 0], u[..., 1]
            b = 1 - np.exp(-th)
            pdf = np.log(th * b) - th * (u1 + u2)
            pdf -= 2 * np.log(b - (1 - np.exp(- th * u1)) *
                              (1 - np.exp(- th * u2)))
            return pdf
        else:
            super(FrankCopula, self).pdf(u, args=args)

    def cdfcond_2g1(self, u, args=()):
        u = np.asarray(u)
        th = args[0]
        if u.shape[-1] == 2:
            # bivariate case
            u1, u2 = u[..., 0], u[..., 1]
            cdfc = np.exp(- th * u1)
            cdfc /= np.expm1(-th) / np.expm1(- th * u2) + np.expm1(- th * u1)
            return cdfc
        else:
            raise NotImplementedError

    def ppfcond_2g1(self, q, u1, args=()):
        u1 = np.asarray(u1)
        th = args[0]
        if u1.shape[-1] == 1:
            # bivariate case, conditional on value of first variable
            ppfc = - np.log(1 + np.expm1(- th) /
                            ((1 / q - 1) * np.exp(-th * u1) + 1)) / th

            return ppfc
        else:
            raise NotImplementedError


class ClaytonCopula(ArchimedeanCopula):
    r"""Clayton copula.

    Dependence is greater in the negative tail than in the positive.

    .. math::

        C_\theta(u,v) = \left[ \max\left\{ u^{-\theta} + v^{-\theta} -1 ;
        0 \right\} \right]^{-1/\theta}

    with :math:`\theta\in[-1,\infty)\backslash\{0\}`.

    """

    def __init__(self, theta=1):
        if theta <= -1 or theta == 0:
            raise ValueError('Theta must be > -1 and !=0')
        self.theta = theta

    def random(self, n=1, random_state=None):
        rng = check_random_state(random_state)
        x = rng.random((n, 2))
        v = stats.gamma(1. / self.theta).rvs(size=(n, 1), random_state=rng)
        return (1 - np.log(x) / v) ** (-1. / self.theta)

    def pdf(self, u):
        a = (self.theta + 1) * np.prod(u, axis=1) ** -(self.theta + 1)
        b = np.sum(u ** -self.theta, axis=1) - 1
        c = -(2 * self.theta + 1) / self.theta
        return a * b ** c

    def cdf(self, u):
        return (np.sum(u ** (-self.theta), axis=1) - 1) ** (-1.0 / self.theta)

    def _theta_from_tau(self, tau):
        return 2 * tau / (1 - tau)


class FrankCopula(ArchimedeanCopula):
    r"""Frank copula.

    Dependence is symmetric.

    .. math::

        C_\theta(\mathbf{u}) = -\frac{1}{\theta} \log \left[ 1-
        \frac{ \prod_j (1-\exp(- \theta u_j)) }{ (1 - \exp(-\theta)-1)^{d -
        1} } \right]

    with :math:`\theta\in \mathbb{R}\backslash\{0\}, \mathbf{u} \in [0, 1]^d`.

    """

    def __init__(self, theta=2):
        if theta == 0:
            raise ValueError('Theta must be !=0')
        self.theta = theta

    def random(self, n=1, random_state=None):
        rng = check_random_state(random_state)
        x = rng.random((n, 2))
        v = stats.logser.rvs(1. - np.exp(-self.theta),
                             size=(n, 1), random_state=rng)

        return -1. / self.theta * np.log(1.
                                         + np.exp(-(-np.log(x) / v))
                                         * (np.exp(-self.theta) - 1.))

    def pdf(self, u):
        g_ = np.exp(-self.theta * np.sum(u, axis=1)) - 1
        g1 = np.exp(-self.theta) - 1

        num = -self.theta * g1 * (1 + g_)
        aux = np.prod(np.exp(-self.theta * u) - 1, axis=1) + g1
        den = aux ** 2
        return num / den

    def cdf(self, u):
        dim = u.shape[1]
        num = np.prod(1 - np.exp(- self.theta * u), axis=1)
        den = (1 - np.exp(-self.theta)) ** (dim - 1)

        return -1.0 / self.theta * np.log(1 - num / den)

    def _theta_from_tau(self, tau):
        MIN_FLOAT_LOG = np.log(sys.float_info.min)
        MAX_FLOAT_LOG = np.log(sys.float_info.max)
        EPSILON = np.finfo(np.float32).eps

        def _theta_from_tau(alpha):
            def debye(t):
                return t / (np.exp(t) - 1)

            debye_value = integrate.quad(debye, EPSILON, alpha)[0] / alpha
            return 4 * (debye_value - 1) / alpha + 1 - tau

        result = least_squares(_theta_from_tau, 1, bounds=(MIN_FLOAT_LOG,
                                                           MAX_FLOAT_LOG))
        self.theta = result.x[0]
        return self.theta


class GumbelCopula(ArchimedeanCopula):
    r"""Gumbel copula.

    Dependence is greater in the positive tail than in the negative.

    .. math::

        C_\theta(u,v) = \exp\!\left[ -\left( (-\log(u))^\theta +
        (-\log(v))^\theta \right)^{1/\theta} \right]

    with :math:`\theta\in[1,\infty)`.

    """

    def __init__(self, theta=2):
        if theta <= 1:
            raise ValueError('Theta must be > 1')
        self.theta = theta

    def random(self, n=1, random_state=None):
        rng = check_random_state(random_state)
        x = rng.random((n, 2))
        v = stats.levy_stable.rvs(
            1. / self.theta, 1., 0,
            np.cos(np.pi / (2 * self.theta)) ** self.theta,
            size=(n, 1), random_state=rng
        )
        return np.exp(-(-np.log(x) / v) ** (1. / self.theta))

    def pdf(self, u):
        xy = -np.log(u)
        xy_theta = xy ** self.theta

        sum_xy_theta = np.sum(xy_theta, axis=1)
        sum_xy_theta_theta = sum_xy_theta ** (1.0 / self.theta)

        a = np.exp(-sum_xy_theta_theta)
        b = sum_xy_theta_theta + self.theta - 1.0
        c = sum_xy_theta ** (1.0 / self.theta - 2)
        d = np.prod(xy, axis=1) ** (self.theta - 1.0)
        e = np.prod(u, axis=1) ** (- 1.0)

        return a * b * c * d * e

    def cdf(self, u):
        h = np.sum((-np.log(u)) ** self.theta, axis=1)
        cdf = np.exp(-h ** (1.0 / self.theta))
        return cdf

    def _theta_from_tau(self, tau):
        return 1 / (1 - tau)
