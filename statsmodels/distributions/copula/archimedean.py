# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:19:45 2021

Author: Josef Perktold
License: BSD-3

"""
import sys

import numpy as np
from scipy import stats, integrate, optimize

from . import transforms
from .copulas import Copula
from statsmodels.tools.rng_qrng import check_random_state


def _debye(alpha):
    EPSILON = np.finfo(np.float32).eps

    def integrand(t):
        return t / (np.exp(t) - 1)

    debye_value = integrate.quad(integrand, EPSILON, alpha)[0] / alpha
    return debye_value


class ArchimedeanCopula(Copula):
    """Base class for Archimedean copulas

    Parameters
    ----------
    transform : instance of transformation class
        Archimedean generator with required methods including first and second
        derivatives
    args : tuple
        Optional copula parameters. Copula parameters can be either provided
        when creating the instance or as arguments when calling methods.
    k_dim : int
        Dimension, number of components in the multivariate random variable.
        Currently only bivariate copulas are verified. Support for more than
        2 dimension is incomplete.
    """

    def __init__(self, transform, args=(), k_dim=2):
        super().__init__(k_dim=k_dim)
        self.args = args
        self.transform = transform
        self.k_args = 1

    def _handle_args(self, args):
        # TODO: how to we handle non-tuple args? two we allow single values?
        # Model fit might give an args that can be empty
        if isinstance(args, np.ndarray):
            args = tuple(args)  # handles empty arrays, unpacks otherwise
        if not isinstance(args, tuple):
            # could still be a scalar or numpy scalar
            args = (args,)
        if len(args) == 0 or args == (None,):
            # second condition because we converted None to tuple
            args = self.args

        return args

    def cdf(self, u, args=()):
        """Evaluate cdf of Archimedean copula."""
        args = self._handle_args(args)
        axis = -1
        phi = self.transform.evaluate
        phi_inv = self.transform.inverse
        cdfv = phi_inv(phi(u, *args).sum(axis), *args)
        # clip numerical noise
        out = cdfv if isinstance(cdfv, np.ndarray) else None
        cdfv = np.clip(cdfv, 0., 1., out=out)  # inplace if possible
        return cdfv

    def pdf(self, u, args=()):
        """Evaluate pdf of Archimedean copula."""
        args = self._handle_args(args)
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
        """Evaluate log pdf of multivariate Archimedean copula."""
        # TODO: replace by formulas, and exp in pdf
        args = self._handle_args(args)
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

    def _arg_from_tau(self, tau):
        # for generic compat
        return self.theta_from_tau(tau)


class ClaytonCopula(ArchimedeanCopula):
    r"""Clayton copula.

    Dependence is greater in the negative tail than in the positive.

    .. math::

        C_\theta(u,v) = \left[ \max\left\{ u^{-\theta} + v^{-\theta} -1 ;
        0 \right\} \right]^{-1/\theta}

    with :math:`\theta\in[-1,\infty)\backslash\{0\}`.

    """

    def __init__(self, theta=None, k_dim=2):
        if theta is not None:
            args = (theta,)
        else:
            args = ()
        super().__init__(transforms.TransfClayton(), args=args, k_dim=k_dim)

        if theta is not None:
            if theta <= -1 or theta == 0:
                raise ValueError('Theta must be > -1 and !=0')
        self.theta = theta

    def rvs(self, nobs=1, args=(), random_state=None):
        if self.k_dim != 2:
            msg = "rvs is only available for bivariate copula"
            raise NotImplementedError(msg)
        rng = check_random_state(random_state)
        th, = self._handle_args(args)
        x = rng.random((nobs, self.k_dim))
        v = stats.gamma(1. / th).rvs(size=(nobs, 1), random_state=rng)
        return (1 - np.log(x) / v) ** (-1. / th)

    def pdf(self, u, args=()):
        u = np.atleast_2d(u)
        th, = self._handle_args(args)
        a = (th + 1) * np.prod(u, axis=1) ** -(th + 1)
        b = np.sum(u ** -th, axis=1) - 1
        c = -(2 * th + 1) / th
        return a * b ** c

    def logpdf(self, u, args=()):
        # we skip Archimedean logpdf, that uses numdiff
        return super(ArchimedeanCopula, self).logpdf(u, args=args)

    def cdf(self, u, args=()):
        u = np.atleast_2d(u)
        th, = self._handle_args(args)
        return (np.sum(u ** (-th), axis=1) - 1) ** (-1.0 / th)

    def tau(self, theta=None):
        # Joe 2014 p. 168
        if theta is None:
            theta = self.theta

        return theta / (theta + 2)

    def theta_from_tau(self, tau):
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

    def __init__(self, theta=None, k_dim=2):
        if theta is not None:
            args = (theta,)
        else:
            args = ()
        super().__init__(transforms.TransfFrank(), args=args, k_dim=k_dim)

        if theta is not None:
            if theta == 0:
                raise ValueError('Theta must be !=0')
        self.theta = theta

    def rvs(self, nobs=1, args=(), random_state=None):
        if self.k_dim != 2:
            msg = "rvs is only available for bivariate copula"
            raise NotImplementedError(msg)
        rng = check_random_state(random_state)
        th, = self._handle_args(args)
        x = rng.random((nobs, self.k_dim))
        v = stats.logser.rvs(1. - np.exp(-th),
                             size=(nobs, 1), random_state=rng)

        return -1. / th * np.log(1. + np.exp(-(-np.log(x) / v))
                                 * (np.exp(-th) - 1.))

    # explicit BV formulas copied from Joe 1997 p. 141
    # todo: check expm1 and log1p for improved numerical precision

    def pdf(self, u, args=()):
        u = np.atleast_2d(u)
        th, = self._handle_args(args)
        if u.shape[-1] != 2:
            return super().pdf(u)

        g_ = np.exp(-th * np.sum(u, axis=1)) - 1
        g1 = np.exp(-th) - 1

        num = -th * g1 * (1 + g_)
        aux = np.prod(np.exp(-th * u) - 1, axis=1) + g1
        den = aux ** 2
        return num / den

    def cdf(self, u, args=()):
        u = np.atleast_2d(u)
        th, = self._handle_args(args)
        dim = u.shape[-1]
        if dim != 2:
            return super().cdf(u)

        num = np.prod(1 - np.exp(- th * u), axis=1)
        den = (1 - np.exp(-th)) ** (dim - 1)

        return -1.0 / th * np.log(1 - num / den)

    def logpdf(self, u, args=()):
        u = np.atleast_2d(u)
        th, = self._handle_args(args)
        if u.shape[-1] == 2:
            # bivariate case
            u1, u2 = u[..., 0], u[..., 1]
            b = 1 - np.exp(-th)
            pdf = np.log(th * b) - th * (u1 + u2)
            pdf -= 2 * np.log(b - (1 - np.exp(- th * u1)) *
                              (1 - np.exp(- th * u2)))
            return pdf
        else:
            # for now use generic from base Copula class, log(self.pdf(...))
            # we skip Archimedean logpdf, that uses numdiff
            super(ArchimedeanCopula, self).logpdf(u, args)

    def cdfcond_2g1(self, u, args=()):
        """Conditional cdf of second component given the value of first.
        """
        u = np.atleast_2d(u)
        th, = self._handle_args(args)
        if u.shape[-1] == 2:
            # bivariate case
            u1, u2 = u[..., 0], u[..., 1]
            cdfc = np.exp(- th * u1)
            cdfc /= np.expm1(-th) / np.expm1(- th * u2) + np.expm1(- th * u1)
            return cdfc
        else:
            raise NotImplementedError("u needs to be bivariate (2 columns)")

    def ppfcond_2g1(self, q, u1, args=()):
        """Conditional pdf of second component given the value of first.
        """
        u1 = np.asarray(u1)
        th, = self._handle_args(args)
        if u1.shape[-1] == 1:
            # bivariate case, conditional on value of first variable
            ppfc = - np.log(1 + np.expm1(- th) /
                            ((1 / q - 1) * np.exp(-th * u1) + 1)) / th

            return ppfc
        else:
            raise NotImplementedError("u needs to be bivariate (2 columns)")

    def tau(self, theta=None):
        # Joe 2014 p. 166
        if theta is None:
            theta = self.theta
        debye_value = _debye(theta)
        return 1 + 4 * (debye_value - 1) / theta

    def theta_from_tau(self, tau):
        MIN_FLOAT_LOG = np.log(sys.float_info.min)
        MAX_FLOAT_LOG = np.log(sys.float_info.max)

        def _theta_from_tau(alpha):
            return self.tau(theta=alpha) - tau

        result = optimize.least_squares(_theta_from_tau, 1, bounds=(
            MIN_FLOAT_LOG, MAX_FLOAT_LOG))
        theta = result.x[0]
        return theta


class GumbelCopula(ArchimedeanCopula):
    r"""Gumbel copula.

    Dependence is greater in the positive tail than in the negative.

    .. math::

        C_\theta(u,v) = \exp\!\left[ -\left( (-\log(u))^\theta +
        (-\log(v))^\theta \right)^{1/\theta} \right]

    with :math:`\theta\in[1,\infty)`.

    """

    def __init__(self, theta=None, k_dim=2):
        if theta is not None:
            args = (theta,)
        else:
            args = ()
        super().__init__(transforms.TransfGumbel(), args=args, k_dim=k_dim)

        if theta is not None:
            if theta <= 1:
                raise ValueError('Theta must be > 1')
        self.theta = theta

    def rvs(self, nobs=1, args=(), random_state=None):
        if self.k_dim != 2:
            msg = "rvs is only available for bivariate copula"
            raise NotImplementedError(msg)
        rng = check_random_state(random_state)
        th, = self._handle_args(args)
        x = rng.random((nobs, self.k_dim))
        v = stats.levy_stable.rvs(
            1. / th, 1., 0,
            np.cos(np.pi / (2 * th)) ** th,
            size=(nobs, 1), random_state=rng
        )
        return np.exp(-(-np.log(x) / v) ** (1. / th))

    def pdf(self, u, args=()):
        u = np.atleast_2d(u)
        th, = self._handle_args(args)
        xy = -np.log(u)
        xy_theta = xy ** th

        sum_xy_theta = np.sum(xy_theta, axis=1)
        sum_xy_theta_theta = sum_xy_theta ** (1.0 / th)

        a = np.exp(-sum_xy_theta_theta)
        b = sum_xy_theta_theta + th - 1.0
        c = sum_xy_theta ** (1.0 / th - 2)
        d = np.prod(xy, axis=1) ** (th - 1.0)
        e = np.prod(u, axis=1) ** (- 1.0)

        return a * b * c * d * e

    def cdf(self, u, args=()):
        u = np.atleast_2d(u)
        th, = self._handle_args(args)
        h = np.sum((-np.log(u)) ** th, axis=1)
        cdf = np.exp(-h ** (1.0 / th))
        return cdf

    def logpdf(self, u, args=()):
        # we skip Archimedean logpdf, that uses numdiff
        return super(ArchimedeanCopula, self).logpdf(u, args=args)

    def tau(self, theta=None):
        # Joe 2014 p. 172
        if theta is None:
            theta = self.theta

        return (theta - 1) / theta

    def theta_from_tau(self, tau):
        return 1 / (1 - tau)
