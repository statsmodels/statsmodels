# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:19:45 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from . import transforms


class ArchimedeanCopula(object):

    def __init__(self, transform):
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


class FrankCopula(ArchimedeanCopula):

    # explicit BV formulas copied from Joe 1997 p. 141
    # todo: check expm1 and log1p for improved numerical precision
    def __init__(self):
        super(FrankCopula, self).__init__(transforms.TransfFrank())

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
