# -*- coding: utf-8 -*-
""" Extreme Value Copulas
Created on Fri Jan 29 19:19:45 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np


def copula_bv_ev(u, transform, args=()):
    '''generic bivariate extreme value copula
    '''
    u, v = u
    return np.exp(np.log(u * v) * (transform(np.log(u)/np.log(u*v), *args)))


class ExtremeValueCopula(object):

    def __init__(self, transform):
        self.transform = transform

    def cdf(self, u, args=()):
        '''evaluate cdf of multivariate Archimedean copula
        '''
        # currently only Bivariate
        u, v = np.asarray(u).T
        cdfv = np.exp(np.log(u * v) *
                      self.transform(np.log(u)/np.log(u*v), *args))
        return cdfv

    def pdf(self, u, args=()):
        """pdf of copula

        """
        tr = self.transform
        u1, u2 = np.asarray(u).T

        log_u12 = np.log(u1 * u2)
        t = np.log(u1) / log_u12
        cdf = self.cdf(u, args)
        dep = tr(t, *args)
        d1 = tr.deriv(t, *args)
        d2 = tr.deriv2(t, *args)
        pdf_ = cdf / (u1 * u2) * ((dep + (1 - t) * d1) * (dep - t * d1) -
                                  d2 * (1 - t) * t / log_u12)

        return pdf_

    def logpdf(self, u, args=()):
        return np.log(self.pdf(u, args=args))

    def conditional_2g1(self, u, args=()):
        """
        C2|1(u2|u1) := ∂C(u1, u2) / ∂u1 = C(u1, u2) / u1 * (A(t) − t A'(t))

        where t = np.log(v)/np.log(u*v)
        """
        pass
