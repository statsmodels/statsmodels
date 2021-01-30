# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:19:45 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np


class CopulaArchimedean(object):

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
        phi = self.transform.evaluate
        phi_inv = self.transform.inverse
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
        return np.log(self.pdf(u, args=args))
