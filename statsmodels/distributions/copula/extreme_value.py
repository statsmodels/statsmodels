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
    return np.exp(np.log(u * v) * (transform(np.log(v)/np.log(u*v), *args)))


class ExtremeValueCopula(object):

    def __init__(self, transform):
        self.transform = transform

    def cdf(self, u, args=()):
        '''evaluate cdf of multivariate Archimedean copula
        '''
        # currently only Bivariate
        u, v = np.asarray(u).T
        cdfv = np.exp(np.log(u * v) *
                      (self.transform(np.log(v)/np.log(u*v), *args)))
        return cdfv
