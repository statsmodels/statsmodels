# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 08:13:38 2010

Author: josef-pktd
License: BSD (3-clause)
"""

from __future__ import print_function
import numpy as np
from numpy import polynomial as npp


class LagPolynomial(npp.Polynomial):

    #def __init__(self, maxlag):

    def pad(self, maxlag):
        return LagPolynomial(np.r_[self.coef, np.zeros(maxlag-len(self.coef))])

    def padflip(self, maxlag):
        return LagPolynomial(np.r_[self.coef, np.zeros(maxlag-len(self.coef))][::-1])

    def flip(self):
        '''reverse polynomial coefficients
        '''
        return LagPolynomial(self.coef[::-1])

    def div(self, other, maxlag=None):
        '''padded division, pads numerator with zeros to maxlag
        '''
        if maxlag is None:
            maxlag = max(len(self.coef), len(other.coef)) + 1
        return (self.padflip(maxlag) / other.flip()).flip()

    def filter(self, arr):
        return (self * arr).coef[:-len(self.coef)]  #trim to end



ar = LagPolynomial([1, -0.8])
arpad = ar.pad(10)

ma = LagPolynomial([1, 0.1])
mapad = ma.pad(10)

unit = LagPolynomial([1])
