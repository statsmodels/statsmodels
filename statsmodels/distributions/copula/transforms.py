# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:33:40 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from scipy.special import expm1
from statsmodels.distributions.copula.depfunc_ev import *  # compat


#not used yet
class Transforms(object):
    def __init__(self):
        pass


class TransfFrank(object):

    def evaluate(self, t, theta):
        t = np.asarray(t)
        return - (np.log(-expm1(-theta*t)) - np.log(-expm1(-theta)))
        #return - np.log(expm1(-theta*t) / expm1(-theta))

    def inverse(self, phi, theta):
        phi = np.asarray(phi)
        return -np.log1p(np.exp(-phi) * expm1(-theta)) / theta

    def deriv(self, t, theta):
        t = np.asarray(t)
        tmp = np.exp(-t*theta)
        return -theta * tmp/(tmp - 1)

    def deriv2(self, t, theta):
        t = np.asarray(t)
        tmp = np.exp(theta * t)
        d2 = - theta**2 * tmp / (tmp - 1)**2
        return d2

    def is_completly_monotonic(self, theta):
        #range of theta for which it is copula for d>2 (more than 2 rvs)
        return theta > 0 & theta < 1


class TransfClayton(object):

    def _checkargs(self, theta):
        return theta > 0

    def evaluate(self, t, theta):
        return np.power(t, -theta) - 1.

    def inverse(self, phi, theta):
        return np.power(1 + phi, -1/theta)

    def deriv(self, t, theta):
        return -theta * np.power(t, -theta-1)

    def deriv2(self, t, theta):
        return theta * (theta + 1) * np.power(t, -theta-2)

    def is_completly_monotonic(self, theta):
        return theta > 0


class TransfGumbel(object):
    '''
    requires theta >=1
    '''

    def _checkargs(self, theta):
        return theta >= 1

    def evaluate(self, t, theta):
        return np.power(-np.log(t), theta)

    def inverse(self, phi, theta):
        return np.exp(-np.power(phi, 1. / theta))

    def deriv(self, t, theta):
        return - theta * (-np.log(t))**(theta - 1) / t

    def deriv2(self, t, theta):
        tmp1 = np.log(t)
        d2 = (theta*(-1)**(1 + theta) * tmp1**(theta-1) * (1 - theta) +
              theta*(-1)**(1 + theta)*tmp1**theta)/(t**2*tmp1)
        #d2 = (theta * tmp1**(-1 + theta) * (1 - theta) + theta * tmp1**theta
        #      ) / (t**2 * tmp1)

        return d2

    def is_completly_monotonic(self, theta):
        return theta > 1


class TransfIndep(object):
    def evaluate(self, t):
        t = np.asarray(t)
        return -np.log(t)

    def inverse(self, phi):
        phi = np.asarray(phi)
        return np.exp(-phi)

    def deriv(self, t):
        t = np.asarray(t)
        return - 1./t

    def deriv2(self, t):
        t = np.asarray(t)
        return 1. / t**2
