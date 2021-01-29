# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:33:40 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np


def transform_tawn(t, a1, a2, theta):
    '''asymmetric logistic model of Tawn 1988

    special case: a1=a2=1 : Gumbel

    restrictions:
     - theta in (0,1]
     - a1, a2 in [0,1]
    '''

    def _check_args(a1, a2, theta):
        condth = (theta > 0) and (theta <= 1)
        conda1 = (a1 >= 0) and (a1 <= 1)
        conda2 = (a2 >= 0) and (a2 <= 1)
        return condth and conda1 and conda2

    if not np.all(_check_args(a1, a2, theta)):
        raise ValueError('invalid args')

    transf = (1 - a2) * (1-t)
    transf += (1 - a1) * t
    transf += ((a1 * t)**(1./theta) + (a2 * (1-t))**(1./theta))**theta

    return transf


def transform_joe(t, a1, a2, theta):
    '''asymmetric negative logistic model of Joe 1990

    special case:  a1=a2=1 : symmetric negative logistic of Galambos 1978

    restrictions:
     - theta in (0,inf)
     - a1, a2 in (0,1]
    '''

    def _check_args(a1, a2, theta):
        condth = (theta > 0)
        conda1 = (a1 > 0) and (a1 <= 1)
        conda2 = (a2 > 0) and (a2 <= 1)
        return condth and conda1 and conda2

    if not np.all(_check_args(a1, a2, theta)):
        raise ValueError('invalid args')

    transf = 1 - ((a1 * (1-t))**(-1./theta) + (a2 * t)**(-1./theta))**(-theta)
    return transf


def transform_tawn2(t, theta, k):
    '''asymmetric mixed model of Tawn 1988

    special case:  k=0, theta in [0,1] : symmetric mixed model of
        Tiago de Oliveira 1980

    restrictions:
     - theta > 0
     - theta + 3*k > 0
     - theta + k <= 1
     - theta + 2*k <= 1
    '''

    def _check_args(theta, k):
        condth = (theta >= 0)
        cond1 = (theta + 3*k > 0) and (theta + k <= 1) and (theta + 2*k <= 1)
        return condth and cond1

    if not np.all(_check_args(theta, k)):
        raise ValueError('invalid args')

    transf = 1 - (theta + k) * t + theta * t*t + k * t**3
    return transf


def transform_bilogistic(t, beta, delta):
    '''bilogistic model of Coles and Tawn 1994, Joe, Smith and Weissman 1992

    restrictions:
     - (beta, delta) in (0,1)^2 or
     - (beta, delta) in (-inf,0)^2

    not vectorized because of numerical integration
    '''

    def _check_args(beta, delta):
        cond1 = (beta > 0) and (beta <= 1) and (delta > 0) and (delta <= 1)
        cond2 = (beta < 0) and (delta < 0)
        return cond1 | cond2

    if not np.all(_check_args(beta, delta)):
        raise ValueError('invalid args')

    def _integrant(w):
        term1 = (1 - beta) * np.power(w, -beta) * (1-t)
        term2 = (1 - delta) * np.power(1-w, -delta) * t
        return np.maximum(term1, term2)

    from scipy.integrate import quad
    transf = quad(_integrant, 0, 1)[0]
    return transf


def transform_hr(t, lamda):
    '''model of Huesler Reiss 1989

    special case:  a1=a2=1 : symmetric negative logistic of Galambos 1978

    restrictions:
     - lambda in (0,inf)
    '''

    def _check_args(lamda):
        cond = (lamda > 0)
        return cond

    if not np.all(_check_args(lamda)):
        raise ValueError('invalid args')

    term = np.log((1. - t) / t) * 0.5 / lamda

    from scipy.stats import norm
    # use special if I want to avoid stats import
    transf = (1 - t) * norm._cdf(lamda + term) + t * norm._cdf(lamda - term)
    return transf


def transform_tev(t, rho, df):
    '''t-EV model of Demarta and McNeil 2005

    restrictions:
     - rho in (-1,1)
     - x > 0
    '''
    x = df  # alias, Genest and Segers use chi, copual package uses df

    def _check_args(rho, x):
        cond1 = (x > 0)
        cond2 = (rho > 0) and (rho < 1)
        return cond1 and cond2

    if not np.all(_check_args(rho, x)):
        raise ValueError('invalid args')

    from scipy.stats import t as stats_t
    # use special if I want to avoid stats import

    term1 = (np.power(t/(1.-t), 1./x) - rho)  # for t
    term2 = (np.power((1.-t)/t, 1./x) - rho)  # for 1-t
    term0 = np.sqrt(1. + x) / np.sqrt(1 - rho*rho)
    z1 = term0 * term1
    z2 = term0 * term2
    transf = t * stats_t._cdf(z1, x+1) + (1 - t) * stats_t._cdf(z2, x+1)
    return transf
