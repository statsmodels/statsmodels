'''

Which Archimedean is Best?
Extreme Value copulas formulas are based on Genest 2009

References
----------

Genest, C., 2009. Rank-based inference for bivariate extreme-value
copulas. The Annals of Statistics, 37(5), pp.2990-3022.



'''


import numpy as np
from scipy.special import expm1

from statsmodels.distributions.copula.depfunc_ev import (
    transform_bilogistic, transform_hr, transform_joe, transform_tawn,
    transform_tawn2, transform_tev
    )


def copula_bv_indep(u, v):
    '''independent bivariate copula
    '''
    return u*v


def copula_bv_min(u, v):
    '''comonotonic bivariate copula
    '''
    return np.minimum(u, v)


def copula_bv_max(u, v):
    '''countermonotonic bivariate copula
    '''
    return np.maximum(u + v - 1, 0)


def copula_bv_clayton(u, v, theta):
    '''Clayton or Cook, Johnson bivariate copula
    '''
    if not theta > 0:
        raise ValueError('theta needs to be strictly positive')
    return np.power(np.power(u, -theta) + np.power(v, -theta) - 1, -theta)


def copula_bv_frank(u, v, theta):
    '''Cook, Johnson bivariate copula
    '''
    if not theta > 0:
        raise ValueError('theta needs to be strictly positive')
    cdfv = -np.log(1 + expm1(-theta*u) * expm1(-theta*v) / expm1(-theta))/theta
    cdfv = np.minimum(cdfv, 1)  # necessary for example if theta=100
    return cdfv


def copula_bv_gauss(u, v, rho):
    raise NotImplementedError


def copula_bv_t(u, v, rho, df):
    raise NotImplementedError


# Archimedean Copulas through generator functions
# ===============================================


def copula_bv_archimedean(u, v, transform, args=()):
    '''
    '''
    phi = transform.evaluate
    phi_inv = transform.inverse
    cdfv = phi_inv(phi(u, *args) + phi(v, *args), *args)
    return cdfv


def copula_mv_archimedean(u, transform, args=(), axis=-1):
    '''generic multivariate Archimedean copula
    '''
    phi = transform.evaluate
    phi_inv = transform.inverse
    cdfv = phi_inv(phi(u, *args).sum(axis), *args)
    return cdfv


def copula_power_mv_archimedean(u, transform, alpha, beta, args=(), axis=-1):
    '''generic multivariate Archimedean copula with additional power transforms

    Nelson p.144, equ. 4.5.2
    '''

    def phi(u, alpha, beta, args=()):
        return np.power(transform.evaluate(np.power(u, alpha), *args), beta)

    def phi_inv(t, alpha, beta, args=()):
        return np.power(transform.evaluate(np.power(t, 1. / beta), *args),
                        1. / alpha)

    cdfv = phi_inv(phi(u, *args).sum(axis), *args)
    return cdfv


class CopulaArchimedean(object):

    def __init__(self, transform):
        self.transform = transform

    def cdf(self, u, args=(), axis=-1):
        '''evaluate cdf of multivariate Archimedean copula
        '''
        phi = self.transform.evaluate
        phi_inv = self.transform.inverse
        cdfv = phi_inv(phi(u, *args).sum(axis), *args)
        return cdfv

    def pdf(self, u, args=(), axis=-1):
        '''evaluate cdf of multivariate Archimedean copula
        '''
        u = np.asarray(u)
        if u.shape[-1] > 2:
            msg = "pdf is currently only available for bivariate copula"
            raise ValueError(msg)
        phi = self.transform.evaluate
        phi_inv = self.transform.inverse
        phi_d1 = self.transform.deriv
        phi_d2 = self.transform.deriv2

        cdfv = self.cdf(u, args=args, axis=axis)

        pdfv = - np.product(phi_d1(u, *args), axis)
        pdfv *= phi_d2(cdfv, *args)
        pdfv /= phi_d1(cdfv, *args)**3

        return pdfv


# Extreme Value Copulas
# =====================

def copula_bv_ev(u, v, transform, args=()):
    '''generic bivariate extreme value copula
    '''
    return np.exp(np.log(u * v) * (transform(np.log(v)/np.log(u*v), *args)))


# ==========================================================================

# define dictionary of copulas by names and aliases
copulanamesbv = {'indep': copula_bv_indep,
                 'i': copula_bv_indep,
                 'min': copula_bv_min,
                 'max': copula_bv_max,
                 'clayton': copula_bv_clayton,
                 'cookjohnson': copula_bv_clayton,
                 'cj': copula_bv_clayton,
                 'frank': copula_bv_frank,
                 'gauss': copula_bv_gauss,
                 'normal': copula_bv_gauss,
                 't': copula_bv_t}


class CopulaDistributionBivariate(object):
    '''bivariate copula class

    Instantiation needs the arguments, cop_args, that are required for copula
    '''
    def __init__(self, marginalcdfs, copula, copargs=()):
        if copula in copulanamesbv:
            self.copula = copulanamesbv[copula]
        else:
            # see if we can call it as a copula function
            try:
                tmp = copula(0.5, 0.5, *copargs)
            except Exception:  # blanket since we throw again
                msg = 'copula needs to be a copula name or callable'
                raise ValueError(msg)
            self.copula = copula

        # no checking done on marginals
        self.marginalcdfs = marginalcdfs
        self.copargs = copargs

    def cdf(self, xy, args=None):
        '''xx needs to be iterable, instead of x,y for extension to multivariate
        '''
        x, y = xy
        if args is None:
            args = self.copargs
        return self.copula(self.marginalcdfs[0](x), self.marginalcdfs[1](y),
                           *args)


class CopulaDistribution(object):
    '''bivariate copula class

    Instantiation needs the arguments, cop_args, that are required for copula
    '''
    def __init__(self, marginalcdfs, copula, copargs=()):
        if copula in copulanamesbv:
            self.copula = copulanamesbv[copula]
        else:
            # see if we can call it as a copula function
            try:
                tmp = copula(0.5, 0.5, *copargs)
            except Exception:  # blanket since we throw again
                msg = 'copula needs to be a copula name or callable'
                raise ValueError(msg)
            self.copula = copula

        # no checking done on marginals
        self.marginalcdfs = marginalcdfs
        self.copargs = copargs

    def cdf(self, xy, args=None):
        '''xx needs to be iterable, instead of x,y for extension to multivariate
        '''
        x, y = xy
        if args is None:
            args = self.copargs
        return self.copula(self.marginalcdfs[0](x), self.marginalcdfs[1](y),
                           *args)
