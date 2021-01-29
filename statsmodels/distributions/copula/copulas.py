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


def copula_bv_clayton(u, theta):
    '''Clayton or Cook, Johnson bivariate copula
    '''
    u, v = u
    if isinstance(theta, tuple) and len(theta) == 1:
        theta = theta[0]

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


# Extreme Value Copulas
# =====================

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
    def __init__(self, marginals, copula, copargs=()):
        if copula in copulanamesbv:
            self.copula = copulanamesbv[copula]
        else:
            # assume it's an appropriate copula class
            self.copula = copula

        # no checking done on marginals
        self.marginals = marginals
        self.copargs = copargs
        self.k_vars = len(marginals)

    def cdf(self, y, args=None):
        '''xx needs to be iterable, instead of x,y for extension to multivariate
        '''
        y = np.asarray(y)
        if args is None:
            args = self.copargs

        cdf_marg = []
        for i in range(self.k_vars):
            cdf_marg.append(self.marginals[i].cdf(y[..., i]))

        u = np.column_stack(cdf_marg)
        if y.ndim == 1:
            u = u.squeeze()
        return self.copula.cdf(u, args)

    def pdf(self, y, args=None):
        ''' log pdf of copula distribution
        '''
        return np.exp(self.logpdf(y, args=args))

    def logpdf(self, y, args=None):
        ''' log pdf of copula distribution
        '''
        y = np.asarray(y)
        if args is None:
            args = self.copargs

        lpdf = 0.0
        cdf_marg = []
        for i in range(self.k_vars):
            lpdf += self.marginals[i].logpdf(y[..., i])
            cdf_marg.append(self.marginals[i].cdf(y[..., i]))

        u = np.column_stack(cdf_marg)
        if y.ndim == 1:
            u = u.squeeze()

        lpdf += self.copula.logpdf(u, args)
        return lpdf
