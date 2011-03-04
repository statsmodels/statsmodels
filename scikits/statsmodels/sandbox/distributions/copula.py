'''

Which Archimedean is Best?
Extreme Value copulas formulas are based on Genest 2009

References
----------

Genest, C., 2009. Rank-based inference for bivariate extreme-value
copulas. The Annals of Statistics, 37(5), pp.2990-3022.



'''


import numpy as np
from scipy.special import expm1, log1p


def copula_bv_indep(u,v):
    '''independent bivariate copula
    '''
    return u*v

def copula_bv_min(u,v):
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
    cdfv = np.minimum(cdfv, 1)  #necessary for example if theta=100
    return cdfv


def copula_bv_gauss(u, v, rho):
    raise NotImplementedError

def copula_bv_t(u, v, rho, df):
    raise NotImplementedError

#not used yet
class Transforms(object):
    def __init__(self):
        pass

class TransfFrank(object):

    def evaluate(self, t, theta):
        return - (np.log(-expm1(-theta*t)) - np.log(-expm1(-theta)))
        #return - np.log(expm1(-theta*t) / expm1(-theta))

    def inverse(self, phi, theta):
        return -np.log1p(np.exp(-phi) * expm1(-theta)) / theta

class TransfClayton(object):

    def _checkargs(theta):
        return theta > 0

    def evaluate(self, t, theta):
        return np.power(t, -theta) - 1.

    def inverse(self, phi, theta):
        return np.power(1 + phi, -theta)

class TransfGumbel(object):
    '''
    requires theta >=1
    '''

    def _checkargs(theta):
        return theta >= 1

    def evaluate(self, t, theta):
        return np.power(-np.log(t), theta)

    def inverse(self, phi, theta):
        return np.exp(-np.power(phi, 1. / theta))

class TransfIndep(object):
    def evaluate(self, t):
        return -np.log(t)

    def inverse(self, phi):
        return np.exp(-phi)

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


def copula_bv_ev(u, v, transform, args=()):
    '''generic bivariate extreme value copula
    '''
    return np.exp(np.log(u * v) * (transform(np.log(v)/np.log(u*v), *args)))

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

    transf = (1 - a1) * (1-t)
    transf += (1 - a2) * t
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
        cond2 = (beta < 0)  and (delta < 0)
        return cond1 | cond2

    if not np.all(_check_args(beta, delta)):
        raise ValueError('invalid args')

    def _integrant(w):
        term1 = (1 - beta) * np.power(w, -beta) * (1-t)
        term2 = (1 - delta) * np.power(1-w, -delta) * t
        np.maximum(term1, term2)

    from scipy.integrate import quad
    transf = quad(_integrant, 0, 1)
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

    from scipy.stats import norm  #use special if I want to avoid stats import
    transf = (1 - t) * norm._cdf(lamda + term) + t * norm._cdf(lamda - term)
    return transf

def transform_tev(t, rho, x):
    '''t-EV model of Demarta and McNeil 2005

    restrictions:
     - rho in (-1,1)
     - x > 0
    '''

    def _check_args(rho, x):
        cond1 = (x > 0)
        cond2 = (rho > 0) and (rho < 1)
        return cond1 and cond2

    if not np.all(_check_args(rho, x)):
        raise ValueError('invalid args')

    from scipy.stats import t as stats_t  #use special if I want to avoid stats import

    z = np.sqrt(1. + x) * (np.power(t/(1.-t), 1./x) - rho)
    z /= np.sqrt(1 - rho*rho)
    transf = (1 - t) * stats_t._cdf(z, x+1) + t * stats_t._cdf(z, x+1)
    return transf

#define dictionary of copulas by names and aliases
copulanames = {'indep' : copula_bv_indep,
               'i' : copula_bv_indep,
               'min' : copula_bv_min,
               'max' : copula_bv_max,
               'clayton' : copula_bv_clayton,
               'cookjohnson' : copula_bv_clayton,
               'cj' : copula_bv_clayton,
               'frank' : copula_bv_frank,
               'gauss' : copula_bv_gauss,
               'normal' : copula_bv_gauss,
               't' : copula_bv_frank}

class CopulaBivariate(object):
    '''bivariate copula class

    Instantiation needs the arguments, cop_args, that are required for copula
    '''
    def __init__(self, marginalcdfs, copula, copargs=()):
        if copula in copulanames:
            self.copula = copulanames[copula]
        else:
            #see if we can call it as a copula function
            try:
                tmp = copula(0.5, 0.5, *copargs)
            except: #blanket since we throw again
                raise ValueError('copula needs to be a copula name or callable')
            self.copula = copula

        #no checking done on marginals
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




