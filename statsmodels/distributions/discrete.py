import numpy as np
from scipy.stats import rv_discrete
from scipy.special import gammaln

class genpoisson_p_gen(rv_discrete):
    '''A generalized Poisson discrete random variable
    '''
    def  _rvs(self, mu, alpha, p):
        pass

    def _logpmf(self, Y, mu, alpha, p):
        mu_p = mu ** (p - 1)
        a1 = 1 + alpha * mu_p
        a2 = mu + (a1 - 1) * Y
        return (np.log(mu) + (Y - 1) * np.log(a2) - Y *
                np.log(a1) - gammaln(Y + 1) - a2 / a1)

    def _pmf(self, Y, mu, alpha, p):
        return np.exp(self._logpmf(Y, mu, alpha, p))

genpoisson_p = genpoisson_p_gen(name='genpoisson_p',
                                longname='Generalized Poisson')
