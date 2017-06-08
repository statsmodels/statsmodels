import numpy as np
from scipy.stats import rv_discrete
from scipy.special import gammaln


class genpoisson_p_gen(rv_discrete):
    '''A generalized Poisson discrete random variable
    '''
    def _argcheck(self, mu, alpha, p):
        return True


    def _logpmf(self, x, mu, alpha, p):
        mu_p = mu ** (p - 1)
        a1 = 1 + alpha * mu_p
        a2 = mu + (a1 - 1) * x
        logpmf_ = np.log(mu) + (x - 1) * np.log(a2)
        logpmf_ -=  x * np.log(a1) + gammaln(x + 1) + a2 / a1
        return logpmf_

    def _pmf(self, x, mu, alpha, p):
        return np.exp(self._logpmf(x, mu, alpha, p))

genpoisson_p = genpoisson_p_gen(name='genpoisson_p',
                                longname='Generalized Poisson')
