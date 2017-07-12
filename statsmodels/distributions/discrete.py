import numpy as np
from scipy.stats import rv_discrete, poisson
from scipy.special import gammaln


class genpoisson_p_gen(rv_discrete):
    '''A generalized Poisson discrete random variable
    '''
    def _argcheck(self, mu, alpha, p):
        return (mu >= 0) & (alpha==alpha) & (p > 0)

    def _logpmf(self, x, mu, alpha, p):
        mu_p = mu ** (p - 1.)
        a1 = np.maximum(np.nextafter(0, 1), 1 + alpha * mu_p)
        a2 = np.maximum(np.nextafter(0, 1), mu + (a1 - 1.) * x)
        logpmf_ = np.log(mu) + (x - 1.) * np.log(a2)
        logpmf_ -=  x * np.log(a1) + gammaln(x + 1.) + a2 / a1
        return logpmf_

    def _pmf(self, x, mu, alpha, p):
        return np.exp(self._logpmf(x, mu, alpha, p))

genpoisson_p = genpoisson_p_gen(name='genpoisson_p',
                                longname='Generalized Poisson')

class truncatedpoisson_gen(rv_discrete):
    '''Truncated Poisson discrete random variable
    '''
    def _argcheck(self, mu, truncation):
        return (mu >= 0) & (truncation >= 0)

    def _logpmf(self, x, mu, truncation):
        pmf = 0
        for i in range(truncation + 1):
            pmf += poisson.pmf(truncation, mu)

        return poisson.logpmf(x, mu) - np.log(1 - pmf)

    def _pmf(self, x, mu, truncation):
        return np.exp(self._logpmf(x, mu, truncation))

truncatedpoisson = truncatedpoisson_gen(name='truncatedpoisson',
                                        longname='Truncated Poisson')



if __name__=="__main__":
    import numpy as np
    import statsmodels.api as sm
