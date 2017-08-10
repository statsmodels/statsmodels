import numpy as np
from scipy.stats import rv_discrete
from scipy.special import gammaln
from statsmodels.compat.scipy import _lazywhere


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

class zipoisson_gen(rv_discrete):
    '''A Zero Inflated Poisson distribution
    '''
    def _argcheck(self, mu, w):
        return (mu > 0) & (w >= 0) & (w<=1)

    def _logpmf(self, x, mu, w):
        return _lazywhere(x != 0, (x, mu, w),
                          (lambda x, mu, w: np.log(1. - w) + x * np.log(mu) -
                          gammaln(x + 1.) - mu),
                          np.log(w + (1. - w) * np.exp(-mu)))

    def _pmf(self, x, mu, w):
        return np.exp(self._logpmf(x, mu, w))

zipoisson = zipoisson_gen(name='zipoisson',
                          longname='Zero Inflated Poisson')

class zigeneralizedpoisson_gen(rv_discrete):
    '''A Zero Inflated Generalized Poisson distribution
    '''
    def _argcheck(self, mu, alpha, p, w):
        return (mu > 0) & (w >= 0) & (w<=1)

    def _logpmf(self, x, mu, alpha, p, w):
        return _lazywhere(x != 0, (x, mu, alpha, p, w),
                          (lambda x, mu, alpha, p, w: np.log(1. - w) + 
                          genpoisson_p.logpmf(x, mu, alpha, p)),
                          np.log(w + (1. - w) *
                          genpoisson_p.pmf(x, mu, alpha, p)))

    def _pmf(self, x, mu, alpha, p, w):
        return np.exp(self._logpmf(x, mu, alpha, p, w))

zigenpoisson = zigeneralizedpoisson_gen(name='zigenpoisson',
                          longname='Zero Inflated Generalized Poisson')
