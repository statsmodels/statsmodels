'''temporary compatibility module

TODO: remove in 0.5.0
'''

from statsmodels.stats.sandwich_covariance import *
#from statsmodels.stats.moment_helpers import se_cov

#not in __all__

def cov_hac_simple(results, nlags=None, weights_func=weights_bartlett,
                   use_correction=True):
    c = cov_hac(results, nlags=nlags, weights_func=weights_func,
                   use_correction=use_correction)
    return c, se_cov(c)
