#
from scipy import stats
from scipy.stats import rv_continuous
import numpy as np
from scipy.special import gammaln, gammaincinv, gamma, gammainc
from numpy import log,exp
#import pymc
np.random.seed(12345)

class igamma_gen(rv_continuous):
    def _pdf(self, x, a, b):
        return exp(self._logpdf(x,a,b))
    def _logpdf(self, x, a, b):
        return a*log(b) - gammaln(a) -(a+1)*log(x) - b/x
    def _cdf(self, x, a, b):
        return 1.0-gammainc(a,b/x) # why is this different than the wiki?
    def _ppf(self, q, a, b):
        return b/gammaincinv(a,1-q)
#NOTE: should be correct, work through invgamma example and 2 param inv gamma
#CDF
    def _munp(self, n, a, b):
        args = (a,b)
        super(igamma_gen, self)._munp(self, n, *args)
#TODO: is this robust for differential entropy in this case? closed form or
#shortcuts in special?
    def _entropy(self, *args):
        def integ(x):
            val = self._pdf(x, *args)
            return val*log(val)

        entr = -integrate.quad(integ, self.a, self.b)[0]
        if not np.isnan(entr):
            return entr
        else:
            raise ValueError("Problem with integration.  Returned nan.")

igamma = igamma_gen(a=0.0, name='invgamma', longname="An inverted gamma",
            shapes = 'a,b', extradoc="""

Inverted gamma distribution

invgamma.pdf(x,a,b) = b**a*x**(-a-1)/gamma(a) * exp(-b/x)
for x > 0, a > 0, b>0.
"""
)

palpha = np.random.gamma(400.,.005, size=10000)
print "First moment: %s\nSecond moment: %s", (palpha.mean(),palpha.std())
palpha = palpha[0]

prho = np.random.beta(49.5,49.5, size=1e5)
print "First moment: %s\nSecond moment: %s", (prho.mean(),prho.std())
prho = prho[0]

psigma = igamma.rvs(1.,4., size=1e5)
