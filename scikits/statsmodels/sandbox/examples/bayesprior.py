#
# This script examines the predictive prior densities of two local level
# models given the same priors for parameters that appear to be the same.
# Reference: Del Negro and Schorfheide.

try:
    import pymc
except:
    print "pymc not imported"
from scipy.stats import gamma, beta, invgamma
import numpy as np
from matplotlib import pyplot as plt
#np.random.seed(12345)
from scipy import stats
from scipy.stats import rv_continuous
from scipy.special import gammaln, gammaincinv, gamma, gammainc
from numpy import log,exp

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

# First do the univariate case
# y_t = theta_t + epsilon_t
# epsilon ~ N(0,1)
# Where theta ~ N(mu,lambda**2)


# or the model
# y_t = theta2_t + theta1_t * y_t-1 + epsilon_t

# Prior 1:
# theta1 ~ uniform(0,1)
# theta2|theta1 ~ N(mu,lambda**2)
# Prior 2:
# theta1 ~ U(0,1)
# theta2|theta1 ~ N(mu(1-theta1),lambda**2(1-theta1)**2)

draws = 400
# prior beliefs
mu_, lambda_2 = 1.,2.

# Model 1
y1y2 = np.zeros((draws,2))
for draw in range(draws):
    theta = np.random.normal(mu_,lambda_2)
    y1 = theta + np.random.normal()
    y2 = theta + np.random.normal()
    y1y2[draw] = y1,y2

# Model 2
pmu_pairsp1 = np.zeros((draws,2))
y1y2pairsp1 = np.zeros((draws,2))
# prior 1
for draw in range(draws):
    y1 = np.random.uniform(-20,20)
    theta1 = np.random.uniform(0,1)
    theta2 = np.random.normal(mu_, lambda_2) # mu_ assumed to be 2
    mu = theta2/(1-theta1)
    pmu_pairsp1[draw] = mu, theta1
    y2 = theta2 + theta1 * y1 + np.random.normal()
    y1y2pairsp1[draw] = y1,y2

pmu_pairsp2 = np.zeros((draws,2))
y1y2pairsp2 = np.zeros((draws,2))
for draw in range(draws):
    y1 = np.random.uniform(-20,20)
    theta1 = np.random.uniform(0,1)
    theta2 = np.random.normal(mu_*(1-theta1), lambda_2*(1-theta1)**2)
    mu = theta2/(1-theta1)
    y2 = theta2 + theta1 * y1 + np.random.normal()
    pmu_pairsp2[draw] = mu, theta1
    y1y2pairsp2[draw] = y1,y2

fig = plt.figure()
fsp = fig.add_subplot(221)
fsp.scatter(pmu_pairsp1[:,0], pmu_pairsp2[:,1], color='b', facecolor='none')

fsp = fig.add_subplot(222)
fsp.scatter(pmu_pairsp2[:,0],pmu_pairsp2[:,1], color='b', facecolor='none')

fsp = fig.add_subplot(223)
fsp.scatter(y1y2pairsp1[:,0], y1y2pairsp1[:,1], color='b', marker='o',
    facecolor='none')
fsp.scatter(y1y2[:,0], y1y2[:,1], color ='g', marker='+')

fsp = fig.add_subplot(224)
fsp.scatter(y1y2pairsp2[:,0], y1y2pairsp2[:,1], color='b', marker='o')
fsp.scatter(y1y2[:,0], y1y2[:,1], color='g',marker='+')

#TODO: this doesn't look the same as the paper?

# Example 2:
# 2 NK Phillips Curves
# Structural form
# M1: y_t = 1/alpha *E_t[y_t+1] + mu_t
# mu_t = p1 * mu_t-1 + epsilon_t
# epsilon_t ~ N(0,sigma2)

# Reduced form Law of Motion
# M1: y_t = p1*y_t-1 + 1/(1-p1/alpha)*epsilon_t

# specify prior for M1
# for i = 1,2
# theta_i = [alpha
#             p_i
#             sigma]
# truncate effective priors by the determinancy region
# for determinancy we need alpha > 1
# p in [0,1)
# palpha ~ Gamma(2.00,.10)
# mean = 2.00
# std = .1 which implies k = 400, theta = .005
palpha = np.random.gamma(400,.005)

# pi ~ Beta(.5,.05)
pi = np.random.beta(49.5, 49.5)

# psigma ~ InvGamma(1.00,4.00)
#def invgamma(a,b):
#    return np.sqrt(b*a**2/np.sum(np.random.random(b,1)**2, axis=1))
#NOTE: Use inverse gamma distribution igamma
psigma = igamma.rvs(1.,4.0, size=1e6) #TODO: parameterization is not correct vs.
# Del Negro and Schorfheide
psigma2 = pymc.rinverse_gamma(1.,4.0, size=1e6)

#Are these supposed to be sampled jointly?

# InvGamma(sigma|v,s) propto sigma**(-v-1)*e**(-vs**2/2*sigma**2)
#igamma =

# M2: y_t = 1/alpha * E_t[y_t+1] + p2*y_t-1 + mu_t
# mu_t ~ epsilon_t
# epsilon_t ~ n(0,sigma2)

# Reduced form Law of Motion
# y_t = 1/2 (alpha-sqrt(alpha**2-4*p2*alpha)) * y_t-1 + 2*alpha/(alpha + \
#        sqrt(alpha**2 - 4*p2*alpha)) * epsilon_t
