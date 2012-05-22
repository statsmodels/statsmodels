"""
Empirical Likelihood Implementation

Start: 21 May 2012
Last Updated: 22 May 2012

General References:

Owen, A. (2001). "Empirical Likelihood." Chapman and Hall



"""
import numpy as np
from scipy import optimize
from scipy.stats import chi2


class ElModel(object):
    """


Initializes data for empirical likelihood.  Not intended for end user.
    """

    def __init__(self, endog):
        self.endog = endog.reshape(max(endog.shape), 1)
        self.nobs = float(endog.shape[0])
        self.weights = np.ones(self.nobs) / float(self.nobs)
        # For now, self. weights should always be a vector of 1's and a
        # variable "new weights" should be created everytime weights are
        # changed.


class OptFuncts(ElModel):
    """


A class that holds functions that are optimized/solved.  Not
intended for the end user.
    """

    def __init__(self, endog):
        super(OptFuncts, self).__init__(endog)

    def find_eta(self, eta):

        """
Finding the root of sum(xi-h0)/(1+eta(xi-mu)) solves for
eta when computing ELR for univariate mean.

See Owen (2001) pg 22.  (eta is his lambda to avoid confusion
with the built-in lambda.

        """
        return np.sum((self.endog - self.mu0) / \
              (1. + eta * (self.endog - self.mu0)))


class DescStat(OptFuncts):
    """


A class for confidence intervals and hypothesis tests invovling mean,
variance and covariance.

Parameters
----------
endog: 1-D array
    Data to be analyzed
    """

    def __init__(self, endog):
        super(DescStat, self).__init__(endog)

    def hy_test_mean(self, mu0, print_weights=False):

        """
Returns the p-value, -2 * log-likelihood ratio and weights
for a hypothesis test of the means.

Parameters
----------
mu0: Mean under the null hypothesis

print_weights: If print_weights = True the funtion returns the weights
of the observations under the null hypothesis | default = False

        """
        self.mu0 = mu0
        eta_min = (1 - (1 / self.nobs)) / (mu0 - max(self.endog))[0]
        eta_max = (1 - (1 / self.nobs)) / (mu0 - min(self.endog))[0]
        eta_star = optimize.brentq(self.find_eta, eta_min, eta_max)
        new_weights = (1 / self.nobs) * \
            1. / (1 + eta_star * (self.endog - self. mu0))
        llr = -2 * np.sum(np.log(self.nobs * new_weights))
        if print_weights:
            return 1 - chi2.cdf(llr, 1), llr, new_weights
        else:
            return 1 - chi2.cdf(llr, 1), llr
