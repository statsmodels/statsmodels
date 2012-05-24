"""
Empirical Likelihood Implementation

Start: 21 May 2012
Last Updated: 23 May 2012

General References:

Owen, A. (2001). "Empirical Likelihood." Chapman and Hall



"""
import numpy as np
from scipy import optimize
from scipy.stats import chi2


class ElModel(object):
    """


    Initializes data for empirical likelihood.  Not intended for end user
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

    def ci_limits(self, mu_test):
        return self.hy_test_mean(mu_test)[1] - self.r0

    def find_gamma(self, gamma):

        """
        Finds gamma that satisfies
        sum(log(n * w(gamma))) - log(r0) = 0

        See Owen (2001) pg. 23.

        """

        denom = np.sum((self.endog - gamma) ** -1)
        new_weights = (self.endog - gamma) ** -1 / denom
        return -2 * np.sum(np.log(self.nobs * new_weights)) - \
            self.r0


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

        print_weights: If print_weights = True the funtion returns
        the weight of the observations under the null hypothesis
        | default = False

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

    def ci_mean(self, sig=.95, method='nested-brent', epsilon=10 ** -6,
                 gamma_low=-10 ** 10, gamma_high=10 ** 10, \
                 tol=10 ** -6):

        """
        Returns the confidence interval for the mean.

        Parameters
        ----------
        sig: Significance level | default=.95

        Optional
        --------

        method: Root finding method,  Can be 'nested-brent or gamma'.
            default | gamma

            'gamma' Tries to solve for the gamma parameter in the
            Lagrangian (see Owen pg 22) and then determine the weights.

            'nested brent' uses brents method to find the confidence
            intervals but must maximize the likelihhod ratio on every
            iteration.

            'bisect' is similar to the nested-brent but instead it
            is a brent nested in a bisection algorithm.

            gamma is much faster.  If the optimizations does not,
            converge, try expanding the gamma_high and gamma_low
            variable.

        gamma_low: lower bound for gamma when finding lower limit.
            If function returns f(a) and f(b) must have different signs,
            consider lowering gamma_low. default | gamma_low =-(10**10)

        gamma_high: upper bound for gamma when finding upper limit.
            If function returns f(a) and f(b) must have different signs,
            consider raising gamma_high. default |gamma_high=10**10

        epsilon: When using 'nested-brent', amount to decrease (increase)
            from the maximum (minimum) of the data when
            starting the search.  This is to protect against the
            likelihood ratio being zero at the maximum (minimum)
            value of the data.  If data is very small in absolute value
            (<10 ** -6) consider shrinking epsilon

            When using 'gamma' amount to decrease (increase) the
            minimum (maximum) by to start the search for gamma.
            If fucntion returns f(a) and f(b) must have differnt signs,
            consider lowering epsilon.

            default| epsilon=10**-6

        tol: Tolerance for the likelihood ratio in the bisect method.
        default | tol=10**-6

        """

        if method == 'nested-brent':
            self.r0 = chi2.ppf(sig, 1)
            middle = np.mean(self.endog)
            epsilon_u = (max(self.endog) - np.mean(self.endog)) * epsilon
            epsilon_l = (np.mean(self.endog) - min(self.endog)) * epsilon
            ul = optimize.brentq(self.ci_limits, middle,
                max(self.endog) - epsilon_u)
            ll = optimize.brentq(self.ci_limits, middle,
                min(self.endog) + epsilon_l)
            return  ll, ul

        if method == 'gamma':
            self.r0 = chi2.ppf(sig, 1)
            gamma_star_l = optimize.brentq(self.find_gamma, gamma_low,
                min(self.endog) - epsilon)
            gamma_star_u = optimize.brentq(self.find_gamma, \
                         max(self.endog) + epsilon, gamma_high)
            weights_low = ((self.endog - gamma_star_l) ** -1) / \
                np.sum((self.endog - gamma_star_l) ** -1)
            weights_high = ((self.endog - gamma_star_u) ** -1) / \
                np.sum((self.endog - gamma_star_u) ** -1)
            mu_low = np.sum(weights_low * self.endog)
            mu_high = np.sum(weights_high * self.endog)
            return mu_low,  mu_high

        if method == 'bisect':
            self.r0 = chi2.ppf(sig, 1)
            self.mu_high = self.endog.mean()
            mu_hmax = max(self.endog)
            while abs(self.hy_test_mean(self.mu_high)[1]
                 - self.r0) > tol:
                self.mu_test = (self.mu_high + mu_hmax) / 2
                if self.hy_test_mean(self.mu_test)[1] - self.r0 < 0:
                    self.mu_high = self.mu_test
                else:
                    mu_hmax = self.mu_test

            self.mu_low = self.endog.mean()
            mu_lmin = min(self.endog)
            while abs(self.hy_test_mean(self.mu_low)[1]
                 - self.r0) > tol:
                self.mu_test = (self.mu_low + mu_lmin) / 2
                if self.hy_test_mean(self.mu_test)[1] - self.r0 < 0:
                    self.mu_low = self.mu_test
                else:
                    mu_lmin = self.mu_test
            return self.mu_low, self.mu_high
