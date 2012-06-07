"""
Empirical Likelihood Implementation

Start: 21 May 2012
Last Updated: 28 May 2012

General References:

Owen, A. (2001). "Empirical Likelihood." Chapman and Hall



"""
import numpy as np
from scipy import optimize
from scipy.stats import chi2
from matplotlib import pyplot as plt


class ElModel(object):
    """


    Initializes data for empirical likelihood.  Not intended for end user
    """

    def __init__(self, endog):
        self.endog = endog
        self.nobs = float(endog.shape[0])
        self.weights = np.ones(self.nobs) / float(self.nobs)
        self.endog = self.endog.reshape(self.nobs, 1)
        self.max_iter = 50  # More iters usually mean problems elsewhere
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

    def get_j_y(self, eta1):
        """
        Calculates J and y via the log*' and log*''.

        Maximizing log* is done via sequential regression of
        y on J.

        See Owen pg. 63

        """

        data = np.copy(self.est_vect.T)
        data_star_prime = np.copy((1 + np.dot(eta1, data)))
        data_star_doub_prime = np.copy((1 + np.dot(eta1, data)))
        for elem in range(int(self.nobs)):
            if data_star_prime[0, elem] <= 1 / self.nobs:
                data_star_prime[0, elem] = 2 * self.nobs - \
                  (self.nobs) ** 2 * data_star_prime[0, elem]
            else:
                data_star_prime[0, elem] = 1. / data_star_prime[0, elem]
            if data_star_doub_prime[0, elem] <= 1 / self.nobs:
                data_star_doub_prime[0, elem] = - self.nobs ** 2
            else:
                data_star_doub_prime[0, elem] = \
                  - (data_star_doub_prime[0, elem]) ** -2
        data_star_prime = data_star_prime.reshape(self.nobs, 1)
        data_star_doub_prime = data_star_doub_prime.reshape(self.nobs, 1)
        J = ((- 1 * data_star_doub_prime) ** .5) * self.est_vect
        y = data_star_prime / ((- 1 * data_star_doub_prime) ** .5)
        return J, y

    def modif_newton(self,  x0):
        """
        Modified Newton's method for maximizing the log* equation.

        See Owen pg. 64

        """
        params = x0.reshape(1, 2)
        diff = 1
        while diff > 10 ** (-10):
            new_J = np.copy(self.get_j_y(params)[0])
            new_y = np.copy(self.get_j_y(params)[1])
            inc = np.dot(np.linalg.pinv(new_J), new_y).reshape(1, 2)
            new_params = np.copy(params + inc)
            diff = np.sum(np.abs(params - new_params))
            params = np.copy(new_params)
        return params

    def find_eta(self, eta):

        """
        Finding the root of sum(xi-h0)/(1+eta(xi-mu)) solves for
        eta when computing ELR for univariate mean.

        See Owen (2001) pg 22.  (eta is his lambda to avoid confusion
        with the built-in lambda.

        """
        return np.sum((self.data - self.mu0) / \
              (1. + eta * (self.data - self.mu0)))

    def ci_limits_mu(self, mu_test):
        return self.hy_test_mean(mu_test)[1] - self.r0

    def find_gamma(self, gamma):

        """
        Finds gamma that satisfies
        sum(log(n * w(gamma))) - log(r0) = 0

        Used for confidence intervals for the mean.

        See Owen (2001) pg. 23.

        """

        denom = np.sum((self.endog - gamma) ** -1)
        new_weights = (self.endog - gamma) ** -1 / denom
        return -2 * np.sum(np.log(self.nobs * new_weights)) - \
            self.r0

    def opt_var(self, nuisance_mu):
        """

        This is the function to be optimized over a nuisance mean parameter
        to determine the likelihood ratio for the variance.  In this function
        is the Newton optimization that finds the optimal weights given
        a mu parameter and sig2_0.

        Also, it contains the creating of self.est_vect (short for estimating
        equations vector).  That then gets read by the log-star equations.

        Not intended for end user.

        """

        sig_data = ((self.endog - nuisance_mu) ** 2 \
                    - self.sig2_0)
        mu_data = (self.endog - nuisance_mu)
        self.est_vect = np.concatenate((mu_data, sig_data), axis=1)
        eta_star = self.modif_newton(np.array([1 / self.nobs,
                                               1 / self.nobs]))
        denom = 1 + np.dot(eta_star, self.est_vect.T)
        self.new_weights = 1 / self.nobs * 1 / denom
        llr = np.sum(np.log(self.nobs * self.new_weights))
        return -2 * llr

    def  ci_limits_var(self, var_test):
        """

        Used to determine the confidence intervals for the variance.
        It calls hy_test_var and when called by an optimizer,
        finds the value of sig2_0 that is chi2.ppf(significance-level)

        """

        return self.hy_test_var(var_test)[1] - self.r0


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

    def hy_test_mean(self, mu0,  trans_data=None, print_weights=False):

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
        if trans_data  is None:
            self.data = self.endog
        else:
            self.data = trans_data
        eta_min = (1 - (1 / self.nobs)) / (self.mu0 - max(self.data))
        eta_max = (1 - (1 / self.nobs)) / (self.mu0 - min(self.data))
        eta_star = optimize.brentq(self.find_eta, eta_min, eta_max)
        new_weights = (1 / self.nobs) * \
            1. / (1 + eta_star * (self.data - self.mu0))
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
            ul = optimize.brentq(self.ci_limits_mu, middle,
                max(self.endog) - epsilon_u)
            ll = optimize.brentq(self.ci_limits_mu, middle,
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

    def hy_test_var(self, sig2_0, print_weights=False):
        """

        Returns the p-value and -2 * log-likelihoog ratio for the
            hypothesized variance.

        Parameters
        ----------

        sig2_0: Hypothesized value to be tested

        Optional
        --------

        print_weights: If True, returns the weights that maximize the
            likelihood of observing sig2_0. default | False.


        Example
        -------
        random_numbers = np.random.standard_normal(1000)*100
        el_analysis = el.DescStat(random_numbers)
        hyp_test = el_analysis.hy_test_var(9500)


        """

        self.sig2_0 = sig2_0
        mu_max = max(self.endog)
        mu_min = min(self.endog)
        llr = optimize.fminbound(self.opt_var, mu_min, mu_max, \
                                 full_output=1)[1]
        p_val = 1 - chi2.cdf(llr, 1)
        if print_weights:
            return p_val, llr, self.new_weights
        else:
            return p_val, llr

    def ci_var(self, ll, ul, sig=.95):
        """

        Returns the confidence interval for the variance.

        Parameters
        ----------

        ll: The minimum value the lower confidence interval can take on.
            The p-value from hy_test_var(lower_l) must be lower than
            1 - significance level.


        ul: The maximum value the upper confidence interval can take.
            The p-value from hy_test_var(upper_h) must be lower than
            1 - significance level.

        sig: The significance level for the conficence interval.
        default | .95


        Example
        -------
        random_numbers = np.random.standard_normal(100)
        el_analysis = el.DescStat(random_numbers)
        # Initialize El
        el_analysis.ci_var(.5, 2)
        # Searches for confidence limits where the lower limit .5 and the
        # upper limit <2.

        Troubleshooting Tips
        --------------------

        If the function returns the error f(a) and f(b) must have
        different signs, consider lowering ll and raising ul.

        """

        self.r0 = chi2.ppf(sig, 1)
        ll = optimize.brentq(self.ci_limits_var, ll, self.endog.var())
        ul = optimize.brentq(self.ci_limits_var, self.endog.var(), ul)
        return   ll, ul

    def var_p_plot(self, lower, upper, step, sig=.95):
        """

        Plots the p-values of the maximum el estimate for the variace

        Parameters
        ----------

        lower: Lowest value of variance to be computed and plotted

        upper: Highest value of the variance to be computed and plotted

        step: Interval between each plot point.


        sig: Will draw a horizontal line at 1- sig. default | .95

        This function can be helpful when trying to determine limits
         in the ci_var function.

        """

        p_vals = []
        for test in np.arange(lower, upper, step):
            p_vals.append(self.hy_test_var(test)[0])
        p_vals = np.asarray(p_vals)
        plt.plot(np.arange(lower, upper, step), p_vals)
        plt.plot(np.arange(lower, upper, step), (1 - sig) * \
                 np.ones(len(p_vals)))
        return 'Type plt.show to see plot'
