"""
Empirical Likelihood Implementation

Start: 21 May 2012
Last Updated: 21 June 2012

General References:
------------------

Owen, A. (2001). "Empirical Likelihood." Chapman and Hall


TODO: Write functions for estimationg equations instead
                    of generating the data every time a hypothesis
                    test is called.
"""
import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from matplotlib import pyplot as plt
from statsmodels.base.model import _fit_mle_newton
import itertools


class ELModel(object):
    """
    Initializes data for empirical likelihood.  Not intended for end user
    """

    def __init__(self, endog):
        self.endog = endog
        self.nobs = float(endog.shape[0])
        self.weights = np.ones(self.nobs) / float(self.nobs)
        if endog.ndim == 1:
            self.endog = self.endog.reshape(self.nobs, 1)
        # For now, self. weights should always be a vector of 1's and a
        # variable "new weights" should be created everytime weights are
        # changed.


class _OptFuncts(ELModel):
    """
    A class that holds functions that are optimized/solved.  Not
    intended for the end user.
    """

    def __init__(self, endog):
        super(_OptFuncts, self).__init__(endog)

    def _get_j_y(self, eta1):
        """
        Calculates J and y via the log*' and log*''.

        Maximizing log* is done via sequential regression of
        y on J.

        Parameters
        ----------

        eta1: 1xm array.

        This is the value of lamba used to write the
        empirical likelihood probabilities in terms of the lagrangian
        multiplier.

        Returns
        -------
        J: n x m matrix
            J'J is the hessian for optimizing

        y: n x 1 array
            -J'y is the gradient for maximizing

        See Owen pg. 63
        """
        nobs = self.nobs
        data = self.est_vect.T
        data_star_prime = (1. + np.dot(eta1, data))
        data_star_doub_prime = np.copy(1. + np.dot(eta1, data))

        # Method 1


        # idx = data_star_prime < 1. / nobs
        # not_idx = ~idx
        # data_star_prime[idx] = 2. * nobs - (nobs) ** 2 * data_star_prime[idx]
        # data_star_prime[not_idx] = 1. / data_star_prime[not_idx]
        # data_star_doub_prime[idx] = - nobs ** 2
        # data_star_doub_prime[not_idx] = - (data_star_doub_prime[not_idx]) ** -2
        # data_star_prime = data_star_prime.reshape(nobs, 1)
        # data_star_doub_prime = data_star_doub_prime.reshape(nobs, 1)
        # root_star = np.sqrt(- 1 * data_star_doub_prime).reshape(nobs,1)
        # JJ = root_star * self.est_vect
        # yy = data_star_prime /  root_star
        # return np.mat(JJ), np.mat(yy)

        # Method 2

        for elem in range(int(self.nobs)):
            if data_star_prime[0, elem] <= 1. / self.nobs:
                data_star_prime[0, elem] = 2. * self.nobs - \
                (self.nobs) ** 2 * data_star_prime[0, elem]
            else:
                data_star_prime[0, elem] = 1. / data_star_prime[0, elem]
            if data_star_doub_prime[0, elem] <= 1. / self.nobs:
                data_star_doub_prime[0, elem] = - self.nobs ** 2
            else:
                data_star_doub_prime[0, elem] = \
                  - (data_star_doub_prime[0, elem]) ** -2
        data_star_prime = data_star_prime.reshape(self.nobs, 1)
        data_star_doub_prime = data_star_doub_prime.reshape(self.nobs, 1)
        J = ((- 1 * data_star_doub_prime) ** .5) * self.est_vect
        y = data_star_prime / ((- 1 * data_star_doub_prime) ** .5)
        return np.mat(J), np.mat(y)

    def _log_star(self, eta1):
        """
        Parameters
        ---------
        eta1: float
            Lagrangian multiplier

        Returns
        ------

        data_star: array
            The logstar of the estimting equations
        """
        data = self.est_vect.T
        data_star = (1 + np.dot(eta1, data))
        for elem in range(int(self.nobs)):
            if data_star[0, elem] < 1. / self.nobs:
                data_star[0, elem] = np.log(1 / self.nobs) - 1.5 +\
                  2 * self.nobs * data_star[0, elem] -\
                  ((self.nobs * data_star[0, elem]) ** 2.) / 2.
            else:
                data_star[0, elem] = np.log(data_star[0, elem])
        return data_star

    def _modif_newton(self,  x0):
        """
        Modified Newton's method for maximizing the log* equation.

        Parameters
        ----------
        x0: 1x m array
            Iitial guess for the lagrangian multiplier

        Returns
        -------
        params: 1xm array
            Lagragian multiplier that maximize the log-likelihood given
            `x0`.

        See Owen pg. 64
        """
        x0 = x0.reshape(self.est_vect.shape[1], 1)
        f = lambda x0: np.sum(self._log_star(x0.T))
        grad = lambda x0: - (self._get_j_y(x0.T)[0]).T * \
                              (self._get_j_y(x0.T)[1])
        hess = lambda x0: (self._get_j_y(x0.T)[0]).T * (self._get_j_y(x0.T)[0])
        kwds = {'tol': 1e-8}
        res = _fit_mle_newton(f, grad, x0, (), kwds, hess=hess, maxiter=50, \
                              disp=0)
        return res[0].T

    def _find_eta(self, eta):
        """
        Finding the root of sum(xi-h0)/(1+eta(xi-mu)) solves for
       `eta` when computing ELR for univariate mean.

        Parameters
        ----------

        eta: float
            Lagrangian multiplier

        Returns
        -------

        llr: float
            n times the Log likelihood value for a given value of eta

        See Owen (2001) pg 22.  (eta is his lambda to avoid confusion
        with the built-in lambda).

        Not intended for end user
        """
        return np.sum((self.data - self.mu0) / \
              (1. + eta * (self.data - self.mu0)))

    def _ci_limits_mu(self, mu_test):
        """
        Parameters
        ----------

        mu0: float
            a hypothesized value of mu

        Returns
        -------

        diff: float
            The difference between the log likelihood value of mu0 and
            a specified value.
        """
        return self.hy_test_mean(mu_test)[1] - self.r0

    def _find_gamma(self, gamma):
        """
        Finds gamma that satisfies
        sum(log(n * w(gamma))) - log(r0) = 0

        Used for confidence intervals for the mean.

        Parameters
        ----------

        gamma: float
            LaGrangian multiplier when computing confidence interval

        Returns
        -------

        diff: float
            The difference between the log-liklihood when the Lagrangian
            multiplier is gamma and a pre-specified value.

        See Owen (2001) pg. 23.
        """
        denom = np.sum((self.endog - gamma) ** -1)
        new_weights = (self.endog - gamma) ** -1 / denom
        return -2 * np.sum(np.log(self.nobs * new_weights)) - \
            self.r0

    def _opt_var(self, nuisance_mu, pval=False):
        """
        This is the function to be optimized over a nuisance mean parameter
        to determine the likelihood ratio for the variance.  In this function
        is the Newton optimization that finds the optimal weights given
        a mu parameter and sig2_0.

        Also, it contains the creating of self.est_vect (short for estimating
        equations vector).  That then gets read by the log-star equations.

        Parameter
        --------
        nuisance_mu: float
            Value of a nuisance mean parameter.

        Returns
        -------

        llr: float:
            Log likelihood of a pre-specified variance holding the nuisance
            parameter constant.
        """
        sig_data = ((self.endog - nuisance_mu) ** 2 \
                    - self.sig2_0)
        mu_data = (self.endog - nuisance_mu)
        self.est_vect = np.concatenate((mu_data, sig_data), axis=1)
        eta_star = self._modif_newton(np.array([1 / self.nobs,
                                               1 / self.nobs]))

        denom = 1 + np.dot(eta_star, self.est_vect.T)
        self.new_weights = 1 / self.nobs * 1 / denom
        llr = np.sum(np.log(self.nobs * self.new_weights))
        if pval:  # Used for contour plotting
            return 1 - chi2.cdf(-2 * llr, 1)
        return -2 * llr

    def  _ci_limits_var(self, var_test):
        """
        Used to determine the confidence intervals for the variance.
        It calls hy_test_var and when called by an optimizer,
        finds the value of sig2_0 that is chi2.ppf(significance-level)

        Parameter
        --------
        var_test: float
            Hypothesized value of the variance

        Returns
        -------

        diff: float
            The difference between the log likelihood ratio at var_test and a
            pre-specified value.
        """
        return self.hy_test_var(var_test)[1] - self.r0

    def _opt_skew(self, nuis_params):
        """
        Called by hy_test_skew.  This function is optimized over
        nuisance parameters mu and sigma

        Parameters
        ----------

        nuis_params: 1x2 array
            A nuisance mean and variance parameter

        Returns
        -------

        llr: float
            The log likelihood ratio of a prespeified skewness holding
            the nuisance parameters constant.
        """
        mu_data = self.endog - nuis_params[0]
        sig_data = ((self.endog - nuis_params[0]) ** 2) - nuis_params[1]
        skew_data = ((((self.endog - nuis_params[0]) ** 3) / \
                    (nuis_params[1] ** 1.5))) - self.skew0
        self.est_vect = np.concatenate((mu_data, sig_data, skew_data), \
                                       axis=1)
        eta_star = self._modif_newton(np.array([1 / self.nobs,
                                               1 / self.nobs,
                                               1 / self.nobs]))
        denom = 1 + np.dot(eta_star, self.est_vect.T)
        self.new_weights = 1 / self.nobs * 1 / denom
        llr = np.sum(np.log(self.nobs * self.new_weights))
        return -2 * llr

    def _opt_kurt(self, nuis_params):
        """
        Called by hy_test_kurt.  This function is optimized over
        nuisance parameters mu and sigma

        Parameters
        -----------

        nuis_params: 1x2 array
            A nuisance mean and variance parameter

        Returns
        ------

        llr: float
            The log likelihood ratio of a pre-speified kurtosis holding the
            nuisance parameters constant.
        """
        mu_data = self.endog - nuis_params[0]
        sig_data = ((self.endog - nuis_params[0]) ** 2) - nuis_params[1]
        kurt_data = (((((self.endog - nuis_params[0]) ** 4) / \
                    (nuis_params[1] ** 2))) - 3) - self.kurt0
        self.est_vect = np.concatenate((mu_data, sig_data, kurt_data), \
                                       axis=1)
        eta_star = self._modif_newton(np.array([1 / self.nobs,
                                               1 / self.nobs,
                                               1 / self.nobs]))
        denom = 1 + np.dot(eta_star, self.est_vect.T)
        self.new_weights = 1 / self.nobs * 1 / denom
        llr = np.sum(np.log(self.nobs * self.new_weights))
        return -2 * llr

    def _opt_skew_kurt(self, nuis_params):
        """
        Called by hy_test_joint_skew_kurt.  This function is optimized over
        nuisance parameters mu and sigma

        Parameters
        -----------

        nuis_params: 1x2 array
            A nuisance mean and variance parameter

        Returns
        ------

        llr: float
            The log likelihood ratio of a pre-speified skewness and
            kurtosis holding the nuisance parameters constant.
        """
        mu_data = self.endog - nuis_params[0]
        sig_data = ((self.endog - nuis_params[0]) ** 2) - nuis_params[1]
        skew_data = ((((self.endog - nuis_params[0]) ** 3) / \
                    (nuis_params[1] ** 1.5))) - self.skew0
        kurt_data = (((((self.endog - nuis_params[0]) ** 4) / \
                    (nuis_params[1] ** 2))) - 3) - self.kurt0
        self.est_vect = np.concatenate((mu_data, sig_data, skew_data,\
                                        kurt_data), axis=1)
        eta_star = self._modif_newton(np.array([1 / self.nobs,
                                               1 / self.nobs,
                                               1 / self.nobs,
                                               1 / self.nobs]))
        denom = 1 + np.dot(eta_star, self.est_vect.T)
        self.new_weights = 1 / self.nobs * 1 / denom
        llr = np.sum(np.log(self.nobs * self.new_weights))
        return -2 * llr

    def _ci_limits_skew(self, skew0):
        """
        Parameters
        ---------
        skew0: float
            Hypothesized value of skewness

        Returns
        -------
        diff: float
            The difference between the log likelihood ratio at skew0 and a
            pre-specified value.
        """
        return self.hy_test_skew(skew0, var_min=self.var_l,
                                 var_max=self.var_u,
                                 mu_min=self.mu_l,
                                 mu_max=self.mu_u)[1] - self.r0

    def _ci_limits_kurt(self, kurt0):
        """
        Parameters
        ---------
        skew0: float
            Hypothesized value of kurtosis

        Returns
        -------
        diff: float
            The difference between the log likelihood ratio at kurt0 and a
            pre-specified value.
        """
        return self.hy_test_kurt(kurt0, var_min=self.var_l,
                                 var_max=self.var_u,
                                 mu_min=self.mu_l,
                                 mu_max=self.mu_u)[1] - self.r0

    def _opt_correl(self, nuis_params):
        """
        Parameters
        ----------

        nuis_params: 4x1 array
            array containing two nuisance means and two nuisance variances

        Returns
        -------

        llr: float
            The log-likelihood of the correlation coefficient holding nuisance
            parameters constant
        """
        mu1_data = (self.endog[:, 0] - nuis_params[0])
        sig1_data = ((self.endog[:, 0] - nuis_params[0]) ** 2) - \
          nuis_params[1]
        mu2_data = self.endog[:, 1] - nuis_params[2]
        sig2_data = ((self.endog[:, 1] - nuis_params[2]) ** 2) -\
           nuis_params[3]
        correl_data = ((self.endog[:, 0] - nuis_params[0]) * \
                      (self.endog[:, 1] - nuis_params[2])) - \
                      (self.corr0 * (nuis_params[1] ** .5) \
                       * (nuis_params[3] ** .5))
        self.est_vect = np.vstack((mu1_data, sig1_data,
                                       mu2_data,
                                       sig2_data, correl_data))
        self.est_vect = self.est_vect.T
        eta_star = self._modif_newton(np.array([1 / self.nobs,
                                               1 / self.nobs,
                                               1 / self.nobs,
                                               1 / self.nobs,
                                               1 / self.nobs]))
        denom = 1 + np.dot(eta_star, self.est_vect.T)
        self.new_weights = 1 / self.nobs * 1 / denom
        llr = np.sum(np.log(self.nobs * self.new_weights))
        return -2 * llr

    def _ci_limits_corr(self, corr0):
        """
        Parameters
        ---------

        corr0: float
            Hypothesized vaalue for the correlation.

        Returns
        -------

        diff: float
            Difference between log-likelihood of corr0 and a pre-specified
            value.
        """
        return self.hy_test_corr(corr0, nuis0=None, mu1_min=self.mu1_lb,
                       mu1_max=self.mu1_ub, mu2_min=self.mu2_lb,
                       mu2_max=self.mu2_ub,
                       var1_min=self.var1_lb, var1_max=self.var1_ub,
                       var2_min=self.var2_lb, var2_max=self.var2_ub,
                       print_weights=0)[1] - self.r0


class DescStatUV(_OptFuncts):
    """
    A class for confidence intervals and hypothesis tests involving mean,
    variance, kurtosis and skewness of a univariate random variable.

    Parameters
    ----------
    endog: nx1 array
        Data to be analyzed

    See Also
    --------

    Method docstring for explicit instructions and uses.
    """

    def __init__(self, endog):
        super(DescStatUV, self).__init__(endog)

    def hy_test_mean(self, mu0,  trans_data=None, print_weights=False):
        """
        Returns the p-value, -2 * log-likelihood ratio and weights
        for a hypothesis test of the means.

        Parameters
        ----------
        mu0: float
            Mean under the null hypothesis

        print_weights: bool, optional
            If print_weights is True the funtion returns
            the weight of the observations under the null hypothesis.
            Default = False

        Returns
        -------

        test_results: tuple
            The p_value and log-likelihood ratio of `mu0`
        """
        self.mu0 = mu0
        if trans_data  is None:
            self.data = self.endog
        else:
            self.data = trans_data
        eta_min = (1 - (1 / self.nobs)) / (self.mu0 - max(self.data))
        eta_max = (1 - (1 / self.nobs)) / (self.mu0 - min(self.data))
        eta_star = optimize.brentq(self._find_eta, eta_min, eta_max)
        new_weights = (1 / self.nobs) * \
            1. / (1 + eta_star * (self.data - self.mu0))
        llr = -2 * np.sum(np.log(self.nobs * new_weights))
        if print_weights:
            return 1 - chi2.cdf(llr, 1), llr, new_weights
        else:
            return 1 - chi2.cdf(llr, 1), llr

    def ci_mean(self, sig=.05, method='gamma', epsilon=10 ** -8,
                 gamma_low=-10 ** 10, gamma_high=10 ** 10, \
                 tol=10 ** -8):
        """
        Returns the confidence interval for the mean.

        Parameters
        ----------
        sig: float
            Significance level. Default=.05

        Optional
        --------

        method: Root finding method,  Can be 'nested-brent' ' gamma'.
            or 'bisect.' Default= 'gamma'

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

        gamma_low: float
            lower bound for gamma when finding lower limit.
            If function returns f(a) and f(b) must have different signs,
            consider lowering gamma_low. Default =-(10``**``10)

        gamma_high: float
            upper bound for gamma when finding upper limit.
            If function returns f(a) and f(b) must have different signs,
            consider raising gamma_high. Default=10``**``10

        epsilon: float
            When using 'nested-brent', amount to decrease (increase)
            from the maximum (minimum) of the data when
            starting the search.  This is to protect against the
            likelihood ratio being zero at the maximum (minimum)
            value of the data.  If data is very small in absolute value
            (<10 ``**`` -6) consider shrinking epsilon

            When using 'gamma' amount to decrease (increase) the
            minimum (maximum) by to start the search for gamma.
            If fucntion returns f(a) and f(b) must have differnt signs,
            consider lowering epsilon.

            Default=10``**``-6

        tol: float
            Tolerance for the likelihood ratio in the bisect method.
            Default=10``**``-6

        Returns
        ------

        Interval: tuple
            Lower and Upper confidence limit
        """
        sig = 1 - sig
        if method == 'nested-brent':
            self.r0 = chi2.ppf(sig, 1)
            middle = np.mean(self.endog)
            epsilon_u = (max(self.endog) - np.mean(self.endog)) * epsilon
            epsilon_l = (np.mean(self.endog) - min(self.endog)) * epsilon
            ul = optimize.brentq(self._ci_limits_mu, middle,
                max(self.endog) - epsilon_u)
            ll = optimize.brentq(self._ci_limits_mu, middle,
                min(self.endog) + epsilon_l)
            return  ll, ul

        if method == 'gamma':
            self.r0 = chi2.ppf(sig, 1)
            gamma_star_l = optimize.brentq(self._find_gamma, gamma_low,
                min(self.endog) - epsilon)
            gamma_star_u = optimize.brentq(self._find_gamma, \
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
        Returns the p-value and -2 ``*`` log-likelihoog ratio for the
            hypothesized variance.

        Parameters
        ----------

        sig2_0: float
            Hypothesized value to be tested

        Optional
        --------

        print_weights: bool
            If True, returns the weights that maximize the
            likelihood of observing sig2_0. Default= False.

        Returns
        --------

        test_results: tuple
            The p_value and log-likelihood ratio of `sig2_0`

        Example
        -------
        random_numbers = np.random.standard_normal(1000)*100
        el_analysis = el.DescStat(random_numbers)
        hyp_test = el_analysis.hy_test_var(9500)
        """
        self.sig2_0 = sig2_0
        mu_max = max(self.endog)
        mu_min = min(self.endog)
        llr = optimize.fminbound(self._opt_var, mu_min, mu_max, \
                                 full_output=1)[1]
        p_val = 1 - chi2.cdf(llr, 1)
        if print_weights:
            return p_val, llr, self.new_weights.T
        else:
            return p_val, llr

    def ci_var(self, lower_bound=None, upper_bound=None, sig=.05):
        """
        Returns the confidence interval for the variance.

        Parameters
        ----------

        lower_bound: float
            The minimum value the lower confidence interval can
            take on. The p-value from hy_test_var(lower_l) must be lower
            than 1 - significance level. Default is calibrated at the .01
            significance level, asusming normality.


        upper_bound: float
            The maximum value the upper confidence interval
            can take. The p-value from hy_test_var(upper_h) must be lower
            than 1 - significance level.  Default is calibrated at the .01
            significance level, asusming normality.


        sig: float
            The significance level for the conficence interval.
            Default= .05

        Returns
        --------

        Interval: tuple
            Lower and Upper confidence limit


        Example
        -------
        random_numbers = np.random.standard_normal(100)
        el_analysis = el.DescStat(random_numbers)
        # Initialize El
        el_analysis.ci_var()
        >>>f(a) and f(b) must have different signs
        el_analysis.ci_var(.5, 2)
        # Searches for confidence limits where the lower limit > .5
        # and the upper limit <2.

        Troubleshooting Tips
        --------------------

        If the function returns the error f(a) and f(b) must have
        different signs, consider lowering lower_bound and raising
        upper_bound.
        """
        if upper_bound is not None:
            ul = upper_bound
        else:
            ul = ((self.nobs - 1) * self.endog.var()) / \
              (chi2.ppf(.0001, self.nobs - 1))
        if lower_bound is not None:
            ll = lower_bound
        else:
            ll = ((self.nobs - 1) * self.endog.var()) / \
              (chi2.ppf(.9999, self.nobs - 1))
        self.r0 = chi2.ppf(1 - sig, 1)
        ll = optimize.brentq(self._ci_limits_var, ll, self.endog.var())
        ul = optimize.brentq(self._ci_limits_var, self.endog.var(), ul)
        return   ll, ul

    def var_p_plot(self, lower, upper, step, sig=.05):
        """
        Plots the p-values of the maximum el estimate for the variance

        Parameters
        ----------

        lower: float
            Lowest value of variance to be computed and plotted

        upper: float
            Highest value of the variance to be computed and plotted

        step: float
            Interval between each plot point.


        sig: float
            Will draw a horizontal line at 1- sig. Default= .05

        This function can be helpful when trying to determine limits
         in the ci_var function.
        """
        sig = 1 - sig
        p_vals = []
        for test in np.arange(lower, upper, step):
            p_vals.append(self.hy_test_var(test)[0])
        p_vals = np.asarray(p_vals)
        plt.plot(np.arange(lower, upper, step), p_vals)
        plt.plot(np.arange(lower, upper, step), (1 - sig) * \
                 np.ones(len(p_vals)))
        return  'Type plt.show to see the figure'

    def mean_var_contour(self, mu_l, mu_h, var_l, var_h, mu_step,
                        var_step,
                        levs=[.2, .1, .05, .01, .001]):
        """
        Returns a plot of the confidence region for a univariate
        mean and variance.

        Parameters
        ----------

        mu_l: float
            Lowest value of the mean to plot

        mu_h: float
            Highest value of the mean to plot

        var_l: float
            Lowest value of the variance to plot

        var_h: float
            Highest value of the variance to plot

        mu_step: float
            Increments to evaluate the mean

        var_step: float
            Increments to evaluate the mean

        Optional
        --------

        levs list
            At Which values of significance the contour lines will be drawn.
            Default: [.2, .1, .05, .01, .001]
        """
        mu_vect = list(np.arange(mu_l, mu_h, mu_step))
        var_vect = list(np.arange(var_l, var_h, var_step))
        z = []
        for sig0 in var_vect:
            self.sig2_0 = sig0
            for mu0 in mu_vect:
                z.append(self._opt_var(mu0, pval=True))
        z = np.asarray(z).reshape(len(var_vect), len(mu_vect))
        fig = plt.contour(mu_vect, var_vect, z, levels=levs)
        plt.clabel(fig)
        return 'Type plt.show to see the figure'

    ## TODO: Use gradient and Hessian to optimize over nuisance params
    ## TODO: Use non-nested optimization to optimize over nuisance
    ## parameters.  See Owen pgs 234- 241

    def hy_test_skew(self, skew0, nuis0=None, mu_min=None,
                     mu_max=None, var_min=None, var_max=None,
                     print_weights=False):
        """
        Returns the p_value and -2 ``*`` log_likelihood for the hypothesized
        skewness.

        Parameters
        ----------
        skew0: float
            Skewness value to be tested

        Optional
        --------

        mu_min, mu_max, var_min, var_max: float
            Minimum and maximum values
            of the nuisance parameters to be optimized over.  If None,
            the function computes the 95% confidence interval for
            the mean and variance and uses the resulting values.

        print_weights: bool
            If True, function also returns the weights that
            maximize the likelihood ratio. Default = False.

        Returns
        --------

        test_results: tuple
            The p_value and log-likelihood ratio of `skew0`
        """
        self.skew0 = skew0
        if nuis0 is not None:
            start_nuisance = nuis0
        else:
            start_nuisance = np.array([self.endog.mean(),
                                       self.endog.var()])
        if mu_min is not None:
            mu_lb = mu_min
        else:
            mu_lb = self.ci_mean()[0]

        if mu_max is not None:
            mu_ub = mu_max
        else:
            mu_ub = self.ci_mean()[1]

        if var_min is None or var_max is None:
            var_ci = self.ci_var()

        if var_min is not None:
            var_lb = var_min
        else:
            var_lb = var_ci[0]

        if var_max is not None:
            var_ub = var_max
        else:
            var_ub = var_ci[1]

        llr = optimize.fmin_l_bfgs_b(self._opt_skew, start_nuisance,
                                     approx_grad=1,
                                     bounds=[(mu_lb, mu_ub),
                                              (var_lb, var_ub)])[1]
        p_val = 1 - chi2.cdf(llr, 1)
        if print_weights:
            return p_val, llr, self.new_weights.T
        return p_val, llr

    def hy_test_kurt(self, kurt0, nuis0=None, mu_min=None,
                     mu_max=None, var_min=None, var_max=None,
                     print_weights=False):
        """
        Returns the p_value and -2 ``*`` log_likelihood for the hypothesized
        kurtosis.

        Parameters
        ----------
        kurt0: float
            kurtosis value to be tested

        Optional
        --------

        mu_min, mu_max, var_min, var_max: float
            Minimum and maximum values
            of the nuisance parameters to be optimized over.  If None,
            the function computes the 95% confidence interval for
            the mean and variance and uses the resulting values.

        print_weights: bool
            If True, function also returns the weights that
            maximize the likelihood ratio. Default = False.

        Returns
        --------

        test_results: tuple
            The p_value and log-likelihood ratio of `kurt0`
        """
        self.kurt0 = kurt0
        if nuis0 is not None:
            start_nuisance = nuis0
        else:
            start_nuisance = np.array([self.endog.mean(),
                                       self.endog.var()])
        if mu_min is not None:
            mu_lb = mu_min
        else:
            mu_lb = self.ci_mean()[0]

        if mu_max is not None:
            mu_ub = mu_max
        else:
            mu_ub = self.ci_mean()[1]

        if var_min is None or var_max is None:
            var_ci = self.ci_var()

        if var_min is not None:
            var_lb = var_min
        else:
            var_lb = var_ci[0]

        if var_max is not None:
            var_ub = var_max
        else:
            var_ub = var_ci[1]

        llr = optimize.fmin_l_bfgs_b(self._opt_kurt, start_nuisance,
                                     approx_grad=1,
                                     bounds=[(mu_lb, mu_ub),
                                              (var_lb, var_ub)])[1]
        p_val = 1 - chi2.cdf(llr, 1)
        if print_weights:
            return p_val, llr, self.new_weights.T
        return p_val, llr

    def hy_test_joint_skew_kurt(self, skew0, kurt0, nuis0=None, mu_min=None,
                     mu_max=None, var_min=None, var_max=None,
                     print_weights=False):
        """
        Returns the p_value and -2 ``*`` log_likelihood for the joint
        hypothesesis test for skewness and kurtosis

        Parameters
        ----------
        skew0: float
            skewness value to be tested
        kurt0: float
            kurtosis value to be tested

        Optional
        --------

        mu_min, mu_max, var_min, var_max: float
            Minimum and maximum values
            of the nuisance parameters to be optimized over.  If None,
            the function computes the 95% confidence interval for
            the mean and variance and uses the resulting values.

        print_weights: bool
            If True, function also returns the weights that
            maximize the likelihood ratio. Default = False.

        Returns
        --------

        test_results: tuple
            The p_value and log-likelihood ratio of the joint hypothesis test.
        """
        self.kurt0 = kurt0
        self.skew0 = skew0
        if nuis0 is not None:
            start_nuisance = nuis0
        else:
            start_nuisance = np.array([self.endog.mean(),
                                       self.endog.var()])
        if mu_min is not None:
            mu_lb = mu_min
        else:
            mu_lb = self.ci_mean()[0]

        if mu_max is not None:
            mu_ub = mu_max
        else:
            mu_ub = self.ci_mean()[1]

        if var_min is None or var_max is None:
            var_ci = self.ci_var()

        if var_min is not None:
            var_lb = var_min
        else:
            var_lb = var_ci[0]

        if var_max is not None:
            var_ub = var_max
        else:
            var_ub = var_ci[1]

        llr = optimize.fmin_l_bfgs_b(self._opt_skew_kurt, start_nuisance,
                                     approx_grad=1,
                                     bounds=[(mu_lb, mu_ub),
                                              (var_lb, var_ub)])[1]
        p_val = 1 - chi2.cdf(llr, 2)
        if print_weights:
            return p_val, llr, self.new_weights.T
        return p_val, llr

    def ci_skew(self, sig=.05, upper_bound=None, lower_bound=None,
                var_min=None, var_max=None, mu_min=None, mu_max=None):
        """
        Returns the confidence interval for skewness.

        Optional
        ----------

        sig: float
            The significance level.  Default = .05

        upper_bound: float
            Maximum Vale of Skewness the upper limit can be.
            Default: .99 confidence assuming normality.

        lower_bound: float
            Minimum value of skewness the lower limit can be.
            Default: .99 confidence level assuming normality.

        var_min, var_max, mu_min, mu_max: float
            Minimum Value of the nuisance
            variance and mean. Default:  sig confidence limits

        Returns
        ------

        Interval: tuple
            Lower and Upper confidence limit

        Tips
        ----

        For large n (approx >25), the default parameters should provide
        successful optimization.

        For small n, var_min and var_max will likely be provided by the
        user.

        If parameters are left at the default and the optimization
        fails, the function will alert as to which parameter it failed
        to compute.

        If function returns f(a) and f(b) must have different signs, consider
        expanding lower and upper bound.
        """
        if upper_bound is not None:
            ul = upper_bound
        else:
            ul = skew(self.endog) + \
            2.5 * ((6. * self.nobs * (self.nobs - 1.)) / \
              ((self.nobs - 2.) * (self.nobs + 1.) * \
               (self.nobs + 3.))) ** .5
        if lower_bound is not None:
            ll = lower_bound
        else:
            ll = skew(self.endog) - \
            2.5 * ((6. * self.nobs * (self.nobs - 1.)) / \
              ((self.nobs - 2.) * (self.nobs + 1.) * \
               (self.nobs + 3.))) ** .5
        # Need to calculate variance and mu limits here to avoid
        # recalculating at every iteration in the maximization.
        if (var_max is None) or (var_min is None):
            print 'Finding CI for the variance'
            var_lims = self.ci_var(sig=sig)
        if (mu_max is None) or (mu_max is None):
            print 'Finding CI for the mean'
            mu_lims = self.ci_mean(sig=sig)
        if var_min is not None:
            self.var_l = var_min
        else:
            self.var_l = var_lims[0]
        if var_max is not None:
            self.var_u = var_max
        else:
            self.var_u = var_lims[1]
        if mu_min is not None:
            self.mu_l = mu_min
        else:
            self.mu_l = mu_lims[0]
        if mu_max is not None:
            self.mu_u = mu_max
        else:
            self.mu_u = mu_lims[1]
        self.r0 = chi2.ppf(1 - sig, 1)
        print 'Finding the lower bound for skewness'
        ll = optimize.brentq(self._ci_limits_skew, ll, skew(self.endog))
        print 'Finding the upper bound for skewness'
        ul = optimize.brentq(self._ci_limits_skew, skew(self.endog), ul)
        return   ll, ul

    def ci_kurt(self, sig=.05, upper_bound=None, lower_bound=None,
                var_min=None, var_max=None, mu_min=None, mu_max=None):
        """
        Returns the confidence interval for kurtosis.

        Optional
        --------

        sig: float
            The significance level.  Default = .05

        upper_bound: float
            Maximum Vale of Kurtosis the upper limit can be.
            Default: .99 confidence assuming normality.

        lower_bound: float
            Minimum value of Kurtosis the lower limit can be.
            Default: .99 confidence level assuming normality.

        var_min, var_max, mu_min, mu_max: float
            Minimum Value of the nuisance
            variance and mean. Default:  sig confidence limits

        Returns
        --------

        Interval: tuple
            Lower and Upper confidence limit

        Tips
        ----

        For large n (approx >25), the default parameters should provide
        successful optimization.

        For small n, upper_bound and lower_bound will likely have to be
        provided.  Consider using hy_test_kurt to find values close to
        the desired significance level.

        If parameters are left at the default and the optimization
        fails, the function will alert as to which parameter it failed
        to compute.

        If function returns f(a) and f(b) must have different signs, consider
        expanding lower and upper bound.
        """
        if upper_bound is not None:
            ul = upper_bound
        else:
            ul = kurtosis(self.endog) + \
            (2.5 * (2. * ((6. * self.nobs * (self.nobs - 1.)) / \
              ((self.nobs - 2.) * (self.nobs + 1.) * \
               (self.nobs + 3.))) ** .5) * \
               (((self.nobs ** 2.) - 1.) / ((self.nobs - 3.) *\
                 (self.nobs + 5.))) ** .5)
        if lower_bound is not None:
            ll = lower_bound
        else:
            ll = kurtosis(self.endog) - \
            (2.5 * (2. * ((6. * self.nobs * (self.nobs - 1.)) / \
              ((self.nobs - 2.) * (self.nobs + 1.) * \
               (self.nobs + 3.))) ** .5) * \
               (((self.nobs ** 2.) - 1.) / ((self.nobs - 3.) *\
                 (self.nobs + 5.))) ** .5)
        # Need to calculate variance and mu limits here to avoid
        # recalculating at every iteration in the maximization.
        if (var_max is None) or (var_min is None):
            print 'Finding CI for the variance'
            var_lims = self.ci_var(sig=sig)
        if (mu_max is None) or (mu_min is None):
            print 'Finding CI for the mean'
            mu_lims = self.ci_mean(sig=sig)
        if var_min is not None:
            self.var_l = var_min
        else:
            self.var_l = var_lims[0]
        if var_max is not None:
            self.var_u = var_max
        else:
            self.var_u = var_lims[1]
        if mu_min is not None:
            self.mu_l = mu_min
        else:
            self.mu_l = mu_lims[0]
        if mu_max is not None:
            self.mu_u = mu_max
        else:
            self.mu_u = mu_lims[1]
        self.r0 = chi2.ppf(1 - sig, 1)
        print 'Finding the lower bound for kurtosis'
        ll = optimize.brentq(self._ci_limits_kurt, ll, \
                             kurtosis(self.endog))
        print 'Finding the upper bound for kurtosis'
        ul = optimize.brentq(self._ci_limits_kurt, kurtosis(self.endog), \
                             ul)
        return   ll, ul


class DescStatMV(_OptFuncts):
    """
    A class for conducting inference on multivariate means and
    correlation

    Parameters
    ----------
    endog: nxk array
        Data to be analyzed

    See Also
    --------

    Method docstring for explicit instructions and uses.
    """

    def __init__(self, endog):
        super(DescStatMV, self).__init__(endog)

    def mv_hy_test_mean(self, mu_array, print_weights=False):
        """
        Returns the -2 ``*`` log likelihood and the p_value
        for a multivariate hypothesis test of the mean

        Parameters
        ----------
        mu_array : 1d array
            hypothesized values for the mean.  Must have same number of
            elements as columns in endog.

        Optional
        --------

        print_weights: bool
            If True, returns the weights that maximize the
            likelihood of mu_array Default= False.

        Returns
        -------

        test_results: tuple
            The p_value and log-likelihood ratio of `mu_array`
        """
        if len(mu_array) != self.endog.shape[1]:
            raise Exception('mu_array must have the same number of \
                           elements as the columns of the data.')
        mu_array = mu_array.reshape(1, self.endog.shape[1])
        means = np.ones((self.endog.shape[0], self.endog.shape[1]))
        means = mu_array * means
        self.est_vect = self.endog - means
        start_vals = 1 / self.nobs * np.ones(self.endog.shape[1])
        eta_star = self._modif_newton(start_vals)
        denom = 1 + np.dot(eta_star, self.est_vect.T)
        self.new_weights = 1 / self.nobs * 1 / denom
        llr = np.sum(np.log(self.nobs * self.new_weights))
        p_val = 1 - chi2.cdf(-2 * llr, mu_array.shape[1])
        if print_weights:
            return p_val, -2 * llr, self.new_weights.T
        else:
            return p_val, -2 * llr

    def mv_mean_contour(self, mu1_l, mu1_u, mu2_l, mu2_u, step1, step2,
                        levs=[.2, .1, .05, .01, .001], plot_dta=False):
        """
        Creates confidence region plot for the mean of bivariate data

        Parameters
        ----------

        m1_l: float
            Minimum value of the mean for variable 1

        m1_u: float
            Maximum value of the mean for variable 1

        mu2_l: float
            Minimum value of the mean for variable 2

        mu2_u: float
            Maximum value of the mean for variable 2

        step1: float
            Increment of evaluations for variable 1

        step2: float
            Increment of evaluations for variable 2


        Optional
        --------
        levs: list
            Levels to be drawn on the contour plot.
            Default =  [.2, .1 .05, .01, .001]

        plot_dta: bool
            If True, makes a scatter plot of the data on
            top of the contour plot. Default =  False.

        Notes
        -----
        The smaller the step size, the more accurate the intervals
        will be.

        If the function returns optimization failed, consider narrowing
        the boundaries of the plot.

        Example
        -------

        two_rvs = np.random.standard_normal((20,2))
        el_analysis = el.DescStat(two_rvs)
        contourp = el_analysis.mv_mean_contour(-2, 2, -2, 2, .1, .1)
        contourp
        >>>Type plt.show() to see plot
        plt.show()
        """
        if self.endog.shape[1] != 2:
            raise Exception('Data must contain exactly two variables')
        x = (np.arange(mu1_l, mu1_u, step1))
        y = (np.arange(mu2_l, mu2_u, step2))
        pairs = itertools.product(x, y)
        z = []
        for i in pairs:
            z.append(self.mv_hy_test_mean(np.asarray(i))[0])
        X, Y = np.meshgrid(x, y)
        z = np.asarray(z)
        z = z.reshape(X.shape[1], Y.shape[0])
        fig = plt.contour(x, y, z.T, levels=levs)
        plt.clabel(fig)
        if plot_dta:
            plt.plot(self.endog[:, 0], self.endog[:, 1], 'bo')
        return 'Type plt.show to see the figure'

    def hy_test_corr(self, corr0, nuis0=None, mu1_min=None,
                       mu1_max=None, mu2_min=None, mu2_max=None,
                       var1_min=None, var1_max=None,
                       var2_min=None, var2_max=None, print_weights=0):
        """
        Returns the p-value and -2 * log-likelihood ratio for the
        correlation coefficient between 2 variables.

        Parameters
        ---------
        corr0: float
            Hypothesized value to be tested

        Optional
        --------
        nuis0: 4x1 array [mu1, var1, mu21, var2]
            Starting value for nuisance parameters. default |
            sample estimate of each Parameters

        mu1_max through var2_max: float
            Limits of nuisance parameters
            to maximize over.  default | 95% confidence limits assuming
            normality

        print_weights: bool
            If true, returns the weights that maximize
            the log-likelihood at the hypothesized value.

        Notes
        -----

        In practice, when optimizing over the nuisance parameters,
        the optimal nuisance parameters are often very close to their
        sample estimate and rarely close to their 95% confidence level.
        Therefore, if the function returns 'Optimization Fails',
        consider narrowing the intervals of the nuisance parameters.

        Also, for very unlikely hypothesized values (ratio > 1000), the
        function may also not be able to optimize successfully.
        """
        if self.endog.shape[1] != 2:
            raise Exception('Correlation matrix not yet implemented')

        self.corr0 = corr0

        if nuis0 is not None:
            start_nuisance = nuis0
        else:
            start_nuisance = np.array([self.endog[:, 0].mean(),
                                       self.endog[:, 0].var(),
                                       self.endog[:, 1].mean(),
                                       self.endog[:, 1].var()])

        if mu1_min is not None:
            mu1_lb = mu1_min
        else:
            mu1_lb = self.endog[:, 0].mean() - ((1.96 * \
              np.sqrt((self.endog[:, 0].var()) / self.nobs)))

        if mu1_max is not None:
            mu1_ub = mu1_max
        else:
            mu1_ub = self.endog[:, 0].mean() + (1.96 * \
              np.sqrt((((self.endog[:, 0].var()) / self.nobs))))

        if mu2_min is not None:
            mu2_lb = mu2_min
        else:
            mu2_lb = self.endog[:, 1].mean() - (1.96 * \
              np.sqrt((((self.endog[:, 1].var()) / self.nobs))))

        if mu2_max is not None:
            mu2_ub = mu2_max
        else:
            mu2_ub = self.endog[:, 1].mean() + (1.96 * \
              np.sqrt((((self.endog[:, 1].var()) / self.nobs))))

        if var1_min is not None:
            var1_lb = var1_min
        else:
            var1_lb = (self.endog[:, 0].var() * (self.nobs - 1)) / \
              chi2.ppf(.975, self.nobs)

        if var1_max is not None:
            var1_ub = var1_max
        else:
            var1_ub = (self.endog[:, 0].var() * (self.nobs - 1)) / \
              chi2.ppf(.025, self.nobs)

        if var2_min is not None:
            var2_lb = var2_min
        else:
            var2_lb = (self.endog[:, 1].var() * (self.nobs - 1)) / \
              chi2.ppf(.975, self.nobs)
        if var2_max is not None:
            var2_ub = var2_max
        else:
            var2_ub = (self.endog[:, 1].var() * (self.nobs - 1)) / \
              chi2.ppf(.025, self.nobs)

      ## TODO: IS there a way to condense the above default Parameters?
        llr = optimize.fmin_l_bfgs_b(self._opt_correl, start_nuisance,
                                     approx_grad=1,
                                     bounds=[(mu1_lb, mu1_ub),
                                              (var1_lb, var1_ub),
                                              (mu2_lb, mu2_ub),
                                              (var2_lb, var2_ub)])[1]
        p_val = 1 - chi2.cdf(llr, 1)
        if print_weights:
            return p_val, llr, self.new_weights.T
        return p_val, llr

    def ci_corr(self, sig=.05, upper_bound=None, lower_bound=None,
                       mu1_min=None,
                       mu1_max=None, mu2_min=None, mu2_max=None,
                       var1_min=None, var1_max=None,
                       var2_min=None, var2_max=None):
        """
        Returns the confidence intervals for the correlation coefficient.

        Parameters
        ----------
        sig: float
            The significance level.  Default= .05

        upper_bound: float
            Maximum value the upper confidence limit can be.
            Default: 99% confidence assuming normality.

        lower_bound: float
            Minimum value the lower condidence limit can be.
            Default: 99% confidence assuming normality.

        mu1_max through var2_max: float
            Limits of nuisance parameters
            to maximize over.  Default:  95% confidence limits assuming
            normality

        Returns
        --------

        Interval: tuple
            Lower and Upper confidence limit

        """
        self.r0 = chi2.ppf(1 - sig, 1)
        point_est = np.corrcoef(self.endog[:, 0], self.endog[:, 1])[0, 1]

        if upper_bound is not None:
            ul = upper_bound
        else:
            ul = min(.999, point_est + \
                          2.5 * ((1. - point_est ** 2.) / \
                          (self.nobs - 2.)) ** .5)

        if lower_bound is not None:
            ll = lower_bound
        else:
            ll = max(- .999, point_est - \
                          2.5 * (np.sqrt((1. - point_est ** 2.) / \
                          (self.nobs - 2.))))

        if mu1_min is not None:
            self.mu1_lb = mu1_min
        else:
            self.mu1_lb = self.endog[:, 0].mean() - np.sqrt((1.96 * \
              (self.endog[:, 0].var()) / self.nobs))

        if mu1_max is not None:
            self.mu1_ub = mu1_max
        else:
            self.mu1_ub = self.endog[:, 0].mean() + (1.96 * \
              np.sqrt(((self.endog[:, 0].var()) / self.nobs)))

        if mu2_min is not None:
            self.mu2_lb = mu2_min
        else:
            self.mu2_lb = self.endog[:, 1].mean() - (1.96 * \
              np.sqrt(((self.endog[:, 1].var()) / self.nobs)))

        if mu2_max is not None:
            self.mu2_ub = mu2_max
        else:
            self.mu2_ub = self.endog[:, 1].mean() + (1.96 * \
              np.sqrt(((self.endog[:, 1].var()) / self.nobs)))

        if var1_min is not None:
            self.var1_lb = var1_min
        else:
            self.var1_lb = (self.endog[:, 0].var() * (self.nobs - 1)) / \
              chi2.ppf(.975, self.nobs)

        if var1_max is not None:
            self.var1_ub = var1_max
        else:
            self.var1_ub = (self.endog[:, 0].var() * (self.nobs - 1)) / \
              chi2.ppf(.025, self.nobs)

        if var2_min is not None:
            self.var2_lb = var2_min
        else:
            self.var2_lb = (self.endog[:, 1].var() * (self.nobs - 1)) / \
              chi2.ppf(.975, self.nobs)

        if var2_max is not None:
            self.var2_ub = var2_max
        else:
            self.var2_ub = (self.endog[:, 1].var() * (self.nobs - 1)) / \
              chi2.ppf(.025, self.nobs)
        print 'Finding the lower bound for correlation'

        ll = optimize.brentq(self._ci_limits_corr, ll, point_est)
        print 'Finding the upper bound for correlation'
        ul = optimize.brentq(self._ci_limits_corr,
                             point_est, ul)
        return ll, ul
