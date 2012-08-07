"""

Accelerated Failure Time (AFT) Model with empirical likelihood inference.

AFT regression analysis is applicable when the researcher has access
to a randomly right censored dependent variable, a matrix of exogenous
variables and an indicatior variable (delta) that takes a value of 0 if the
observation is censored and 1 otherwise.

AFT References
--------------

Stute, W. (1993). "Consistent Estimation Under Random Censorship when
Covariables are Present." Journal of Multivariate Analysis.
Vol. 45. Iss. 1. 89-103

EL and AFT References
---------------------

Zhou, Kim And Bathke. "Empirical Likelihood Analysis for the Heteroskedastic
Accelerated Failure Time Model." Manuscript:
URL: www.ms.uky.edu/~mai/research/CasewiseEL20080724.pdf

Zhou, M. (2005). Empirical Likelihood Ratio with Arbitrarily Censored/
Truncated Data by EM Algorithm.  Journal of Computational and Graphical
Statistics. 14:3, 643-656.


"""

import numpy as np
from statsmodels.api import WLS, add_constant
from elregress import ElLinReg
from statsmodels.base.model import _fit_mle_newton
from scipy import optimize
from scipy.stats import chi2

class OptAFT:
    def __init__(self):
        pass


    def _wtd_grad(self, eta1, est_vect, wts):
        """
        Calculates the gradient of a weighted empirical likelihood
        problem.


        Parameters
        ----------
        eta1: 1xm array.

        This is the value of lamba used to write the
        empirical likelihood probabilities in terms of the lagrangian
        multiplier.

        Returns
        -------
        gradient: m x 1 array
            The gradient used in _wtd_modif_newton
        """
        wts = wts.reshape(-1,1)
        nobs = self.uncens_nobs
        data = est_vect.T
        data_star_prime = (np.sum(wts) + np.dot(eta1, data))
        idx = data_star_prime < 1. / nobs
        not_idx = ~idx
        data_star_prime[idx] = 2. * nobs - (nobs) ** 2 * data_star_prime[idx]
        data_star_prime[not_idx] = 1. / data_star_prime[not_idx]
        data_star_prime = data_star_prime.reshape(nobs, 1) # log*'
        return np.dot(data, wts*data_star_prime)

    def _wtd_hess(self, eta1, est_vect, wts):
        """
        Calculates the hessian of a weighted empirical likelihood
        provlem.

        Parameters
        ----------
        eta1: 1xm array.

        This is the value of lamba used to write the
        empirical likelihood probabilities in terms of the lagrangian
        multiplier.

        Returns
        -------
        hess: m x m array
            Weighted hessian used in _wtd_modif_newton
        """
        nobs = self.uncens_nobs
        data = est_vect.T
        wts = wts.reshape(-1, 1)
        data_star_doub_prime = np.copy(np.sum(wts) + np.dot(eta1, data))
        idx = data_star_doub_prime < 1. / nobs
        not_idx = ~idx
        data_star_doub_prime[idx] = - nobs ** 2
        data_star_doub_prime[not_idx] = - (data_star_doub_prime[not_idx]) ** -2
        data_star_doub_prime = data_star_doub_prime.reshape(nobs, 1)
        wtd_dsdp = wts * data_star_doub_prime
        return np.dot(data, wtd_dsdp*data.T)


    def _wtd_log_star(self, eta1, est_vect, wts):
        """
        Parameters
        ---------
        eta1: float
            Lagrangian multiplier

        Returns
        ------

        data_star: array
            The weighted logstar of the estimting equations

        Note
        ----

        This function is really only a placeholder for the _fit_mle_Newton.
        The function value is not used in optimization and the optimal value
        is disregarded when computng the log likelihood ratio.
        """
        nobs = self.uncens_nobs
        data = est_vect.T
        data_star = np.log(wts).reshape(-1,1)\
           + (np.sum(wts) + np.dot(eta1, data)).reshape(-1,1)
        idx = data_star < 1. / nobs
        not_idx = ~idx
        data_star[idx] = np.log(1 / nobs) - 1.5 +\
                  2. * nobs * data_star[idx] -\
                  ((nobs * data_star[idx]) ** 2.) / 2
        data_star[not_idx] = np.log(data_star[not_idx])
        return data_star

    def _wtd_modif_newton(self,  x0, est_vect, wts):
        """
        Weighted Modified Newton's method for maximizing the log* equation.

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
        x0 = x0.reshape(est_vect.shape[1], 1)
        f = lambda x0: - np.sum(self._wtd_log_star(x0.T, est_vect, wts))
        grad = lambda x0: - self._wtd_grad(x0.T, est_vect, wts)
        hess = lambda x0: - self._wtd_hess(x0.T, est_vect, wts)
        kwds = {'tol': 1e-8}
        res = _fit_mle_newton(f, grad, x0, (), kwds, hess=hess, maxiter=50, \
                              disp=0)
        return res[0].T


    def _opt_wtd_nuis_regress(self, test_vals):
        """
        A function that is optimized over nuisance parameters to conduct a
        hypothesis test for the parameters of interest

        Parameters
        ----------

        params: 1d array
            The regression coefficients of the model.  This includes the
            nuisance and parameters of interests.

        Returns
        -------

        llr: float
            -2 times the log likelihood of the nuisance parameters and the
            hypothesized value of the parameter(s) of interest.

        """
        self.new_params = test_vals.reshape(self.nvar, 1)
        self.est_vect = self.uncens_exog * (self.uncens_endog -
                                            np.dot(self.uncens_exog,
                                                         self.new_params))
        eta_star = self._wtd_modif_newton(self.start_lbda, self.est_vect,
                                         self._fit_weights)
        self.eta_star = eta_star
        denom = np.sum(self._fit_weights) + np.dot(eta_star, self.est_vect.T)
        self.new_weights = self._fit_weights / denom
        return -1 * np.sum(np.log(self.new_weights))

    def _EM_test(self, nuisance_params,
                F=None, survidx=None, uncens_nobs=None,
                numcensbelow=None, km=None, uncensored=None, censored=None,maxiter=None):
        """
        Uses EM algorithm to compute the maximum likelihood of a test

        Parameters
        ---------

        Nuisance Params: array
            Vector of values to be used as nuisance params.

        maxiter: int
            Number of iterations in the EM algorithm for a parameter vector

        Returns
        -------
        -2 ''*'' log likelihood ratio at hypothesized values and nuisance params

        Notes
        -----
        F, survidx, uncens_nobs, numcensbelow, km, uncensored, censored are provided by
        the test_beta function.
        """
        iters=0
        params = np.copy(self.params())
        params[self.param_nums] = self.b0_vals

        nuis_param_index = np.int_(np.delete(np.arange(self.nvar),
                                           self.param_nums))
        params[nuis_param_index] = nuisance_params
        to_test = params.reshape(self.nvar, 1)
        while iters < maxiter:
            F = F.flatten()
            death=np.cumsum(F[::-1])
            survivalprob = death[::-1]
            surv_point_mat = np.dot(F.reshape(-1, 1),
                                1./survivalprob[survidx].reshape(1,-1))
            surv_point_mat = add_constant(surv_point_mat, prepend=1)
            summed_wts = np.cumsum(surv_point_mat, axis=1)
            wts = summed_wts[np.int_(np.arange(uncens_nobs)),
                             numcensbelow[uncensored]]
            # ^E step
            # See Zhou 2005, section 3.
            self._fit_weights = wts
            opt_res = self._opt_wtd_nuis_regress(to_test)
                # ^ Uncensored weights' contribution to likelihood value.
            F = self.new_weights
                # ^ M step
            iters = iters+1
        death = np.cumsum(F.flatten()[::-1])
        survivalprob = death[::-1]
        llike = -opt_res + np.sum(np.log(survivalprob[survidx]))

        wtd_km = km.flatten()/np.sum(km)
        survivalmax = np.cumsum(wtd_km[::-1])[::-1]
        llikemax = np.sum(np.log(wtd_km[uncensored])) + \
          np.sum(np.log(survivalmax[censored]))
        return -2 * (llike - llikemax)



class emplikeAFT(OptAFT):
    def __init__(self, endog, exog, censors):
        self.nobs = float(np.shape(exog)[0])
        self.endog = endog.reshape(self.nobs, 1)
        self.exog = exog.reshape(self.nobs, -1)
        self.censors = censors.reshape(self.nobs, 1)
        self.nvar = self.exog.shape[1]
        self.idx = np.lexsort((-self.censors[:, 0], self.endog[:, 0]))
        self.endog = self.endog[self.idx]
        self.exog = self.exog[self.idx]
        self.censors = self.censors[self.idx]
        self.censors[-1]=1 # Sort in init, not in function
        self.uncens_nobs = np.sum(self.censors)
        self.uncens_endog = self.endog[np.bool_(self.censors),:].reshape(-1,1)
        self.uncens_exog = self.exog[np.bool_(self.censors.flatten()),:]


    def _is_tied(self, endog, censors):
        """
        Indicated if an observation takes the same value as the next
        ordered observation.

        Parameters
        ----------
        endog: array
            Models endogenous variable
        censors: array
            arrat indicating a censored array

        Returns
        -------
        indic_ties: array
            ties[i]=1 if endog[i]==endog[i+1] and
            censors[i]=censors[i+1]
        """
        nobs = int(self.nobs)
        endog_idx = endog[np.arange(nobs - 1)] == (
            endog[np.arange(nobs - 1) + 1])
        censors_idx = censors[np.arange(nobs - 1)] == (
            censors[np.arange(nobs - 1) + 1])
        indic_ties = endog_idx * censors_idx  # Both true
        return np.int_(indic_ties)

    def _km_w_ties(self, tie_indic, untied_km):
        """
        Computes KM estimator value at each observation, taking into acocunt
        ties in the data.

        Parameters:
        ----------

        tie_indic: 1d array
            Indicates if the i'th observation is the same as the ith +1

        untied_km: 1d array
            Km estimates at each observation assuming no ties.

        """
        # TODO: Vectorize, even though it is only 1 pass through for any
        # function call
        num_same = 1
        idx_nums = []
        for obs_num in np.arange(int(self.nobs - 1))[::-1]:
            if tie_indic[obs_num] == 1:
                idx_nums.append(obs_num)
                num_same = num_same + 1
                untied_km[obs_num] = untied_km[obs_num + 1]
            elif tie_indic[obs_num] == 0 and num_same > 1:
                idx_nums.append(max(idx_nums) + 1)
                idx_nums = np.asarray(idx_nums)
                untied_km[idx_nums] = untied_km[idx_nums]
                num_same = 1
                idx_nums = []
        return untied_km.reshape(self.nobs, 1)

    def _make_km(self, endog, censors):
        """

        Computes the Kaplan-Meier estimate for the weights in the AFT model

        Parameters
        ----------

        endog: nx1 array
            Array of response variables

        censors: nx1 array
            Censor-indicating variable

        Returns
        ------

        Kaplan Meier estimate for each observation

        Notes
        -----

        This function makes calls to _is_tied and km_w_ties to handle ties in
        the data.If a censored observation and an uncensored observation has
        the same value, it is assumed that the uncensored happened first.

        """
        nobs = self.nobs
        num = (nobs - (np.arange(nobs) + 1.))
        denom = ((nobs - (np.arange(nobs) + 1.) + 1.))
        km = (num / denom).reshape(nobs, 1)
        km = km ** np.abs(censors - 1.)
        km = np.cumprod(km)  # If no ties, this is kaplan-meier
        tied = self._is_tied(endog, censors)
        wtd_km = self._km_w_ties(tied, km)
        return  (censors / wtd_km).reshape(nobs, 1)

    def params(self):
        """

        Fits an AFT model and returns parameters.

        Parameters
        ---------
        None


        Returns
        -------
        Fitted params

        Notes
        -----
        To avoid dividing by zero, max(endog) is assumed to be uncensored.
        """
        self.modif_censors = np.copy(self.censors)
        self.modif_censors[-1] = 1
        wts = self._make_km(self.endog, self.modif_censors)
        res = WLS(self.endog, self.exog, wts).fit()
        params = res.params
        return params

    def test_beta(self, value, param_num, ftol=10 **-5, maxiter=30,
                  print_weights=1):
        """
        Returns the profile log likelihood for regression parameters 'param_num' and value
        'value'

        Parameters
        ----------
        value: list
            The value of parameters to be tested

        param_num: list
            Which parameters to be tested

        maxiter: int, optional
            How many iterations to use in the EM algorithm.  Default is 30

        print_weights: bool
            If true, returns the weights the maximize the profile log likelihood.
            Default is False



        Returns
        -------
        test_results: tuple
            The log-likelihood and p-pvalue of the test.

        Examples
        -------

        import statsmodels.api as sm
        import numpy as np

        # Test parameter is .05 in one regressor no intercept model
        data=sm.datasets.heart.load()
        y = np.log10(data.endog)
        x = data.exog
        cens = data.censors
        model = sm.emplike.emplikeAFT(y, x, cens)
        res=model.test_beta([0], [0])
        >>>res
        >>>(1.4657739632606308, 0.22601365256959183)

        #Test slope is 0 in  model with intercept

        data=sm.datasets.heart.load()
        y = np.log10(data.endog)
        x = data.exog
        cens = data.censors
        model = sm.emplike.emplikeAFT(y, sm.add_constant(x, prepend=1), cens)
        res=model.test_beta([0], [1])
        >>>res
        >>>(1.4657739632606308, 0.22601365256959183)

        """

        self.start_lbda = np.zeros(self.nvar)
        # Starting value for Newton Optimizer
        censors = self.censors
        endog = self.endog
        exog = self.exog
        censors[-1] = 1
        uncensored = (censors == 1).flatten()
        censored = (censors == 0).flatten()
        uncens_endog = endog[uncensored]
        uncens_exog = exog[uncensored, :]
        reg_model = ElLinReg(uncens_endog, uncens_exog)
        reg_model.hy_test_beta(value, param_num)
        km = self._make_km(self.endog, self.censors).flatten()
        uncens_nobs = self.uncens_nobs
        F = np.asarray(reg_model.new_weights).reshape(uncens_nobs)
        # Step 0 ^
        self.b0_vals = value
        self.param_nums =param_num
        params=self.params()
        survidx = np.where(censors==0)
        survidx = survidx[0] - np.arange(len(survidx[0])) #Zhou's K
        numcensbelow =  np.int_(np.cumsum(1-censors))
        if len(self.param_nums) == len(params):
            llr = self._EM_test([], F=F , survidx=survidx,
                             uncens_nobs=uncens_nobs, numcensbelow=numcensbelow,
                             km=km, uncensored=uncensored, censored=censored, maxiter=25)
            return llr, chi2.sf(llr, self.nvar)
        else:
            x0 = np.delete(params, self.param_nums)
            res = optimize.fmin_powell(self._EM_test, x0,
                                   (F, survidx, uncens_nobs,
                                numcensbelow, km, uncensored, censored, maxiter), full_output=1)

            llr = res[1]
            return llr, chi2.sf(llr, len(param_num))





