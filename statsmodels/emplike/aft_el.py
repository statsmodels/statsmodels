"""

Accelerated Failure Time (AFT) Model with empirical likelihood inference.

AFT regression analysis is applicable when the researcher has access
to a randomly right censored dependent variable, a matrix of exogenous
variables and an indicatior variable (delta) that takes a value of 0 if the
observation is censored and 1 otherwise.

AFT References
--------------

Koul, Susarla and Ryzin. (1981).  "Regression Analysis with Randomly
Right-Censored Data." Ann. Statistics. Vol. 9. N. 6. 1276-1288


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

Li, G and Wang, Q.(2003). "Empirical Likelihood Regression Analysis for Right
Censored Data." Statistica Sinica 13. 51-68.

Qin, G and Jing, B. (2001). "Empirical Likelihood for Censored Linear
Regression." Scandanavian Journal of Statistics. Vol 28. Iss. 4. 661-673.
"""

import numpy as np
from statsmodels.api import OLS, WLS, add_constant
from elregress import ElLinReg

class emplikeAFT(object):
    def __init__(self, endog, exog, censors):
        self.nobs = float(np.shape(exog)[0])
        self.endog = endog.reshape(self.nobs, 1)
        self.exog = exog.reshape(self.nobs, -1)
        self.censors = censors.reshape(self.nobs, 1)
        self.idx = np.lexsort((-self.censors[:, 0], self.endog[:, 0]))
        self.endog = self.endog[self.idx]
        self.exog = self.exog[self.idx]
        self.censors = self.censors[self.idx]  # Sort in init, not in function

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
        if self.km_est == 'koul':
            num = (nobs - (np.arange(nobs) + 1.) + 1.)
            denom = ((nobs - (np.arange(nobs) + 1.) + 2.))
        if self.km_est == 'li':
            num = (nobs - (np.arange(nobs) + 1.))
            denom = ((nobs - (np.arange(nobs) + 1.) + 1.))
        km = (num / denom).reshape(nobs, 1)
        km = km ** np.abs(censors - 1.)
        km = np.cumprod(km)  # If no ties, this is kaplan-meier
        tied = self._is_tied(endog, censors)
        wtd_km = self._km_w_ties(tied, km)
        return  (censors / wtd_km).reshape(nobs, 1)

    def fit(self, km_est='koul', wt_method='koul', last_uncensored=False):
        """

        Fits an AFT model and returns results instance.

        Parameters
        ---------

        km: string, optional
            Version of Kaplan-Meier estimate to use.  If 'koul', uses the
            estimator in Koul (1981).  This ensures no divide by 0 errors.
            If 'li,' uses the estimate in Qin (2001) and Li (2003).
            If max(endog)  is censored and km='Qin', estimation and inference
            is impossible.  See Owen (2001). Default is 'koul'

        wt_method: string, optional
            If 'koul', estimation is conducted using synthetic data of
            Koul et al (1981). If 'stute', inference is conducted by WLS
            as in Stute (1993).  If wt_method is 'kout' and max(endog) is
            censored and km='koul', estimation andinference is impossible.
            Default is 'koul.'

        last_uncensored: bool, optional
            If True, the model assumes that max(endog) is uncensored,
            regardless of value in censors.  This prevents divide by 0 errors.
            Default is False.

        Returns
        -------

        Fitted results instance.  See also RegressionResults.

        """
        self.km_est = km_est
        self.modif_censors = np.copy(self.censors)
        if last_uncensored:
            self.modif_censors[-1] = 1
        wts = self._make_km(self.endog, self.modif_censors)
        if wt_method == 'koul':
            res = OLS(self.endog * wts, self.exog).fit()
        if wt_method == 'stute':
            res = WLS(self.endog, self.exog, wts).fit()
            res.bse = None      # Inference will be done with EL.
        res.km = wts
        res.conf_int = None
        res.t = None
        res.pvalues = None
        res.f_test = None
        res.fvalue = None
        return res

    def test_beta(self, value, param_num, ftol=10 **-5, maxiter=1,
                  print_weights=1 ):
        """
        Some notes:

        Censored Observations have weight 0.
        """
        censors = self.censors
        endog = self.endog
        exog = self.exog
        censors[-1] = 1
        uncensored = (censors == 1).flatten()
        censored = (censors == 0).flatten()
        uncens_endog = endog[uncensored]
        cens_endog = endog[censored]
        uncens_exog = exog[uncensored, :]
        cens_exog = exog[censored, :]
        reg_model = ElLinReg(uncens_endog, uncens_exog)
        uncens_nobs = reg_model.nobs
        reg_model.hy_test_beta(value, param_num)
        init_F = np.asarray(reg_model.new_weights).reshape(uncens_nobs)
        # init_F is step 0 in Zhou (2005)
        iters=1
        #while iters < maxiter
        survidx = np.where(censors==0)
        survidx = survidx[0] - np.arange(len(survidx[0])) #Zhou's K
        numcensbelow =  np.int_(np.cumsum(1-censors))
        death = np.cumsum(init_F[::-1])
        survivalprob = death[::-1]
        surv_point_mat = np.dot(init_F.reshape(-1, 1),
                                1./survivalprob[survidx].reshape(1,-1))
        surv_point_mat = add_constant(surv_point_mat, prepend=1)
        summed_wts = np.cumsum(surv_point_mat, axis=1)
        wts = summed_wts[np.int_(np.arange(uncens_nobs)),
                         numcensbelow[uncensored]]
        new_model = ElLinReg(uncens_endog, uncens_exog)
        new_F = np.asarray(new_model.hy_test_beta(value, param_num,
                                fit_weights=wts, print_weights=1, is_censored=1)[2])
        return new_F, wts, new_model.est_vect




koul_data = np.genfromtxt('/home/justin/rverify.csv', delimiter=';')
# ^ Change path to where file is located.
koul_y = np.log10(koul_data[:, 0])
#koul_x = sm.add_constant(koul_data[:, 2], prepend=2)
koul_x = koul_data[:,2]
koul_censors = koul_data[:, 1]
koul_model = emplikeAFT(koul_y, koul_x, koul_censors,).fit(km_est = 'li', wt_method='stute', last_uncensored=1)
koul_censors[14] =1
newky = koul_y[koul_censors==1]
newkx = koul_x[koul_censors==1]




