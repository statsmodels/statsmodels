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


Li, G and Wang, Q.(2003). "Empirical Likelihood Regression Analysis for Right
Censored Data." Statistica Sinica 13. 51-68.

Qin, G and Jing, B. (2001). "Empirical Likelihood for Censored Linear
Regression." Scandanavian Journal of Statistics. Vol 28. Iss. 4. 661-673.
"""

import numpy as np
from statsmodels.api import OLS, WLS


class emplikeAFT(object):
    def __init__(self, endog, exog, censors):
        self.nobs = float(np.shape(exog)[0])
        self.endog = endog.reshape(self.nobs, 1)
        self.exog = exog
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
        else:
            self.modif_censors[-1] = self.censors[-1]
        wts = self._make_km(self.endog, self.modif_censors)
        if wt_method == 'koul':
            res = OLS(self.endog * wts, self.exog).fit()
        if wt_method == 'stute':
            res = WLS(self.endog, self.exog, wts).fit()
            res.bse = None      # Inference will be done with EL.
        res.conf_int = None
        res.t = None
        res.pvalues = None
        res.f_test = None
        res.fvalue = None
        return res
