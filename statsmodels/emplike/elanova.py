class _ANOVAOpt(_OptFuncts):
    """

    Class containing functions that are optimized over when
    conducting ANOVA

    """
    def _opt_common_mu(self, mu):
        empt_array = np.zeros((self.nobs, self.num_groups))
        obs_num = 0
        for arr_num in range(len(self.data)):
            new_obs_num = obs_num + len(self.data[arr_num])
            empt_array[obs_num: new_obs_num, arr_num] = self.data[arr_num] - \
              mu
            obs_num = new_obs_num
        est_vect = empt_array
        wts = np.ones(est_vect.shape[0]) * (1. / (est_vect.shape[0]))
        eta_star = self._modif_newton(np.zeros(self.num_groups), est_vect, wts)
        self.eta_star = eta_star
        denom = 1. + np.dot(eta_star, est_vect.T)
        self.new_weights = 1. / self.nobs * 1. / denom
        llr = np.sum(np.log(self.nobs * self.new_weights))
        return -2 * llr


class ANOVA(_ANOVAOpt):
    """

    A class for ANOVA and comparing means.

    Parameters
    ---------

    data: list of arrays
        data should be a list containing 1 dimensional arrays.  Each array
        is the data collected from a certain group.


    """

    def __init__(self, data):
        self.data = data
        self.num_groups = len(self.data)
        self.nobs = 0
        for i in self.data:
            self.nobs = self.nobs + len(i)

    def compute_ANOVA(self, mu=None, mu_start=0, print_weights=0):

        """
        Returns -2 log likelihood, the pvalue and the maximum likelihood
        estimate for a common mean.

        Parameters
        ----------

        mu: float, optional
            If a mu is specified, ANOVA is conducted with mu as the
            common mean.  Otherwise, the common mean is the maximum
            empirical likelihood estimate of the common mean.
            Default is None.

        mu_start: float, optional
            Starting value for commean mean if specific mu is not specified.
            Default = 0

        print_weights: bool, optional
            if TRUE, returns the weights on observations that maximize the
            likelihood.  Default is FALSE

        Returns
        -------

        res: tuple
            The p-vale, log-likelihood and estimate for the common mean.

        """

        if mu is not None:
            llr = self._opt_common_mu(mu)
            pval = 1 - chi2.cdf(llr, self.num_groups - 1)
            if print_weights:
                return llr, pval, mu, self.new_weights
            else:
                return llr, pval, mu
        else:
            res = optimize.fmin_powell(self._opt_common_mu, mu_start,
                                       full_output=1)
            llr = res[1]
            mu_common = res[0]
            pval = 1 - chi2.cdf(llr, self.num_groups - 1)
            if print_weights:
                return llr, pval, mu_common, self.new_weights
            else:
                return llr, pval, mu_common
