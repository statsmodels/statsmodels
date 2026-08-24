"""
This script contains empirical likelihood ANOVA

Currently the script only contains one feature that allows the user to compare
means of multiple groups.

References
----------

Owen, A. B. (2001). Empirical Likelihood. Chapman and Hall.
"""

from typing import NamedTuple
import warnings

import numpy as np
from scipy import optimize
from scipy.stats import chi2

from statsmodels.tools.validation import bool_like

from .descriptive import _OptFuncts


class ANOVAResult(NamedTuple):
    """
    Result of :meth:`ANOVA.compute_ANOVA`.

    Parameters
    ----------
    statistic : float
        -2 times the log-likelihood ratio, the test statistic.
    pvalue : float
        The p-value of the test statistic.
    mu : float
        The common mean, either the value supplied by the caller or the
        maximum empirical likelihood estimate.
    weights : ndarray or None
        The observation weights that maximize the likelihood.
    """

    statistic: float
    pvalue: float
    mu: float
    weights: np.ndarray | None


class _ANOVAOpt(_OptFuncts):
    """
    Class containing functions that are optimized over when
    conducting ANOVA
    """

    def _opt_common_mu(self, mu):
        """
        Optimizes the likelihood under the null hypothesis that all groups have
        mean mu

        Parameters
        ----------
        mu : float
            The common mean.

        Returns
        -------
        llr : float
            -2 times the llr ratio, which is the test statistic.
        """
        nobs = self.nobs
        endog = self.endog
        num_groups = self.num_groups
        endog_asarray = np.zeros((nobs, num_groups))
        obs_num = 0
        for arr_num in range(len(endog)):
            new_obs_num = obs_num + len(endog[arr_num])
            endog_asarray[obs_num:new_obs_num, arr_num] = endog[arr_num] - mu
            obs_num = new_obs_num
        est_vect = endog_asarray
        wts = np.ones(est_vect.shape[0]) * (1.0 / (est_vect.shape[0]))
        eta_star = self._modif_newton(np.zeros(num_groups), est_vect, wts)
        denom = 1.0 + np.dot(eta_star, est_vect.T)
        self.new_weights = 1.0 / nobs * 1.0 / denom
        llr = np.sum(np.log(nobs * self.new_weights))
        return -2 * llr


class ANOVA(_ANOVAOpt):
    """
    A class for ANOVA and comparing means

    Parameters
    ----------
    endog : list of array_like
        endog should be a list containing 1 dimensional arrays.  Each array
        is the data collected from a certain group.

    Attributes
    ----------
    endog : list of array_like
        The data passed by the caller.
    num_groups : int
        The number of groups.
    nobs : int
        The total number of observations across all groups.
    """

    def __init__(self, endog):
        self.endog = endog
        self.num_groups = len(self.endog)
        self.nobs = 0
        for i in self.endog:
            self.nobs = self.nobs + len(i)

    def compute_ANOVA(
        self, mu=None, mu_start=0, return_weights=False, *, result_object=None
    ):
        """
        Returns -2 log likelihood, the pvalue and the maximum likelihood
        estimate for a common mean

        Parameters
        ----------
        mu : float, optional
            If a mu is specified, ANOVA is conducted with mu as the
            common mean.  Otherwise, the common mean is the maximum
            empirical likelihood estimate of the common mean.
            Default is None.
        mu_start : float, optional
            Starting value for common mean if specific mu is not specified.
            Default is 0.
        return_weights : bool, optional
            if True, returns the weights on observations that maximize the
            likelihood.  Default is False.
        result_object : bool, optional
            Flag indicating whether to return the results as an
            ``ANOVAResult`` NamedTuple instead of a plain tuple. When
            ``return_weights=True`` the NamedTuple holds the same four
            elements as the legacy tuple, so it unpacks identically and is
            always returned, with no warning. When ``return_weights=False``
            the legacy three-element tuple is returned by default and a
            ``FutureWarning`` is issued.

            .. deprecated:: 0.15.0

                In release 0.16.0 or after July 2027, whichever is later, the
                default will change to always return an ``ANOVAResult``. Set
                ``result_object=True`` to opt in now, or
                ``result_object=False`` to silence the warning and keep the
                current return type.

        Returns
        -------
        ANOVAResult or tuple
            If ``result_object=True`` or ``return_weights=True``, a
            NamedTuple with fields ``statistic``, ``pvalue``, ``mu`` and
            ``weights``. See
            :class:`~statsmodels.emplike.elanova.ANOVAResult`.

            Otherwise (the deprecated default), the plain
            ``(llr, pvalue, mu)`` tuple.
        """
        result_object = bool_like(result_object, "result_object", optional=True)
        if mu is not None:
            llr = self._opt_common_mu(mu)
            mu_common = mu
        else:
            res = optimize.fmin_powell(
                self._opt_common_mu, mu_start, full_output=1, disp=False
            )
            llr = res[1]
            mu_common = float(np.squeeze(res[0]))
        pval = 1 - chi2.cdf(llr, self.num_groups - 1)

        if result_object is None and not return_weights:
            warnings.warn(
                "ANOVA.compute_ANOVA currently returns a plain tuple whose "
                "length depends on the return_weights argument. In release "
                "0.16.0 or after July 2027, whichever is later, the default "
                "behavior will switch to always returning an ANOVAResult "
                "NamedTuple. Set result_object=True to switch now, or "
                "result_object=False to keep the current behavior and "
                "silence this warning.",
                FutureWarning,
                stacklevel=2,
            )
        if result_object or return_weights:
            return ANOVAResult(llr, pval, mu_common, self.new_weights)
        return llr, pval, mu_common
