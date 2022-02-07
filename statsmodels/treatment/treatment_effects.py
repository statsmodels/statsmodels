# -*- coding: utf-8 -*-
"""Treatment effect estimators

follows largely Stata's teffects in Stata 13 manual

Created on Tue Jun  9 22:45:23 2015

Author: Josef Perktold
License: BSD-3

currently available

                     ATE        POM_0        POM_1
res_ipw       230.688598  3172.774059  3403.462658
res_aipw     -230.989201  3403.355253  3172.366052
res_aipw_wls -227.195618  3403.250651  3176.055033
res_ra       -239.639211  3403.242272  3163.603060
res_ipwra    -229.967078  3403.335639  3173.368561


Lots of todos, just the beginning, but most effects are available but not
standard errors, and no code structure that has a useful pattern

see https://github.com/statsmodels/statsmodels/issues/2443

Note: script requires cattaneo2 data file from Stata 14, hardcoded file path
could be loaded with webuse

"""

import numpy as np
from scipy.linalg import block_diag
import statsmodels.api as sm

from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.contrast import ContrastResults


def mom_ate(te, endog, tind, prob, weighted=True):
    """moment condition for average treatment effect

    This does not include a moment condition for potential outcome mean (POM).

    """
    if weighted:
        w1 = (tind / prob)
        w2 = (1. - tind) / (1. - prob)
        wdiff = w1 / w1.mean() - w2 / w2.mean()
        # wdiff = w1 / w1.sum() - w2 / w2.sum()
    else:
        wdiff = (tind / prob) - (1 - tind) / (1 - prob)

    return endog * wdiff - te


def mom_atm(tm, endog, tind, prob, weighted=True):
    """moment conditions for average treatment means (POM)

    """
    w1 = (tind / prob)
    w0 = (1. - tind) / (1. - prob)
    if weighted:
        w1 /= w1.mean()
        w0 /= w0.mean()

    return np.column_stack((endog * w0 - tm[0], endog * w1 - tm[1]))


def mom_ols(tm, endog, tind, prob, weighted=True):
    """
    moment condition for average treatment mean based on OLS dummy regression

    """
    w = tind / prob + (1-tind) / (1 - prob)

    treat_ind = np.column_stack((1 - tind, tind))
    mom = (w * (endog - treat_ind.dot(tm)))[:, None] * treat_ind

    return mom


def mom_ols_te(tm, endog, tind, prob, weighted=True):
    """
    moment condition for average treatment mean based on OLS dummy regression

    first moment is ATE
    second moment is POM0  (control)

    """
    w = tind / prob + (1-tind) / (1 - prob)

    treat_ind = np.column_stack((tind, np.ones(len(tind))))
    mom = (w * (endog - treat_ind.dot(tm)))[:, None] * treat_ind

    return mom


def mom_olsex(params, model=None, exog=None, scale=None):
    exog = exog if exog is not None else model.exog
    fitted = model.predict(params, exog)
    resid = model.endog - fitted
    if scale is not None:
        resid /= scale
    mom = resid[:, None] * exog
    return mom


def ate_ipw(endog, tind, prob, weighted=True):
    """average treatment effect based on basic inverse propensity weighting.

    """
    w1 = (tind / prob)
    w0 = (1. - tind) / (1. - prob)
    if weighted:
        w0 /= w0.mean()
        w1 /= w1.mean()

    wdiff = w1 - w0

    return (endog * wdiff).mean(), (endog * w0).mean(), (endog * w1).mean()


class TEGMMGeneric1(GMM):
    """GMM class to get cov_params for treatment effects

    This combines moment conditions for the selection/treatment model and the
    outcome model to get the standard errors for the treatment effect that
    takes the first step estimation of the treatment model into account.

    this also matches standard errors of ATE and POM in Stata

    """

    def __init__(self, endog, res_select, mom_outcome, exclude_tmoms=False,
                 **kwargs):
        super(TEGMMGeneric1, self).__init__(endog, None, None)
        self.results_select = res_select
        self.mom_outcome = mom_outcome
        self.exclude_tmoms = exclude_tmoms
        self.__dict__.update(kwargs)

        # add xnames so it's not None
        # we don't have exog in init in this version
        if self.data.xnames is None:
            self.data.xnames = []

        # need information about decomposition of parameters
        if exclude_tmoms:
            self.k_select = 0
        else:
            self.k_select = len(res_select.model.data.param_names)

        if exclude_tmoms:
            # fittedvalues is still linpred
            self.prob = self.results_select.predict()
        else:
            self.prob = None

    def momcond(self, params):
        k_outcome = len(params) - self.k_select
        tm = params[:k_outcome]
        p_tm = params[k_outcome:]

        tind = self.results_select.model.endog

        if self.exclude_tmoms:
            prob = self.prob
        else:
            prob = self.results_select.model.predict(p_tm)

        moms_list = []
        mom_o = self.mom_outcome(tm, self.endog, tind, prob, weighted=True)
        moms_list.append(mom_o)

        if not self.exclude_tmoms:
            mom_t = self.results_select.model.score_obs(p_tm)
            moms_list.append(mom_t)

        moms = np.column_stack(moms_list)
        return moms

    def momcond_aipw(self, params):
        k_outcome = len(params) - self.k_select
        tm = params[:k_outcome]
        p_tm = params[k_outcome:]

        tind = self.results_select.model.endog
        prob = self.results_select.model.predict(p_tm)

        momt = self.mom_outcome(tm, self.endog, tind, prob, weighted=True)
        moms = np.column_stack((momt,
                                self.results_select.model.score_obs(p_tm)))
        return moms


class TEGMM(GMM):
    """GMM class to get cov_params for treatment effects

    This combines moment conditions for the selection/treatment model and the
    outcome model to get the standard errors for the treatment effect that
    takes the first step estimation of the treatment model into account.

    this also matches standard errors of ATE and POM in Stata

    """

    def __init__(self, endog, res_select, mom_outcome):
        super(TEGMM, self).__init__(endog, None, None)
        self.results_select = res_select
        self.mom_outcome = mom_outcome

        # add xnames so it's not None
        # we don't have exog in init in this version
        if self.data.xnames is None:
            self.data.xnames = []

    def momcond(self, params):
        tm = params[:2]
        p_tm = params[2:]

        tind = self.results_select.model.endog
        prob = self.results_select.model.predict(p_tm)
        momt = self.mom_outcome(tm, self.endog, tind, prob)  # weighted=True)
        moms = np.column_stack((momt,
                                self.results_select.model.score_obs(p_tm)))
        return moms


class _AIPWGMM(TEGMMGeneric1):
    """ GMM for aipw treatment effect and potential outcome

    uses unweighted outcome regression
    """

    def momcond(self, params):
        ra = self.teff
        treat_mask = ra.treat_mask
        res_select = ra.results_select

        add_pom0 = True
        if add_pom0:
            ppom = params[1]
            mask = np.arange(len(params)) != 1
            params = params[mask]

        k = ra.results0.model.exog.shape[1]
        pm = params[0]  # ATE parameter
        p0 = params[1:k+1]
        p1 = params[k+1:2*k+1]
        ps = params[2*k+1:]
        mod0 = ra.results0.model
        mod1 = ra.results1.model
        # reorder exog so it matches sub models by group
        exog = np.concatenate((mod0.exog, mod1.exog), axis=0)
        endog = np.concatenate((mod0.endog, mod1.endog), axis=0)

        # todo: need weights in outcome models
        prob_sel = np.asarray(res_select.model.predict(ps))

        prob_sel = np.clip(prob_sel, 0.01, 0.99)

        prob0 = prob_sel[~treat_mask]
        prob1 = prob_sel[treat_mask]
        prob = np.concatenate((prob0, prob1))

        # outcome models by treatment unweighted
        fitted0 = mod0.predict(p0, exog)
        mom0 = mom_olsex(p0, model=mod0)

        fitted1 = mod1.predict(p1, exog)
        mom1 = mom_olsex(p1, model=mod1)

        mom_outcome = block_diag(mom0, mom1)

        # moments for target statistics, ATE and POM
        tind = ra.treatment
        tind = np.concatenate((tind[~treat_mask], tind[treat_mask]))
        correct0 = (endog - fitted0) / (1 - prob) * (1 - tind)
        correct1 = (endog - fitted1) / prob * tind

        tmean0 = fitted0 + correct0
        tmean1 = fitted1 + correct1
        ate = tmean1 - tmean0

        mm = ate - pm
        # mf = np.concatenate((fitted0, fitted1)) - pm
        if add_pom0:
            mpom = tmean0 - ppom
            mm = np.column_stack((mm, mpom))

        # Note: res_select has original data order,
        # mom_outcome and mm use grouped observations
        mom_select = res_select.model.score_obs(ps)
        mom_select = np.concatenate((mom_select[~treat_mask],
                                     mom_select[treat_mask]), axis=0)

        moms = np.column_stack((mm, mom_outcome, mom_select))
        return moms


class _AIPWWLSGMM(TEGMMGeneric1):
    """ GMM for aipw-wls treatment effect and potential outcome

    uses weighted outcome regression
    """

    def momcond(self, params):
        ra = self.teff
        treat_mask = ra.treat_mask
        res_select = ra.results_select

        add_pom0 = True
        if add_pom0:
            ppom = params[1]
            mask = np.arange(len(params)) != 1
            params = params[mask]

        k = ra.results0.model.exog.shape[1]
        pm = params[0]  # ATE parameter
        p0 = params[1:k+1]
        p1 = params[k+1:2*k+1]
        ps = params[-6:]
        mod0 = ra.results0.model
        mod1 = ra.results1.model
        # reorder exog so it matches sub models by group
        exog = np.concatenate((mod0.exog, mod1.exog), axis=0)
        endog = np.concatenate((mod0.endog, mod1.endog), axis=0)

        # todo: need weights in outcome models
        prob_sel = np.asarray(res_select.model.predict(ps))

        prob_sel = np.clip(prob_sel, 0.001, 0.999)

        prob0 = prob_sel[~treat_mask]
        prob1 = prob_sel[treat_mask]
        prob = np.concatenate((prob0, prob1))

        tind = 0
        ww0 = (1 - tind) / (1 - prob0) * ((1 - tind) / (1 - prob0) - 1)
        tind = 1
        ww1 = tind / prob1 * (tind / prob1 - 1)

        # outcome models by treatment using IPW weights
        fitted0 = mod0.predict(p0, exog)
        mom0 = mom_olsex(p0, model=mod0) * ww0[:, None]

        fitted1 = mod1.predict(p1, exog)
        mom1 = mom_olsex(p1, model=mod1) * ww1[:, None]

        mom_outcome = block_diag(mom0, mom1)

        # moments for target statistics, ATE and POM
        tind = ra.treatment
        tind = np.concatenate((tind[~treat_mask], tind[treat_mask]))

        correct0 = (endog - fitted0) / (1 - prob) * (1 - tind)
        correct1 = (endog - fitted1) / prob * tind

        tmean0 = fitted0 + correct0
        tmean1 = fitted1 + correct1
        ate = tmean1 - tmean0

        mm = ate - pm
        # mf = np.concatenate((fitted0, fitted1)) - pm
        if add_pom0:
            mpom = tmean0 - ppom
            mm = np.column_stack((mm, mpom))

        # Note: res_select has original data order,
        # mom_outcome and mm use grouped observations
        mom_select = res_select.model.score_obs(ps)
        mom_select = np.concatenate((mom_select[~treat_mask],
                                     mom_select[treat_mask]), axis=0)

        moms = np.column_stack((mm, mom_outcome, mom_select))
        return moms


class _RAGMM(TEGMMGeneric1):
    """GMM for regression adjustment treatment effect and potential outcome

    uses unweighted outcome regression
    """

    def momcond(self, params):
        ra = self.teff

        add_pom0 = True
        if add_pom0:
            ppom = params[1]
            mask = np.arange(len(params)) != 1
            params = params[mask]

        k = ra.results0.model.exog.shape[1]
        pm = params[0]
        p0 = params[1:k+1]
        p1 = params[-k:]
        mod0 = ra.results0.model
        mod1 = ra.results1.model
        exog = np.concatenate((mod0.exog, mod1.exog), axis=0)

        fitted0 = mod0.predict(p0, exog)
        mom0 = mom_olsex(p0, model=mod0)

        fitted1 = mod1.predict(p1, exog)
        mom1 = mom_olsex(p1, model=mod1)

        momout = block_diag(mom0, mom1)

        mm = fitted1 - fitted0 - pm
        # mf = np.concatenate((fitted0, fitted1)) - pm
        if add_pom0:
            mpom = fitted0 - ppom
            mm = np.column_stack((mm, mpom))

        moms = np.column_stack((mm, momout))
        return moms


class _IPWRAGMM(TEGMMGeneric1):
    """ GMM for ipwra treatment effect and potential outcome
    """

    def momcond(self, params):
        ra = self.teff
        treat_mask = ra.treat_mask
        res_select = ra.results_select

        add_pom0 = True
        if add_pom0:
            ppom = params[1]
            mask = np.arange(len(params)) != 1
            params = params[mask]

        k = ra.results0.model.exog.shape[1]
        pm = params[0]  # ATE parameter
        p0 = params[1:k+1]
        p1 = params[k+1:2*k+1]
        ps = params[-6:]
        mod0 = ra.results0.model
        mod1 = ra.results1.model
        # reorder exog so it matches sub models by group
        exog = np.concatenate((mod0.exog, mod1.exog), axis=0)

        # todo: need weights in outcome models
        prob_sel = np.asarray(res_select.model.predict(ps))
        prob_sel = np.clip(prob_sel, 0.001, 0.999)
        prob0 = prob_sel[~treat_mask]
        prob1 = prob_sel[treat_mask]

        # outcome models by treatment using IPW weights
        fitted0 = mod0.predict(p0, exog)
        mom0 = mom_olsex(p0, model=mod0) / (1 - prob0[:, None])

        fitted1 = mod1.predict(p1, exog)
        mom1 = mom_olsex(p1, model=mod1) / prob1[:, None]

        mom_outcome = block_diag(mom0, mom1)

        # moments for target statistics, ATE and POM
        mm = fitted1 - fitted0 - pm
        # mf = np.concatenate((fitted0, fitted1)) - pm
        if add_pom0:
            mpom = fitted0 - ppom
            mm = np.column_stack((mm, mpom))

        # Note: res_select has original data order,
        # mom_outcome and mm use grouped observations
        mom_select = res_select.model.score_obs(ps)
        mom_select = np.concatenate((mom_select[~treat_mask],
                                     mom_select[treat_mask]), axis=0)

        moms = np.column_stack((mm, mom_outcome, mom_select))
        return moms


class TreatmentEffectResults(ContrastResults):

    def __init__(self, teff, results_gmm, method, **kwds):

        k_params = len(results_gmm.params)
        constraints = np.zeros((3, k_params))
        constraints[0, 0] = 1
        constraints[1, 1] = 1
        constraints[2, :2] = [1, 1]
        tt = results_gmm.t_test(constraints)
        self.__dict__.update(tt.__dict__)
        self.teff = teff
        self.results_gmm = results_gmm
        self.method = method
        # TODO: make those explicit?
        self.__dict__.update(kwds)

        self.c_names = ["ATE", "POM0", "POM1"]


class TreatmentEffect(object):
    """Estimate average treatment effect under conditional independence

    This class estimates treatment effect and potential outcome using 5
    different methods, ipw, ra, aipw, aipw-wls, ipw-ra.
    Standard errors and inference are based on the joint GMM representation of
    selection or treatment model, outcome model and effect functions.

    Parameters
    ----------
    model : instance of a model class
        The model class should contain endog and exog for the full model.
    treatment : ndarray
        indicator array for observations with treatment (1) or without (0)
    results_select : results instance
        The results instance for the treatment or selection model.
    _cov_type : "HC0"
        Internal keyword. The keyword oes not affect GMMResults which always
        corresponds to HC0 standard errors.
    kwds : keyword arguments
        currently not used

    Notes
    -----
    The outcome model is currently limited to a linear model based on OLS or
    WLS.
    Other outcome models, like Logit and Poisson, will become available in
    future.

    """

    def __init__(self, model, treatment, results_select=None, _cov_type="HC0",
                 **kwds):
        # Note _cov_type is only for preliminary estimators,
        # cov in GMM alwasy corresponds to HC0
        self.__dict__.update(kwds)  # currently not used
        self.treatment = np.asarray(treatment)
        self.treat_mask = treat_mask = (treatment == 1)

        if results_select is not None:
            self.results_select = results_select
            self.prob_select = results_select.predict()

        self.model_pool = model
        endog = model.endog
        exog = model.exog
        self.nobs = endog.shape[0]
        self._cov_type = _cov_type

        results_select
        # no init keys are supported
        mod0 = model.__class__(endog[~treat_mask], exog[~treat_mask])
        self.results0 = mod0.fit(cov_type='HC0')
        mod1 = model.__class__(endog[treat_mask], exog[treat_mask])
        self.results1 = mod1.fit(cov_type='HC0')
        self.predict_mean0 = self.model_pool.predict(self.results0.params
                                                     ).mean()
        self.predict_mean1 = self.model_pool.predict(self.results1.params
                                                     ).mean()

        # this only works for linear model, need margins for discrete
        exog_mean = exog.mean(0)
        self.tt0 = self.results0.t_test(exog_mean)
        self.tt1 = self.results1.t_test(exog_mean)
        self.ate = self.tt1.effect - self.tt0.effect
        self.se_ate = np.sqrt(self.tt1.sd**2 + self.tt0.sd**2)

    @classmethod
    def from_data(cls, endog, exog, treatment, model='ols', **kwds):
        """create models from data

        not yet implemented

        """
        raise NotImplementedError

    def ipw(self, return_results=True, disp=False):
        endog = self.model_pool.endog
        tind = self.treatment
        prob = self.prob_select
        res_ipw = ate_ipw(endog, tind, prob, weighted=True)

        if not return_results:
            return res_ipw

        gmm = TEGMMGeneric1(endog, self.results_select, mom_ols_te)
        start_params = np.concatenate((res_ipw[:2],
                                       self.results_select.params))
        res_gmm = gmm.fit(start_params=start_params, optim_method='nm',
                          inv_weights=np.eye(len(start_params)), maxiter=1,
                          optim_args={"maxiter": 5000, "disp": disp},
                          )

        res = TreatmentEffectResults(self, res_gmm, "IPW",
                                     start_params=start_params)
        return res

    def ra(self, return_results=True, disp=False):
        """
        ATE and POM from regression adjustment
        """
        if not return_results:
            # Note: currently 1-d arrays, get scalar
            return self.ate[0], self.tt0.effect[0], self.tt1.effect[0]

        endog = self.model_pool.endog
        mod_gmm = _RAGMM(endog, self.results_select, mom_ols_te, teff=self)
        start_params = np.concatenate((
            self.ate, self.tt0.effect,
            self.results0.params,
            self.results1.params))
        res_gmm = mod_gmm.fit(start_params=start_params,
                              inv_weights=np.eye(len(start_params)),
                              optim_args={"maxiter": 5000, "disp": disp},
                              )
        res = TreatmentEffectResults(self, res_gmm, "IPW",
                                     start_params=start_params)
        return res

    def aipw(self, return_results=True, disp=False):
        """ATE and POM from double robust augmented inverse probability weighting

        replicates Stata's `teffects aipw`

        no standard errors yet

        """
        nobs = self.nobs
        prob = self.prob_select
        tind = self.treatment
        correct0 = (self.results0.resid / (1 - prob[tind == 0])).sum() / nobs
        correct1 = (self.results1.resid / (prob[tind == 1])).sum() / nobs
        tmean0 = self.tt0.effect + correct0
        tmean1 = self.tt1.effect + correct1
        ate = tmean1 - tmean0
        if not return_results:
            # Note: currently 1-d arrays, get scalar
            return ate[0], tmean0[0], tmean1[0]

        endog = self.model_pool.endog
        p2_aipw = np.asarray([ate, tmean0]).squeeze()

        mag_aipw1 = _AIPWGMM(endog, self.results_select, mom_ols_te, teff=self)
        start_params = np.concatenate((
            p2_aipw,
            self.results0.params, self.results1.params,
            self.results_select.params))
        res_gmm = mag_aipw1.fit(
            start_params=start_params,
            inv_weights=np.eye(len(start_params)),
            optim_method='nm',
            optim_args={"maxiter": 5000, "disp": disp},
            maxiter=1)

        res = TreatmentEffectResults(self, res_gmm, "IPW",
                                     start_params=start_params)
        return res

    def aipw_wls(self, return_results=True, disp=False):
        """double robust augmented inverse probability weighting

        replicates Stata's `teffects aipw`

        no standard errors yet

        """
        nobs = self.nobs
        prob = self.prob_select

        endog = self.model_pool.endog
        exog = self.model_pool.exog
        tind = self.treatment
        treat_mask = self.treat_mask

        ww1 = tind / prob * (tind / prob - 1)
        mod1 = sm.WLS(endog[treat_mask], exog[treat_mask],
                      weights=ww1[treat_mask])
        result1 = mod1.fit(cov_type='HC1')
        mean1_ipw2 = result1.predict(exog).mean()

        ww0 = (1 - tind) / (1 - prob) * ((1 - tind) / (1 - prob) - 1)
        mod0 = sm.WLS(endog[~treat_mask], exog[~treat_mask],
                      weights=ww0[~treat_mask])
        result0 = mod0.fit(cov_type='HC1')
        mean0_ipw2 = result0.predict(exog).mean()

        self.results_ipwwls0 = result0
        self.results_ipwwls1 = result1

        correct0 = (result0.resid / (1 - prob[tind == 0])).sum() / nobs
        correct1 = (result1.resid / (prob[tind == 1])).sum() / nobs
        tmean0 = mean0_ipw2 + correct0
        tmean1 = mean1_ipw2 + correct1
        ate = tmean1 - tmean0
        if not return_results:
            return ate, tmean0, tmean1

        p2_aipw_wls = np.asarray([ate, tmean0]).squeeze()

        # GMM
        mod_gmm = _AIPWWLSGMM(endog, self.results_select, mom_ols_te,
                              teff=self)
        start_params = np.concatenate((
            p2_aipw_wls,
            result0.params,
            result1.params,
            self.results_select.params))
        res_gmm = mod_gmm.fit(
            start_params=start_params,
            inv_weights=np.eye(len(start_params)),
            optim_method='nm',
            optim_args={"maxiter": 5000, "disp": disp},
            maxiter=0)
        res = TreatmentEffectResults(self, res_gmm, "IPW",
                                     start_params=start_params)
        return res

    def ipw_ra(self, return_results=True, disp=False):
        """ATE and POM for ip-weighted regression adjustment

        """
        treat_mask = self.treat_mask
        endog = self.model_pool.endog
        exog = self.model_pool.exog
        prob = self.prob_select

        mod0 = sm.WLS(endog[~treat_mask], exog[~treat_mask],
                      weights=1/(1 - prob[~treat_mask]))
        result0 = mod0.fit(cov_type='HC1')

        mean0_ipwra = result0.predict(exog).mean()
        mod1 = sm.WLS(endog[treat_mask], exog[treat_mask],
                      weights=1/prob[treat_mask])
        result1 = mod1.fit(cov_type='HC1')
        mean1_ipwra = result1.predict(exog).mean()

        if not return_results:
            return mean1_ipwra - mean0_ipwra, mean0_ipwra, mean1_ipwra

        # GMM
        mod_gmm = _IPWRAGMM(endog, self.results_select, mom_ols_te, teff=self)
        start_params = np.concatenate((
            [mean1_ipwra - mean0_ipwra, mean0_ipwra],
            result0.params,
            result1.params,
            np.asarray(self.results_select.params)
            ))
        res_gmm = mod_gmm.fit(
            start_params=start_params,
            inv_weights=np.eye(len(start_params)),
            optim_method='nm',
            optim_args={"maxiter": 2000, "disp": disp},
            maxiter=1
            )

        res = TreatmentEffectResults(self, res_gmm, "IPW",
                                     start_params=start_params)
        return res
