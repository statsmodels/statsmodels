"""
Treatment effect estimators

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

from statsmodels.compat.pandas import Substitution

import numpy as np
import pandas as pd
from scipy.linalg import block_diag

from statsmodels.regression.linear_model import WLS
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.docstring import indent
from statsmodels.tools.validation import array_like


def _mom_ate(params, endog, tind, prob, weighted=True):
    """
    Moment condition for average treatment effect

    This does not include a moment condition for potential outcome mean (POM).

    Parameters
    ----------
    params : ndarray
        Parameter at which the moment condition is evaluated, the average
        treatment effect.
    endog : ndarray
        Outcome variable.
    tind : ndarray
        Treatment indicator, 1 for treated and 0 for untreated observations.
    prob : ndarray
        Estimated propensity score, probability of treatment.
    weighted : bool, optional
        If True, weights are normalized so that each of the treated and
        untreated weights average to one.

    Returns
    -------
    ndarray
        Moment condition, evaluated at each observation.
    """
    w1 = (tind / prob)
    w0 = (1. - tind) / (1. - prob)
    if weighted:
        w0 /= w0.mean()
        w1 /= w1.mean()

    wdiff = w1 - w0

    return endog * wdiff - params


def _mom_atm(params, endog, tind, prob, weighted=True):
    """
    Moment conditions for average treatment means (POM)

    Moment conditions are POM0 and POM1.

    Parameters
    ----------
    params : ndarray
        Parameters, POM0 and POM1, at which the moment conditions are
        evaluated.
    endog : ndarray
        Outcome variable.
    tind : ndarray
        Treatment indicator, 1 for treated and 0 for untreated observations.
    prob : ndarray
        Estimated propensity score, probability of treatment.
    weighted : bool, optional
        If True, weights are normalized so that each of the treated and
        untreated weights average to one.

    Returns
    -------
    ndarray
        Moment conditions, evaluated at each observation, 2 columns for
        POM0 and POM1.
    """
    w1 = (tind / prob)
    w0 = (1. - tind) / (1. - prob)
    if weighted:
        w1 /= w1.mean()
        w0 /= w0.mean()

    return np.column_stack((endog * w0 - params[0], endog * w1 - params[1]))


def _mom_ols(params, endog, tind, prob, weighted=True):
    """
    Moment condition for average treatment mean based on OLS dummy regression

    Moment conditions are POM0 and POM1.

    Parameters
    ----------
    params : ndarray
        Parameters, POM0 and POM1, at which the moment conditions are
        evaluated.
    endog : ndarray
        Outcome variable.
    tind : ndarray
        Treatment indicator, 1 for treated and 0 for untreated observations.
    prob : ndarray
        Estimated propensity score, probability of treatment.
    weighted : bool, optional
        Not used. Kept for signature compatibility with other moment
        condition functions.

    Returns
    -------
    ndarray
        Moment conditions, evaluated at each observation, 2 columns for
        POM0 and POM1.
    """
    w = tind / prob + (1-tind) / (1 - prob)

    treat_ind = np.column_stack((1 - tind, tind))
    mom = (w * (endog - treat_ind.dot(params)))[:, None] * treat_ind

    return mom


def _mom_ols_te(tm, endog, tind, prob, weighted=True):
    """
    Moment condition for average treatment mean based on OLS dummy regression

    First moment is ATE, second moment is POM0 (control).

    Parameters
    ----------
    tm : ndarray
        Parameters, ATE and POM0, at which the moment conditions are
        evaluated.
    endog : ndarray
        Outcome variable.
    tind : ndarray
        Treatment indicator, 1 for treated and 0 for untreated observations.
    prob : ndarray
        Estimated propensity score, probability of treatment.
    weighted : bool, optional
        Not used. Kept for signature compatibility with other moment
        condition functions.

    Returns
    -------
    ndarray
        Moment conditions, evaluated at each observation, 2 columns for
        ATE and POM0.
    """
    w = tind / prob + (1-tind) / (1 - prob)

    treat_ind = np.column_stack((tind, np.ones(len(tind))))
    mom = (w * (endog - treat_ind.dot(tm)))[:, None] * treat_ind

    return mom


def _mom_olsex(params, model=None, exog=None, scale=None):
    """
    Moment condition for OLS with an optionally external exog and scale

    Parameters
    ----------
    params : ndarray
        Parameters of the outcome model.
    model : instance of a model class
        Model with ``endog`` and, if `exog` is None, ``exog`` attributes,
        and a ``predict`` method.
    exog : ndarray, optional
        Explanatory variables used to compute fitted values. If None, then
        ``model.exog`` is used instead.
    scale : float, optional
        If provided, the residual based moment condition is divided by
        this scale.

    Returns
    -------
    ndarray
        Moment condition, evaluated at each observation, weighted by exog.
    """
    exog = exog if exog is not None else model.exog
    fitted = model.predict(params, exog)
    resid = model.endog - fitted
    if scale is not None:
        resid /= scale
    mom = resid[:, None] * exog
    return mom


def ate_ipw(endog, tind, prob, weighted=True, probt=None):
    """
    Average treatment effect based on basic inverse propensity weighting

    Parameters
    ----------
    endog : ndarray
        Outcome variable.
    tind : ndarray
        Treatment indicator, 1 for treated and 0 for untreated observations.
    prob : ndarray
        Estimated propensity score, probability of treatment.
    weighted : bool, optional
        If True, weights are normalized so that each of the treated and
        untreated weights average to one.
    probt : ndarray, optional
        Additional weight, e.g., propensity score, used to target the effect
        on a subgroup such as the treated or untreated population. If None,
        the sample average treatment effect is computed.

    Returns
    -------
    ate : float
        Average treatment effect.
    pom0 : float
        Potential outcome mean for the untreated (control) group.
    pom1 : float
        Potential outcome mean for the treated group.
    """
    w1 = (tind / prob)
    w0 = (1. - tind) / (1. - prob)

    if probt is not None:
        w1 *= probt
        w0 *= probt

    if weighted:
        w0 /= w0.mean()
        w1 /= w1.mean()

    wdiff = w1 - w0

    return (endog * wdiff).mean(), (endog * w0).mean(), (endog * w1).mean()


class _TEGMMGeneric1(GMM):
    """
    GMM class to get cov_params for treatment effects

    This combines moment conditions for the selection/treatment model and the
    outcome model to get the standard errors for the treatment effect that
    takes the first step estimation of the treatment model into account.

    This also matches standard errors of ATE and POM in Stata.

    Parameters
    ----------
    endog : ndarray
        Outcome variable, endog of the outcome model.
    res_select : results instance
        Results instance of the treatment or selection model.
    mom_outcome : callable
        Function that computes the moment conditions of the outcome model.
    exclude_tmoms : bool, optional
        If True, then the moment conditions of the treatment or selection
        model are not included and `params` does not contain the
        parameters of the selection model.
    **kwargs
        Additional keyword arguments that are attached to the instance,
        e.g., `teff` and `effect_group` used by the subclasses.
    """

    def __init__(self, endog, res_select, mom_outcome, exclude_tmoms=False,
                 **kwargs):
        super().__init__(endog, None, None)
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


class _TEGMM(GMM):
    """
    GMM class to get cov_params for treatment effects

    This combines moment conditions for the selection/treatment model and the
    outcome model to get the standard errors for the treatment effect that
    takes the first step estimation of the treatment model into account.

    This also matches standard errors of ATE and POM in Stata.

    Parameters
    ----------
    endog : ndarray
        Outcome variable, endog of the outcome model.
    res_select : results instance
        Results instance of the treatment or selection model.
    mom_outcome : callable
        Function that computes the moment conditions of the outcome model.
    """

    def __init__(self, endog, res_select, mom_outcome):
        super().__init__(endog, None, None)
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


class _IPWGMM(_TEGMMGeneric1):
    """
    GMM for ipw treatment effect and potential outcome

    Notes
    -----
    Uses a single, pooled outcome moment condition weighted by the inverse
    probability weights.
    """

    def momcond(self, params):
        # Note: momcond in original order of observations
        ra = self.teff
        res_select = ra.results_select
        tind = ra.treatment
        endog = ra.model_pool.endog
        effect_group = self.effect_group

        tm = params[:2]
        ps = params[2:]

        prob_sel = np.asarray(res_select.model.predict(ps))
        prob_sel = np.clip(prob_sel, *ra.ps_bounds)
        prob = prob_sel

        if effect_group == "all":
            probt = None
        elif effect_group in [1, "treated"]:
            probt = prob
        elif effect_group in [0, "untreated", "control"]:
            probt = 1 - prob
        elif not isinstance(effect_group, np.ndarray):
            raise ValueError("incorrect option for effect_group")
        # default else probt = probt

        w = tind / prob + (1 - tind) / (1 - prob)
        # Are we supposed to use scaled weights? doesn't cloesely match Stata
        # w1 = tind / prob
        # w2 = (1 - tind) / (1 - prob)
        # w = w1 / w1.sum() * tind.sum() + w2 / w2.sum() * (1 - tind).sum()
        if probt is not None:
            w *= probt

        treat_ind = np.column_stack((tind, np.ones(len(tind))))
        mm = (w * (endog - treat_ind.dot(tm)))[:, None] * treat_ind

        mom_select = res_select.model.score_obs(ps)
        moms = np.column_stack((mm, mom_select))
        return moms


class _AIPWGMM(_TEGMMGeneric1):
    """
    GMM for aipw treatment effect and potential outcome

    Notes
    -----
    Uses unweighted outcome regression.
    """

    def momcond(self, params):
        ra = self.teff
        treat_mask = ra.treat_mask
        res_select = ra.results_select

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
        # use reordered exog, endog so it matches sub models by group
        exog = ra.exog_grouped
        endog = ra.endog_grouped

        prob_sel = np.asarray(res_select.model.predict(ps))
        prob_sel = np.clip(prob_sel, *ra.ps_bounds)

        prob0 = prob_sel[~treat_mask]
        prob1 = prob_sel[treat_mask]
        prob = np.concatenate((prob0, prob1))

        # outcome models by treatment unweighted
        fitted0 = mod0.predict(p0, exog)
        mom0 = _mom_olsex(p0, model=mod0)

        fitted1 = mod1.predict(p1, exog)
        mom1 = _mom_olsex(p1, model=mod1)

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
        mpom = tmean0 - ppom
        mm = np.column_stack((mm, mpom))

        # Note: res_select has original data order,
        # mom_outcome and mm use grouped observations
        mom_select = res_select.model.score_obs(ps)
        mom_select = np.concatenate((mom_select[~treat_mask],
                                     mom_select[treat_mask]), axis=0)

        moms = np.column_stack((mm, mom_outcome, mom_select))
        return moms


class _AIPWWLSGMM(_TEGMMGeneric1):
    """
    GMM for aipw-wls treatment effect and potential outcome

    Notes
    -----
    Uses weighted outcome regression.
    """

    def momcond(self, params):
        ra = self.teff
        treat_mask = ra.treat_mask
        res_select = ra.results_select

        ppom = params[1]
        mask = np.arange(len(params)) != 1
        params = params[mask]

        k = ra.results0.model.exog.shape[1]
        pm = params[0]  # ATE parameter
        p0 = params[1:k+1]
        p1 = params[k+1:2*k+1]
        ps = params[-self.k_select:]
        mod0 = ra.results0.model
        mod1 = ra.results1.model
        # use reordered exog, endog so it matches sub models by group
        exog = ra.exog_grouped
        endog = ra.endog_grouped

        # todo: need weights in outcome models
        prob_sel = np.asarray(res_select.model.predict(ps))

        prob_sel = np.clip(prob_sel, *ra.ps_bounds)

        prob0 = prob_sel[~treat_mask]
        prob1 = prob_sel[treat_mask]
        prob = np.concatenate((prob0, prob1))

        tind = 0
        ww0 = (1 - tind) / (1 - prob0) * ((1 - tind) / (1 - prob0) - 1)
        tind = 1
        ww1 = tind / prob1 * (tind / prob1 - 1)

        # outcome models by treatment using IPW weights
        fitted0 = mod0.predict(p0, exog)
        mom0 = _mom_olsex(p0, model=mod0) * ww0[:, None]

        fitted1 = mod1.predict(p1, exog)
        mom1 = _mom_olsex(p1, model=mod1) * ww1[:, None]

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
        mpom = tmean0 - ppom
        mm = np.column_stack((mm, mpom))

        # Note: res_select has original data order,
        # mom_outcome and mm use grouped observations
        mom_select = res_select.model.score_obs(ps)
        mom_select = np.concatenate((mom_select[~treat_mask],
                                     mom_select[treat_mask]), axis=0)

        moms = np.column_stack((mm, mom_outcome, mom_select))
        return moms


class _RAGMM(_TEGMMGeneric1):
    """
    GMM for regression adjustment treatment effect and potential outcome

    Notes
    -----
    Uses unweighted outcome regression.
    """

    def momcond(self, params):
        ra = self.teff

        ppom = params[1]
        mask = np.arange(len(params)) != 1
        params = params[mask]

        k = ra.results0.model.exog.shape[1]
        pm = params[0]
        p0 = params[1:k+1]
        p1 = params[-k:]
        mod0 = ra.results0.model
        mod1 = ra.results1.model
        # use reordered exog, endog so it matches sub models by group
        exog = ra.exog_grouped

        fitted0 = mod0.predict(p0, exog)
        mom0 = _mom_olsex(p0, model=mod0)

        fitted1 = mod1.predict(p1, exog)
        mom1 = _mom_olsex(p1, model=mod1)

        momout = block_diag(mom0, mom1)

        mm = fitted1 - fitted0 - pm
        mpom = fitted0 - ppom
        mm = np.column_stack((mm, mpom))
        if self.probt is not None:
            mm *= (self.probt / self.probt.mean())[:, None]

        moms = np.column_stack((mm, momout))
        return moms


class _IPWRAGMM(_TEGMMGeneric1):
    """GMM for ipwra treatment effect and potential outcome"""

    def momcond(self, params):
        ra = self.teff
        treat_mask = ra.treat_mask
        res_select = ra.results_select

        ppom = params[1]
        mask = np.arange(len(params)) != 1
        params = params[mask]

        k = ra.results0.model.exog.shape[1]
        pm = params[0]  # ATE parameter
        p0 = params[1:k+1]
        p1 = params[k+1:2*k+1]
        ps = params[-self.k_select:]
        mod0 = ra.results0.model
        mod1 = ra.results1.model

        # use reordered exog so it matches sub models by group
        exog = ra.exog_grouped
        tind = np.zeros(len(treat_mask))
        tind[-treat_mask.sum():] = 1

        # selection probability by group, propensity score
        prob_sel = np.asarray(res_select.model.predict(ps))
        prob_sel = np.clip(prob_sel, *ra.ps_bounds)
        prob0 = prob_sel[~treat_mask]
        prob1 = prob_sel[treat_mask]

        effect_group = self.effect_group
        if effect_group == "all":
            w0 = 1 / (1 - prob0)
            w1 = 1 / prob1
            sind = 1
        elif effect_group in [1, "treated"]:
            w0 = prob0 / (1 - prob0)
            w1 = prob1 / prob1
            # for averaging effect on treated
            sind = tind / tind.mean()
        elif effect_group in [0, "untreated", "control"]:
            w0 = (1 - prob0) / (1 - prob0)
            w1 = (1 - prob1) / prob1

            sind = (1 - tind)
            sind /= sind.mean()
        else:
            raise ValueError("incorrect option for effect_group")

        # outcome models by treatment using IPW weights
        fitted0 = mod0.predict(p0, exog)
        mom0 = _mom_olsex(p0, model=mod0) * w0[:, None]

        fitted1 = mod1.predict(p1, exog)
        mom1 = _mom_olsex(p1, model=mod1) * w1[:, None]

        mom_outcome = block_diag(mom0, mom1)

        # moments for target statistics, ATE and POM
        mm = (fitted1 - fitted0 - pm) * sind
        mpom = (fitted0 - ppom) * sind
        mm = np.column_stack((mm, mpom))

        # Note: res_select has original data order,
        # mom_outcome and mm use grouped observations
        mom_select = res_select.model.score_obs(ps)
        mom_select = np.concatenate((mom_select[~treat_mask],
                                     mom_select[treat_mask]), axis=0)

        moms = np.column_stack((mm, mom_outcome, mom_select))
        return moms


class TreatmentEffectResults(ContrastResults):
    """
    Results class for treatment effect estimation

    Parameters
    ----------
    teff : instance of TreatmentEffect class
        The treatment effect instance that produced the results.
    results_gmm : instance of GMMResults class
        The GMM results instance used to compute the treatment effect
        parameters and their covariance.
    method : {"IPW", "RA", "AIPW", "AIPW-WLS", "IPW-RA"}
        Method and estimator of treatment effect, corresponding to the
        ``TreatmentEffect`` method that produced the results.
    **kwds
        Other keywords with additional information.

    Notes
    -----
    This class is a subclass of ContrastResults and inherits methods like
    summary, summary_frame and conf_int. Attributes correspond to a z-test
    given by ``GMMResults.t_test``.
    """

    def __init__(self, teff, results_gmm, method, **kwds):
        super().__init__()
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


doc_params_returns = """\
Parameters
----------
return_results : bool, optional
    If True, then a results instance is returned.
    If False, just ATE, POM0 and POM1 are returned.
effect_group : {"all", 0, 1}
    ``effectgroup`` determines for which population the effects are
    estimated.
    If effect_group is "all", then sample average treatment effect and
    potential outcomes are returned
    If effect_group is 1 or "treated", then effects on treated are
    returned.
    If effect_group is 0, "untreated" or "control", then effects on
    untreated, i.e., control group, are returned.
disp : bool, optional
    Indicates whether the scipy optimizer should display the
    optimization results

Returns
-------
TreatmentEffectResults or tuple
    Results instance if `return_results` is True, otherwise the tuple
    (ATE, POM0, POM1).
"""

doc_params_returns2 = """\
Parameters
----------
return_results : bool, optional
    If True, then a results instance is returned.
    If False, just ATE, POM0 and POM1 are returned.
disp : bool, optional
    Indicates whether the scipy optimizer should display the
    optimization results

Returns
-------
TreatmentEffectResults or tuple
    Results instance if `return_results` is True, otherwise the tuple
    (ATE, POM0, POM1).
"""


class TreatmentEffect:
    """
    Estimate average treatment effect under conditional independence

    .. versionadded:: 0.14.0

    This class estimates treatment effect and potential outcome using 5
    different methods, ipw, ra, aipw, aipw-wls, ipw-ra.
    Standard errors and inference are based on the joint GMM representation of
    selection or treatment model, outcome model and effect functions.

    Parameters
    ----------
    model : instance of a model class
        The model class should contain endog and exog for the outcome model.
    treatment : array_like
        indicator array for observations with treatment (1) or without (0)
    results_select : results instance, optional
        The results instance for the treatment or selection model.
    _cov_type : str, optional
        Internal keyword. The keyword does not affect GMMResults which always
        corresponds to HC0 standard errors.
    ps_bounds : array_like of float, optional
        Lower and upper bounds for clipping the propensity score, i.e. the
        predicted probabilities of the selection model. The same bounds are
        used for point estimates and for the GMM moment conditions of all
        estimation methods. Default is (0.001, 0.999).
    **kwds
        Currently not used.

    Notes
    -----
    The outcome model is currently limited to a linear model based on OLS.
    Other outcome models, like Logit and Poisson, will become available in
    future.

    See `Treatment Effect notebook
    <../examples/notebooks/generated/treatment_effect.html>`__
    for an overview.

    """

    def __init__(self, model, treatment, results_select=None, _cov_type="HC0",
                 ps_bounds=(0.001, 0.999), **kwds):
        # Note _cov_type is only for preliminary estimators,
        # cov in GMM always corresponds to HC0
        self.__dict__.update(kwds)  # currently not used
        self.treatment = np.asarray(treatment)
        self.treat_mask = treat_mask = (treatment == 1)
        ps_bounds = array_like(ps_bounds, "ps_bounds", shape=(2,))
        if not 0 < ps_bounds[0] < ps_bounds[1] < 1:
            raise ValueError(
                "ps_bounds values must satisfy 0 < lower < upper < 1"
                )
        self.ps_bounds = ps_bounds

        if results_select is not None:
            self.results_select = results_select
            self.prob_select = np.clip(results_select.predict(), *ps_bounds)

        self.model_pool = model
        endog = model.endog
        exog = model.exog
        self.nobs = endog.shape[0]
        self._cov_type = _cov_type

        # no init keys are supported
        mod0 = model.__class__(endog[~treat_mask], exog[~treat_mask])
        self.results0 = mod0.fit(cov_type=_cov_type)
        mod1 = model.__class__(endog[treat_mask], exog[treat_mask])
        self.results1 = mod1.fit(cov_type=_cov_type)
        # self.predict_mean0 = self.model_pool.predict(self.results0.params
        #                                             ).mean()
        # self.predict_mean1 = self.model_pool.predict(self.results1.params
        #                                             ).mean()

        self.exog_grouped = np.concatenate((mod0.exog, mod1.exog), axis=0)
        self.endog_grouped = np.concatenate((mod0.endog, mod1.endog), axis=0)

    def _diagnostic_sample(self):
        if not hasattr(self, "results_select"):
            raise ValueError("diagnostics require results_select")
        treatment = np.asarray(self.treatment)
        if (treatment.shape != (self.nobs,)
                or not np.isin(treatment, [0, 1]).all()):
            raise ValueError("treatment must be a one-dimensional 0/1 array")
        mask = treatment == 1
        if not mask.any() or mask.all():
            raise ValueError("diagnostics require both treatment groups")
        return mask

    def overlap_summary(self):
        """
        Summarize unclipped propensity scores by treatment group.

        Returns
        -------
        DataFrame
            Rows ``control`` and ``treated`` contain the observation count
            (``nobs``), minimum, 25th percentile, median, 75th percentile,
            maximum, and counts strictly below and above ``ps_bounds``
            (``n_below`` and ``n_above``).

        Notes
        -----
        Uses predictions from the selection model before clipping. Extreme
        scores can indicate poor practical overlap; these summaries do not
        establish the population overlap assumption. No observations are
        dropped and no model is refitted.
        """
        mask = self._diagnostic_sample()
        prob = np.asarray(self.results_select.predict(), dtype=float)
        if (prob.shape != (self.nobs,) or not np.isfinite(prob).all()
                or np.any((prob < 0) | (prob > 1))):
            raise ValueError("selection predictions must be finite probabilities "
                             "with one value per observation")
        rows = []
        for group in [~mask, mask]:
            values = prob[group]
            quantiles = np.quantile(values, [0, 0.25, 0.5, 0.75, 1])
            rows.append(dict(zip(
                ["min", "q25", "median", "q75", "max"], quantiles,
                strict=True)))
            rows[-1].update(
                nobs=values.size,
                n_below=int(np.sum(values < self.ps_bounds[0])),
                n_above=int(np.sum(values > self.ps_bounds[1])),
            )
        return pd.DataFrame(rows, index=["control", "treated"])[
            ["nobs", "min", "q25", "median", "q75", "max", "n_below",
             "n_above"]]

    def balance_table(self, exog=None, effect_group="all"):
        """
        Compare covariate means before and after inverse probability weighting.

        Parameters
        ----------
        exog : array_like, optional
            Two-dimensional finite numeric covariates in the original sample
            order, with one row per observation. The default is the selection
            model's design matrix. DataFrame column names are retained.
            Categorical covariates must first be encoded numerically.
        effect_group : {"all", 0, 1, "treated", "untreated", "control"}, optional
            Weighting target: ``"all"`` for ATE, 1 or ``"treated"`` for ATET,
            and 0, ``"untreated"`` or ``"control"`` for ATC.

        Returns
        -------
        DataFrame
            One row per covariate, with ``mean_control``, ``mean_treated``,
            ``mean_control_weighted``, ``mean_treated_weighted``, ``smd`` and
            ``smd_weighted``. Selection-model covariate names are used by
            default; unnamed supplied columns are labelled x0, x1, etc.

        Notes
        -----
        Weights use the clipped ``prob_select`` used by the estimators.
        Means are normalized separately within each treatment group.
        SMD is treated minus control mean divided by
        ``sqrt((var_treated + var_control) / 2)``, using unweighted sample
        variances (ddof=1). This same denominator is held fixed before and
        after weighting, for all weighting targets. Binary columns use this
        same numeric convention rather than population Bernoulli variances.
        This follows the fixed-scale approach in [1]_, but does not adopt
        its target-specific or binary-covariate defaults.

        Columns with zero pooled variance, including intercepts, have NaN
        SMDs. At least two observations are required in each treatment group.
        These descriptive statistics do not certify absence of confounding.

        References
        ----------
        .. [1] Greifer, N. cobalt: Frequently Asked Questions, "How are
           standardized mean differences computed in cobalt?"
           https://ngreifer.github.io/cobalt/articles/faq.html
        """
        mask = self._diagnostic_sample()
        if min(mask.sum(), (~mask).sum()) < 2:
            raise ValueError("balance requires at least two observations per group")
        if exog is None:
            exog = self.results_select.model.exog
            names = self.results_select.model.exog_names
        else:
            names = getattr(exog, "columns", None)
        exog = np.asarray(exog, dtype=float)
        if (exog.ndim != 2 or exog.shape[0] != self.nobs
                or exog.shape[1] == 0 or not np.isfinite(exog).all()):
            raise ValueError("exog must be a finite nonempty two-dimensional "
                             "array with one row per observation")
        if names is None:
            names = [f"x{i}" for i in range(exog.shape[1])]
        prob = np.asarray(self.prob_select)
        if (prob.shape != (self.nobs,) or not np.isfinite(prob).all()
                or np.any((prob <= 0) | (prob >= 1))):
            raise ValueError("prob_select must contain finite probabilities "
                             "strictly between 0 and 1")
        if effect_group == "all":
            target = np.ones(self.nobs)
        elif effect_group in [1, "treated"]:
            target = prob
        elif effect_group in [0, "untreated", "control"]:
            target = 1 - prob
        else:
            raise ValueError("incorrect option for effect_group")
        w0 = target[~mask] / (1 - prob[~mask])
        w1 = target[mask] / prob[mask]
        x0, x1 = exog[~mask], exog[mask]
        mean0, mean1 = x0.mean(axis=0), x1.mean(axis=0)
        weighted0 = np.average(x0, axis=0, weights=w0)
        weighted1 = np.average(x1, axis=0, weights=w1)
        scale = np.sqrt((x0.var(axis=0, ddof=1)
                         + x1.var(axis=0, ddof=1)) / 2)
        scale[(np.ptp(x0, axis=0) == 0) & (np.ptp(x1, axis=0) == 0)] = np.nan
        return pd.DataFrame({
            "mean_control": mean0,
            "mean_treated": mean1,
            "mean_control_weighted": weighted0,
            "mean_treated_weighted": weighted1,
            "smd": (mean1 - mean0) / scale,
            "smd_weighted": (weighted1 - weighted0) / scale,
        }, index=names)

    @classmethod
    def from_data(cls, endog, exog, treatment, model="ols", **kwds):
        """
        Create models from data

        Not yet implemented.

        Parameters
        ----------
        endog : ndarray
            Outcome variable for the outcome model.
        exog : ndarray
            Explanatory variables for the outcome model.
        treatment : ndarray
            Indicator array for observations with treatment (1) or
            without (0).
        model : str, optional
            Name of the model class to use for the outcome model.
        **kwds
            Additional keyword arguments passed to model classes.

        Returns
        -------
        TreatmentEffect
            Instance of the treatment effect class created from the data.
        """
        raise NotImplementedError

    def ipw(self, return_results=True, effect_group="all", disp=False):
        """
        Inverse Probability Weighted treatment effect estimation

        Parameters
        ----------
        return_results : bool, optional
            If True, then a results instance is returned.
            If False, just ATE, POM0 and POM1 are returned.
        effect_group : {"all", 0, 1}
            ``effectgroup`` determines for which population the effects are
            estimated.
            If effect_group is "all", then sample average treatment effect and
            potential outcomes are returned.
            If effect_group is 1 or "treated", then effects on treated are
            returned.
            If effect_group is 0, "untreated" or "control", then effects on
            untreated, i.e., control group, are returned.
        disp : bool, optional
            Indicates whether the scipy optimizer should display the
            optimization results

        Returns
        -------
        TreatmentEffectResults or tuple
            Results instance if `return_results` is True, otherwise the
            tuple (ATE, POM0, POM1).

        See Also
        --------
        TreatmentEffectResults
        """
        endog = self.model_pool.endog
        tind = self.treatment
        prob = self.prob_select
        if effect_group == "all":
            probt = None
        elif effect_group in [1, "treated"]:
            probt = prob
            effect_group = 1  # standardize effect_group name
        elif effect_group in [0, "untreated", "control"]:
            probt = 1 - prob
            effect_group = 0  # standardize effect_group name
        elif isinstance(effect_group, np.ndarray):
            probt = effect_group
            effect_group = "user"  # standardize effect_group name
        else:
            raise ValueError("incorrect option for effect_group")

        res_ipw = ate_ipw(endog, tind, prob, weighted=True, probt=probt)

        if not return_results:
            return res_ipw

        # gmm = _TEGMMGeneric1(endog, self.results_select, _mom_ols_te,
        #                     probt=probt)
        gmm = _IPWGMM(endog, self.results_select, None, teff=self,
                      effect_group=effect_group)
        start_params = np.concatenate((res_ipw[:2],
                                       self.results_select.params))
        res_gmm = gmm.fit(start_params=start_params,
                          inv_weights=np.eye(len(start_params)),
                          optim_method="nm",
                          optim_args={"maxiter": 5000, "disp": disp},
                          maxiter=1,
                          )

        res = TreatmentEffectResults(self, res_gmm, "IPW",
                                     start_params=start_params,
                                     effect_group=effect_group,
                                     )
        return res

    @Substitution(params_returns=indent(doc_params_returns, " " * 8))
    def ra(self, return_results=True, effect_group="all", disp=False):
        """
        Regression Adjustment treatment effect estimation
        \n%(params_returns)s
        See Also
        --------
        TreatmentEffectResults
        """
        # need indicator for reordered observations
        tind = np.zeros(len(self.treatment))
        tind[-self.treatment.sum():] = 1
        if effect_group == "all":
            probt = None
        elif effect_group in [1, "treated"]:
            probt = tind
            effect_group = 1  # standardize effect_group name
        elif effect_group in [0, "untreated", "control"]:
            probt = 1 - tind
            effect_group = 0  # standardize effect_group name
        elif isinstance(effect_group, np.ndarray):
            # TODO: do we keep this?
            probt = effect_group
            effect_group = "user"  # standardize effect_group name
        else:
            raise ValueError("incorrect option for effect_group")

        exog = self.exog_grouped

        # weight or indicator for effect_group
        if probt is not None:
            cw = (probt / probt.mean())
        else:
            cw = 1

        pom0 = (self.results0.predict(exog) * cw).mean()
        pom1 = (self.results1.predict(exog) * cw).mean()
        if not return_results:
            return pom1 - pom0, pom0, pom1

        endog = self.model_pool.endog
        mod_gmm = _RAGMM(endog, self.results_select, None, teff=self,
                         probt=probt)
        start_params = np.concatenate((
            # ate, tt0.effect,
            [pom1 - pom0, pom0],
            self.results0.params,
            self.results1.params))
        res_gmm = mod_gmm.fit(start_params=start_params,
                              inv_weights=np.eye(len(start_params)),
                              optim_method="nm",
                              optim_args={"maxiter": 5000, "disp": disp},
                              maxiter=1,
                              )
        res = TreatmentEffectResults(self, res_gmm, "RA",
                                     start_params=start_params,
                                     effect_group=effect_group,
                                     )
        return res

    @Substitution(params_returns=indent(doc_params_returns2, " " * 8))
    def aipw(self, return_results=True, disp=False):
        """
        ATE and POM from double robust augmented inverse probability weighting
        \n%(params_returns)s
        See Also
        --------
        TreatmentEffectResults
        """
        nobs = self.nobs
        prob = self.prob_select
        tind = self.treatment
        exog = self.model_pool.exog  # in original order
        correct0 = (self.results0.resid / (1 - prob[tind == 0])).sum() / nobs
        correct1 = (self.results1.resid / (prob[tind == 1])).sum() / nobs
        tmean0 = self.results0.predict(exog).mean() + correct0
        tmean1 = self.results1.predict(exog).mean() + correct1
        ate = tmean1 - tmean0
        if not return_results:
            return ate, tmean0, tmean1

        endog = self.model_pool.endog
        p2_aipw = np.asarray([ate, tmean0])

        mag_aipw1 = _AIPWGMM(endog, self.results_select, None, teff=self)
        start_params = np.concatenate((
            p2_aipw,
            self.results0.params, self.results1.params,
            self.results_select.params))
        res_gmm = mag_aipw1.fit(
            start_params=start_params,
            inv_weights=np.eye(len(start_params)),
            optim_method="nm",
            optim_args={"maxiter": 5000, "disp": disp},
            maxiter=1)

        res = TreatmentEffectResults(self, res_gmm, "AIPW",
                                     start_params=start_params,
                                     effect_group="all",
                                     )
        return res

    @Substitution(params_returns=indent(doc_params_returns2, " " * 8))
    def aipw_wls(self, return_results=True, disp=False):
        """
        ATE and POM from double robust augmented inverse probability weighting

        This uses weighted outcome regression, while `aipw` uses unweighted
        outcome regression.
        Option for effect on treated or on untreated is not available.
        \n%(params_returns)s
        See Also
        --------
        TreatmentEffectResults
        """
        nobs = self.nobs
        prob = self.prob_select

        endog = self.model_pool.endog
        exog = self.model_pool.exog
        tind = self.treatment
        treat_mask = self.treat_mask

        ww1 = tind / prob * (tind / prob - 1)
        mod1 = WLS(endog[treat_mask], exog[treat_mask],
                   weights=ww1[treat_mask])
        result1 = mod1.fit(cov_type="HC1")
        mean1_ipw2 = result1.predict(exog).mean()

        ww0 = (1 - tind) / (1 - prob) * ((1 - tind) / (1 - prob) - 1)
        mod0 = WLS(endog[~treat_mask], exog[~treat_mask],
                   weights=ww0[~treat_mask])
        result0 = mod0.fit(cov_type="HC1")
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
        mod_gmm = _AIPWWLSGMM(endog, self.results_select, None,
                              teff=self)
        start_params = np.concatenate((
            p2_aipw_wls,
            result0.params,
            result1.params,
            self.results_select.params))
        res_gmm = mod_gmm.fit(
            start_params=start_params,
            inv_weights=np.eye(len(start_params)),
            optim_method="nm",
            optim_args={"maxiter": 5000, "disp": disp},
            maxiter=1)
        res = TreatmentEffectResults(self, res_gmm, "AIPW-WLS",
                                     start_params=start_params,
                                     effect_group="all",
                                     )
        return res

    @Substitution(params_returns=indent(doc_params_returns, " " * 8))
    def ipw_ra(self, return_results=True, effect_group="all", disp=False):
        """
        ATE and POM from inverse probability weighted regression adjustment

        \n%(params_returns)s
        See Also
        --------
        TreatmentEffectResults
        """
        treat_mask = self.treat_mask
        endog = self.model_pool.endog
        exog = self.model_pool.exog
        prob = self.prob_select

        prob0 = prob[~treat_mask]
        prob1 = prob[treat_mask]
        if effect_group == "all":
            w0 = 1 / (1 - prob0)
            w1 = 1 / prob1
            exogt = exog
        elif effect_group in [1, "treated"]:
            w0 = prob0 / (1 - prob0)
            w1 = prob1 / prob1
            exogt = exog[treat_mask]
            effect_group = 1  # standardize effect_group name
        elif effect_group in [0, "untreated", "control"]:
            w0 = (1 - prob0) / (1 - prob0)
            w1 = (1 - prob1) / prob1
            exogt = exog[~treat_mask]
            effect_group = 0  # standardize effect_group name
        else:
            raise ValueError("incorrect option for effect_group")

        mod0 = WLS(endog[~treat_mask], exog[~treat_mask],
                   weights=w0)
        result0 = mod0.fit(cov_type="HC1")
        # mean0_ipwra = (result0.predict(exog) * (prob / prob.mean())).mean()
        mean0_ipwra = result0.predict(exogt).mean()

        mod1 = WLS(endog[treat_mask], exog[treat_mask],
                   weights=w1)
        result1 = mod1.fit(cov_type="HC1")
        # mean1_ipwra = (result1.predict(exog) * (prob / prob.mean())).mean()
        mean1_ipwra = result1.predict(exogt).mean()

        if not return_results:
            return mean1_ipwra - mean0_ipwra, mean0_ipwra, mean1_ipwra

        # GMM
        mod_gmm = _IPWRAGMM(endog, self.results_select, None, teff=self,
                            effect_group=effect_group)
        start_params = np.concatenate((
            [mean1_ipwra - mean0_ipwra, mean0_ipwra],
            result0.params,
            result1.params,
            np.asarray(self.results_select.params)
            ))
        res_gmm = mod_gmm.fit(
            start_params=start_params,
            inv_weights=np.eye(len(start_params)),
            optim_method="nm",
            optim_args={"maxiter": 2000, "disp": disp},
            maxiter=1
            )

        res = TreatmentEffectResults(self, res_gmm, "IPW-RA",
                                     start_params=start_params,
                                     effect_group=effect_group,
                                     )
        return res
