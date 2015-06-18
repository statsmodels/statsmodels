# -*- coding: utf-8 -*-
"""Treatment effect estimators, follows largely Stata's teffects in Stata 13 manual

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
import pandas as pd
#import statsmodels.iolib.foreign as smio
import statsmodels.api as sm




def mom_ate(te, endog, tind, prob, weighted=True):
    if weighted:
        w1 = (tind / prob)
        w2 = (1. - tind) / (1. - prob)
        wdiff = w1 / w1.mean() - w2 / w2.mean()
        #wdiff = w1 / w1.sum() - w2 / w2.sum()
    else:
        wdiff = (tind / prob) - (1 - tind) / (1 - prob)

    return endog * wdiff - te

def mom_atm(tm, endog, tind, prob, weighted=True):
    w1 = (tind / prob)
    w0 = (1. - tind) / (1. - prob)
    if weighted:
        w1 /= w1.mean()
        w0 /= w0.mean()

    return np.column_stack((endog * w0 - tm[0], endog * w1 - tm[1]))


def mom_ols(tm, endog, tind, prob, weighted=True):
    w = tind / prob + (1-tind) / (1 - prob)

    treat_ind = np.column_stack((1 - tind, tind))
    mom = (w * (endog - treat_ind.dot(tm)))[:,None] * treat_ind

    return mom


def ate_ipw(endog, tind, prob, weighted=True):
    if weighted:
        w1 = (tind / prob)
        w2 = (1. - tind) / (1. - prob)
        wdiff = w1 / w1.mean() - w2 / w2.mean()
        #wdiff = w1 / w1.sum() - w2 / w2.sum()
    else:
        wdiff = (tind / prob) - (1 - tind) / (1 - prob)

    return (endog * wdiff).mean()



from statsmodels.sandbox.regression.gmm import GMM


class TEGMMGeneric1(GMM):
    """GMM class to get cov_params for treatment effects

    This combines moment conditions for the selection/treatment model and the
    outcome model to get the standard errors for the treatment effect that
    takes the first step estimation of the treatment model into account.

    this also matches standard errors of ATE and POM in Stata

    """

    def __init__(self, endog, res_select, mom_outcome, exclude_tmoms=False):
        super(TEGMMGeneric1, self).__init__(endog, None, None)
        self.res_select = res_select
        self.mom_outcome = mom_outcome
        self.exclude_tmoms = exclude_tmoms

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
            self.prob = self.res_select.predict() # fittedvalues is still linpred
        else:
            self.prob = None

    def momcond(self, params):
        k_outcome = len(params) - self.k_select
        tm = params[:k_outcome]
        p_tm = params[k_outcome:]

        tind = self.res_select.model.endog


        if self.exclude_tmoms:
            prob = self.prob
        else:
            prob = self.res_select.model.predict(p_tm)

        moms_list = []
        mom_o = self.mom_outcome(tm, self.endog, tind, prob, weighted=True)
        moms_list.append(mom_o)

        if not self.exclude_tmoms:
            mom_t = self.res_select.model.score_obs(p_tm)
            moms_list.append(mom_t)

        moms = np.column_stack(moms_list)
        return moms


    def momcond_aipw(self, params):
        k_outcome = len(params) - self.k_select
        tm = params[:k_outcome]
        p_tm = params[k_outcome:]

        tind = self.res_select.model.endog
        prob = self.res_select.model.predict(p_tm)

        momt = self.mom_outcome(tm, self.endog, tind, prob, weighted=True)
        moms = np.column_stack((momt,
                                self.res_select.model.score_obs(p_tm)))
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
        self.res_select = res_select
        self.mom_outcome = mom_outcome

        # add xnames so it's not None
        # we don't have exog in init in this version
        if self.data.xnames is None:
            self.data.xnames = []


    def momcond(self, params):
        tm = params[:2]
        p_tm = params[2:]

        tind = self.res_select.model.endog
        prob = self.res_select.model.predict(p_tm)
        momt = self.mom_outcome(tm, self.endog, tind, prob, weighted=True)
        moms = np.column_stack((momt,
                                self.res_select.model.score_obs(p_tm)))
        return moms




class RegAdjustment(object):
    """estimate average treatment effect using regression adjustment

    The class is written for regressio adjustment estimator but now also
    includes methods to calculate POM and ATE for other estimators
    (no standard errors for those yet, summary is only for RA).


    Parameters
    ----------
    model : instance of a model class
        The model class should contain endog and exog for the full model.
    treatment : ndarray
        indicator array for observations with treatment (1) or without (0)
    kwds : keyword arguments
        currently not used

    Notes
    -----
    The results are attached after creating the instance.
    Calling the `summary` will return a string with an overview of the results.

    This is currently a basic implementation targeting OLS
    Other models will need new methods for the calculations, e.g. nonlinear
    prediction standard errors, or we need to use a more generic method to
    calculate ATE.

    This currently only looks at the outcome model and takes the probabilities
    of the selection model (propensity score) as argument in methods.

    Other limitations
    Does not use any `model.__init__` keywords.


    """

    def __init__(self, model, treatment, **kwds):
        self.treatment = np.asarray(treatment)
        self.treat_mask = treat_mask = (treatment == 1)


        self.model_pool = model
        endog = model.endog
        exog = model.exog
        self.nobs = endog.shape[0]
        # no init keys are supported
        mod0 = model.__class__(endog[~treat_mask], exog[~treat_mask])
        self.result0 = mod0.fit(cov_type='HC1')
        mod1 = model.__class__(endog[treat_mask], exog[treat_mask])
        self.result1 = mod1.fit(cov_type='HC1')
        self.predict_mean0 = self.model_pool.predict(self.result0.params).mean()
        self.predict_mean1 = self.model_pool.predict(self.result1.params).mean()

        # this only works for linear model, need margins for discrete
        exog_mean = exog.mean(0)
        self.tt0 = self.result0.t_test(exog_mean)
        self.tt1 = self.result1.t_test(exog_mean)
        self.ate = self.tt1.effect - self.tt0.effect
        self.se_ate = np.sqrt(self.tt1.sd**2 + self.tt0.sd**2)

    @classmethod
    def from_data(cls, endog, exog, treatment, model='ols', **kwds):
        raise NotImplementedError

    def ra(self):
        return self.ate, self.tt0.effect, self.tt1.effect

    def aipw(self, prob=None):
        """double robust augmented inverse probability weighting

        replicates Stata's `teffects aipw`

        no standard errors yet

        """
        nobs = self.nobs
        if prob is None:
            raise NotImplementedError
            #prob = ???   # need selection model or probability
        correct0 = (self.result0.resid / (1 - prob[tind == 0])).sum() / nobs
        correct1 = (self.result1.resid / (prob[tind == 1])).sum() / nobs
        tmean0 = self.tt0.effect + correct0
        tmean1 = self.tt1.effect + correct1
        ate = tmean1 - tmean0
        return ate, tmean0, tmean1

    def aipw_wls(self, prob=None):
        """double robust augmented inverse probability weighting

        replicates Stata's `teffects aipw`

        no standard errors yet

        """
        nobs = self.nobs
        if prob is None:
            raise NotImplementedError
            #prob = ???   # need selection model or probability

        endog = self.model_pool.endog
        exog = self.model_pool.exog

        ww1 = tind / prob * (tind / prob - 1)
        mod1 = sm.WLS(endog[treat_mask], exog[treat_mask], weights=ww1[treat_mask])
        result1 = mod1.fit(cov_type='HC1')
        mean1_ipw2 = result1.predict(exog).mean()

        ww0 = (1 - tind) / (1 - prob) * ((1 - tind) / (1 - prob) - 1)
        mod0 = sm.WLS(endog[~treat_mask], exog[~treat_mask], weights=ww0[~treat_mask])
        result0 = mod0.fit(cov_type='HC1')
        mean0_ipw2 = result0.predict(exog).mean()

        correct0 = (result0.resid / (1 - prob[tind == 0])).sum() / nobs
        correct1 = (result1.resid / (prob[tind == 1])).sum() / nobs
        tmean0 = mean0_ipw2 + correct0
        tmean1 = mean1_ipw2 + correct1
        ate = tmean1 - tmean0
        return ate, tmean0, tmean1

    def ipw_ra(self, prob=None):
        treat_mask = self.treat_mask
        endog = self.model_pool.endog
        exog = self.model_pool.exog

        mod0 = sm.WLS(endog[~treat_mask], exog[~treat_mask], weights=w[~treat_mask])
        result0 = mod0.fit(cov_type='HC1')

        mean0_ipwra = result0.predict(mod_ra01.exog).mean()
        mod1 = sm.WLS(endog[treat_mask], exog[treat_mask], weights=1/prob[treat_mask])
        result1 = mod1.fit(cov_type='HC1')
        mean1_ipwra = result1.predict(mod_ra01.exog).mean()

        #res_ipwra = np.array((mean1_ipwra - mean0_ipwra, mean0_ipwra, mean1_ipwra))
        return mean1_ipwra - mean0_ipwra, mean0_ipwra, mean1_ipwra


    def summary(self):
        txt = [str(self.tt0.summary(title='POM Treatment 0'))]
        txt.append(str(self.tt1.summary(title='POM Treatment 1')))
        txt.append('ATE = %f10.4   std.dev. = %f10.4' % (self.ate, self.se_ate))
        return '\n'.join(txt)
