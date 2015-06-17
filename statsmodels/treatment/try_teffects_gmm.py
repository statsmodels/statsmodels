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
import statsmodels.iolib.foreign as smio
import statsmodels.api as sm

# not used here
#dta_spm = smio.genfromdta(r"M:\josef_new\work_oth_pdf\data\stata\spmdata.dta", pandas=True)

#dta_cat = smio.genfromdta(r"M:\josef_new\work_oth_pdf\data\stata\r14_\cattaneo2.dta", pandas=True)

dta_cat = pd.read_stata(r"M:\josef_new\work_oth_pdf\data\stata\r14_\cattaneo2.dta")

dta_cat['mbsmoke_'] = (dta_cat['mbsmoke'] == 'smoker').astype(int)
dta_cat['mmarried_'] = dta_cat['mmarried'] == 'married'
dta_cat['fbaby_'] = dta_cat['fbaby'] == 'Yes'
dta_cat['prenatal1_'] = dta_cat['prenatal1'] == 'Yes'
dta_cat['mage2'] = dta_cat['mage']**2.  #.values.astype(float)**2.

#res_logit = sm.Probit.from_formula('mbsmoke ~ mmarried + mage:mage + fbaby + medu', dta_cat).fit()
res_logit = sm.Probit.from_formula('mbsmoke_ ~ mmarried_ + mage + mage2 + fbaby_ + medu', dta_cat).fit()
#print(res_logit.summary())

prob = res_logit.predict()

# the following is IPW weighted outcome regression (treatment dummies only)
# robust standard errors are the same as gmmo, i.e. ignores that prob is estimated
treat_ind = (dta_cat['mbsmoke_'].values[:, None] == [False, True]).astype(int)
probt = treat_ind[:, 1] * prob + (1-treat_ind[:, 1]) * (1 - prob)
w = treat_ind[:, 1] / prob + (1-treat_ind[:, 1]) / (1 - prob)
bw = dta_cat['bweight'].values.astype(float)
res_out = sm.WLS(bw, treat_ind, weights=w).fit(cov_type='HC0')
#print(res_out.summary())
print(res_out.t_test([-1, 1]))
print(res_out.t_test([[1,0], [0,1],[-1, 1]]))


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
    w2 = (1. - tind) / (1. - prob)
    if weighted:
        w1 = w1 / w1.mean()
        w2 = w2 / w2.mean()

    return np.column_stack((endog * w1 - tm[0], endog * w2 - tm[1]))


def mom_ols(tm, endog, tind, prob, weighted=True):
    w = tind / prob + (1-tind) / (1 - prob)

    treat_ind = np.column_stack((tind, 1 - tind))
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

def alpha_hat(endog, treat_ind, prob, tm_exog):
    w = treat_ind / prob + (1 - treat_ind) / (1 - prob)
    res_aux = sm.OLS(endog * w**2, res_logit.model.exog).fit()
    ahat = - res_aux.predict() * (treat_ind - prob)
    return ahat, res_aux

endog = bw
tind = treat_ind[:, 1]
def mom(params):
    te = params[0]
    p_tm = params[1:]
    prob = res_logit.model.predict(p_tm)
    moms = np.column_stack((mom_ate(te, endog, tind, prob, weighted=True),
                            res_logit.model.score_obs(p_tm)))
    return moms

from statsmodels.sandbox.regression.gmm import GMM

# The next classes use hardcoded functions and models from module globals
class TEGMM(GMM):
    def momcond(self, params):
        return mom(params)

class TEGMMs(GMM):
    def momcond(self, params):
        # moment condition for ATE only, not even one POM
        # standard errors are very large
        te = params
        prob = res_logit.predict()
        return mom_ate(te, endog, tind, prob, weighted=True)[:,None]

class TEGMMm(GMM):
    def momcond(self, params):
        # moment condition for POM without "instrument" conditioning variable
        # standard errors are very large
        tm = params
        prob = res_logit.predict()
        return mom_atm(tm, endog, tind, prob, weighted=True)

class TEGMMo(GMM):
    def momcond(self, params):
        # this does not take the affect of the parameter estimation in the
        # treatment effect into account for the standard errors of the
        # outcome model.
        # It's still first order correct with robust standard errors.
        tm = params
        prob = res_logit.predict()
        return mom_ols(tm, endog, tind, prob, weighted=True)


class TEGMM2(GMM):
    """GMM class to get cov_params for treatment effects

    This combines moment conditions for the selection/treatment model and the
    outcome model to get the standard errors for the treatment effect that
    takes the first step estimation of the treatment model into account.

    this also matches standard errors of ATE and POM in Stata

    """

    def __init__(self, endog, res_select, mom_outcome):
        super(TEGMM2, self).__init__(endog, None, None)
        self.res_select = res_select
        self.mom_outcome = mom_outcome

        # add xnames so it's not None
        # we don't have exog in init in this version
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


gmm = TEGMM(endog, res_logit.model.exog, res_logit.model.exog)
te = ate_ipw(endog, tind, prob, weighted=True)
start_params=np.concatenate(([te], res_logit.params))
#res_gmm = gmm.fit(start_params=start_params)
res_gmm = gmm.fit(start_params=start_params, optim_method='nm', inv_weights=np.eye(7), maxiter=2)

gmms = TEGMMs(endog, res_logit.model.exog, res_logit.model.exog)
res_gmms = gmms.fit(start_params=start_params[0], optim_method='nm', inv_weights=np.eye(1), maxiter=2)

gmmm = TEGMMm(endog, res_logit.model.exog, res_logit.model.exog)
res_gmmm = gmmm.fit(start_params=[3000, 3000], optim_method='nm', inv_weights=np.eye(2), maxiter=2)

gmmo = TEGMMo(endog, res_logit.model.exog, res_logit.model.exog)
res_gmmo = gmmo.fit(start_params=[3000, 3000], optim_method='nm', inv_weights=np.eye(2), maxiter=2)


gmm2 = TEGMM2(endog, res_logit, mom_ols)
#te = ate_ipw(endog, tind, prob, weighted=True)
start_params=np.concatenate(([3000, 3000], res_logit.params))
#res_gmm = gmm.fit(start_params=start_params)
res_gmm2 = gmm2.fit(start_params=start_params, optim_method='nm', inv_weights=np.eye(8), maxiter=2)
res_gmm2.model.data.param_names = ['par%d' % i for i in range(8)]
constraint = np.zeros((3, 8))
constraint[0,0] = 1
constraint[1,1] = 1
constraint[2,1] = 1
constraint[2,0] = -1
res_gmm2.bse
print(res_gmm2.t_test(constraint))



class RegAdjustment(object):
    """estimate average treatment effect using regression adjustment

    This includes now methods to calculate POM and ATE for other estimators
    (no standard errors for those yet)


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


mod_ra01 = sm.OLS.from_formula('bweight ~ prenatal1_ + mmarried_ + mage + fbaby_', dta_cat)
ra = RegAdjustment(mod_ra01, tind)
print(ra.summary())
print(ra.aipw(prob))

treatment = treat_ind[:,1]
treat_mask = treatment == 1

self = ra
model = self.model_pool
endog = model.endog
exog = model.exog
self.nobs = endog.shape[0]
# no init keys are supported
mod0 = model.__class__(endog[~treat_mask], exog[~treat_mask])
self.result0 = mod0.fit(cov_type='HC1')
mod1 = model.__class__(endog[treat_mask], exog[treat_mask])
self.result1 = mod1.fit(cov_type='HC1')

w_norm = (treat_ind * w[:,None]) / (treat_ind * w[:,None]).sum(0)
w_norm0 = w_norm[~treat_mask]
w_norm1 = w_norm[treat_mask]

mod0 = sm.WLS(endog[~treat_mask], exog[~treat_mask], weights=w[~treat_mask])
result0 = mod0.fit(cov_type='HC1')

mean0_ipwra = result0.predict(mod_ra01.exog).mean()
mod1 = sm.WLS(endog[treat_mask], exog[treat_mask], weights=1/prob[treat_mask])
result1 = mod1.fit(cov_type='HC1')
mean1_ipwra = result1.predict(mod_ra01.exog).mean()
print(mean0_ipwra, mean1_ipwra, mean1_ipwra - mean0_ipwra)

res_ipwra = np.array((mean1_ipwra - mean0_ipwra, mean0_ipwra, mean1_ipwra))
ttg = res_gmm2.t_test(constraint[[2, 0, 1]]) # reorder so ATE is first
res_ipw = ttg.effect
res_ra = np.array((ra.ate, ra.tt0.effect, ra.tt1.effect))

res_aipw = ra.aipw(prob)
res_aipw_wls = ra.aipw_wls(prob)

import pandas as pd
res_all = pd.DataFrame(np.column_stack((res_ipw, res_aipw, res_aipw_wls, res_ra, res_ipwra)),
                       columns = 'res_ipw res_aipw res_aipw_wls res_ra res_ipwra'.split(),
                       index = 'ATE POM_0 POM_1'.split())
print(res_all.T)

# regression values matching Stata documentation
res_all_values = np.array([[  230.68859809,  3172.77405949,  3403.46265758],
                           [ -230.98920111,  3403.35525317,  3172.36605206],
                           [ -227.19561819,  3403.25065098,  3176.05503279],
                           [ -239.63921146,  3403.24227194,  3163.60306047],
                           [ -229.96707794,  3403.33563931,  3173.36856137]])

from numpy.testing import assert_allclose

assert_allclose(res_all.T.values, res_all_values, rtol=1e-6)
