# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 23:19:45 2015

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
import pandas as pd
#import statsmodels.iolib.foreign as smio
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.treatment.treatment_effects import (TEGMM, TEGMMGeneric1,
                              RegAdjustment,
                              mom_ate, mom_atm, mom_ols, ate_ipw)

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


########

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

###########


# The next classes use hardcoded functions and models from module globals
class _TEGMM(GMM):
    # uses ate and treatment/selection model for moment conditions, no POM
    # standard errors for ATE don't agree with TEGMM
    def momcond(self, params):
        return mom(params)

class _TEGMMs(GMM):
    def momcond(self, params):
        # moment condition for ATE only, not even one POM
        # standard errors are very large
        te = params
        prob = res_logit.predict()
        return mom_ate(te, endog, tind, prob, weighted=True)[:,None]

class _TEGMMm(GMM):
    def momcond(self, params):
        # moment condition for POM without "instrument" conditioning variable
        # standard errors are very large
        tm = params
        prob = res_logit.predict()
        return mom_atm(tm, endog, tind, prob, weighted=True)

class _TEGMMo(GMM):
    def momcond(self, params):
        # this does not take the affect of the parameter estimation in the
        # treatment effect into account for the standard errors of the
        # outcome model.
        # It's still first order correct with robust standard errors.
        tm = params
        prob = res_logit.predict()
        return mom_ols(tm, endog, tind, prob, weighted=True)


################


gmm = _TEGMM(endog, res_logit.model.exog, res_logit.model.exog)
te = ate_ipw(endog, tind, prob, weighted=True)
start_params=np.concatenate(([te], res_logit.params))
#res_gmm = gmm.fit(start_params=start_params)
res_gmm = gmm.fit(start_params=start_params, optim_method='nm', inv_weights=np.eye(7), maxiter=2)

gmms = _TEGMMs(endog, res_logit.model.exog, res_logit.model.exog)
res_gmms = gmms.fit(start_params=start_params[0], optim_method='nm', inv_weights=np.eye(1), maxiter=2)

gmmm = _TEGMMm(endog, res_logit.model.exog, res_logit.model.exog)
res_gmmm = gmmm.fit(start_params=[3000, 3000], optim_method='nm', inv_weights=np.eye(2), maxiter=2)

gmmo = _TEGMMo(endog, res_logit.model.exog, res_logit.model.exog)
res_gmmo = gmmo.fit(start_params=[3000, 3000], optim_method='nm', inv_weights=np.eye(2), maxiter=2)


gmm2 = TEGMM(endog, res_logit, mom_ols)
#te = ate_ipw(endog, tind, prob, weighted=True)
start_params=np.concatenate(([3000, 3000], res_logit.params))
#res_gmm = gmm.fit(start_params=start_params)
res_gmm2 = gmm2.fit(start_params=start_params, optim_method='nm', inv_weights=np.eye(8), maxiter=2)
res_gmm2.model.data.param_names = ['par%d' % i for i in range(8)]
constraint = np.zeros((3, 8))
constraint[0,1] = 1
constraint[0,0] = -1
constraint[1,0] = 1
constraint[2,1] = 1

res_gmm2.bse
print(res_gmm2.t_test(constraint))


###################


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
ttg = res_gmm2.t_test(constraint)
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
res_all_values = np.array([[ -230.68859809,  3403.46265758,  3172.77405949],
                           [ -230.98920111,  3403.35525317,  3172.36605206],
                           [ -227.19561819,  3403.25065098,  3176.05503279],
                           [ -239.63921146,  3403.24227194,  3163.60306047],
                           [ -229.96707794,  3403.33563931,  3173.36856137]])

from numpy.testing import assert_allclose

assert_allclose(res_all.T.values, res_all_values, rtol=1e-6)

# first 2 values agree with stata docs at around 2 or 3 decimals
assert_allclose(ttg.sd, np.array([ 25.81674561,   9.57141232,  24.00104938]))


# compare IPWeighted regression (WLS) with IPW GMM
tt_wls = res_out.t_test([[-1, 1],[1,0], [0,1]])
# should be close at optimization precision
assert_allclose(tt_wls.effect, ttg.effect, rtol=1e-6)
# uses different cov_params, but they are close to each other in this case
assert_allclose(tt_wls.sd, ttg.sd, rtol=0.005)


# regression tests, keep results during refactoring
# ATE or POM for IPW close at optimization precision to IPW res_gmm2, ttg
res_gmm_params = np.array([ -2.30688638e+02,  -1.55825526e+00,  -6.48482138e-01,
        -2.17596162e-01,   1.74432697e-01,  -3.25591262e-03,
        -8.63630870e-02])

assert_allclose(res_gmm.params, res_gmm_params)

res_gmm_params = np.array([ -2.30688638e+02,  -1.55825526e+00,  -6.48482138e-01,
        -2.17596162e-01,   1.74432697e-01,  -3.25591262e-03,
        -8.63630870e-02])

assert_allclose(res_gmm.params, res_gmm_params)

assert_allclose(res_gmms.params, np.array([-230.6886378]))
assert_allclose(res_gmmm.params, np.array([ 3403.46270016,  3172.77407107]))
assert_allclose(res_gmmo.params, np.array([ 3403.46272906,  3172.77402981]))

# Using generic GMM version to replicate original version

gmm2b = TEGMMGeneric1(endog, res_logit, mom_ols)
res_gmm2b = gmm2b.fit(start_params=start_params, optim_method='nm',
                      inv_weights=np.eye(8), maxiter=2)

# generic versus original model
assert_allclose(res_gmm2b.params, res_gmm2.params, rtol=1e-10)
assert_allclose(res_gmm2b.bse, res_gmm2.bse, rtol=1e-10)

gmmo2 = TEGMMGeneric1(endog, res_logit, mom_ols, exclude_tmoms=True)
res_gmmo2 = gmmo2.fit(start_params=[3000, 3000], optim_method='nm',
                          inv_weights=np.eye(2), maxiter=2)

# equivalence to IPWLS ?
assert_allclose(res_gmmo2.params, res_out.params, rtol=1e-7)
assert_allclose(res_gmmo2.bse, res_out.bse, rtol=1e-7)
# refactored versus original model
assert_allclose(res_gmmo2.params, res_gmmo.params, rtol=1e-10)
assert_allclose(res_gmmo2.bse, res_gmmo.bse, rtol=1e-10)

gmms2 = TEGMMGeneric1(endog, res_logit, mom_ate, exclude_tmoms=True)
res_gmms2 = gmms2.fit(start_params=start_params[0]*0.9, optim_method='nm', inv_weights=np.eye(1), maxiter=2)
# refactored versus original model
assert_allclose(res_gmms2.params, res_gmms.params, rtol=1e-7)
assert_allclose(res_gmms2.bse, res_gmms.bse, rtol=1e-7)

gmmm2 = TEGMMGeneric1(endog, res_logit, mom_atm, exclude_tmoms=True)
res_gmmm2 = gmmm2.fit(start_params=[3000, 3000], optim_method='nm', inv_weights=np.eye(2), maxiter=2)

# refactored versus original model
assert_allclose(res_gmmm2.params, res_gmmm.params, rtol=1e-7)
assert_allclose(res_gmmm2.bse, res_gmmm.bse, rtol=1e-7)

gmm_ = TEGMMGeneric1(endog, res_logit, mom_ate, exclude_tmoms=False)
te = ate_ipw(endog, tind, prob, weighted=True)
start_params_ = np.concatenate(([te], res_logit.params))
res_gmm_ = gmm_.fit(start_params=start_params_, optim_method='nm', inv_weights=np.eye(7), maxiter=2)

# refactored versus original model
assert_allclose(res_gmm_.params, res_gmm.params, rtol=1e-7)
assert_allclose(res_gmm_.bse, res_gmm.bse, rtol=1e-7)

