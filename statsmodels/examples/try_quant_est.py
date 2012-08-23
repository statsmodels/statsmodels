# -*- coding: utf-8 -*-
"""

Created on Thu Aug 23 15:10:08 2012

Author: Josef Perktold
"""
import numpy as np
from scipy import stats
from statsmodels.distributions.quantiles_estimation import (LocScaleQEst,
                                                            locscale)
#locscale is unnecessary cosmetic helper function
import statsmodels.api as sm

np.random.seed(9853781)

#estimate loc and scale using OLS
distargs = (3,)
nobs = 500
x = 1 + 2 * np.random.randn(nobs)
x = 1 + 2 * np.random.standard_t(*distargs, **dict(size=nobs))
x[:10] -= 10

x.sort()
nobs = 4 #
nobs = len(x)
kth = np.arange(1, nobs+1, dtype=float)
probs = kth / (nobs + 1.)    #lambda
#probs =   # check unequal spacing of probs
ppf_x = stats.norm.ppf(probs)  #U
pdf_x = stats.norm.pdf(ppf_x)  #f
d_probs = np.diff(probs)
probs_e = np.concatenate(([0], probs, [1]))
d_probs_e = np.diff(probs_e) #extended
ds2_probs_e = probs_e[2:] - probs_e[:-2]
vii = pdf_x**2 * ds2_probs_e / d_probs_e[1:] / d_probs_e[:-1]
#incomplete

fact_low = probs / pdf_x
fact_high = (1. - probs) / pdf_x
v = fact_low[:,None] * fact_high  #upper triangle is correct
larger = kth[:,None] > kth
smaller = kth[:,None] < kth
v[larger] = v.T[larger]

v /= nobs

check_v = 0
if check_v:
    print np.linalg.inv(v)

    #mistake in formula in Hassanein ?
    #works for uniform probs
    print 'diag'
    print pdf_x[:]**2 / d_probs_e[1:] * nobs * 2
    print 'off-diag'
    print - pdf_x[1:] * pdf_x[:-1] / d_probs_e[2:] * nobs

more=1
if more:
    exog = sm.add_constant(ppf_x, prepend=True)
    params = sm.OLS(x, exog).fit().params
    print params
    print x.mean(), x.std(ddof=1)
    print sm.GLS(x, exog, sigma=v).fit().params, 'GLS'

    dist = stats.t
    ppf_x = dist.ppf(probs, *distargs)
    pdf_x = dist.pdf(ppf_x, *distargs)
    exog = sm.add_constant(ppf_x, prepend=True)
    params = sm.OLS(x, exog).fit().params

    print '\nunsing t distribution'
    print params, 'OLS'
    print x.mean(), x.std(ddof=1), 'mom'
    print sm.RLM(x, exog).fit().params, 'RLM'
    print stats.t.fit_loc_scale(x, 3), 'mom distr'
    #BUG: this uses v from normal
    print sm.GLS(x, exog, sigma=v).fit().params, 'GLS'

    stats.t.logpdf(x, 3, *params).sum()
    stats.t.logpdf(x, 3, **locscale(params)).sum()

    g = sm.graphics.qqplot(x, dist=stats.t, distargs=(3,))
    g.show()

def print_mod(mod, true=None):
    print mod.dist.name, mod.distargs
    print true, 'true'
    print mod.fit(method='OLS').params, 'OLS'
    print mod.fit().params, 'GLS'
    print mod.fit(method='GLS2').params, 'GLS2'
    print mod.fit(method='RLM').params, 'RLM'
    print mod.fit(method='G-RLM').params, 'G-RLM'
    print mod.fit(method='G-RLM2').params, 'G-RLM2'
    print mod.fit(method='sp_mom'), 'sp_mom'

#mod = LocScaleQEst(x, dist=stats.t, distargs=(3,))
mod = LocScaleQEst(x) #, dist=stats.t, distargs=(3,))
print
print mod.fit(method='OLS').params, 'OLS'
print mod.fit().params, 'GLS'
print mod.fit(method='RLM').params, 'RLM'
print mod.fit(method='G-RLM').params, 'G-RLM'
print mod.fit(method='GLS2').params, 'GLS2'
print mod.fit(method='G-RLM').params, 'G-RLM2'

true = [5., 7.]
xwei = stats.weibull_min.rvs(1, loc=5, scale=7, size=200)
modwei = LocScaleQEst(xwei, dist=stats.weibull_min, distargs=(1,))
print '\nWeibull_min'
print_mod(modwei, true)

xgev = stats.genextreme.rvs(1, loc=5, scale=7, size=200)
modgev = LocScaleQEst(xgev, dist=stats.genextreme, distargs=(1,))
print '\nGEV'
print_mod(modgev, true)

xchi = stats.chi.rvs(1, loc=5, scale=7, size=200)
modchi = LocScaleQEst(xchi, dist=stats.chi, distargs=(1,))
print '\nChi'
print_mod(modchi, true)

xcauchy = stats.cauchy.rvs(loc=5, scale=7, size=200)
modcauchy = LocScaleQEst(xcauchy, dist=stats.cauchy, distargs=())
print '\nChi'
print_mod(modcauchy, true)

xt = stats.t.rvs(2, loc=5, scale=7, size=200)
modt = LocScaleQEst(xt, dist=stats.t, distargs=(2,))
print '\nChi'
print_mod(modt, true)

res_gls = mod.fit()
print res_gls.model.wendog[:5]
wend = mod.whiten(mod.endog)
print wend[:5]
