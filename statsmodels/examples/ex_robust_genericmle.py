# -*- coding: utf-8 -*-
"""

Created on Thu Feb 27 13:32:36 2014

Author: Josef Perktold
"""

import numpy as np

from statsmodels.miscmodels.robust_genericmle import *


#import statsmodels.api as sm

if 0:
    data = sm.datasets.stackloss.load()
    data.exog = sm.add_constant(data.exog)
    endog, exog = data.endog, data.exog

    seed = 976537
    np.random.seed(seed)
    nobs, k_vars = 100, 5
    n_outliers = 10
    sig_e = 0.5
    beta = np.ones(k_vars)
    beta[-2:] *= 0.25
    exog = sm.add_constant(np.random.uniform(0, 1, size=(nobs, k_vars - 1)))
    y_true = np.dot(exog, beta)
    endog = y_true + sig_e * np.random.randn(nobs)
    endog[-n_outliers:] += 100

    print_summary = False
    # Huber's T norm with the (default) median absolute deviation scaling

    huber_t = sm.RLM(endog, exog, M=sm.robust.norms.HuberT())
    hub_results = huber_t.fit(scale_est=rscale.HuberScale())
    print(hub_results.params)
    print(hub_results.bse)
    if print_summary:
        print(hub_results.summary(yname='y',
                xname=['var_%d' % i for i in range(len(hub_results.params))]))

    res_ols = sm.OLS(endog, exog).fit()
    start_ols = np.concatenate((res_ols.params, [np.sqrt(res_ols.scale)]))
    start_ols = np.concatenate((res_ols.params, [rscale.mad(res_ols.resid, center=0)]))

    start_hub = np.concatenate((hub_results.params, [hub_results.scale]))
    start_params = start_hub

    mod = MEstimator(endog, exog)
    print('loss RLM', mod.loglike(start_hub), 'scale RLM', hub_results.scale)
    print(start_params + 0.5, (start_params * 0.5).shape)
    start_params = np.ones(mod.exog.shape[1] + 1)
    start_params = start_ols
    res = mod.fit(start_params=start_params, method='bfgs') # nm')#
    print('\n M-E joint with scale')
    print(res.params)
    if print_summary:
        print(res.summary())  # no normalized_params

    mod2 = MEstimator(endog, exog)
    mod2.scale_fixed = hub_results.scale / 0.491
    res2 = mod2.fit(start_params=start_params[:-1]*0.5, method='bfgs')

    print('\n M-E fixed scale')
    print(res2.params)
    if print_summary:
        print(res2.summary())  # no normalized_params


    # chekc gradient, jac, score
    print(mod.jac(start_ols).sum(0))
    print(mod.score(start_ols))
    import statsmodels.tools.numdiff as nd
    print(nd.approx_fprime(start_ols, mod.loglike))



    mod4 = MEstimatorHD(endog, exog)
    #mod4.scale_fixed = hub_results.scale / 0.491
    res4 = mod4.fit(start_params=start_params, method='bfgs')

    print('\n M-E HD jointly estimated scale')
    print(res4.params)
    print(res4.summary())  # no normalized_params


    # chekc gradient, jac, score
    print(mod4.jac(start_ols).sum(0))
    print(mod4.score(start_ols))
    import statsmodels.tools.numdiff as nd
    print(nd.approx_fprime(start_ols, mod4.loglike))

    print('\ncompare params')
    print(res_ols.params)
    print(hub_results.params)
    print(res2.params)
    print(res.params)
    print(res4.params)

    mod_mit = RLMIterative(endog, exog, M=rnorms.HuberT(), meef_scale=lambda x:rnorms.TukeyBiweight().rho(x)-0)
    res_mit = mod_mit.fit()

    mod_mit2 = RLMIterative(endog, exog, M=rnorms.HuberT(),
                           meef_scale=lambda x:rnorms.TukeyBiweight().rho(x)-0,
                           update_scale=False)
    res_mit2 = mod_mit2.fit(start_params=res2.params, start_scale=0.79598, update_scale=False)

    mod_mit3 = RLMIterative(endog, exog, M=rnorms.HuberT(), meef_scale=lambda x:rnorms.HuberT(2.5).psi(np.abs(x)**2)-0)
    mod_mit3.scale_bias=0.97755998345280681
    res_mit3a = mod_mit3.fit()
    print(res_mit3a[0].params)


chem = np.array([2.20, 2.20, 2.4, 2.4, 2.5, 2.7, 2.8, 2.9, 3.03,
            3.03, 3.10, 3.37, 3.4, 3.4, 3.4, 3.5, 3.6, 3.7, 3.7, 3.7, 3.7,
            3.77, 5.28, 28.95])

example = ['menton', 'exp', 'sigmoid'][:]

if 'menton' in example:
    def func_m(params, x):
        a, b = params
        return a * x / (np.exp(b) + x)

    nobs2 = 30
    sig_e = 0.75
    np.random.seed(456753)
    x = np.random.uniform(0, 10, size=nobs2)
    x.sort()
    beta_m = [10, 1]
    y_true = func_m(beta_m, x)
    y = y_true + np.random.randn(nobs2)
    y[-8::2] += 5





    modm = MentenNL(y, x)
    resm0 = modm.fit([0.5, 0.5, 0.5], method='nm')
    resm = modm.fit(resm0.params, method='bfgs')
    print(resm.summary())
    fittedvalues = resm.predict()
    resid = y - fittedvalues

    from scipy.optimize import leastsq
    res_cf = leastsq(lambda p: y - func_m(p,x), x0=[0.5,0.5])
    fitted_ls = func_m(res_cf[0], x)



    modm2 = MentenNL(y, x, norm=rnorms.TukeyBiweight())
    modm2.scale_fixed = rscale.mad(resid, center=0)
    resm2 = modm2.fit(resm.params[:-1], method='bfgs')
    store = []
    for i in range(10):
        fittedvalues2 = resm2.predict()
        modm2.scale_fixed = rscale.mad(y - fittedvalues2, center=0)
        resm2 = modm2.fit(resm.params[:-1], method='bfgs')
        store.append(resm2.params)



####################### mostly copy and paste

if 'exp' in example:
    func_m = lambda beta, x: np.exp(beta[0] + np.dot(x, beta[1:]))
    func_m = lambda beta, x: np.exp(np.dot(x, beta))

    nobs2 = 30
    sig_e = 10
    np.random.seed(456753)
    x = np.random.uniform(0, 10, size=nobs2)
    x.sort()
    exog = np.column_stack((np.ones(x.shape[0]), x))
    beta_m = [-5, 0.75]
    y_true = func_m(beta_m, exog)
    y = y_true + np.random.randn(nobs2)
    y[-8::2] += 15

    from scipy.optimize import leastsq
    res_cf = leastsq(lambda p: y - func_m(p,exog), x0=[0.5,0.5])
    fitted_ls = func_m(res_cf[0], exog)

    modm = ExpNL(y, exog)
    resm0 = modm.fit([0.5, 0.5, 0.5], method='nm')
    resm = modm.fit(resm0.params, method='bfgs')
    print(resm.summary())
    fittedvalues = resm.predict()
    resid = y - fittedvalues



    modm2 = ExpNL(y, exog, norm=rnorms.TukeyBiweight())
    modm2.scale_fixed = rscale.mad(resid, center=0)
    resm2 = modm2.fit(resm.params[:-1], method='nm')
    store = []
    for i in range(10):
        fittedvalues2 = resm2.predict()
        modm2.scale_fixed = rscale.mad(y - fittedvalues2, center=0)
        resm2 = modm2.fit(resm2.params, method='bfgs')
        store.append(np.concatenate((resm2.params, [modm2.scale_fixed])))
    fittedvalues2 = resm2.predict()

##########################

if 'sigmoid' in example:

    func_m = sigmoid

    nobs2 = 50
    sig_e = 2
    seed = 456753
    seed = 139146
    seed = 581472
    seed = np.random.RandomState().randint(999999)
    print('seed', seed)
    np.random.seed(seed)
    x = np.random.uniform(0, 10, size=nobs2)
    x.sort()
    exog = np.column_stack((np.ones(x.shape[0]), x))
    x0, y0, c, k = 5, 8, 10, 1.
    beta_m = [x0, y0, c, k]
    y_true = func_m(beta_m, x)
    y = y_true + np.random.randn(nobs2)
    y[-10::2] += 10

    start = sig_start(y, x)

    from scipy.optimize import leastsq
    res_cf = leastsq(lambda p: y - func_m(p, x), x0=start)
    fitted_ls = func_m(res_cf[0], x)

    start1 = res_cf[0]
    modm = SigmoidNL(y, x)
    resm0 = modm.fit(np.concatenate((start1,[0.5])), method='nm')
    resm = modm.fit(resm0.params, method='bfgs')
    print(resm.summary())
    fittedvalues = resm.predict()
    resid = y - fittedvalues





    modm2 = SigmoidNL(y, x, norm=rnorms.TukeyBiweight())
    modm2.scale_fixed = rscale.mad(resid, center=0)
    resm2 = modm2.fit(resm.params[:-1], method='nm')
    store = []
    for i in range(10):
        fittedvalues2 = resm2.predict()
        modm2.scale_fixed = rscale.mad(y - fittedvalues2, center=0)
        resm2 = modm2.fit(resm2.params, method='bfgs')
        store.append(np.concatenate((resm2.params, [modm2.scale_fixed])))
    fittedvalues2 = resm2.predict()
##########################

print(np.array(store))


import matplotlib.pyplot as plt
#fig = plt.figure()
#ax = fig.add_subplot()
plt.plot(x, y, 'o')
plt.plot(x, y_true, '-', color='b', lw=2, alpha = 0.75, label='true')
plt.plot(x, fittedvalues, '-', lw=2, alpha = 0.75, label='fit_robust')
plt.plot(x, fittedvalues2, '-', lw=2, alpha = 0.75, label='fit_robust_bs')
plt.plot(x, fitted_ls, '-', lw=2, alpha = 0.75, label='fit_leastsq')
plt.legend(loc='upper left')
plt.title('Robust Nonlinear M-Estimation - outliers')
#plt.title('Robust Nonlinear M-Estimation - no outliers')
plt.show()


'''Initially we planned this year to focus on maintenance and skip GSOC, even though it's the tenth. However, currently we have a student interested in the high priority "maintenance" project (preparing a backlog of pull requests for merge and increase test coverage). Additionally, we have a statistics professor that started to contribute last year, who is interested in mentoring a statistics enhancement.
'''

def predict_jac(model, params_loc, exog=None):
    if exog is None:
        exog = model.exog
    from statsmodels.tools.numdiff import approx_fprime
    return approx_fprime(params_loc, model.predict)


print(resm2.model.score(resm2.params))
print(resm2.model.predict_jac(resm2.params).sum(0))
print(resm2.model._predict_jac(resm2.params).sum(0))
print(predict_jac(modm2, resm2.params, exog=None).sum(0))


resm_it = modm2.fit_iterative(resm.params[:-1], method='nm')
print(resm_it.params)
print(resm2.params)

modm_biw = SigmoidNL(y, x, norm=rnorms.TukeyBiweight())
resm_biw = modm_biw.fit(start_params=resm_it.history[-1]) #history includes scale
print(resm_biw.params)

s_params = resm_it.history[-1]
print(resm_biw.model.score(s_params))
print(resm_biw.model.predict_jac(s_params).sum(0))
print(resm_biw.model._predict_jac(s_params).sum(0))
print(predict_jac(modm_biw, s_params, exog=None).sum(0))

'''
>>> from scipy import stats
>>> norm=rnorms.TukeyBiweight()
>>> stats.norm.expect(lambda t: t*norm.psi(t) - norm.rho(t))
3.9791304551565183
'''

''' with corrected norm, branch fix_robust_scale
>>> norm=rnorms.TukeyBiweight()
>>> from scipy import stats
>>> stats.norm.expect(lambda t: t*norm.psi(t) - norm.rho(t))
0.3209262884898523
'''
