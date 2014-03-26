
from __future__ import print_function
import numpy as np

from scipy import stats, special, optimize
import statsmodels.api as sm
from statsmodels.miscmodels import TLinearModel

#Example:
#np.random.seed(98765678)
nobs = 50
nvars = 6
df = 3
rvs = np.random.randn(nobs, nvars-1)
data_exog = sm.add_constant(rvs, prepend=False)
xbeta = 0.9 + 0.1*rvs.sum(1)
data_endog = xbeta + 0.1*np.random.standard_t(df, size=nobs)
print('variance of endog:', data_endog.var())
print('true parameters:', [0.1]*nvars + [0.9])

res_ols = sm.OLS(data_endog, data_exog).fit()
print('\nResults with ols')
print('----------------')
print(res_ols.scale)
print(np.sqrt(res_ols.scale))
print(res_ols.params)
print(res_ols.bse)
kurt = stats.kurtosis(res_ols.resid)
df_fromkurt = 6./kurt + 4
print('df_fromkurt from ols residuals', df_fromkurt)
print(stats.t.stats(df_fromkurt, moments='mvsk'))
print(stats.t.stats(df, moments='mvsk'))

modp = TLinearModel(data_endog, data_exog)
start_value = 0.1*np.ones(data_exog.shape[1]+2)
#start_value = np.zeros(data_exog.shape[1]+2)
#start_value[:nvars] = sm.OLS(data_endog, data_exog).fit().params
start_value[:nvars] = res_ols.params
start_value[-2] = df_fromkurt #10
start_value[-1] = np.sqrt(res_ols.scale) #0.5
modp.start_params = start_value

#adding fixed parameters

fixdf = np.nan * np.zeros(modp.start_params.shape)
fixdf[-2] = 5

fixone = 0
if fixone:
    modp.fixed_params = fixdf
    modp.fixed_paramsmask = np.isnan(fixdf)
    modp.start_params = modp.start_params[modp.fixed_paramsmask]
else:
    modp.fixed_params = None
    modp.fixed_paramsmask = None


print('\nResults with TLinearModel')
print('-------------------------')
resp = modp.fit(start_params = modp.start_params, disp=1, method='nm',
                maxfun=10000, maxiter=5000)#'newton')
#resp = modp.fit(start_params = modp.start_params, disp=1, method='newton')

print('using Nelder-Mead')
print(resp.params)
print(resp.bse)
resp2 = modp.fit(start_params = resp.params, method='Newton')
print('using Newton')
print(resp2.params)
print(resp2.bse)

from statsmodels.tools.numdiff import approx_fprime, approx_hess

hb=-approx_hess(modp.start_params, modp.loglike, epsilon=-1e-4)
tmp = modp.loglike(modp.start_params)
print(tmp.shape)
print('eigenvalues of numerical Hessian')
print(np.linalg.eigh(np.linalg.inv(hb))[0])

#store_params is only available in original test script
##pp=np.array(store_params)
##print pp.min(0)
##print pp.max(0)


