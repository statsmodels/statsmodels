"""
Example: scikits.statsmodels.WLS
"""
import numpy as np
import scikits.statsmodels as sm
import matplotlib.pyplot as plt

data = sm.datasets.ccard.Load()
data.exog = sm.add_constant(data.exog)
ols_test_fit = sm.OLS(data.endog, data.exog).fit()

# perhaps the residuals from this fit depend on the square of income
incomesq = data.exog[:,2]
plt.scatter(incomesq, ols_test_fit.resid)
plt.grid()


# If we think that the variance is proportional to income**2
# we would want to weight the regression by income
# the weights argument in WLS weights the regression by its square root
# and since income enters the equation, if we have income/income
# it becomes the constant, so we would want to perform
# this type of regression without an explicit constant in the design

#data.exog = data.exog[:,:-1]
wls_fit = sm.WLS(data.endog, data.exog[:,:-1], weights=1/incomesq).fit()

# This however, leads to difficulties in interpreting the post-estimation
# statistics.  Statsmodels does not yet handle this elegantly, but
# the following may be more appropriate

# explained sum of squares
ess = wls_fit.uncentered_tss - wls_fit.ssr
# rsquared
rsquared = ess/wls_fit.uncentered_tss
# mean squared error of the model
mse_model = ess/(wls_fit.df_model + 1) # add back the dof of the constant
# f statistic
fvalue = mse_model/wls_fit.mse_resid
# adjusted r-squared
rsquared_adj = 1 -(wls_fit.nobs)/(wls_fit.df_resid)*(1-rsquared)


#JP: I need to look at this again. Even if I exclude the weight variable
# from the regressors and keep the constant in then the reported rsquared
# stays small
# need to add 45 degree line to graphs
wls_fit3 = sm.WLS(data.endog, data.exog[:,(0,1,3,4)], weights=1/incomesq).fit()
print wls_fit3.summary()
print 'corrected rsquared',
print (wls_fit3.uncentered_tss - wls_fit3.ssr)/wls_fit3.uncentered_tss
plt.figure()
plt.title('WLS dropping heteroscedasticity variable from regressors')
plt.plot(data.endog, wls_fit3.fittedvalues, 'o')
plt.xlim([0,2000])
plt.ylim([0,2000])
print 'raw correlation of endog and fittedvalues squared'
print np.corrcoef(data.endog, wls_fit.fittedvalues)**2

# compare with robust regression,
# heteroscedasticity correction downweights the outliers
rlm_fit = sm.RLM(data.endog, data.exog).fit()
plt.figure()
plt.title('using robust for comparison')
plt.plot(data.endog, rlm_fit.fittedvalues, 'o')
plt.xlim([0,2000])
plt.ylim([0,2000])
plt.show()
