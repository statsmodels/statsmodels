"""
Example: scikits.statsmodels.WLS
"""
import scikits.statsmodels as sm
import pylab

data = sm.datasets.ccard.Load()
data.exog = sm.add_constant(data.exog)
ols_test_fit = sm.OLS(data.endog, data.exog).fit()

# perhaps the residuals from this fit depend on the square of income
incomesq = data.exog[:,2]
pylab.scatter(incomesq, ols_test_fit.resid)
pylab.grid()
#pylab.show()

# If we think that the variance is proportional to income**2
# we would want to weight the regression by income
# the weights argument in WLS weights the regression by its square root
# and since income enters the equation, if we have income/income
# it becomes the constant, so we would want to perform
# this type of regression without an explicit constant in the design

data.exog = data.exog[:,:-1]
wls_fit = sm.WLS(data.endog, data.exog, weights=1/incomesq).fit()

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
