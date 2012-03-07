"""Ordinary Least Squares
"""

from statsmodels.datasets.longley import load
import statsmodels.api as sm
import numpy as np

data = load()
data.exog = sm.tools.add_constant(data.exog)

ols_model = sm.OLS(data.endog, data.exog)
ols_results = ols_model.fit()

# the Longley dataset is well known to have high multicollinearity
# one way to find the condition number is as follows

# normalize the independent variables to have unit length, Greene 4.9
norm_x = np.ones_like(data.exog)
for i in range(int(ols_model.df_model)):
    norm_x[:,i] = data.exog[:,i]/np.linalg.norm(data.exog[:,i])
norm_xtx = np.dot(norm_x.T,norm_x)
eigs = np.linalg.eigvals(norm_xtx)
collin = np.sqrt(eigs.max()/eigs.min())
print collin
# clearly there is a big problem with multicollinearity
# the rule of thumb is any number of 20 requires attention

# for instance, consider the longley dataset with the last observation dropped
ols_results2 = sm.OLS(data.endog[:-1], data.exog[:-1,:]).fit()

# all of our coefficients change considerably in percentages
# of the original coefficients
print "Percentage change %4.2f%%\n"*7 % tuple([i for i in ols_results.params/ols_results2.params*100 - 100])





