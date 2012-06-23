"""Ordinary Least Squares
"""

# Load modules and the Longley dataset which ships with statsmodels
import statsmodels.api as sm
import numpy as np
from statsmodels.datasets.longley import load
data = load()

# Print the first 5 observations of the Longley exogenous variables matrix
# (i.e. independent variables) and the endogenous variable (i.e. dependent
# variable).
print data.exog[:5,:]
print data.endog[:5]

# Our model needs an intercept, so we add a column with a constant
data.exog = sm.tools.add_constant(data.exog)

# Fit the model and print summary
ols_model = sm.OLS(data.endog, data.exog)
ols_results = ols_model.fit()
print ols_results.summary()

# The Longley dataset is well known to have high multicollinearity. One way to
# find the condition number is to normalize the independent variables to have
# unit length (see Greene 4.9).
norm_x = np.ones_like(data.exog)
for i in range(int(ols_model.df_model)):
    norm_x[:,i] = data.exog[:,i]/np.linalg.norm(data.exog[:,i])
norm_xtx = np.dot(norm_x.T,norm_x)
eigs = np.linalg.eigvals(norm_xtx)
collin = np.sqrt(eigs.max()/eigs.min())
print collin

# There seems to be a big problem with multicollinearity; the rule of thumb is
# that any number above 20 requires attention. We consider the Longley dataset
# with the last observation dropped, and we compare the coefficient estimates
# in this new model to our original ones. The changes are considerable.
ols_results2 = sm.OLS(data.endog[:-1], data.exog[:-1,:]).fit()
print "Percentage change %4.2f%%\n"*7 % tuple([i for i in ols_results.params/ols_results2.params*100 - 100])

