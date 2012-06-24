"""Ordinary Least Squares
"""
import numpy as np
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# Fix a random seed for these examples
np.random.seed(9876789)

#OLS Estimation 
#--------------

#Artificial data
#^^^^^^^^^^^^^^^^
nsample = 100
x = np.linspace(0, 10, 100)
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)
y = np.dot(X, beta) + e
X = np.column_stack((x, x**2))

# Our model needs an intercept so we add a column of 1s:
X = sm.add_constant(X)

# Inspect data
print X[:5,:]
print y[:5]

#Fit and summary
#^^^^^^^^^^^^^^^

model = sm.OLS(y, X)
results = model.fit()
print results.summary()

# Quantities of interest can be extracted directly from the fitted model. Type
# ``dir(results)`` for a full list. Here are some examples:  
print results.params
print results.rsquared

#OLS non-linear curve but linear in parameters
#---------------------------------------------

#Artificial data 
#^^^^^^^^^^^^^^^
# Non-linear relationship between x and y 
nsample = 50
sig = 0.5
x = np.linspace(0, 20, nsample)
X = np.c_[x, np.sin(x), (x-5)**2, np.ones(nsample)]
beta = [0.5, 0.5, -0.02, 5.]
y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)

# Fit and summary
#^^^^^^^^^^^^^^^^
res = sm.OLS(y, X).fit()
print res.summary()

# Extract other quantities of interest
print res.params
print res.bse
print res.predict()

# Draw a plot to compare the true relationship to OLS predictions. Confidence
# intervals around the predictions are built using the ``wls_prediction_std``
# command.
plt.figure()
plt.plot(x, y, 'o', x, y_true, 'b-')
prstd, iv_l, iv_u = wls_prediction_std(res)
plt.plot(x, res.fittedvalues, 'r--.')
plt.plot(x, iv_u, 'r--')
plt.plot(x, iv_l, 'r--')
plt.title('blue: true,   red: OLS')

#OLS with dummy variables
#------------------------

#Artificial data 
#^^^^^^^^^^^^^^^^
# We create 3 groups which will be modelled using dummy variables. Group 0 is
# the omitted/benchmark category.
nsample = 50
groups = np.zeros(nsample, int)
groups[20:40] = 1
groups[40:] = 2
dummy = (groups[:,None] == np.unique(groups)).astype(float)
x = np.linspace(0, 20, nsample)
X = np.c_[x, dummy[:,1:], np.ones(nsample)]
beta = [1., 3, -3, 10]
y_true = np.dot(X, beta)
e = np.random.normal(size=nsample)
y = y_true + e

# Inspect the data
print X[:5,:]
print y[:5]
print groups
print dummy[:5,:]

#Fit and summary
#^^^^^^^^^^^^^^^
res2 = sm.OLS(y, X).fit()
print res.summary()
print res2.params
print res2.bse
print res.predict()

# Draw a plot to compare the true relationship to OLS predictions. 
prstd, iv_l, iv_u = wls_prediction_std(res2)
plt.figure()
plt.plot(x, y, 'o', x, y_true, 'b-')
plt.plot(x, res2.fittedvalues, 'r--.')
plt.plot(x, iv_u, 'r--')
plt.plot(x, iv_l, 'r--')
plt.title('blue: true,   red: OLS')

#Joint hypothesis tests
#----------------------

#F test
#^^^^^^
# We want to test the hypothesis that both coefficients on the dummy variables
# are equal to zero, that is, :math:`R \times \beta = 0`. An F test leads us to
# strongly reject the null hypothesis of identical constant in the 3 groups:

R = [[0, 1, 0, 0], [0, 0, 1, 0]]
print np.array(R)
print res2.f_test(R)

#T test
#^^^^^^
# We want to test the null hypothesis that the effects of the 2nd and 3rd
# groups add to zero. The T-test allows us to reject the Null (but note the
# one-sided p-value): 
R = [0, 1, -1, 0]
print res2.t_test(R)

#Small group effects
#^^^^^^^^^^^^^^^^^^^ 
# If we generate artificial data with smaller group effects, the T test can no
# longer reject the Null hypothesis: 
beta = [1., 0.3, -0.0, 10]
y_true = np.dot(X, beta)
y = y_true + np.random.normal(size=nsample)
res3 = sm.OLS(y, X).fit()
print res3.f_test(R)

#Multicollinearity
#-----------------

#Data
#^^^^
# The Longley dataset is well known to have high multicollinearity. 
from statsmodels.datasets.longley import load
y = load().endog
X = load().exog
X = sm.tools.add_constant(X)

#Fit and summary
#^^^^^^^^^^^^^^^
ols_model = sm.OLS(y, X)
ols_results = ols_model.fit()
print ols_results.summary()

#Condition number
#^^^^^^^^^^^^^^^^
# One way to assess multicollinearity is to compute the condition number.
# Values over 20 are worrisome (see Greene 4.9). The first step is to normalize
# the independent variables to have unit length: 
norm_x = np.ones_like(X)
for i in range(int(ols_model.df_model)):
    norm_x[:,i] = X[:,i]/np.linalg.norm(X[:,i])
norm_xtx = np.dot(norm_x.T,norm_x)

# Then, we take the square root of the ratio of the biggest to the smallest
# eigen values. 
eigs = np.linalg.eigvals(norm_xtx)
condition_number = np.sqrt(eigs.max() / eigs.min())
print condition_number

#Dropping an observation
#^^^^^^^^^^^^^^^^^^^^^^^
# Greene also points out that dropping a single observation can have a dramatic
# effect on the coefficient estimates: 
ols_results2 = sm.OLS(y[:-1], X[:-1,:]).fit()
print "Percentage change %4.2f%%\n"*7 % tuple([i for i in ols_results.params/ols_results2.params*100 - 100])

