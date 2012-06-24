'''OLS Example with joint significance tests
'''

import numpy as np
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
#..from scipy import stats
#..matplotlib.use('Qt4Agg')#, warn=True)   #for Spyder

# Fix a random seed for these examples
np.random.seed(9876789)

#OLS non-linear curve but linear in parameters
#---------------------------------------------

# Artificial data: non-linear relationship between x and y 
nsample = 50
sig = 0.5
x = np.linspace(0, 20, nsample)
X = np.c_[x, np.sin(x), (x-5)**2, np.ones(nsample)]
beta = [0.5, 0.5, -0.02, 5.]
y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)

# Fit OLS model and print results
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

# Artificial data: 3 groups will be modelled using dummy variables. Group 0 is
# the omitted/benchmark category.
nsample = 50
groups = np.zeros(nsample, int)
groups[20:40] = 1
groups[40:] = 2
dummy = (groups[:,None] == np.unique(groups)).astype(float)
X = np.c_[x, dummy[:,1:], np.ones(nsample)]
beta = [1., 3, -3, 10]
y_true = np.dot(X, beta)
y = y_true + np.random.normal(size=nsample)

# Inspect the data
print X[:5,:]
print y[:5]
print groups
print dummy[:5,:]

# Fit OLS model, print results and other quantities of interest
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
print R
print res2.f_test(R)
print float(res2.f_test(R).pvalue)

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

