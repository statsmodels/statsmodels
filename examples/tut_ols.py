'''Examples OLS

Note: uncomment plt.show() to display graphs
'''

import numpy as np
#from scipy import stats
import statsmodels.api as sm
import matplotlib
#matplotlib.use('Qt4Agg')#, warn=True)   #for Spyder
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

#fix a seed for these examples
np.random.seed(9876789)

# OLS non-linear curve but linear in parameters
# ---------------------------------------------

nsample = 50
sig = 0.5
x1 = np.linspace(0, 20, nsample)
X = np.c_[x1, np.sin(x1), (x1-5)**2, np.ones(nsample)]
beta = [0.5, 0.5, -0.02, 5.]
y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)

plt.figure()
plt.plot(x1, y, 'o', x1, y_true, 'b-')

res = sm.OLS(y, X).fit()
print res.params
print res.bse
#current bug predict requires call to model.results
#print res.model.predict
prstd, iv_l, iv_u = wls_prediction_std(res)
plt.plot(x1, res.fittedvalues, 'r--.')
plt.plot(x1, iv_u, 'r--')
plt.plot(x1, iv_l, 'r--')
plt.title('blue: true,   red: OLS')

print res.summary()


#OLS with dummy variables
#------------------------

sig = 1.
#suppose observations from 3 groups
xg = np.zeros(nsample, int)
xg[20:40] = 1
xg[40:] = 2
print xg
dummy = (xg[:,None] == np.unique(xg)).astype(float)
#use group 0 as benchmark
X = np.c_[x1, dummy[:,1:], np.ones(nsample)]
beta = [1., 3, -3, 10]
y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)

plt.figure()
plt.plot(x1, y, 'o', x1, y_true, 'b-')
plt.figure()
plt.plot(x1, y, 'o', x1, y_true, 'b-')
res2 = sm.OLS(y, X).fit()
print res2.params
print res2.bse
#current bug predict requires call to model.results
#print res.model.predict
prstd, iv_l, iv_u = wls_prediction_std(res2)
plt.plot(x1, res2.fittedvalues, 'r--.')
plt.plot(x1, iv_u, 'r--')
plt.plot(x1, iv_l, 'r--')
plt.title('blue: true,   red: OLS')

#print res.summary()

R = [[0, 1, 0, 0],
     [0, 0, 1, 0]]

# F test joint hypothesis R * beta = 0
# i.e. coefficient on both dummy variables equal zero
print res2.f_test(R)
# strongly rejected Null of identical constant in 3 groups
#<F test: F=124.19050615860911, p=2.87411973729e-019, df_denom=46, df_num=2>
# see also: help(res2.f_test)

# t test for Null hypothesis effects of 2nd and 3rd group add to zero
R = [0, 1, -1, 0]
print res2.t_test(R)
# don't reject Null at 5% confidence level (note one sided p-value)
#<T test: effect=1.0363792917100714, sd=0.52675137730463362, t=1.9674923243925513, p=0.027586676754860262, df_denom=46>


# OLS with small group effects

beta = [1., 0.3, -0.0, 10]
y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)
res3 = sm.OLS(y, X).fit()
print res3.f_test(R)
# don't reject Null of identical constant in 3 groups
#<F test: F=1.9715385826285652, p=0.15083366806, df_denom=46, df_num=2>


#plt.draw()
#plt.show()
