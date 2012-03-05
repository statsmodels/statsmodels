'''Examples: comparing OLS and RLM

robust estimators and outliers

Note: uncomment plt.show() to display graphs
'''

import numpy as np
#from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std


nsample = 50
x1 = np.linspace(0, 20, nsample)
X = np.c_[x1, (x1-5)**2, np.ones(nsample)]

sig = 0.3   # smaller error variance makes OLS<->RLM contrast bigger
beta = [0.5, -0.0, 5.]
y_true2 = np.dot(X, beta)
y2 = y_true2 + sig*1. * np.random.normal(size=nsample)
y2[[39,41,43,45,48]] -= 5   # add some outliers (10% of nsample)

#Example: estimate quadratic function (true is linear)

plt.figure()
plt.plot(x1, y2, 'o', x1, y_true2, 'b-')

res = sm.OLS(y2, X).fit()
print res.params
# Note: quadratic term captures outlier effect
print res.bse
#print res.predict
#plt.plot(x1, res.fittedvalues, 'r--')
prstd, iv_l, iv_u = wls_prediction_std(res)
plt.plot(x1, res.fittedvalues, 'r-')
plt.plot(x1, iv_u, 'r--')
plt.plot(x1, iv_l, 'r--')


#compare with robust estimator

resrlm = sm.RLM(y2, X).fit()
print resrlm.params
print resrlm.bse
# Note different spelling fittedvalues with underline -> corrected spelling
plt.plot(x1, resrlm.fittedvalues, 'g.-')
plt.title('blue: true,   red: OLS,   green: RLM')


# Example: estimate linear function (true is linear)

X2 = X[:,[0,2]]  # use only linear term and constant
plt.figure()
plt.plot(x1, y2, 'o', x1, y_true2, 'b-')


res2 = sm.OLS(y2, X2).fit()
print res2.params
# Note: quadratic term captures outlier effect
print res2.bse
#print res2.predict
prstd, iv_l, iv_u = wls_prediction_std(res2)
plt.plot(x1, res2.fittedvalues, 'r-')
plt.plot(x1, iv_u, 'r--')
plt.plot(x1, iv_l, 'r--')


#compare with robust estimator

resrlm2 = sm.RLM(y2, X2).fit()
print resrlm2.params
print resrlm2.bse
# Note different spelling fittedvalues with underline -> corrected spelling
plt.plot(x1, resrlm2.fittedvalues, 'g.-')
plt.title('blue: true,   red: OLS,   green: RLM')


# see also help(sm.RLM.fit) for more options and
# module sm.robust.scale for scale options

#plt.show()
