'''Examples: comparing OLS and RLM

robust estimators and outliers

'''

import numpy as np
#from scipy import stats
import scikits.statsmodels as sm
import matplotlib.pyplot as plt
from predstd import wls_prediction_std


nsample = 50
x1 = np.linspace(0, 20, nsample)
X = np.c_[x1, (x1-5)**2, np.ones(nsample)]

sig = 0.5
beta = [0.5, -0.0, 5.]
y_true2 = np.dot(X, beta)
y2 = y_true2 + sig*1. * np.random.normal(size=nsample)
y2[[39,41,43,45,48]] -= 5   # add some outliers

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
# Note different spelling fitted_values with underline
plt.plot(x1, resrlm.fitted_values, 'g.-')
plt.title('blue: true,   red: OLS,   green: RLM')

plt.show()
