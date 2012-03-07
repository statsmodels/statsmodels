# -*- coding: utf-8 -*-
"""Out of sample prediction
"""

import numpy as np
import statsmodels.api as sm

#Create some data

nsample = 50
sig = 0.25
x1 = np.linspace(0, 20, nsample)
X = np.c_[x1, np.sin(x1), (x1-5)**2, np.ones(nsample)]
beta = [0.5, 0.5, -0.02, 5.]
y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)

#Setup and estimate the model

olsmod = sm.OLS(y, X)
olsres = olsmod.fit()
print olsres.params
print olsres.bse

#In-sample prediction

ypred = olsres.predict(X)

#Create a new sample of explanatory variables Xnew, predict and plot

x1n = np.linspace(20.5,25, 10)
Xnew = np.c_[x1n, np.sin(x1n), (x1n-5)**2, np.ones(10)]
ynewpred =  olsres.predict(Xnew) # predict out of sample
print ypred

import matplotlib.pyplot as plt
plt.figure()
plt.plot(x1, y, 'o', x1, y_true, 'b-')
plt.plot(np.hstack((x1, x1n)), np.hstack((ypred, ynewpred)),'r')
#@savefig ols_predict.png
plt.title('OLS prediction, blue: true and data, fitted/predicted values:red')
