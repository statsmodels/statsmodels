# -*- coding: utf-8 -*-
'''
Using the state crime dataset separately plot the effect of the each
variable on the on the outcome, murder rate while accounting for the effect
of all other variables in the model.

'''

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

np.random.seed(12345)
X = sm.add_constant(np.random.normal(0, 20, size=30))
y = np.dot(X, [25, 3.5]) + np.random.normal(0, 30, size=30)
mod = sm.OLS(y, X).fit()
fig = sm.graphics.abline_plot(model_results=mod)
ax = fig.axes[0]
ax.scatter(X[:, 1], y)
ax.margins(.1)
plt.show()
