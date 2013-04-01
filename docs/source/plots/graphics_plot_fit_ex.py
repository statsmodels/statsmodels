# -*- coding: utf-8 -*-
"""

Created on Monday April 1st 2013

Author: Padarn Wilson

"""

# Load the Statewide Crime data set and perform linear regression with 
#    'poverty' and 'hs_grad' as variables and 'muder' as the response


import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

data = sm.datasets.statecrime.load()
murder = data['data']['murder']
poverty = data['data']['poverty']
hs_grad = data['data']['hs_grad']

X = np.column_stack((poverty, hs_grad))
X = sm.add_constant(X, prepend=False)
y = murder
model = sm.OLS(y, X)
results = model.fit()

# Create a plot just for the variable 'Poverty':

fig = plt.figure()
ax = fig.add_subplot(111)
res = sm.graphics.plot_fit(results, 0, ax=ax)
ax.set_ylabel("Murder Rate")
ax.axes.set_xlabel("Poverty Level")
ax.axes.set_title("Linear Regression")

plt.show()
