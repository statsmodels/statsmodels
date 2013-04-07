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

data = sm.datasets.statecrime.load_pandas().data
murder = data['murder']
X = data[['poverty', 'hs_grad']]
X["constant"] = 1

y = murder
model = sm.OLS(y, X)
results = model.fit()

# Create a plot just for the variable 'Poverty':

fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(results, 0, ax=ax)
ax.set_ylabel("Murder Rate")
ax.set_xlabel("Poverty Level")
ax.set_title("Linear Regression")

plt.show()
