# -*- coding: utf-8 -*-
'''
Using the state crime dataset separately plot the effect of the each
variable on the on the outcome, murder rate while accounting for the effect
of all other variables in the model.

'''

import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

fig = plt.figure(figsize=(8, 8))
crime_data = sm.datasets.statecrime.load_pandas()
results = smf.ols('murder ~ hs_grad + urban + poverty + single',
                  data=crime_data.data).fit()
sm.graphics.plot_ccpr_grid(results, fig=fig)
plt.show()
