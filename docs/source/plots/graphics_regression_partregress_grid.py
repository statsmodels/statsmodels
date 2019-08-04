# -*- coding: utf-8 -*-
'''
Using the state crime dataset separately plot the effect of the each
variable on the on the outcome, murder rate while accounting for the effect
of all other variables in the model visualized with a grid of partial
regression plots.

'''

from statsmodels.graphics.regressionplots import plot_partregress_grid
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

fig = plt.figure(figsize=(8, 6))
crime_data = sm.datasets.statecrime.load_pandas()
results = smf.ols('murder ~ hs_grad + urban + poverty + single',
                  data=crime_data.data).fit()
plot_partregress_grid(results, fig=fig)
plt.show()
