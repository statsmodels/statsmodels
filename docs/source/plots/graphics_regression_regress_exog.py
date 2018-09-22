# -*- coding: utf-8 -*-
'''
Load the Statewide Crime data set and build a model with regressors
including the rate of high school graduation (hs_grad), population in urban
areas (urban), households below poverty line (poverty), and single person
households (single).  Outcome variable is the muder rate (murder).

Build a 2 by 2 figure based on poverty showing fitted versus actual murder
rate, residuals versus the poverty rate, partial regression plot of poverty,
and CCPR plot for poverty rate.

'''

import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

fig = plt.figure(figsize=(8, 6))
crime_data = sm.datasets.statecrime.load_pandas()
results = smf.ols('murder ~ hs_grad + urban + poverty + single',
                  data=crime_data.data).fit()
sm.graphics.plot_regress_exog(results, 'poverty', fig=fig)
plt.show()
