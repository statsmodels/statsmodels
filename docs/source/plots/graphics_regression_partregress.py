# -*- coding: utf-8 -*-
'''
Load the statewide dataset and plot partial regression of high school
graduation on murder rate after removing the effects of rate of urbanization,
poverty, and rate of single household.

'''

import matplotlib.pyplot as plt

import statsmodels.api as sm

crime_data = sm.datasets.statecrime.load_pandas()
sm.graphics.plot_partregress(endog='murder', exog_i='hs_grad',
                             exog_others=['urban', 'poverty', 'single'],
                             data=crime_data.data, obs_labels=False)
plt.show()
