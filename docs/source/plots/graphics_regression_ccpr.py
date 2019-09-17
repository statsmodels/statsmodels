# -*- coding: utf-8 -*-
'''
Using the state crime dataset plot the effect of the rate of single
households ('single') on the murder rate while accounting for high school
graduation rate ('hs_grad'), percentage of people in an urban area, and rate
of poverty ('poverty').

'''

import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

crime_data = sm.datasets.statecrime.load_pandas()
results = smf.ols('murder ~ hs_grad + urban + poverty + single',
                  data=crime_data.data).fit()
sm.graphics.plot_ccpr(results, 'single')
plt.show()
