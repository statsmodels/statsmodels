# -*- coding: utf-8 -*-
'''
    Using a model built from the the state crime dataset, plot the leverage
    statistics vs. normalized residuals squared.  Observations with
    Large-standardized Residuals will be labeled in the plot.
'''

import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

crime_data = sm.datasets.statecrime.load_pandas()
results = smf.ols('murder ~ hs_grad + urban + poverty + single',
                  data=crime_data.data).fit()
sm.graphics.plot_leverage_resid2(results)
plt.show()
