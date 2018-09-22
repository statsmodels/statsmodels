# -*- coding: utf-8 -*-
'''
    Using a model built from the the state crime dataset, plot the influence in
    regression.  Observations with high leverage, or large residuals will be
    labeled in the plot to show potential influence points.
'''
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

crime_data = sm.datasets.statecrime.load_pandas()
results = smf.ols('murder ~ hs_grad + urban + poverty + single',
                  data=crime_data.data).fit()
sm.graphics.influence_plot(results)
plt.show()
