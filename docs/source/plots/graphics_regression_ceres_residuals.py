# -*- coding: utf-8 -*-
'''
    Using a model built from the the state crime dataset, make a CERES plot
    with the rate of Poverty as the focus variable.
'''
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import plot_ceres_residuals

crime_data = sm.datasets.statecrime.load_pandas()
results = smf.ols('murder ~ hs_grad + urban + poverty + single',
                  data=crime_data.data).fit()
plot_ceres_residuals(results, 'poverty')
plt.show()
