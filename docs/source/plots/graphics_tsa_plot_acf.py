# -*- coding: utf-8 -*-
'''
    Load Sunspots Data and plot the autocorrelation of the number of sunspots
    per year.
'''

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

dta = sm.datasets.sunspots.load_pandas().data
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del dta["YEAR"]
sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40)
plt.show()
