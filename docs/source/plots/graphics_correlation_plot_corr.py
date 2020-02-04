# -*- coding: utf-8 -*-
"""
Create a plot of correlation among many variables in a grid

"""

import matplotlib.pyplot as plt
import numpy as np

import statsmodels.api as sm
import statsmodels.graphics.api as smg

hie_data = sm.datasets.randhie.load_pandas()
corr_matrix = np.corrcoef(hie_data.data.T)
smg.plot_corr(corr_matrix, xnames=hie_data.names)
plt.show()
