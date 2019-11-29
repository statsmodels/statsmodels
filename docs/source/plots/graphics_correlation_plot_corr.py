# -*- coding: utf-8 -*-
"""
Create a plot of correlation among many variables in a grid

"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.api as smg
import statsmodels.api as sm

hie_data = sm.datasets.randhie.load_pandas()
corr_matrix = np.corrcoef(hie_data.data.T)
smg.plot_corr(corr_matrix, xnames=hie_data.names)
plt.show()
