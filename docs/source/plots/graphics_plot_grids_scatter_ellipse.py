# -*- coding: utf-8 -*-
"""
Create of grid of scatter plots with confidence ellipses from the statecrime
dataset
"""

import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.graphics.plot_grids import scatter_ellipse

data = sm.datasets.statecrime.load_pandas().data
fig = plt.figure(figsize=(8, 8))
scatter_ellipse(data, varnames=data.columns, fig=fig)
plt.show()
