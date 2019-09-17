# -*- coding: utf-8 -*-
'''
    In this example we just reuse the same correlation matrix several times.
    Of course in reality one would show a different correlation (measuring a
    another type of correlation, for example Pearson (linear) and Spearman,
    Kendall (nonlinear) correlations) for the same variables.
'''
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

hie_data = sm.datasets.randhie.load_pandas()
corr_matrix = np.corrcoef(hie_data.data.T)
sm.graphics.plot_corr_grid([corr_matrix] * 8, xnames=hie_data.names)
plt.show()
