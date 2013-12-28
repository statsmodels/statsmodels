# -*- coding: utf-8 -*-
"""
Author: Padarn Wilson

Performance of normal reference plug-in estimator vs silverman. Plots the kde
estimate based on 200 pts, and a histogram based on 4000 to give an idea of the
true density.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.nonparametric.api as npar
from statsmodels.sandbox.nonparametric import kernels
from statsmodels.distributions.mixture_rvs import mixture_rvs

# example from test_kde.py mixture of two normal distributions
np.random.seed(12345)
x = mixture_rvs([.25, .75], size=200, dist=[stats.norm, stats.norm],
                kwargs=(dict(loc=-1, scale=.5), dict(loc=1, scale=.5)))

# create a second data set to give a better view of true distribution
x2 = mixture_rvs([.25, .75], size=4000, dist=[stats.norm, stats.norm],
                kwargs=(dict(loc=-1, scale=.5), dict(loc=1, scale=.5)))

kde = npar.KDEUnivariate(x)


kernel_names = ['Gaussian', 'Epanechnikov', 'Biweight',
                'Triangular', 'Triweight', 'Cosine'
                ]

kernel_switch = ['gau', 'epa', 'tri', 'biw',
                 'triw', 'cos'
                 ]

fig = plt.figure()
for ii, kn in enumerate(kernel_switch):

    ax = fig.add_subplot(2, 3, ii + 1)   # without uniform
    ax.hist(x2, bins=40, normed=True, alpha=0.25)

    kde.fit(kernel=kn, bw='silverman', fft=False)
    ax.plot(kde.support, kde.density)

    kde.fit(kernel=kn, bw='normal_reference', fft=False)
    ax.plot(kde.support, kde.density)

    ax.set_title(kernel_names[ii])


ax.legend(['silverman', 'normal reference'], loc='lower right')
plt.show()