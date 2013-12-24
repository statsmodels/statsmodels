# -*- coding: utf-8 -*-
"""
Author: Padarn Wilson

Plot an example of the available properties of a kernel density estimate.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.nonparametric.api as npar
from statsmodels.sandbox.nonparametric import kernels
from statsmodels.distributions.mixture_rvs import mixture_rvs

# example from test_kde.py mixture of two normal distributions
np.random.seed(12345)
x = mixture_rvs([.25,.75], size=200, dist=[stats.norm, stats.norm],
                kwargs = (dict(loc=-1, scale=.5),dict(loc=1, scale=.5)))

kde = npar.KDEUnivariate(x)
kde.fit('gau')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.hist(x, bins=15, normed=True, alpha=0.25)
ax.plot(kde.support, kde.density, lw=2, color='red')
ax.set_title('Kernel Density Gaussian (bw = %4.2f)' % kde.bw)

x_grid = np.linspace(np.min(x), np.max(x), 51)
x_grid = np.linspace(-3, 3, 51)

function_names = ['cdf','cumhazard','sf','icdf']

fig = plt.figure()
for ii, func in enumerate(function_names):
    ax = fig.add_subplot(2, 2, ii+1)

    cur_func = getattr(kde, func)
    ax.plot(kde.support, cur_func, lw=2, color='red')

    ax.set_title(func)

plt.show()
