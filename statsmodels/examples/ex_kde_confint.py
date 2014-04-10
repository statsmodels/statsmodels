# -*- coding: utf-8 -*-
"""

Created on Mon Dec 16 11:02:59 2013

Author: Josef Perktold
"""

from __future__ import print_function
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

x.sort() # not needed

kde = npar.KDEUnivariate(x)
kde.fit('gau')
ci = kde.kernel.density_confint(kde.density, len(x))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.hist(x, bins=15, normed=True, alpha=0.25)
ax.plot(kde.support, kde.density, lw=2, color='red')
ax.fill_between(kde.support, ci[:,0], ci[:,1],
                    color='grey', alpha='0.7')
ax.set_title('Kernel Density Gaussian (bw = %4.2f)' % kde.bw)


# use all kernels directly

x_grid = np.linspace(np.min(x), np.max(x), 51)
x_grid = np.linspace(-3, 3, 51)

kernel_names = ['Biweight', 'Cosine', 'Epanechnikov', 'Gaussian',
                'Triangular', 'Triweight', #'Uniform',
                ]

fig = plt.figure()
for ii, kn in enumerate(kernel_names):
    ax = fig.add_subplot(2, 3, ii+1)   # without uniform
    ax.hist(x, bins=10, normed=True, alpha=0.25)
    #reduce bandwidth for Gaussian and Uniform which are to large in example
    if kn in ['Gaussian', 'Uniform']:
        args = (0.5,)
    else:
        args = ()
    kernel = getattr(kernels, kn)(*args)

    kde_grid = [kernel.density(x, xi) for xi in x_grid]
    confint_grid = kernel.density_confint(kde_grid, len(x))

    ax.plot(x_grid, kde_grid, lw=2, color='red', label=kn)
    ax.fill_between(x_grid, confint_grid[:,0], confint_grid[:,1],
                    color='grey', alpha='0.7')
    ax.legend(loc='upper left')

plt.show()
