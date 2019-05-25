'''example for grid of scatter plots with probability ellipses


Author: Josef Perktold
License: BSD-3
'''
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.graphics.plot_grids import scatter_ellipse


nvars = 6
mmean = np.arange(1.,nvars+1)/nvars * 1.5

# Randomly generate a valid covariance matrix with shape (nvars, nvars)
#  Do this by writing our 6 variables as linear combinations of 6 independent
#  normally distributed variables random weights.
weights = np.random.randn(nvars**2).reshape(nvars, nvars)
mcov = weights.dot(weights.T)
evals = np.linalg.eigvalsh(mcov)
assert evals.min() > 0, evals.min()  # assert positive definite

nobs = 100
data = np.random.multivariate_normal(mmean, mcov, size=nobs)
dmean = data.mean(0)
dcov = np.cov(data, rowvar=0)
print(dmean)
print(dcov)
dcorr = np.corrcoef(data, rowvar=0)
dcorr[np.triu_indices(nvars)] = 0
print(dcorr)

#default
#fig = scatter_ellipse(data, level=[0.5, 0.75, 0.95])
#used for checking
#fig = scatter_ellipse(data, level=[0.5, 0.75, 0.95], add_titles=True, keep_ticks=True)
#check varnames
varnames = ['var%d' % i for i in range(nvars)]
fig = scatter_ellipse(data, level=0.9, varnames=varnames)
plt.show()
