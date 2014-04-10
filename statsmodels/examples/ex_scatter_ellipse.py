'''example for grid of scatter plots with probability ellipses


Author: Josef Perktold
License: BSD-3
'''


from statsmodels.compat.python import lrange
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.graphics.plot_grids import scatter_ellipse


nvars = 6
mmean = np.arange(1.,nvars+1)/nvars * 1.5
rho = 0.5
#dcorr = rho*np.ones((nvars, nvars)) + (1-rho)*np.eye(nvars)
r = np.random.uniform(-0.99, 0.99, size=(nvars, nvars))
##from scipy import stats
##r = stats.rdist.rvs(1, size=(nvars, nvars))
r = (r + r.T) / 2.
assert np.allclose(r, r.T)
mcorr = r
mcorr[lrange(nvars), lrange(nvars)] = 1
#dcorr = np.array([[1, 0.5, 0.1],[0.5, 1, -0.2], [0.1, -0.2, 1]])
mstd = np.arange(1.,nvars+1)/nvars
mcov = mcorr * np.outer(mstd, mstd)
evals = np.linalg.eigvalsh(mcov)
assert evals.min > 0 #assert positive definite

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
