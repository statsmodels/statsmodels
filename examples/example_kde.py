from scipy import stats
import numpy as np
from scikits.statsmodels.sandbox.distributions.mixture_rvs import mixture_rvs
from scikits.statsmodels.nonparametric.kde import (kdensity, kdensityfft)
import matplotlib.pyplot as plt

np.random.seed(12345)
obs_dist = mixture_rvs([.25,.75], size=10000, dist=[stats.norm, stats.norm],
                kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))
#obs_dist = mixture_rvs([.25,.75], size=10000, dist=[stats.norm, stats.beta],
#                kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=1,args=(1,.5))))


f_hat, grid, bw = kdensityfft(obs_dist, kernel="gauss", bw="scott")

# check the plot

plt.hist(obs_dist, bins=50, normed=True, color='red')
plt.plot(grid, f_hat, lw=2, color='black')
plt.show()

# do some timings
# get bw first because they're not streamlined
from scikits.statsmodels.nonparametric import bandwidths
bw = bandwidths.bw_scott(obs_dist)

#timeit kdensity(obs_dist, kernel="gauss", bw=bw, gridsize=2**10)
#timeit kdensityfft(obs_dist, kernel="gauss", bw=bw, gridsize=2**10)
