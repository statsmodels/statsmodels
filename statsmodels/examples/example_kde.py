
from __future__ import print_function
import numpy as np
from scipy import stats
from statsmodels.distributions.mixture_rvs import mixture_rvs
from statsmodels.nonparametric.kde import kdensityfft
import matplotlib.pyplot as plt


np.random.seed(12345)
obs_dist = mixture_rvs([.25,.75], size=10000, dist=[stats.norm, stats.norm],
                kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))
#.. obs_dist = mixture_rvs([.25,.75], size=10000, dist=[stats.norm, stats.beta],
#..            kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=1,args=(1,.5))))


f_hat, grid, bw = kdensityfft(obs_dist, kernel="gauss", bw="scott")

# Check the plot

plt.figure()
plt.hist(obs_dist, bins=50, normed=True, color='red')
plt.plot(grid, f_hat, lw=2, color='black')
plt.show()

# do some timings
# get bw first because they're not streamlined
from statsmodels.nonparametric import bandwidths
bw = bandwidths.bw_scott(obs_dist)

#.. timeit kdensity(obs_dist, kernel="gauss", bw=bw, gridsize=2**10)
#.. timeit kdensityfft(obs_dist, kernel="gauss", bw=bw, gridsize=2**10)
