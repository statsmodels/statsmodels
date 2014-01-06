
## Kernel Density Estimation

import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.distributions.mixture_rvs import mixture_rvs


##### A univariate example.

np.random.seed(12345)


obs_dist1 = mixture_rvs([.25,.75], size=10000, dist=[stats.norm, stats.norm],
                kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))


kde = sm.nonparametric.KDEUnivariate(obs_dist1)
kde.fit()


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.hist(obs_dist1, bins=50, normed=True, color='red')
ax.plot(kde.support, kde.density, lw=2, color='black');


obs_dist2 = mixture_rvs([.25,.75], size=10000, dist=[stats.norm, stats.beta],
            kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=1,args=(1,.5))))

kde2 = sm.nonparametric.KDEUnivariate(obs_dist2)
kde2.fit()


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.hist(obs_dist2, bins=50, normed=True, color='red')
ax.plot(kde2.support, kde2.density, lw=2, color='black');


# The fitted KDE object is a full non-parametric distribution.

obs_dist3 = mixture_rvs([.25,.75], size=1000, dist=[stats.norm, stats.norm],
                kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))
kde3 = sm.nonparametric.KDEUnivariate(obs_dist3)
kde3.fit()


kde3.entropy


kde3.evaluate(-1)


##### CDF

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(kde3.support, kde3.cdf);


##### Cumulative Hazard Function

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(kde3.support, kde3.cumhazard);


##### Inverse CDF

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(kde3.support, kde3.icdf);


##### Survival Function

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(kde3.support, kde3.sf);

