
## Kernel Density Estimation

# In[ ]:

import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.distributions.mixture_rvs import mixture_rvs


##### A univariate example.

# In[ ]:

np.random.seed(12345)


# In[ ]:

obs_dist1 = mixture_rvs([.25,.75], size=10000, dist=[stats.norm, stats.norm],
                kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))


# In[ ]:

kde = sm.nonparametric.KDEUnivariate(obs_dist1)
kde.fit()


# In[ ]:

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.hist(obs_dist1, bins=50, normed=True, color='red')
ax.plot(kde.support, kde.density, lw=2, color='black');


# In[ ]:

obs_dist2 = mixture_rvs([.25,.75], size=10000, dist=[stats.norm, stats.beta],
            kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=1,args=(1,.5))))

kde2 = sm.nonparametric.KDEUnivariate(obs_dist2)
kde2.fit()


# In[ ]:

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.hist(obs_dist2, bins=50, normed=True, color='red')
ax.plot(kde2.support, kde2.density, lw=2, color='black');


# The fitted KDE object is a full non-parametric distribution.

# In[ ]:

obs_dist3 = mixture_rvs([.25,.75], size=1000, dist=[stats.norm, stats.norm],
                kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))
kde3 = sm.nonparametric.KDEUnivariate(obs_dist3)
kde3.fit()


# In[ ]:

kde3.entropy


# In[ ]:

kde3.evaluate(-1)


##### CDF

# In[ ]:

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(kde3.support, kde3.cdf);


##### Cumulative Hazard Function

# In[ ]:

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(kde3.support, kde3.cumhazard);


##### Inverse CDF

# In[ ]:

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(kde3.support, kde3.icdf);


##### Survival Function

# In[ ]:

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(kde3.support, kde3.sf);

