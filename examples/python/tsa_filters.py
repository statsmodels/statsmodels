
## Time Series Filters

from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm


dta = sm.datasets.macrodata.load_pandas().data


index = pd.Index(sm.tsa.datetools.dates_from_range('1959Q1', '2009Q3'))
print(index)


dta.index = index
del dta['year']
del dta['quarter']


print(sm.datasets.macrodata.NOTE)


print(dta.head(10))


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
dta.realgdp.plot(ax=ax);
legend = ax.legend(loc = 'upper left');
legend.prop.set_size(20);


#### Hodrick-Prescott Filter

# The Hodrick-Prescott filter separates a time-series $y_t$ into a trend $\tau_t$ and a cyclical component $\zeta_t$
#
# $$y_t = \tau_t + \zeta_t$$
#
# The components are determined by minimizing the following quadratic loss function
#
# $$\min_{\\{ \tau_{t}\\} }\sum_{t}^{T}\zeta_{t}^{2}+\lambda\sum_{t=1}^{T}\left[\left(\tau_{t}-\tau_{t-1}\right)-\left(\tau_{t-1}-\tau_{t-2}\right)\right]^{2}$$

gdp_cycle, gdp_trend = sm.tsa.filters.hpfilter(dta.realgdp)


gdp_decomp = dta[['realgdp']]
gdp_decomp["cycle"] = gdp_cycle
gdp_decomp["trend"] = gdp_trend


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
gdp_decomp[["realgdp", "trend"]]["2000-03-31":].plot(ax=ax, fontsize=16);
legend = ax.get_legend()
legend.prop.set_size(20);


#### Baxter-King approximate band-pass filter: Inflation and Unemployment

##### Explore the hypothesis that inflation and unemployment are counter-cyclical.

# The Baxter-King filter is intended to explictly deal with the periodicty of the business cycle. By applying their band-pass filter to a series, they produce a new series that does not contain fluctuations at higher or lower than those of the business cycle. Specifically, the BK filter takes the form of a symmetric moving average
#
# $$y_{t}^{*}=\sum_{k=-K}^{k=K}a_ky_{t-k}$$
#
# where $a_{-k}=a_k$ and $\sum_{k=-k}^{K}a_k=0$ to eliminate any trend in the series and render it stationary if the series is I(1) or I(2).
#
# For completeness, the filter weights are determined as follows
#
# $$a_{j} = B_{j}+\theta\text{ for }j=0,\pm1,\pm2,\dots,\pm K$$
#
# $$B_{0} = \frac{\left(\omega_{2}-\omega_{1}\right)}{\pi}$$
# $$B_{j} = \frac{1}{\pi j}\left(\sin\left(\omega_{2}j\right)-\sin\left(\omega_{1}j\right)\right)\text{ for }j=0,\pm1,\pm2,\dots,\pm K$$
#
# where $\theta$ is a normalizing constant such that the weights sum to zero.
#
# $$\theta=\frac{-\sum_{j=-K^{K}b_{j}}}{2K+1}$$
#
# $$\omega_{1}=\frac{2\pi}{P_{H}}$$
#
# $$\omega_{2}=\frac{2\pi}{P_{L}}$$
#
# $P_L$ and $P_H$ are the periodicity of the low and high cut-off frequencies. Following Burns and Mitchell's work on US business cycles which suggests cycles last from 1.5 to 8 years, we use $P_L=6$ and $P_H=32$ by default.

bk_cycles = sm.tsa.filters.bkfilter(dta[["infl","unemp"]])


# * We lose K observations on both ends. It is suggested to use K=12 for quarterly data.

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
bk_cycles.plot(ax=ax, style=['r--', 'b-']);


#### Christiano-Fitzgerald approximate band-pass filter: Inflation and Unemployment

# The Christiano-Fitzgerald filter is a generalization of BK and can thus also be seen as weighted moving average. However, the CF filter is asymmetric about $t$ as well as using the entire series. The implementation of their filter involves the
# calculations of the weights in
#
# $$y_{t}^{*}=B_{0}y_{t}+B_{1}y_{t+1}+\dots+B_{T-1-t}y_{T-1}+\tilde B_{T-t}y_{T}+B_{1}y_{t-1}+\dots+B_{t-2}y_{2}+\tilde B_{t-1}y_{1}$$
#
# for $t=3,4,...,T-2$, where
#
# $$B_{j} = \frac{\sin(jb)-\sin(ja)}{\pi j},j\geq1$$
#
# $$B_{0} = \frac{b-a}{\pi},a=\frac{2\pi}{P_{u}},b=\frac{2\pi}{P_{L}}$$
#
# $\tilde B_{T-t}$ and $\tilde B_{t-1}$ are linear functions of the $B_{j}$'s, and the values for $t=1,2,T-1,$ and $T$ are also calculated in much the same way. $P_{U}$ and $P_{L}$ are as described above with the same interpretation.

# The CF filter is appropriate for series that may follow a random walk.

print(sm.tsa.stattools.adfuller(dta['unemp'])[:3])


print(sm.tsa.stattools.adfuller(dta['infl'])[:3])


cf_cycles, cf_trend = sm.tsa.filters.cffilter(dta[["infl","unemp"]])
print(cf_cycles.head(10))


fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
cf_cycles.plot(ax=ax, style=['r--','b-']);


# Filtering assumes *a priori* that business cycles exist. Due to this assumption, many macroeconomic models seek to create models that match the shape of impulse response functions rather than replicating properties of filtered series. See VAR notebook.
