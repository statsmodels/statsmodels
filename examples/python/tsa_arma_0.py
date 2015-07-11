
## Autoregressive Moving Average (ARMA): Sunspots data

from __future__ import print_function
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm


from statsmodels.graphics.api import qqplot


### Sunpots Data

print(sm.datasets.sunspots.NOTE)


dta = sm.datasets.sunspots.load_pandas().data


dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del dta["YEAR"]


dta.plot(figsize=(12,8));


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)


arma_mod20 = sm.tsa.ARMA(dta, (2,0)).fit()
print(arma_mod20.params)


arma_mod30 = sm.tsa.ARMA(dta, (3,0)).fit()


print(arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)


print(arma_mod30.params)


print(arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic)


# * Does our model obey the theory?

sm.stats.durbin_watson(arma_mod30.resid.values)


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax = arma_mod30.resid.plot(ax=ax);


resid = arma_mod30.resid


stats.normaltest(resid)


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)


r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))


# * This indicates a lack of fit.

# * In-sample dynamic prediction. How good does our model do?

predict_sunspots = arma_mod30.predict('1990', '2012', dynamic=True)
print(predict_sunspots)


ax = dta.ix['1950':].plot(figsize=(12,8))
ax = predict_sunspots.plot(ax=ax, style='r--', label='Dynamic Prediction')
ax.legend()
ax.axis((-20.0, 38.0, -4.0, 200.0))


def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()


mean_forecast_err(dta.SUNACTIVITY, predict_sunspots)


#### Exercise: Can you obtain a better fit for the Sunspots model? (Hint: sm.tsa.AR has a method select_order)

#### Simulated ARMA(4,1): Model Identification is Difficult

from statsmodels.tsa.arima_process import arma_generate_sample, ArmaProcess


np.random.seed(1234)
# include zero-th lag
arparams = np.array([1, .75, -.65, -.55, .9])
maparams = np.array([1, .65])


# Let's make sure this model is estimable.

arma_t = ArmaProcess(arparams, maparams)


arma_t.isinvertible


arma_t.isstationary


# * What does this mean?

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(arma_t.generate_sample(nsample=50));


arparams = np.array([1, .35, -.15, .55, .1])
maparams = np.array([1, .65])
arma_t = ArmaProcess(arparams, maparams)
arma_t.isstationary


arma_rvs = arma_t.generate_sample(nsample=500, burnin=250, scale=2.5)


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(arma_rvs, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(arma_rvs, lags=40, ax=ax2)


# * For mixed ARMA processes the Autocorrelation function is a mixture of exponentials and damped sine waves after (q-p) lags.
# * The partial autocorrelation function is a mixture of exponentials and dampened sine waves after (p-q) lags.

arma11 = sm.tsa.ARMA(arma_rvs, (1,1)).fit()
resid = arma11.resid
r,q,p = sm.tsa.acf(resid, qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))


arma41 = sm.tsa.ARMA(arma_rvs, (4,1)).fit()
resid = arma41.resid
r,q,p = sm.tsa.acf(resid, qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))


#### Exercise: How good of in-sample prediction can you do for another series, say, CPI

macrodta = sm.datasets.macrodata.load_pandas().data
macrodta.index = pd.Index(sm.tsa.datetools.dates_from_range('1959Q1', '2009Q3'))
cpi = macrodta["cpi"]


##### Hint:

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax = cpi.plot(ax=ax)
ax.legend()


# P-value of the unit-root test, resoundly rejects the null of no unit-root.

print(sm.tsa.adfuller(cpi)[1])

