import warnings

import matplotlib.pyplot as plt
import pandas as pd

import statsmodels.api as sm

dta = sm.datasets.sunspots.load_pandas().data[["SUNACTIVITY"]]
dta.index = pd.date_range(start="1700", end="2009", freq="YE")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res = sm.tsa.arima.ARIMA(dta, order=(3, 0, 0)).fit()
fig, ax = plt.subplots()
ax = dta.loc["1950":].plot(ax=ax)
pred = res.get_prediction("1990", "2012", dynamic=True)
ci = pred.conf_int()
ax.plot(pred.predicted_mean.index, pred.predicted_mean, label="forecast")
ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="k", alpha=0.1)
ax.legend()
