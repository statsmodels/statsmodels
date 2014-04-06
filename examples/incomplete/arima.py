from __future__ import print_function
from statsmodels.datasets.macrodata import load_pandas
from statsmodels.tsa.base.datetools import dates_from_range
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
plt.interactive(False)

# let's examine an ARIMA model of CPI

cpi = load_pandas().data['cpi']
dates = dates_from_range('1959q1', '2009q3')
cpi.index = dates

res = ARIMA(cpi, (1, 1, 1), freq='Q').fit()
print(res.summary())

# we can look at the series
cpi.diff().plot()

# maybe logs are better
log_cpi = np.log(cpi)

# check the ACF and PCF plots
acf, confint_acf = sm.tsa.acf(log_cpi.diff().values[1:], confint=95)
# center the confidence intervals about zero
#confint_acf -= confint_acf.mean(1)[:, None]
pacf = sm.tsa.pacf(log_cpi.diff().values[1:], method='ols')
# confidence interval is now an option to pacf
from scipy import stats
confint_pacf = stats.norm.ppf(1 - .025) * np.sqrt(1 / 202.)

fig = plt.figure()
ax = fig.add_subplot(121)
ax.set_title('Autocorrelation')
ax.plot(range(41), acf, 'bo', markersize=5)
ax.vlines(range(41), 0, acf)
ax.fill_between(range(41), confint_acf[:, 0], confint_acf[:, 1], alpha=.25)
fig.tight_layout()
ax = fig.add_subplot(122, sharey=ax)
ax.vlines(range(41), 0, pacf)
ax.plot(range(41), pacf, 'bo', markersize=5)
ax.fill_between(range(41), -confint_pacf, confint_pacf, alpha=.25)


#NOTE: you'll be able to just to this when tsa-plots is in master
#sm.graphics.acf_plot(x, nlags=40)
#sm.graphics.pacf_plot(x, nlags=40)


# still some seasonality
# try an arma(1, 1) with ma(4) term
