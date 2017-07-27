
## Dates in timeseries models

from __future__ import print_function
import statsmodels.api as sm
import pandas as pd


# ## Getting started

data = sm.datasets.sunspots.load()


# Right now an annual date series must be datetimes at the end of the year.

dates = sm.tsa.datetools.dates_from_range('1700', length=len(data.endog))


# ## Using Pandas
#
# Make a pandas Series or DataFrame with DatetimeIndex

endog = pd.Series(data.endog, index=dates)


# Instantiate the model

ar_model = sm.tsa.AR(endog, freq='A')
pandas_ar_res = ar_model.fit(maxlag=9, method='mle', disp=-1)


# Out-of-sample prediction

pred = pandas_ar_res.predict(start='2005', end='2015')
print(pred)


# ## Using explicit dates

ar_model = sm.tsa.AR(data.endog, dates=dates, freq='A')
ar_res = ar_model.fit(maxlag=9, method='mle', disp=-1)
pred = ar_res.predict(start='2005', end='2015')
print(pred)


# This just returns a regular array, but since the model has date information attached, you can get the prediction dates in a roundabout way.

print(ar_res.data.predict_dates)


# Note: This attribute only exists if predict has been called. It holds the dates associated with the last call to predict.
