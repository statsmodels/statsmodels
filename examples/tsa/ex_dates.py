"""
Using dates with timeseries models
"""
import statsmodels.api as sm
import numpy as np
import pandas

# Getting started
# ---------------

data = sm.datasets.sunspots.load()

# Right now an annual date series must be datetimes at the end of the year.

from datetime import datetime
dates = sm.tsa.datetools.dates_from_range('1700', length=len(data.endog))

# Using Pandas
# ------------

# Make a pandas TimeSeries or DataFrame
endog = pandas.TimeSeries(data.endog, index=dates)

# and instantiate the model
ar_model = sm.tsa.AR(endog, freq='A')
pandas_ar_res = ar_model.fit(maxlag=9, method='mle', disp=-1)

# Let's do some out-of-sample prediction
pred = pandas_ar_res.predict(start='2005', end='2015')
print pred

# Using explicit dates
# --------------------

ar_model = sm.tsa.AR(data.endog, dates=dates, freq='A')
ar_res = ar_model.fit(maxlag=9, method='mle', disp=-1)
pred = ar_res.predict(start='2005', end='2015')
print pred

# This just returns a regular array, but since the model has date information
# attached, you can get the prediction dates in a roundabout way.

print ar_res._data.predict_dates

# This attribute only exists if predict has been called. It holds the dates
# associated with the last call to predict.
#..TODO: should this be attached to the results instance?
