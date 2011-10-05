import scikits.statsmodels.api as sm
import numpy as np
import pandas

# Getting started
# ---------------

data = sm.datasets.sunspots.load()

# Right now an annual date series must be datetimes at the end of the year.
# We can use scikits.timeseries and datetime to create this array.

import datetime
import scikits.timeseries as ts
dates = ts.date_array(start_date=1700, length=len(data.endog), freq='A')

# To make an array of datetime types, we need an integer array of ordinals

#.. from datetime import datetime
#.. dt_dates = dates.toordinal().astype(int)
#.. dt_dates = np.asarray([datetime.fromordinal(i) for i in dt_dates])
dt_dates = dates.tolist()

# Using Pandas
# ------------

# Make a pandas TimeSeries or DataFrame
endog = pandas.Series(data.endog, index=dt_dates)

# and instantiate the model
ar_model = sm.tsa.AR(endog, freq='A')
pandas_ar_res = ar_model.fit(maxlag=9, method='mle', disp=-1)

# Let's do some out-of-sample prediction
pred = pandas_ar_res.predict(start='2005', end='2015')
print pred

# Using explicit dates
# --------------------

ar_model = sm.tsa.AR(data.endog, dates=dt_dates, freq='A')
ar_res = ar_model.fit(maxlag=9, method='mle', disp=-1)
pred = ar_res.predict(start='2005', end='2015')
print pred

# This just returns a regular array, but since the model has date information
# attached, you can get the prediction dates in a roundabout way.

print ar_res._data.predict_dates

# This attribute only exists if predict has been called. It holds the dates
# associated with the last call to predict.
#..TODO: should this be attached to the results instance?

# Using scikits.timeseries
# ------------------------

ts_data = ts.time_series(data.endog, dates=dates)
ts_ar_model = sm.tsa.AR(ts_data, freq='A')
ts_ar_res = ts_ar_model.fit(maxlag=9)

# Using Larry
# -----------

import la
larr = la.larry(data.endog, [dt_dates])
la_ar_model = sm.tsa.AR(larr, freq='A')
la_ar_res = la_ar_model.fit(maxlag=9)
