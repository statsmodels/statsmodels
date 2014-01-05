
## Dates in timeseries models

# In[ ]:

import statsmodels.api as sm
import numpy as np
import pandas


# ## Getting started

# In[ ]:

data = sm.datasets.sunspots.load()


# Right now an annual date series must be datetimes at the end of the year.

# In[ ]:

from datetime import datetime
dates = sm.tsa.datetools.dates_from_range('1700', length=len(data.endog))


# ## Using Pandas
# 
# Make a pandas TimeSeries or DataFrame

# In[ ]:

endog = pandas.TimeSeries(data.endog, index=dates)


# Instantiate the model

# In[ ]:

ar_model = sm.tsa.AR(endog, freq='A')
pandas_ar_res = ar_model.fit(maxlag=9, method='mle', disp=-1)


# Out-of-sample prediction

# In[ ]:

pred = pandas_ar_res.predict(start='2005', end='2015')
print pred


# ## Using explicit dates

# In[ ]:

ar_model = sm.tsa.AR(data.endog, dates=dates, freq='A')
ar_res = ar_model.fit(maxlag=9, method='mle', disp=-1)
pred = ar_res.predict(start='2005', end='2015')
print pred


# This just returns a regular array, but since the model has date information attached, you can get the prediction dates in a roundabout way.

# In[ ]:

print ar_res.data.predict_dates


# Note: This attribute only exists if predict has been called. It holds the dates associated with the last call to predict.
