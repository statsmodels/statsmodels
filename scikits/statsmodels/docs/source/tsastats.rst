.. currentmodule:: scikits.statsmodels.tsa.tsatools

Time Series Analysis
====================

These are some of the helper functions for doing time series analysis. First
we can load some a some data from the US Macro Economy 1959:Q1 - 2009:Q3. ::

  >>> data = sm.datasets.macrodata.load()

The macro dataset is a structured array. ::

 >>> data = data.data[['year','quarter','realgdp','tbilrate','cpi','unemp']]

We can add a lag like so ::

 >>> data = sm.tsa.add_lag(data, 'realgdp', lags=2)

TODO:
-scikits.timeseries
-link in to var docs

