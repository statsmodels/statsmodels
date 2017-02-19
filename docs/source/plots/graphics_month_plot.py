import statsmodels.api as sm
import pandas as pd

dta = sm.datasets.elnino.load_pandas().data
dta['YEAR'] = dta.YEAR.astype(int).astype(str)
dta = dta.set_index('YEAR').T.unstack()
dates = pd.to_datetime(list(map(lambda x : '-'.join(x) + '-1', dta.index.values)))

dta.index = pd.DatetimeIndex(list(dates), freq='MS')
dta.name = 'temp'
fig = sm.graphics.tsa.month_plot(dta)
