import statsmodels.api as sm
import pandas as pd

dta = sm.datasets.elnino.load_pandas().data
dta['YEAR'] = dta.YEAR.astype(int).astype(str)
dta = dta.set_index('YEAR').T.unstack()
dates = map(lambda x : pd.datetools.parse_time_string('1 '+' '.join(x)),
                                       dta.index.values)

dta.index = pd.DatetimeIndex(dates, freq='M')
dta.name = 'temp'
fig = sm.graphics.tsa.month_plot(dta)
