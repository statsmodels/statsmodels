import statsmodels.api as sm
import pandas as pd

dta = sm.datasets.elnino.load_pandas().data
dta['YEAR'] = dta.YEAR.astype(int).astype(str)
dta = dta.set_index('YEAR').T.unstack()
dates = pd.to_datetime(list(map(lambda x: '-'.join(x) + '-1',
                                dta.index.values)))

dta.index = dates.to_period('Q')
dta.name = 'temp'
fig = sm.graphics.tsa.quarter_plot(dta)
