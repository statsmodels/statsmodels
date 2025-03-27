'''
    Load El Niño dataset, perform STL, plot Seasonal Diagnostic Plot
'''

from statsmodels.datasets import elnino
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import seasonal_diagnostic_plot
import pandas as pd

month_dict = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
              'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
data = (elnino.load().data
        .rename(columns=month_dict)
        .melt(id_vars=['YEAR'], var_name='month')
        .assign(day=1,
                date=lambda df: pd.to_datetime(
                        df[['YEAR', 'month', 'day']]))
        .drop(columns=['YEAR', 'month', 'day'])
        .set_index('date')
        .sort_index()
        )

res = STL(data, period=12, seasonal=53).fit()
labels = ['January', 'February', 'March', 'November']
fig = seasonal_diagnostic_plot(res, period=12, subplots=[0, 1, 2, 10],
                               labels=labels, nrows=2)

fig.show()
