'''
    Load Weekly CO2 dataset, perform STL, plot Seasonal Diagnostic Plot
'''

from statsmodels.datasets import co2
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import seasonal_diagnostic_plot
data = (co2.load().data
        .loc[lambda df: df.index.isocalendar().week < 53]
        .loc['1986-01-01':]
        )
res = STL(data, period=52, seasonal=21).fit()
fig = seasonal_diagnostic_plot(res, period=52, subplots=6, nrows=2)
fig.show()
