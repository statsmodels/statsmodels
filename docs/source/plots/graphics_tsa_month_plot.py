import pandas as pd

import statsmodels.api as sm

dta = sm.datasets.elnino.load_pandas().data
dta["YEAR"] = dta.YEAR.astype(int).astype(str)
dta = dta.set_index("YEAR").T.unstack()
dates = pd.to_datetime(
    ["-".join(x) + "-01" for x in dta.index.values], format="%Y-%b-%d"
)

dta.index = pd.DatetimeIndex(list(dates), freq="MS")
dta.name = "temp"
fig = sm.graphics.tsa.month_plot(dta)
