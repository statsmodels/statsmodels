import pandas as pd

import statsmodels.api as sm

dta = sm.datasets.elnino.load_pandas().data
dta["YEAR"] = dta.YEAR.astype(int).astype(str)
dta = dta.set_index("YEAR").T.unstack()
dates = pd.to_datetime(["-".join(x) + "-1" for x in dta.index.values])

dta.index = dates.to_period("Q")
dta.name = "temp"
fig = sm.graphics.tsa.quarter_plot(dta)
