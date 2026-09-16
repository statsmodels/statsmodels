import matplotlib.pyplot as plt
import pandas as pd

import statsmodels.api as sm

dta = sm.datasets.macrodata.load_pandas().data
index = pd.period_range("1959Q1", "2009Q3", freq="Q")
dta.set_index(index, inplace=True)
cycle, trend = sm.tsa.filters.hamilton_filter(dta[["infl", "unemp"]], h=8, p=4)


fig, ax = plt.subplots()
cycle.plot(ax=ax, style=["r--", "b-"])
ax.set_title("Cycle extracted using Hamilton Filter (h=8, p=4)")
plt.show()
