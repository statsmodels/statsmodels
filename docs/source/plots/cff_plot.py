from load_macrodata import dta
import matplotlib.pyplot as plt

import statsmodels.api as sm

cf_cycles, cf_trend = sm.tsa.filters.cffilter(dta[["infl", "unemp"]])

fig, ax = plt.subplots()
cf_cycles.plot(ax=ax, style=['r--', 'b-'])
