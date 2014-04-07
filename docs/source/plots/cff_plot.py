import statsmodels.api as sm

from load_macrodata import dta

cf_cycles, cf_trend = sm.tsa.filters.cffilter(dta[["infl", "unemp"]])

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
cf_cycles.plot(ax=ax, style=['r--', 'b-'])
