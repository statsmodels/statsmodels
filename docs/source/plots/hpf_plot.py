import statsmodels.api as sm
import pandas as pd
from load_macrodata import dta

cycle, trend = sm.tsa.filters.hpfilter(dta.realgdp, 1600)
gdp_decomp = dta[['realgdp']]
gdp_decomp["cycle"] = cycle
gdp_decomp["trend"] = trend

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
gdp_decomp[["realgdp", "trend"]]["2000-03-31":].plot(ax=ax,
                                                     fontsize=16)
