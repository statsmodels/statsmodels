from load_macrodata import dta
import matplotlib.pyplot as plt
import pandas as pd

import statsmodels.api as sm

cycle, trend = sm.tsa.filters.hpfilter(dta.realgdp, 1600)
gdp_decomp = dta[['realgdp']].copy()
gdp_decomp["cycle"] = cycle
gdp_decomp["trend"] = trend

fig, ax = plt.subplots()
gdp_decomp[["realgdp", "trend"]]["2000-03-31":].plot(ax=ax,
                                                     fontsize=16)
