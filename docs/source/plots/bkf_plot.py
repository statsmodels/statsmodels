import statsmodels.api as sm

from load_macrodata import dta

cycles = sm.tsa.filters.bkfilter(dta[['realinv']], 6, 24, 12)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
cycles.plot(ax=ax, style=['r--', 'b-'])
