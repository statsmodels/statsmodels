import matplotlib.pyplot as plt

from load_macrodata import dta
import statsmodels.api as sm

cycles = sm.tsa.filters.bkfilter(dta[['realinv']], 6, 24, 12)

fig, ax = plt.subplots()
cycles.plot(ax=ax, style=['r--', 'b-'])
