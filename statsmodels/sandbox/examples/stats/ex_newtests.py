
from __future__ import print_function
from .diagnostic import unitroot_adf

import statsmodels.datasets.macrodata.data as macro

macrod = macro.load(as_pandas=False).data

print(macro.NOTE)

print(macrod.dtype.names)

datatrendli = [
               ('realgdp', 1),
               ('realcons', 1),
               ('realinv', 1),
               ('realgovt', 1),
               ('realdpi', 1),
               ('cpi', 1),
               ('m1', 1),
               ('tbilrate', 0),
               ('unemp',0),
               ('pop', 1),
               ('infl',0),
               ('realint', 0)
               ]

print('%-10s %5s %-8s' % ('variable', 'trend', '  adf'))
for name, torder in datatrendli:
    adf_, pval = unitroot_adf(macrod[name], trendorder=torder)[:2]
    print('%-10s %5d %8.4f %8.4f' % (name, torder, adf_, pval))
