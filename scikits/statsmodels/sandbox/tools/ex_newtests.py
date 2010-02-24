

from stattools import unitroot_adf

import scikits.statsmodels.datasets.macrodata.data as macro

macrod = macro.Load().data

print macro.NOTE

print macrod.dtype.names

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

print '%-10s %5s %-8s' % ('variable', 'trend', '  adf')
for name, torder in datatrendli:
    print '%-10s %5d %8.4f' % (name, torder, unitroot_adf(macrod[name], trendorder=torder))


