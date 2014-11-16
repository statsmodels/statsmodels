from __future__ import print_function
from statsmodels.compat import iteritems, cStringIO

import numpy as np
import pandas as pd


sio = cStringIO.StringIO()
c = pd.read_hdf('kpss_critical_values.h5', 'c')
ct = pd.read_hdf('kpss_critical_values.h5', 'ct')

data = {'c': c, 'ct': ct}
for k, v in iteritems(data):
    n = v.shape[0]
    selected = np.zeros((n, 1), dtype=np.bool)
    selected[0] = True
    selected[-1] = True
    selected[v.index == 10.0] = True
    selected[v.index == 5.0] = True
    selected[v.index == 2.5] = True
    selected[v.index == 1.0] = True
    max_diff = 1.0
    while max_diff > 0.05:
        xp = np.squeeze(v[selected].values)
        yp = np.asarray(v[selected].index, dtype=np.float64)
        x = np.squeeze(v.values)
        y = np.asarray(v.index, dtype=np.float64)
        yi = np.interp(x, xp, yp)
        abs_diff = np.abs(y - yi)
        max_diff = np.max(abs_diff)
        if max_diff > 0.05:
            selected[np.where(abs_diff == max_diff)] = True

    quantiles = list(np.squeeze(v[selected].index.values))
    critical_values = list(np.squeeze(v[selected].values))
    # Fix for first CV
    critical_values[0] = 0.0
    sio.write("from numpy import asarray\n\n")
    sio.write("kpss_critical_values = {}\n")
    sio.write("kpss_critical_values['" + k + "'] = (")
    count = 0
    for c, q in zip(critical_values, quantiles):
        sio.write(
            '(' + '{0:0.1f}'.format(q) + ', ' + '{0:0.4f}'.format(c) + ')')
        count += 1
        if count % 3 == 0:
            sio.write(',\n                             ')
        else:
            sio.write(', ')
    sio.write(")\n")
    sio.write("kpss_critical_values['" + k + "'] = ")
    sio.write("asarray(kpss_critical_values['" + k + "'])")
    sio.write("\n")

sio.seek(0)
print(sio.read())