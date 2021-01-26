import sys
sys.path.insert(0, "/afs/umich.edu/user/k/s/kshedden/statsmodels_fork/statsmodels")

import statsmodels.api as sm
import numpy as np
import pandas as pd

data = sm.datasets.vision_ordnance.load()
df = data.data
tab = df.set_index(['left', 'right'])
tab = tab.unstack()

st = sm.stats.TableSymmetry(tab)

print(st.summary())

"""
data = sm.datasets.china_smoking.load()

mat = np.asarray(data.data)
tables = [np.reshape(x, (2, 2)) for x in mat]

st = sm.stats.StratifiedTables(tables)

print(st.summary())

print("\n\n")
st = sm.stats.TableSymmetry([[34, 23, 11], [55, 57, 75], [34, 52, 34]])

print(st.summary())
"""
