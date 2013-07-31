from statsmodels.datasets.macrodata import load_pandas
from statsmodels.tsa.base.datetools import dates_from_range
from statsmodels.tsa.setar_model import SETAR
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import statsmodels.api as sm
import datetime
plt.interactive(False)

# Load the sunspots dataset
dta = sm.datasets.sunspots.load_pandas().data
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))

# Reconcile with Hansen (1999) dataset
dta = dta[dta.YEAR <= 1988]
dta.sun = 2*(np.sqrt(1 + dta.SUNACTIVITY) - 1)
# Adjustments to match Hansen (1999)
dta.SUNACTIVITY.iloc[[262, 280, 281, 287]] = [
    10.40967365,
    22.95596121,
    21.79075451,
    8.99090533
]

# Replicate Hansen (1999) Figure 1
f1 = plt.figure(figsize=(10,6))
ax = f1.add_subplot(111)
ax.plot_date(x=dta.index, y=dta.sun, fmt='k-')
ax.set(title='Figure 1. Annual sunspot means, 1700-1988.',
       ylim=(0,28),
       xlim=(datetime.date(1680,1,1), datetime.date(2000,1,1))
);
ax.yaxis.set_ticks(range(0,32,4));

# Replicate Hansen (1999) Table 1
res = SETAR(dta.sun, order=1, ar_order=11).fit()
res.summary()

# Replicate Hansen (1999) Table 2
res = SETAR(dta.sun, order=2, ar_order=11,
			delay=2, thresholds=[7.4233751915117967]).fit()
res.summary()

# Recalculate results, but this time using a comprehensive grid search
res = SETAR(dta.sun, order=2, ar_order=11, threshold_grid_size=300).fit()
res.summary()