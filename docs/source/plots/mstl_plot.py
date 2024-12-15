import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import MSTL

pd.plotting.register_matplotlib_converters()
plt.rc("figure", figsize=(8, 10))
plt.rc("font", size=10)
np.random.seed(0)

t = np.arange(1, 1000)
trend = 0.0001 * t ** 2 + 100
daily_seasonality = 5 * np.sin(2 * np.pi * t / 24)
weekly_seasonality = 10 * np.sin(2 * np.pi * t / (24 * 7))
noise = np.random.randn(len(t))
y = trend + daily_seasonality + weekly_seasonality + noise
index = pd.date_range(start='2000-01-01', periods=len(t), freq='h')
data = pd.DataFrame(data=y, index=index)

res = MSTL(data, periods=(24, 24*7)).fit()
res.plot()
plt.tight_layout()
plt.show()
