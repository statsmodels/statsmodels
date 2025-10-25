import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters

from statsmodels.datasets import co2
from statsmodels.tsa.seasonal import STL

register_matplotlib_converters()
data = co2.load().data
data = data.resample('ME').mean().ffill()

res = STL(np.squeeze(data)).fit()
res.plot()
plt.show()
