import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

from statsmodels.datasets import co2
from statsmodels.tsa.seasonal import STL

register_matplotlib_converters()
data = co2.load(True).data
data = data.resample('M').mean().ffill()

res = STL(data).fit()
res.plot()
plt.show()
