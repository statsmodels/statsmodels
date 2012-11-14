
import numpy as np
from statsmodels.graphics.factorplots import interaction_plot
from pandas import Series

np.random.seed(12345)
weight = Series(np.repeat(['low', 'hi', 'low', 'hi'], 15))
nutrition = Series(np.repeat(['lo_carb', 'hi_carb'], 30))
days = np.log(np.random.randint(1, 30, size=60))
levels = dict(low=0, hi=1)
fig = interaction_plot(weight, nutrition, days, levels,
                 colors=['red', 'blue'], markers=['D', '^'],
                 ms=10)

import matplotlib.pyplot as plt
plt.show()
