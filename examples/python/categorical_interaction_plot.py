
## Plot Interaction of Categorical Factors

# In this example, we will vizualize the interaction between categorical factors. First, we will create some categorical data are initialized. Then plotted using the interaction_plot function which internally recodes the x-factor categories to ingegers.

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.factorplots import interaction_plot
from pandas import Series
np.random.seed(12345)
weight = Series(np.repeat(['low', 'hi', 'low', 'hi'], 15), name='weight')
nutrition = Series(np.repeat(['lo_carb', 'hi_carb'], 30), name='nutrition')
days = np.log(np.random.randint(1, 30, size=60))
plt.figure(figsize=(6, 6));
interaction_plot(x=weight, trace=nutrition, response=days,
                 colors=['red', 'blue'], markers=['D', '^'], ms=10)


#     <matplotlib.figure.Figure at 0x106dd2a10>

# image file:

# image file:
