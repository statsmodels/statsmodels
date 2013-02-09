# -*- coding: utf-8 -*-
"""Plot Interaction of Categorical Factors
"""

# In this example, we will vizualize the interaction between
# categorical factors. First, categorical data are initialized
# and then plotted using the interaction_plot function.
#
# Author: Denis A. Engemann


print __doc__

import numpy as np
from statsmodels.graphics.factorplots import interaction_plot
from pandas import Series

np.random.seed(12345)
weight = Series(np.repeat(['low', 'hi', 'low', 'hi'], 15), name='weight')
nutrition = Series(np.repeat(['lo_carb', 'hi_carb'], 30), name='nutrition')
days = np.log(np.random.randint(1, 30, size=60))

fig = interaction_plot(weight, nutrition, days, colors=['red', 'blue'],
                       markers=['D', '^'], ms=10)

import matplotlib.pylab as plt
plt.show()
