# -*- coding: utf-8 -*-
"""

Created on Fri May 04 11:08:56 2012

Author: Ralf Gommers

"""

#Load the El Nino dataset.  Consists of 60 years worth of Pacific Ocean sea
#surface temperature data.

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
data = sm.datasets.elnino.load()

#Create a rainbow plot:

fig = plt.figure()
ax = fig.add_subplot(111)
res = sm.graphics.rainbowplot(data.raw_data[:, 1:], ax=ax)

ax.set_xlabel("Month of the year")
ax.set_ylabel("Sea surface temperature (C)")
ax.set_xticks(np.arange(13, step=3) - 1)
ax.set_xticklabels(["", "Mar", "Jun", "Sep", "Dec"])
ax.set_xlim([-0.2, 11.2])

#plt.show()
