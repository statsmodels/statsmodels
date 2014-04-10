'''Functional boxplots and rainbow plots

see docstrings for an explanation


Author: Ralf Gommers

'''

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

#Load the El Nino dataset.  Consists of 60 years worth of Pacific Ocean sea
#surface temperature data.

data = sm.datasets.elnino.load()

#Create a functional boxplot:

#We see that the years 1982-83 and 1997-98 are outliers; these are
#the years where El Nino (a climate pattern characterized by warming
#up of the sea surface and higher air pressures) occurred with unusual
#intensity.

fig = plt.figure()
ax = fig.add_subplot(111)
res = sm.graphics.fboxplot(data.raw_data[:, 1:], wfactor=2.58,
                           labels=data.raw_data[:, 0].astype(int),
                           ax=ax)

ax.set_xlabel("Month of the year")
ax.set_ylabel("Sea surface temperature (C)")
ax.set_xticks(np.arange(13, step=3) - 1)
ax.set_xticklabels(["", "Mar", "Jun", "Sep", "Dec"])
ax.set_xlim([-0.2, 11.2])



#Create a rainbow plot:

fig = plt.figure()
ax = fig.add_subplot(111)
res = sm.graphics.rainbowplot(data.raw_data[:, 1:], ax=ax)

ax.set_xlabel("Month of the year")
ax.set_ylabel("Sea surface temperature (C)")
ax.set_xticks(np.arange(13, step=3) - 1)
ax.set_xticklabels(["", "Mar", "Jun", "Sep", "Dec"])
ax.set_xlim([-0.2, 11.2])
plt.show()
