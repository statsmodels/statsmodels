import numpy as np
import matplotlib.pyplot as plt

import scikits.statsmodels.api as sm


# Necessary to make horizontal axis labels fit
plt.rcParams['figure.subplot.bottom'] = 0.23

data = sm.datasets.anes96.load_pandas()
party_ID = np.arange(7)
labels = ["Strong Democrat", "Weak Democrat", "Independent-Democrat",
          "Independent-Independent", "Independent-Republican",
          "Weak Republican", "Strong Republican"]

# Group age by party ID.
age = [data.exog['age'][data.endog == id] for id in party_ID]

# Create a violin plot.
fig = plt.figure()
ax = fig.add_subplot(111)

sm.graphics.violinplot(age, ax=ax, labels=labels,
                       plot_opts={'cutoff_val':5, 'cutoff_type':'abs',
                                  'label_fontsize':'small',
                                  'label_rotation':30})

ax.set_xlabel("Party identification of respondent.")
ax.set_ylabel("Age")
ax.set_title("US national election '96 - Age & Party Identification")

# Create a bean plot.
fig2 = plt.figure()
ax = fig2.add_subplot(111)

sm.graphics.beanplot(age, ax=ax, labels=labels,
                    plot_opts={'cutoff_val':5, 'cutoff_type':'abs',
                               'label_fontsize':'small',
                               'label_rotation':30})

ax.set_xlabel("Party identification of respondent.")
ax.set_ylabel("Age")
ax.set_title("US national election '96 - Age & Party Identification")

# Show both plots.
plt.show()

