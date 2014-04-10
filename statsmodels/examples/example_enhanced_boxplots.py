
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm


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


# Create a jitter plot.
fig3 = plt.figure()
ax = fig3.add_subplot(111)

plot_opts={'cutoff_val':5, 'cutoff_type':'abs', 'label_fontsize':'small',
           'label_rotation':30, 'violin_fc':(0.8, 0.8, 0.8),
           'jitter_marker':'.', 'jitter_marker_size':3, 'bean_color':'#FF6F00',
           'bean_mean_color':'#009D91'}
sm.graphics.beanplot(age, ax=ax, labels=labels, jitter=True,
                    plot_opts=plot_opts)

ax.set_xlabel("Party identification of respondent.")
ax.set_ylabel("Age")
ax.set_title("US national election '96 - Age & Party Identification")


# Create an asymmetrical jitter plot.
ix = data.exog['income'] < 16  # incomes < $30k
age = data.exog['age'][ix]
endog = data.endog[ix]
age_lower_income = [age[endog == id] for id in party_ID]

ix = data.exog['income'] >= 20  # incomes > $50k
age = data.exog['age'][ix]
endog = data.endog[ix]
age_higher_income = [age[endog == id] for id in party_ID]

fig = plt.figure()
ax = fig.add_subplot(111)

plot_opts['violin_fc'] = (0.5, 0.5, 0.5)
plot_opts['bean_show_mean'] = False
plot_opts['bean_show_median'] = False
plot_opts['bean_legend_text'] = 'Income < \$30k'
plot_opts['cutoff_val'] = 10
sm.graphics.beanplot(age_lower_income, ax=ax, labels=labels, side='left',
                     jitter=True, plot_opts=plot_opts)
plot_opts['violin_fc'] = (0.7, 0.7, 0.7)
plot_opts['bean_color'] = '#009D91'
plot_opts['bean_legend_text'] = 'Income > \$50k'
sm.graphics.beanplot(age_higher_income, ax=ax, labels=labels, side='right',
                     jitter=True, plot_opts=plot_opts)

ax.set_xlabel("Party identification of respondent.")
ax.set_ylabel("Age")
ax.set_title("US national election '96 - Age & Party Identification")


# Show all plots.
plt.show()

