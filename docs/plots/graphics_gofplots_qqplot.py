# -*- coding: utf-8 -*-
"""
Created on Sun May 06 05:32:15 2012

Author: Josef Perktold

"""
from scipy import stats
from matplotlib import pyplot as plt
import statsmodels.api as sm

#example from docstring
data = sm.datasets.longley.load()
data.exog = sm.add_constant(data.exog, prepend=True)
mod_fit = sm.OLS(data.endog, data.exog).fit()
res = mod_fit.resid

left = -1.8   #x coordinate for text insert

fig = plt.figure()

ax = fig.add_subplot(2, 2, 1)
sm.graphics.qqplot(res, ax=ax)
top = ax.get_ylim()[1] * 0.75
txt = ax.text(left, top, 'no keywords', verticalalignment='top')
txt.set_bbox(dict(facecolor='k', alpha=0.1))

ax = fig.add_subplot(2, 2, 2)
sm.graphics.qqplot(res, line='s', ax=ax)
top = ax.get_ylim()[1] * 0.75
txt = ax.text(left, top, "line='s'", verticalalignment='top')
txt.set_bbox(dict(facecolor='k', alpha=0.1))

ax = fig.add_subplot(2, 2, 3)
sm.graphics.qqplot(res, line='45', fit=True, ax=ax)
ax.set_xlim(-2, 2)
top = ax.get_ylim()[1] * 0.75
txt = ax.text(left, top, "line='45', \nfit=True", verticalalignment='top')
txt.set_bbox(dict(facecolor='k', alpha=0.1))

ax = fig.add_subplot(2, 2, 4)
sm.graphics.qqplot(res, dist=stats.t, line='45', fit=True, ax=ax)
ax.set_xlim(-2, 2)
top = ax.get_ylim()[1] * 0.75
txt = ax.text(left, top, "dist=stats.t, \nline='45', \nfit=True",
              verticalalignment='top')
txt.set_bbox(dict(facecolor='k', alpha=0.1))

fig.tight_layout()

plt.gcf()
