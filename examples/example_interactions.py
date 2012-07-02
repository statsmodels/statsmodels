"""Interactions and ANOVA
"""
#NOTE: This script is based heavily on Jonathan Taylor's class notes
#http://www.stanford.edu/class/stats191/interactions.html
from urllib2 import urlopen

import numpy as np
import statsmodels.api as sm
import pandas
import matplotlib.pyplot as plt
from matplotlib import figure

from statsmodels.formula.api import ols
from statsmodels.graphics.api import interaction_plot, abline_plot
from statsmodels.stats.anova import anova_lm

try:
    salary_table = pandas.read_csv('salary.table')
except: # recent pandas should be able to read URL
    url = 'http://stats191.stanford.edu/data/salary.table'
    fh = urlopen(url)
    salary_table = pandas.read_table(fh)
    salary_table.to_csv('salary.table')

E = salary_table.E
M = salary_table.M
X = salary_table.X
S = salary_table.S

# Take a look at the data

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='Experience', ylabel='Salary')
#ax.set_xlim(0, 21)
symbols = ['D', '^']
colors = ['r', 'g', 'blue']
factor_groups = salary_table.groupby(['E','M'])
for values, group in factor_groups:
    i,j = values
    ax.scatter(group['X'], group['S'], marker=symbols[j], color=colors[i-1],
               s=144)
ax.axis('tight')
plt.show()

# Fit a linear model


formula = 'S ~ C(E) + C(M) + X'
lm = ols(formula, salary_table).fit()
print lm.summary()

# Have a look at the created design matrix

lm.model.exog[:20]

# Or since we initially passed in a DataFrame, we have a DataFrame available in

lm.model._data._orig_exog

# We keep a reference to the original untouched data in

lm.model._data.frame

# Get influence statistics
infl = lm.get_influence()

print infl.summary_table()

# or get a dataframe
df_infl = infl.summary_frame()

#Now plot the reiduals within the groups separately
resid = lm.resid

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='Group', ylabel='Residuals')
for values, group in factor_groups:
    i,j = values
    group_num = i*2 + j - 1 # for plotting purposes
    x = [group_num] * len(group)
    ax.scatter(x, resid[group.index], marker=symbols[j], color=colors[i-1],
            s=144, edgecolors='black')

ax.axis('tight')
plt.show()

# now we will test some interactions using anova or f_test

interX_lm = ols("S ~ C(E) * X + C(M)", salary_table).fit()
print interX_lm.summary()

# Do an ANOVA check
from statsmodels.stats.api import anova_lm

table1 = anova_lm(lm, interX_lm)
print table1

interM_lm = ols("S ~ X + C(E)*C(M)",
                                 df=salary_table).fit()
print interM_lm.summary()

table2 = anova_lm(lm, interM_lm)
print table2

# The design matrix as a DataFrame
interM_lm.model._data._orig_exog
# The design matrix as an ndarray
interM_lm.model.exog
interM_lm.model.exog_names

infl = interM_lm.get_influence()
resid = infl.resid_studentized_internal

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='X', ylabel='standardized resids')

for values, group in factor_groups:
    i,j = values
    idx = group.index
    ax.scatter(X[idx], resid[idx], marker=symbols[j], color=colors[i-1],
            s=144, edgecolors='black')
ax.axis('tight')
plt.show()

# Looks like one observation is an outlier.
#TODO: do we have Bonferonni outlier test?

drop_idx = abs(resid).argmax()
print drop_idx # zero-based index
idx = salary_table.index.drop(drop_idx)

lm32 = ols('S ~ C(E) + X + C(M)', df=salary_table, subset=idx).fit()

print lm32.summary()

interX_lm32 = ols('S ~ C(E) * X + C(M)', df=salary_table, subset=idx).fit()

print interX_lm32.summary()

table3 = anova_lm(lm32, interX_lm32)
print table3

interM_lm32 = ols('S ~ X + C(E) * C(M)', df=salary_table, subset=idx).fit()

table4 = anova_lm(lm32, interM_lm32)
print table4

# Replot the residuals
try:
    resid = interM_lm32.get_influence().summary_frame()['standard_resid']
except:
    resid = interM_lm32.get_influence().summary_frame()['standard_resid']

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='X[~[32]]', ylabel='standardized resids')

for values, group in factor_groups:
    i,j = values
    idx = group.index
    ax.scatter(X[idx], resid[idx], marker=symbols[j], color=colors[i-1],
            s=144, edgecolors='black')
ax.axis('tight')
plt.show()


# Plot the fitted values

lm_final = ols('S ~ X + C(E)*C(M)',
                    df = salary_table.drop([drop_idx])).fit()
mf = lm_final.model._data._orig_exog
lstyle = ['-','--']

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='Experience', ylabel='Salary')

for values, group in factor_groups:
    i,j = values
    idx = group.index
    ax.scatter(X[idx], S[idx], marker=symbols[j], color=colors[i-1],
            s=144, edgecolors='black')
    # drop NA because there is no idx 32 in the final model
    ax.plot(mf.X[idx].dropna(), lm_final.fittedvalues[idx].dropna(),
            ls=lstyle[j], color=colors[i-1])
ax.axis('tight')
plt.show()

#From our first look at the data, the difference between Master's and PhD in the management group is different than in the non-management group. This is an interaction between the two qualitative variables management,M and education,E. We can visualize this by first removing the effect of experience, then plotting the means within each of the 6 groups using interaction.plot.

U = S - X * interX_lm32.params['X']

ax = interaction_plot(E, M, U, colors=['red','blue'], markers=['^','D'],
        markersize=10)
plt.show()

# Minority Employment Data
# ------------------------

try:
    minority_table = pandas.read_table('minority.table')
except: # don't have data already
    url = 'http://stats191.stanford.edu/data/minority.table'
    try:
        minority_table = pandas.read_table(url)
    except: # don't have recent pandas
        fh = urlopen(url)
        minority_table = pandas.read_table(fh)

factor_group = minority_table.groupby(['ETHN'])

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='TEST', ylabel='JPERF')
colors = ['purple', 'green']
markers = ['o', 'v']
for factor, group in factor_group:
    ax.scatter(group['TEST'], group['JPERF'], color=colors[factor],
                marker=markers[factor], s=12**2)

plt.show()

min_lm = ols('JPERF ~ TEST', df=minority_table).fit()
print min_lm.summary()

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='TEST', ylabel='JPERF')
for factor, group in factor_group:
    ax.scatter(group['TEST'], group['JPERF'], color=colors[factor],
                marker=markers[factor], s=12**2)

abline_plot(model_results = min_lm, ax=ax)
plt.show()

min_lm2 = ols('JPERF ~ TEST + TEST:ETHN',
        df=minority_table).fit()

print min_lm2.summary()

fig = plt.figure()
ax = fig.add_subplot(111)
for factor, group in factor_group:
    ax.scatter(group['TEST'], group['JPERF'], color=colors[factor],
                marker=markers[factor], s=12**2)

fig = abline_plot(intercept = min_lm2.params['Intercept'],
                 slope = min_lm2.params['TEST'], ax=ax, color='purple')
ax = fig.axes[0]
fig = abline_plot(intercept = min_lm2.params['Intercept'],
        slope = min_lm2.params['TEST'] + min_lm2.params['TEST:ETHN'],
        ax=ax, color='green')
plt.show()


min_lm3 = ols('JPERF ~ TEST + ETHN', df = minority_table).fit()
print min_lm3.summary()

fig = plt.figure()
ax = fig.add_subplot(111)
for factor, group in factor_group:
    ax.scatter(group['TEST'], group['JPERF'], color=colors[factor],
                marker=markers[factor], s=12**2)

fig = abline_plot(intercept = min_lm3.params['Intercept'],
                 slope = min_lm3.params['TEST'], ax=ax, color='purple')
ax = fig.axes[0]
fig = abline_plot(intercept = min_lm3.params['Intercept'] + min_lm3.params['ETHN'],
        slope = min_lm3.params['TEST'], ax=ax, color='green')
plt.show()


min_lm4 = ols('JPERF ~ TEST * ETHN', df = minority_table).fit()
print min_lm4.summary()

fig = plt.figure()
ax = fig.add_subplot(111)
for factor, group in factor_group:
    ax.scatter(group['TEST'], group['JPERF'], color=colors[factor],
                marker=markers[factor], s=12**2)

fig = abline_plot(intercept = min_lm4.params['Intercept'],
                 slope = min_lm4.params['TEST'], ax=ax, color='purple')
ax = fig.axes[0]
fig = abline_plot(intercept = min_lm4.params['Intercept'] + min_lm4.params['ETHN'],
        slope = min_lm4.params['TEST'] + min_lm4.params['TEST:ETHN'],
        ax=ax, color='green')
plt.show()

# is there any effect of ETHN on slope or intercept
table5 = anova_lm(min_lm, min_lm4)
print table5
# is there any effect of ETHN on intercept
table6 = anova_lm(min_lm, min_lm3)
print table6
# is there any effect of ETHN on slope
table7 = anova_lm(min_lm, min_lm2)
print table7
# is it just the slope or both?
table8 = anova_lm(min_lm2, min_lm4)
print table8


# One-way ANOVA
# -------------


try:
    rehab_table = pandas.read_csv('rehab.table')
except:
    url = 'http://stats191.stanford.edu/data/rehab.csv'
    try:
        rehab_table = pandas.read_table(url, delimiter=",")
    except:
        rehab_table = pandas.read_table(urlopen(url), delimiter=",")
    rehab_table.to_csv('rehab.table')

ax = rehab_table.boxplot('Time', 'Fitness')
plt.show()

rehab_lm = ols('Time ~ C(Fitness)',
                df=rehab_table).fit()
table9 = anova_lm(rehab_lm)
print table9

print rehab_lm.model._data._orig_exog

print rehab_lm.summary()

# Two-way ANOVA
# -------------

# pandas fails on this table
try:
    kidney_table = pandas.read_table('./kidney.table')
except:
    url = 'http://stats191.stanford.edu/data/kidney.table'
    kidney_table = pandas.read_table(url, delimiter=" *")

# Explore the dataset
kidney_table.groupby(['Weight', 'Duration']).size()
# balanced panel

kt = kidney_table
interaction_plot(kt['Weight'], kt['Duration'], np.log(kt['Days']+1),
        colors=['red', 'blue'], markers=['D','^'], ms=10)
plt.show()

# You have things available in the calling namespace available
# in the formula evaluation namespace
kidney_lm = ols('np.log(Days+1) ~ C(Duration) * C(Weight)',
        df=kt).fit()

table10 = anova_lm(kidney_lm)

print anova_lm(ols('np.log(Days+1) ~ C(Duration) + C(Weight)',
                df=kt).fit(), kidney_lm)
print anova_lm(ols('np.log(Days+1) ~ C(Duration)', df=kt).fit(),
               ols('np.log(Days+1) ~ C(Duration) + C(Weight, Sum)',
                   df=kt).fit())
print anova_lm(ols('np.log(Days+1) ~ C(Weight)', df=kt).fit(),
               ols('np.log(Days+1) ~ C(Duration) + C(Weight, Sum)',
                   df=kt).fit())

# Sum of squares
# --------------
#
# Illustrates the use of different types of sums of squares (I,II,II)
# and how the Sum contrast can be used to produce the same output between
# the 3.

# Types I and II are equivalent under a balanced design.

# Don't use Type III with non-orthogonal contrast - ie., Treatment

sum_lm = ols('np.log(Days+1) ~ C(Duration, Sum) * C(Weight, Sum)',
            df=kt).fit()

print anova_lm(sum_lm)
print anova_lm(sum_lm, typ=2)
print anova_lm(sum_lm, typ=3)

nosum_lm = ols('np.log(Days+1) ~ C(Duration, Treatment) * C(Weight, Treatment)',
            df=kt).fit()
print anova_lm(nosum_lm)
print anova_lm(nosum_lm, typ=2)
print anova_lm(nosum_lm, typ=3)
