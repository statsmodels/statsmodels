#!/usr/bin/env python

# DO NOT EDIT
# Autogenerated from the notebook variance_components.ipynb.
# Edit the notebook and then sync the output with this file.
#
# flake8: noqa
# DO NOT EDIT

# # Variance Component Analysis
#
# This notebook illustrates variance components analysis for two-level
# nested and crossed designs.

import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import VCSpec
import pandas as pd

# Make the notebook reproducible

np.random.seed(3123)

# ## Nested analysis

# In our discussion below, "Group 2" is nested within "Group 1".  As a
# concrete example, "Group 1" might be school districts, with "Group
# 2" being individual schools.  The function below generates data from
# such a population.  In a nested analysis, the group 2 labels that
# are nested within different group 1 labels are treated as
# independent groups, even if they have the same label.  For example,
# two schools labeled "school 1" that are in two different school
# districts are treated as independent schools, even though they have
# the same label.


def generate_nested(n_group1=200,
                    n_group2=20,
                    n_rep=10,
                    group1_sd=2,
                    group2_sd=3,
                    unexplained_sd=4):

    # Group 1 indicators
    group1 = np.kron(np.arange(n_group1), np.ones(n_group2 * n_rep))

    # Group 1 effects
    u = group1_sd * np.random.normal(size=n_group1)
    effects1 = np.kron(u, np.ones(n_group2 * n_rep))

    # Group 2 indicators
    group2 = np.kron(np.ones(n_group1),
                     np.kron(np.arange(n_group2), np.ones(n_rep)))

    # Group 2 effects
    u = group2_sd * np.random.normal(size=n_group1 * n_group2)
    effects2 = np.kron(u, np.ones(n_rep))

    e = unexplained_sd * np.random.normal(size=n_group1 * n_group2 * n_rep)
    y = effects1 + effects2 + e

    df = pd.DataFrame({"y": y, "group1": group1, "group2": group2})

    return df


# Generate a data set to analyze.

df = generate_nested()

# Using all the default arguments for `generate_nested`, the population
# values of "group 1 Var" and "group 2 Var" are 2^2=4 and 3^2=9,
# respectively.  The unexplained variance, listed as "scale" at the
# top of the summary table, has population value 4^2=16.

model1 = sm.MixedLM.from_formula(
    "y ~ 1",
    re_formula="1",
    vc_formula={"group2": "0 + C(group2)"},
    groups="group1",
    data=df,
)
result1 = model1.fit()
print(result1.summary())

# If we wish to avoid the formula interface, we can fit the same model
# by building the design matrices manually.


def f(x):
    n = x.shape[0]
    g2 = x.group2
    u = g2.unique()
    u.sort()
    uv = {v: k for k, v in enumerate(u)}
    mat = np.zeros((n, len(u)))
    for i in range(n):
        mat[i, uv[g2.iloc[i]]] = 1
    colnames = ["%d" % z for z in u]
    return mat, colnames


# Then we set up the variance components using the VCSpec class.

vcm = df.groupby("group1").apply(f).to_list()
mats = [x[0] for x in vcm]
colnames = [x[1] for x in vcm]
names = ["group2"]
vcs = VCSpec(names, [colnames], [mats])

# Finally we fit the model.  It can be seen that the results of the
# two fits are identical.

oo = np.ones(df.shape[0])
model2 = sm.MixedLM(df.y, oo, exog_re=oo, groups=df.group1, exog_vc=vcs)
result2 = model2.fit()
print(result2.summary())

# ## Crossed analysis

# In a crossed analysis, the levels of one group can occur in any
# combination with the levels of the another group.  The groups in
# Statsmodels MixedLM are always nested, but it is possible to fit a
# crossed model by having only one group, and specifying all random
# effects as variance components.  Many, but not all crossed models
# can be fit in this way.  The function below generates a crossed data
# set with two levels of random structure.


def generate_crossed(n_group1=100,
                     n_group2=100,
                     n_rep=4,
                     group1_sd=2,
                     group2_sd=3,
                     unexplained_sd=4):

    # Group 1 indicators
    group1 = np.kron(np.arange(n_group1, dtype=int),
                     np.ones(n_group2 * n_rep, dtype=int))
    group1 = group1[np.random.permutation(len(group1))]

    # Group 1 effects
    u = group1_sd * np.random.normal(size=n_group1)
    effects1 = u[group1]

    # Group 2 indicators
    group2 = np.kron(np.arange(n_group2, dtype=int),
                     np.ones(n_group2 * n_rep, dtype=int))
    group2 = group2[np.random.permutation(len(group2))]

    # Group 2 effects
    u = group2_sd * np.random.normal(size=n_group2)
    effects2 = u[group2]

    e = unexplained_sd * np.random.normal(size=n_group1 * n_group2 * n_rep)
    y = effects1 + effects2 + e

    df = pd.DataFrame({"y": y, "group1": group1, "group2": group2})

    return df


# Generate a data set to analyze.

df = generate_crossed()

# Next we fit the model, note that the `groups` vector is constant.
# Using the default parameters for `generate_crossed`, the level 1
# variance should be 2^2=4, the level 2 variance should be 3^2=9, and
# the unexplained variance should be 4^2=16.

vc = {"g1": "0 + C(group1)", "g2": "0 + C(group2)"}
oo = np.ones(df.shape[0])
model3 = sm.MixedLM.from_formula("y ~ 1", groups=oo, vc_formula=vc, data=df)
result3 = model3.fit()
print(result3.summary())

# If we wish to avoid the formula interface, we can fit the same model
# by building the design matrices manually.


def f(g):
    n = len(g)
    u = g.unique()
    u.sort()
    uv = {v: k for k, v in enumerate(u)}
    mat = np.zeros((n, len(u)))
    for i in range(n):
        mat[i, uv[g[i]]] = 1
    colnames = ["%d" % z for z in u]
    return [mat], [colnames]


vcm = [f(df.group1), f(df.group2)]
mats = [x[0] for x in vcm]
colnames = [x[1] for x in vcm]
names = ["group1", "group2"]
vcs = VCSpec(names, colnames, mats)

# Here we fit the model without using formulas, it is simple to check
# that the results for models 3 and 4 are identical.

oo = np.ones(df.shape[0])
model4 = sm.MixedLM(df.y, oo[:, None], exog_re=None, groups=oo, exog_vc=vcs)
result4 = model4.fit()
print(result4.summary())
