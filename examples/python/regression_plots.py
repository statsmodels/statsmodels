
## Regression Plots

from __future__ import print_function
from statsmodels.compat import lzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols


### Duncan's Prestige Dataset

#### Load the Data

# We can use a utility function to load any R dataset available from the great <a href="http://vincentarelbundock.github.com/Rdatasets/">Rdatasets package</a>.

prestige = sm.datasets.get_rdataset("Duncan", "car", cache=True).data


prestige.head()


prestige_model = ols("prestige ~ income + education", data=prestige).fit()


print(prestige_model.summary())


#### Influence plots

# Influence plots show the (externally) studentized residuals vs. the leverage of each observation as measured by the hat matrix.
#
# Externally studentized residuals are residuals that are scaled by their standard deviation where
#
# $$var(\\hat{\epsilon}_i)=\hat{\sigma}^2_i(1-h_{ii})$$
#
# with
#
# $$\hat{\sigma}^2_i=\frac{1}{n - p - 1 \;\;}\sum_{j}^{n}\;\;\;\forall \;\;\; j \neq i$$
#
# $n$ is the number of observations and $p$ is the number of regressors. $h_{ii}$ is the $i$-th diagonal element of the hat matrix
#
# $$H=X(X^{\;\prime}X)^{-1}X^{\;\prime}$$
#
# The influence of each point can be visualized by the criterion keyword argument. Options are Cook's distance and DFFITS, two measures of influence.

fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.influence_plot(prestige_model, ax=ax, criterion="cooks")


# As you can see there are a few worrisome observations. Both contractor and reporter have low leverage but a large residual. <br />
# RR.engineer has small residual and large leverage. Conductor and minister have both high leverage and large residuals, and, <br />
# therefore, large influence.

#### Partial Regression Plots

# Since we are doing multivariate regressions, we cannot just look at individual bivariate plots to discern relationships. <br />
# Instead, we want to look at the relationship of the dependent variable and independent variables conditional on the other <br />
# independent variables. We can do this through using partial regression plots, otherwise known as added variable plots. <br />
#
# In a partial regression plot, to discern the relationship between the response variable and the $k$-th variabe, we compute <br />
# the residuals by regressing the response variable versus the independent variables excluding $X_k$. We can denote this by <br />
# $X_{\sim k}$. We then compute the residuals by regressing $X_k$ on $X_{\sim k}$. The partial regression plot is the plot <br />
# of the former versus the latter residuals. <br />
#
# The notable points of this plot are that the fitted line has slope $\beta_k$ and intercept zero. The residuals of this plot <br />
# are the same as those of the least squares fit of the original model with full $X$. You can discern the effects of the <br />
# individual data values on the estimation of a coefficient easily. If obs_labels is True, then these points are annotated <br />
# with their observation label. You can also see the violation of underlying assumptions such as homooskedasticity and <br />
# linearity.

fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.plot_partregress("prestige", "income", ["income", "education"], data=prestige, ax=ax)
ax = fig.axes[0]

ax.set_xlim(-2e-15, 1e-14)
ax.set_ylim(-25, 30);


fix, ax = plt.subplots(figsize=(12,14))
fig = sm.graphics.plot_partregress("prestige", "income", ["education"], data=prestige, ax=ax)


# As you can see the partial regression plot confirms the influence of conductor, minister, and RR.engineer on the partial relationship between income and prestige. The cases greatly decrease the effect of income on prestige. Dropping these cases confirms this.

subset = ~prestige.index.isin(["conductor", "RR.engineer", "minister"])
prestige_model2 = ols("prestige ~ income + education", data=prestige, subset=subset).fit()
print(prestige_model2.summary())


# For a quick check of all the regressors, you can use plot_partregress_grid. These plots will not label the <br />
# points, but you can use them to identify problems and then use plot_partregress to get more information.

fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(prestige_model, fig=fig)


#### Component-Component plus Residual (CCPR) Plots

# The CCPR plot provides a way to judge the effect of one regressor on the <br />
# response variable by taking into account the effects of the other  <br />
# independent variables. The partial residuals plot is defined as  <br />
# $\text{Residuals} + B_iX_i \text{ }\text{ }$   versus $X_i$. The component adds $B_iX_i$ versus  <br />
# $X_i$ to show where the fitted line would lie. Care should be taken if $X_i$  <br />
# is highly correlated with any of the other independent variables. If this  <br />
# is the case, the variance evident in the plot will be an underestimate of  <br />
# the true variance.

fig, ax = plt.subplots(figsize=(12, 8))
fig = sm.graphics.plot_ccpr(prestige_model, "education", ax=ax)


# As you can see the relationship between the variation in prestige explained by education conditional on income seems to be linear, though you can see there are some observations that are exerting considerable influence on the relationship. We can quickly look at more than one variable by using plot_ccpr_grid.

fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_ccpr_grid(prestige_model, fig=fig)


#### Regression Plots

# The plot_regress_exog function is a convenience function that gives a 2x2 plot containing the dependent variable and fitted values with confidence intervals vs. the independent variable chosen, the residuals of the model vs. the chosen independent variable, a partial regression plot, and a CCPR plot. This function can be used for quickly checking modeling assumptions with respect to a single regressor.

fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(prestige_model, "education", fig=fig)


#### Fit Plot

# The plot_fit function plots the fitted values versus a chosen independent variable. It includes prediction confidence intervals and optionally plots the true dependent variable.

fig, ax = plt.subplots(figsize=(12, 8))
fig = sm.graphics.plot_fit(prestige_model, "education", ax=ax)


### Statewide Crime 2009 Dataset

# Compare the following to http://www.ats.ucla.edu/stat/stata/webbooks/reg/chapter4/statareg_self_assessment_answers4.htm
#
# Though the data here is not the same as in that example. You could run that example by uncommenting the necessary cells below.

#dta = pd.read_csv("http://www.stat.ufl.edu/~aa/social/csv_files/statewide-crime-2.csv")
#dta = dta.set_index("State", inplace=True).dropna()
#dta.rename(columns={"VR" : "crime",
#                    "MR" : "murder",
#                    "M"  : "pctmetro",
#                    "W"  : "pctwhite",
#                    "H"  : "pcths",
#                    "P"  : "poverty",
#                    "S"  : "single"
#                    }, inplace=True)
#
#crime_model = ols("murder ~ pctmetro + poverty + pcths + single", data=dta).fit()


dta = sm.datasets.statecrime.load_pandas().data


crime_model = ols("murder ~ urban + poverty + hs_grad + single", data=dta).fit()
print(crime_model.summary())


#### Partial Regression Plots

fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(crime_model, fig=fig)


fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.plot_partregress("murder", "hs_grad", ["urban", "poverty", "single"],  ax=ax, data=dta)


#### Leverage-Resid<sup>2</sup> Plot

# Closely related to the influence_plot is the leverage-resid<sup>2</sup> plot.

fig, ax = plt.subplots(figsize=(8,6))
fig = sm.graphics.plot_leverage_resid2(crime_model, ax=ax)


#### Influence Plot

fig, ax = plt.subplots(figsize=(8,6))
fig = sm.graphics.influence_plot(crime_model, ax=ax)


#### Using robust regression to correct for outliers.

# Part of the problem here in recreating the Stata results is that M-estimators are not robust to leverage points. MM-estimators should do better with this examples.

from statsmodels.formula.api import rlm


rob_crime_model = rlm("murder ~ urban + poverty + hs_grad + single", data=dta,
                      M=sm.robust.norms.TukeyBiweight(3)).fit(conv="weights")
print(rob_crime_model.summary())


#rob_crime_model = rlm("murder ~ pctmetro + poverty + pcths + single", data=dta, M=sm.robust.norms.TukeyBiweight()).fit(conv="weights")
#print(rob_crime_model.summary())


# There aren't yet an influence diagnostics as part of RLM, but we can recreate them. (This depends on the status of [issue #888](https://github.com/statsmodels/statsmodels/issues/808))

weights = rob_crime_model.weights
idx = weights > 0
X = rob_crime_model.model.exog[idx]
ww = weights[idx] / weights[idx].mean()
hat_matrix_diag = ww*(X*np.linalg.pinv(X).T).sum(1)
resid = rob_crime_model.resid
resid2 = resid**2
resid2 /= resid2.sum()
nobs = int(idx.sum())
hm = hat_matrix_diag.mean()
rm = resid2.mean()


from statsmodels.graphics import utils
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(resid2[idx], hat_matrix_diag, 'o')
ax = utils.annotate_axes(range(nobs), labels=rob_crime_model.model.data.row_labels[idx],
                    points=lzip(resid2[idx], hat_matrix_diag), offset_points=[(-5,5)]*nobs,
                    size="large", ax=ax)
ax.set_xlabel("resid2")
ax.set_ylabel("leverage")
ylim = ax.get_ylim()
ax.vlines(rm, *ylim)
xlim = ax.get_xlim()
ax.hlines(hm, *xlim)
ax.margins(0,0)

