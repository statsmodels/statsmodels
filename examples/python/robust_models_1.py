
## M-Estimators for Robust Linear Modeling

# In[ ]:

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import statsmodels.api as sm


# * An M-estimator minimizes the function 
# 
# $$Q(e_i, \rho) = \sum_i~\rho \left (\frac{e_i}{s}\right )$$
# 
# where $\rho$ is a symmetric function of the residuals 
# 
# * The effect of $\rho$ is to reduce the influence of outliers
# * $s$ is an estimate of scale. 
# * The robust estimates $\hat{\beta}$ are computed by the iteratively re-weighted least squares algorithm

# * We have several choices available for the weighting functions to be used

# In[ ]:

norms = sm.robust.norms


# In[ ]:

def plot_weights(support, weights_func, xlabels, xticks):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax.plot(support, weights_func(support))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=16)
    ax.set_ylim(-.1, 1.1)
    return ax


#### Andrew's Wave

# In[ ]:

help(norms.AndrewWave.weights)


# In[ ]:

a = 1.339
support = np.linspace(-np.pi*a, np.pi*a, 100)
andrew = norms.AndrewWave(a=a)
plot_weights(support, andrew.weights, ['$-\pi*a$', '0', '$\pi*a$'], [-np.pi*a, 0, np.pi*a]);


#### Hampel's 17A

# In[ ]:

help(norms.Hampel.weights)


# In[ ]:

c = 8
support = np.linspace(-3*c, 3*c, 1000)
hampel = norms.Hampel(a=2., b=4., c=c)
plot_weights(support, hampel.weights, ['3*c', '0', '3*c'], [-3*c, 0, 3*c]);


#### Huber's t

# In[ ]:

help(norms.HuberT.weights)


# In[ ]:

t = 1.345
support = np.linspace(-3*t, 3*t, 1000)
huber = norms.HuberT(t=t)
plot_weights(support, huber.weights, ['-3*t', '0', '3*t'], [-3*t, 0, 3*t]);


#### Least Squares

# In[ ]:

help(norms.LeastSquares.weights)


# In[ ]:

support = np.linspace(-3, 3, 1000)
lst_sq = norms.LeastSquares()
plot_weights(support, lst_sq.weights, ['-3', '0', '3'], [-3, 0, 3]);


#### Ramsay's Ea

# In[ ]:

help(norms.RamsayE.weights)


# In[ ]:

a = .3
support = np.linspace(-3*a, 3*a, 1000)
ramsay = norms.RamsayE(a=a)
plot_weights(support, ramsay.weights, ['-3*a', '0', '3*a'], [-3*a, 0, 3*a]);


#### Trimmed Mean

# In[ ]:

help(norms.TrimmedMean.weights)


# In[ ]:

c = 2
support = np.linspace(-3*c, 3*c, 1000)
trimmed = norms.TrimmedMean(c=c)
plot_weights(support, trimmed.weights, ['-3*c', '0', '3*c'], [-3*c, 0, 3*c]);


#### Tukey's Biweight

# In[ ]:

help(norms.TukeyBiweight.weights)


# In[ ]:

c = 4.685
support = np.linspace(-3*c, 3*c, 1000)
tukey = norms.TukeyBiweight(c=c)
plot_weights(support, tukey.weights, ['-3*c', '0', '3*c'], [-3*c, 0, 3*c]);


#### Scale Estimators

# * Robust estimates of the location

# In[ ]:

x = np.array([1, 2, 3, 4, 500])


# * The mean is not a robust estimator of location

# In[ ]:

x.mean()


# * The median, on the other hand, is a robust estimator with a breakdown point of 50%

# In[ ]:

np.median(x)


# * Analagously for the scale
# * The standard deviation is not robust

# In[ ]:

x.std()


# Median Absolute Deviation
# 
# $$ median_i |X_i - median_j(X_j)|) $$

# Standardized Median Absolute Deviation is a consistent estimator for $\hat{\sigma}$
# 
# $$\hat{\sigma}=K \cdot MAD$$
# 
# where $K$ depends on the distribution. For the normal distribution for example,
# 
# $$K = \Phi^{-1}(.75)$$

# In[ ]:

stats.norm.ppf(.75)


# In[ ]:

print x


# In[ ]:

sm.robust.scale.stand_mad(x)


# In[ ]:

np.array([1,2,3,4,5.]).std()


# * The default for Robust Linear Models is MAD
# * another popular choice is Huber's proposal 2

# In[ ]:

np.random.seed(12345)
fat_tails = stats.t(6).rvs(40)


# In[ ]:

kde = sm.nonparametric.KDE(fat_tails)
kde.fit()
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(kde.support, kde.density);


# In[ ]:

print fat_tails.mean(), fat_tails.std()


# In[ ]:

print stats.norm.fit(fat_tails)


# In[ ]:

print stats.t.fit(fat_tails, f0=6)


# In[ ]:

huber = sm.robust.scale.Huber()
loc, scale = huber(fat_tails)
print loc, scale


# In[ ]:

sm.robust.stand_mad(fat_tails)


# In[ ]:

sm.robust.stand_mad(fat_tails, c=stats.t(6).ppf(.75))


# In[ ]:

sm.robust.scale.mad(fat_tails)


#### Duncan's Occupational Prestige data - M-estimation for outliers

# In[ ]:

from statsmodels.graphics.api import abline_plot
from statsmodels.formula.api import ols, rlm


# In[ ]:

prestige = sm.datasets.get_rdataset("Duncan", "car", cache=True).data


# In[ ]:

print prestige.head(10)


# In[ ]:

fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot(211, xlabel='Income', ylabel='Prestige')
ax1.scatter(prestige.income, prestige.prestige)
xy_outlier = prestige.ix['minister'][['income','prestige']]
ax1.annotate('Minister', xy_outlier, xy_outlier+1, fontsize=16)
ax2 = fig.add_subplot(212, xlabel='Education',
                           ylabel='Prestige')
ax2.scatter(prestige.education, prestige.prestige);


# In[ ]:

ols_model = ols('prestige ~ income + education', prestige).fit()
print ols_model.summary()


# In[ ]:

infl = ols_model.get_influence()
student = infl.summary_frame()['student_resid']
print student


# In[ ]:

print student.ix[np.abs(student) > 2]


# In[ ]:

print infl.summary_frame().ix['minister']


# In[ ]:

sidak = ols_model.outlier_test('sidak')
sidak.sort('unadj_p', inplace=True)
print sidak


# In[ ]:

fdr = ols_model.outlier_test('fdr_bh')
fdr.sort('unadj_p', inplace=True)
print fdr


# In[ ]:

rlm_model = rlm('prestige ~ income + education', prestige).fit()
print rlm_model.summary()


# In[ ]:

print rlm_model.weights


#### Hertzprung Russell data for Star Cluster CYG 0B1 - Leverage Points

# * Data is on the luminosity and temperature of 47 stars in the direction of Cygnus.

# In[ ]:

dta = sm.datasets.get_rdataset("starsCYG", "robustbase", cache=True).data


# In[ ]:

from matplotlib.patches import Ellipse
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, xlabel='log(Temp)', ylabel='log(Light)', title='Hertzsprung-Russell Diagram of Star Cluster CYG OB1')
ax.scatter(*dta.values.T)
# highlight outliers
e = Ellipse((3.5, 6), .2, 1, alpha=.25, color='r')
ax.add_patch(e);
ax.annotate('Red giants', xy=(3.6, 6), xytext=(3.8, 6),
            arrowprops=dict(facecolor='black', shrink=0.05, width=2),
            horizontalalignment='left', verticalalignment='bottom',
            clip_on=True, # clip to the axes bounding box
            fontsize=16,
     )
# annotate these with their index
for i,row in dta.ix[dta['log.Te'] < 3.8].iterrows():
    ax.annotate(i, row, row + .01, fontsize=14)
xlim, ylim = ax.get_xlim(), ax.get_ylim()


# In[ ]:

from IPython.display import Image
Image(filename='star_diagram.png')


# In[ ]:

y = dta['log.light']
X = sm.add_constant(dta['log.Te'], prepend=True)
ols_model = sm.OLS(y, X).fit()
abline_plot(model_results=ols_model, ax=ax)


# In[ ]:

rlm_mod = sm.RLM(y, X, sm.robust.norms.TrimmedMean(.5)).fit()
abline_plot(model_results=rlm_mod, ax=ax, color='red')


# * Why? Because M-estimators are not robust to leverage points.

# In[ ]:

infl = ols_model.get_influence()


# In[ ]:

h_bar = 2*(ols_model.df_model + 1 )/ols_model.nobs
hat_diag = infl.summary_frame()['hat_diag']
hat_diag.ix[hat_diag > h_bar]


# In[ ]:

sidak2 = ols_model.outlier_test('sidak')
sidak2.sort('unadj_p', inplace=True)
print sidak2


# In[ ]:

fdr2 = ols_model.outlier_test('fdr_bh')
fdr2.sort('unadj_p', inplace=True)
print fdr2


# * Let's delete that line

# In[ ]:

del ax.lines[-1]


# In[ ]:

weights = np.ones(len(X))
weights[X[X['log.Te'] < 3.8].index.values - 1] = 0
wls_model = sm.WLS(y, X, weights=weights).fit()
abline_plot(model_results=wls_model, ax=ax, color='green')


# * MM estimators are good for this type of problem, unfortunately, we don't yet have these yet. 
# * It's being worked on, but it gives a good excuse to look at the R cell magics in the notebook.

# In[ ]:

yy = y.values[:,None]
xx = X['log.Te'].values[:,None]


# In[ ]:

get_ipython().magic(u'load_ext rmagic')

get_ipython().magic(u'R library(robustbase)')
get_ipython().magic(u'Rpush yy xx')
get_ipython().magic(u'R mod <- lmrob(yy ~ xx);')
get_ipython().magic(u'R params <- mod$coefficients;')
get_ipython().magic(u'Rpull params')


# In[ ]:

get_ipython().magic(u'R print(mod)')


# In[ ]:

print params


# In[ ]:

abline_plot(intercept=params[0], slope=params[1], ax=ax, color='green')


#### Exercise: Breakdown points of M-estimator

# In[ ]:

np.random.seed(12345)
nobs = 200
beta_true = np.array([3, 1, 2.5, 3, -4])
X = np.random.uniform(-20,20, size=(nobs, len(beta_true)-1))
# stack a constant in front
X = sm.add_constant(X, prepend=True) # np.c_[np.ones(nobs), X]
mc_iter = 500
contaminate = .25 # percentage of response variables to contaminate


# In[ ]:

all_betas = []
for i in range(mc_iter):
    y = np.dot(X, beta_true) + np.random.normal(size=200)
    random_idx = np.random.randint(0, nobs, size=int(contaminate * nobs))
    y[random_idx] = np.random.uniform(-750, 750)
    beta_hat = sm.RLM(y, X).fit().params
    all_betas.append(beta_hat)


# In[ ]:

all_betas = np.asarray(all_betas)
se_loss = lambda x : np.linalg.norm(x, ord=2)**2
se_beta = map(se_loss, all_betas - beta_true)


##### Squared error loss

# In[ ]:

np.array(se_beta).mean()


# In[ ]:

all_betas.mean(0)


# In[ ]:

beta_true


# In[ ]:

se_loss(all_betas.mean(0) - beta_true)

