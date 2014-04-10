"""
This example tests the nonparametric estimator
for several popular univariate distributions with the different
bandwidth selction methods - CV-ML; CV-LS; Scott's rule of thumb.

Produces six different plots for each distribution
1) Beta
2) f
3) Pareto
4) Laplace
5) Weibull
6) Poisson

"""


from __future__ import print_function
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

KDEMultivariate = sm.nonparametric.KDEMultivariate


np.random.seed(123456)

# Beta distribution

# Parameters
a = 2
b = 5
nobs = 250

support = np.random.beta(a, b, size=nobs)
rv = stats.beta(a, b)
ix = np.argsort(support)

dens_normal = KDEMultivariate(data=[support], var_type='c', bw='normal_reference')
dens_cvls = KDEMultivariate(data=[support], var_type='c', bw='cv_ls')
dens_cvml = KDEMultivariate(data=[support], var_type='c', bw='cv_ml')

plt.figure(1)
plt.plot(support[ix], rv.pdf(support[ix]), label='Actual')
plt.plot(support[ix], dens_normal.pdf()[ix], label='Scott')
plt.plot(support[ix], dens_cvls.pdf()[ix], label='CV_LS')
plt.plot(support[ix], dens_cvml.pdf()[ix], label='CV_ML')
plt.title("Nonparametric Estimation of the Density of Beta Distributed " \
          "Random Variable")
plt.legend(('Actual', 'Scott', 'CV_LS', 'CV_ML'))

# f distribution
df = 100
dn = 100
nobs = 250

support = np.random.f(dn, df, size=nobs)
rv = stats.f(df, dn)
ix = np.argsort(support)

dens_normal = KDEMultivariate(data=[support], var_type='c', bw='normal_reference')
dens_cvls = KDEMultivariate(data=[support], var_type='c', bw='cv_ls')
dens_cvml = KDEMultivariate(data=[support], var_type='c', bw='cv_ml')

plt.figure(2)
plt.plot(support[ix], rv.pdf(support[ix]), label='Actual')
plt.plot(support[ix], dens_normal.pdf()[ix], label='Scott')
plt.plot(support[ix], dens_cvls.pdf()[ix], label='CV_LS')
plt.plot(support[ix], dens_cvml.pdf()[ix], label='CV_ML')
plt.title("Nonparametric Estimation of the Density of f Distributed " \
          "Random Variable")
plt.legend(('Actual', 'Scott', 'CV_LS', 'CV_ML'))

# Pareto distribution
a = 2
nobs = 150
support = np.random.pareto(a, size=nobs)
rv = stats.pareto(a)
ix = np.argsort(support)

dens_normal = KDEMultivariate(data=[support], var_type='c', bw='normal_reference')
dens_cvls = KDEMultivariate(data=[support], var_type='c', bw='cv_ls')
dens_cvml = KDEMultivariate(data=[support], var_type='c', bw='cv_ml')
plt.figure(3)
plt.plot(support[ix], rv.pdf(support[ix]), label='Actual')
plt.plot(support[ix], dens_normal.pdf()[ix], label='Scott')
plt.plot(support[ix], dens_cvls.pdf()[ix], label='CV_LS')
plt.plot(support[ix], dens_cvml.pdf()[ix], label='CV_ML')
plt.title("Nonparametric Estimation of the Density of Pareto " \
          "Distributed Random Variable")
plt.legend(('Actual', 'Scott', 'CV_LS', 'CV_ML'))

# Laplace Distribution
mu = 0
s = 1
nobs = 250

support = np.random.laplace(mu, s, size=nobs)
rv = stats.laplace(mu, s)
ix = np.argsort(support)

dens_normal = KDEMultivariate(data=[support], var_type='c', bw='normal_reference')
dens_cvls = KDEMultivariate(data=[support], var_type='c', bw='cv_ls')
dens_cvml = KDEMultivariate(data=[support], var_type='c', bw='cv_ml')

plt.figure(4)
plt.plot(support[ix], rv.pdf(support[ix]), label='Actual')
plt.plot(support[ix], dens_normal.pdf()[ix], label='Scott')
plt.plot(support[ix], dens_cvls.pdf()[ix], label='CV_LS')
plt.plot(support[ix], dens_cvml.pdf()[ix], label='CV_ML')
plt.title("Nonparametric Estimation of the Density of Laplace " \
          "Distributed Random Variable")
plt.legend(('Actual', 'Scott', 'CV_LS', 'CV_ML'))

# Weibull Distribution
a = 1
nobs = 250

support = np.random.weibull(a, size=nobs)
rv = stats.weibull_min(a)

ix = np.argsort(support)
dens_normal = KDEMultivariate(data=[support], var_type='c', bw='normal_reference')
dens_cvls = KDEMultivariate(data=[support], var_type='c', bw='cv_ls')
dens_cvml = KDEMultivariate(data=[support], var_type='c', bw='cv_ml')

plt.figure(5)
plt.plot(support[ix], rv.pdf(support[ix]), label='Actual')
plt.plot(support[ix], dens_normal.pdf()[ix], label='Scott')
plt.plot(support[ix], dens_cvls.pdf()[ix], label='CV_LS')
plt.plot(support[ix], dens_cvml.pdf()[ix], label='CV_ML')
plt.title("Nonparametric Estimation of the Density of Weibull " \
          "Distributed Random Variable")
plt.legend(('Actual', 'Scott', 'CV_LS', 'CV_ML'))

# Poisson Distribution
a = 2
nobs = 250
support = np.random.poisson(a, size=nobs)
rv = stats.poisson(a)

ix = np.argsort(support)
dens_normal = KDEMultivariate(data=[support], var_type='o', bw='normal_reference')
dens_cvls = KDEMultivariate(data=[support], var_type='o', bw='cv_ls')
dens_cvml = KDEMultivariate(data=[support], var_type='o', bw='cv_ml')

plt.figure(6)
plt.plot(support[ix], rv.pmf(support[ix]), label='Actual')
plt.plot(support[ix], dens_normal.pdf()[ix], label='Scott')
plt.plot(support[ix], dens_cvls.pdf()[ix], label='CV_LS')
plt.plot(support[ix], dens_cvml.pdf()[ix], label='CV_ML')
plt.title("Nonparametric Estimation of the Density of Poisson " \
          "Distributed Random Variable")
plt.legend(('Actual', 'Scott', 'CV_LS', 'CV_ML'))

plt.show()
