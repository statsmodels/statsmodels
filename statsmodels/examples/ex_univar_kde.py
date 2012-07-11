#import nonparametric2 as nparam
import statsmodels.nonparametric as nparam
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
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

author: George Panterov
"""

# Beta distribution

# Parameters
a=2
b=5
N=250

support = np.random.beta(a,b, size = N)
rv = stats.beta(a,b)
ix = np.argsort(support)

dens_normal = nparam.UKDE(tdat=[support], var_type='c', bw='normal_reference')
dens_cvls = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ls')
dens_cvml = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ml')

plt.figure(1)
plt.plot(support[ix],rv.pdf(support[ix]), label='Actual')
plt.plot(support[ix],dens_normal.pdf()[ix],label='Scott')
plt.plot(support[ix],dens_cvls.pdf()[ix], label='CV_LS')
plt.plot(support[ix],dens_cvml.pdf()[ix], label='CV_ML')
plt.title("Nonparametric Estimation of the Density of Beta Distributed Random Variable")
plt.legend(('Actual','Scott','CV_LS','CV_ML'))

# f distribution
df=100
dn=100
N=250

support = np.random.f(dn,df, size = N)
rv = stats.f(df,dn)
ix = np.argsort(support)

dens_normal = nparam.UKDE(tdat=[support], var_type='c', bw='normal_reference')
dens_cvls = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ls')
dens_cvml = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ml')

plt.figure(2)
plt.plot(support[ix],rv.pdf(support[ix]), label='Actual')
plt.plot(support[ix],dens_normal.pdf()[ix],label='Scott')
plt.plot(support[ix],dens_cvls.pdf()[ix], label='CV_LS')
plt.plot(support[ix],dens_cvml.pdf()[ix], label='CV_ML')
plt.title("Nonparametric Estimation of the Density of f Distributed Random Variable")
plt.legend(('Actual','Scott','CV_LS','CV_ML'))

# Pareto distribution
a=2
N=150
support = np.random.pareto(a, size = N)
rv = stats.pareto(a)
ix = np.argsort(support)

dens_normal = nparam.UKDE(tdat=[support], var_type='c', bw='normal_reference')
dens_cvls = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ls')
dens_cvml = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ml')
plt.figure(3)
plt.plot(support[ix],rv.pdf(support[ix]), label='Actual')
plt.plot(support[ix],dens_normal.pdf()[ix],label='Scott')
plt.plot(support[ix],dens_cvls.pdf()[ix], label='CV_LS')
plt.plot(support[ix],dens_cvml.pdf()[ix], label='CV_ML')
plt.title("Nonparametric Estimation of the Density of Pareto Distributed Random Variable")
plt.legend(('Actual','Scott','CV_LS','CV_ML'))

# Laplace Distribution

mu=0
s=1
N=250

support = np.random.laplace(mu, s, size = N)
rv = stats.laplace(mu, s)
ix = np.argsort(support)

dens_normal = nparam.UKDE(tdat=[support], var_type='c', bw='normal_reference')
dens_cvls = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ls')
dens_cvml = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ml')

plt.figure(4)
plt.plot(support[ix],rv.pdf(support[ix]), label='Actual')
plt.plot(support[ix],dens_normal.pdf()[ix],label='Scott')
plt.plot(support[ix],dens_cvls.pdf()[ix], label='CV_LS')
plt.plot(support[ix],dens_cvml.pdf()[ix], label='CV_ML')
plt.title("Nonparametric Estimation of the Density of Laplace Distributed Random Variable")
plt.legend(('Actual','Scott','CV_LS','CV_ML'))

# Weibull Distribution
a=1
N=250

support = np.random.weibull(a, size = N)
rv = stats.weibull_min(a)

ix = np.argsort(support)
dens_normal = nparam.UKDE(tdat=[support], var_type='c', bw='normal_reference')
dens_cvls = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ls')
dens_cvml = nparam.UKDE(tdat=[support], var_type='c', bw='cv_ml')

plt.figure(5)
plt.plot(support[ix],rv.pdf(support[ix]), label='Actual')
plt.plot(support[ix],dens_normal.pdf()[ix],label='Scott')
plt.plot(support[ix],dens_cvls.pdf()[ix], label='CV_LS')
plt.plot(support[ix],dens_cvml.pdf()[ix], label='CV_ML')
plt.title("Nonparametric Estimation of the Density of Weibull Distributed Random Variable")
plt.legend(('Actual','Scott','CV_LS','CV_ML'))

# Poisson Distribution
a=2
N=250
support = np.random.poisson(a, size = N)
rv = stats.poisson(a)

ix = np.argsort(support)
dens_normal = nparam.UKDE(tdat=[support], var_type='o', bw='normal_reference')
dens_cvls = nparam.UKDE(tdat=[support], var_type='o', bw='cv_ls')
dens_cvml = nparam.UKDE(tdat=[support], var_type='o', bw='cv_ml')
plt.figure(6)
plt.plot(support[ix],rv.pmf(support[ix]), label='Actual')
plt.plot(support[ix],dens_normal.pdf()[ix],label='Scott')
plt.plot(support[ix],dens_cvls.pdf()[ix], label='CV_LS')
plt.plot(support[ix],dens_cvml.pdf()[ix], label='CV_ML')
plt.title("Nonparametric Estimation of the Density of Poisson Distributed Random Variable")
plt.legend(('Actual','Scott','CV_LS','CV_ML'))
plt.show()
