#import nonparametric2 as nparam
import statsmodels.nonparametric as nparam
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


"""
This example illustrates the nonparametric estimation of a
bivariate bi-modal distribution that is a mixture of two normal
distributions.

author: George Panterov
"""
np.random.seed(123456)

# generate the data
N = 500
BW = 'cv_ml'

mu1 = [3, 4]
mu2 = [6, 1]
cov1 = np.asarray([[1, 0.7], [0.7, 1]])
cov2 = np.asarray([[1, -0.7], [-0.7, 1]])


A = np.random.multivariate_normal(mu1, cov1, size=N)
B = np.random.multivariate_normal(mu2, cov2, size=N)
V = np.empty((N, 2))
#generate the bimodal distribution
for i in range(N):
    if np.random.uniform() > 0.5:
        V[i, :] = A[i, :]
    else:
        V[i, :] = B[i, :]

x = V[:, 0]
y = V[:, 1]
dens = nparam.UKDE(tdat=[x, y], var_type='cc', bw=BW)

supportx = np.linspace(min(x), max(x), 60)
supporty = np.linspace(min(y), max(y), 60)
X, Y = np.meshgrid(supportx, supporty)
Z = np.empty(np.shape(X))
for i in range(np.shape(X)[0]):
    Z[i, :] = dens.pdf(edat=[X[i], Y[i]])

# plot
fig = plt.figure(1)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
        linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
