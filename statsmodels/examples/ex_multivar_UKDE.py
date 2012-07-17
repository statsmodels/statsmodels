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

ix = np.random.uniform(size=N) > 0.5
V = np.random.multivariate_normal(mu1, cov1, size=N)
V[ix, :] = np.random.multivariate_normal(mu2, cov2, size=N)[ix, :]

x = V[:, 0]
y = V[:, 1]

dens = nparam.UKDE(tdat=[x, y], var_type='cc', bw=BW)

supportx = np.linspace(min(x), max(x), 60)
supporty = np.linspace(min(y), max(y), 60)
X, Y = np.meshgrid(supportx, supporty)

edat = np.column_stack([X.ravel(), Y.ravel()])
Z = dens.pdf(edat).reshape(X.shape)

# plot
fig = plt.figure(1)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
        linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.figure(2)
plt.imshow(Z)

plt.show()
