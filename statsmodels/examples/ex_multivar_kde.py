
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

import statsmodels.api as sm

"""
This example illustrates the nonparametric estimation of a
bivariate bi-modal distribution that is a mixture of two normal
distributions.

author: George Panterov
"""
if __name__ == '__main__':
    np.random.seed(123456)

    # generate the data
    nobs = 500
    BW = 'cv_ml'

    mu1 = [3, 4]
    mu2 = [6, 1]
    cov1 = np.asarray([[1, 0.7], [0.7, 1]])
    cov2 = np.asarray([[1, -0.7], [-0.7, 1]])

    ix = np.random.uniform(size=nobs) > 0.5
    V = np.random.multivariate_normal(mu1, cov1, size=nobs)
    V[ix, :] = np.random.multivariate_normal(mu2, cov2, size=nobs)[ix, :]

    x = V[:, 0]
    y = V[:, 1]

    dens = sm.nonparametric.KDEMultivariate(data=[x, y], var_type='cc', bw=BW,
                                            defaults=sm.nonparametric.EstimatorSettings(efficient=True))

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
