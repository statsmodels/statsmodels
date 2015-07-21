from statsmodels.sandbox.gam_gsoc2015.gam import GLMGam
from statsmodels.sandbox.gam_gsoc2015.smooth_basis import CubicSplines
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
cur_dir = os.path.dirname(os.path.abspath('__file__'))
file_path = os.path.join(cur_dir, "tests/results", "gam_PIRLS_results.csv")
data = pd.read_csv(file_path)

print('Univariate GAM ')
X = data['x'].as_matrix()
Y = data['y'].as_matrix()

XK = np.array([0.2, .4, .6, .8])


cs = CubicSplines(X, 4).fit()

print(cs.x.shape, cs.xs.shape, cs.s.shape)



for i, alpha in enumerate([0, .1, 10, 200]):

    gam = GLMGam(Y, X, penal=0)
    gam_results = gam._fit_pirls(Y, cs.xs, cs.s, alpha)
    Y_EST = np.dot(cs.xs, gam_results.params)
    plt.subplot(2, 2, i+1)
    plt.title('Alpha=' + str(alpha))
    plt.plot(X, Y, '.')
    plt.plot(X, Y_EST, '.')
plt.show()
