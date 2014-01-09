# -*- coding: utf-8 -*-
"""

Created on Wed Jan 08 18:26:30 2014

Author: Josef Perktold

"""


import numpy as np
import statsmodels.graphics.gofplots as smgg
import matplotlib.pyplot as plt


# get Iris data

import os
import statsmodels.genmod
iris_dir = os.path.join(statsmodels.genmod.__path__[0], 'tests', 'results')
iris_dir = os.path.abspath(iris_dir)
iris = np.genfromtxt(os.path.join(iris_dir, 'iris.csv'), delimiter=",",
                        skip_header=1)

x = iris[iris[:,-1] == 0,:2]#-1] #.view(float)
x = iris[:, :-1]
x2 = iris[iris[:,-1] == 2,:2]#-1]


pp = smgg.get_mvnormal_probplot(x, use_chi2=True)
fig = pp.ppplot(line='45')
fig.axes[0].set_title('Chi2 PP-PLot for Multivariate Normality for Iris Data')


plt.show()
