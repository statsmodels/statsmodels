# -*- coding: utf-8 -*-
"""

Created on Wed Jan 02 09:17:40 2013

Author: Josef Perktold based on test file by George Panterov
"""

from __future__ import print_function
import numpy as np
import numpy.testing as npt

import statsmodels.nonparametric.api as nparam
#import statsmodels.api as sm
#nparam = sm.nonparametric



italy_gdp = \
        [8.556, 12.262, 9.587, 8.119, 5.537, 6.796, 8.638,
         6.483, 6.212, 5.111, 6.001, 7.027, 4.616, 3.922,
         4.688, 3.957, 3.159, 3.763, 3.829, 5.242, 6.275,
         8.518, 11.542, 9.348, 8.02, 5.527, 6.865, 8.666,
         6.672, 6.289, 5.286, 6.271, 7.94, 4.72, 4.357,
         4.672, 3.883, 3.065, 3.489, 3.635, 5.443, 6.302,
         9.054, 12.485, 9.896, 8.33, 6.161, 7.055, 8.717,
         6.95]

italy_year = \
        [1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951,
       1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1952,
       1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952,
       1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1953, 1953,
       1953, 1953, 1953, 1953, 1953, 1953]

italy_year = np.asarray(italy_year, float)

model = nparam.KernelReg(endog=[italy_gdp],
                         exog=[italy_year], reg_type='lc',
                         var_type='o', bw='cv_ls')

sm_bw = model.bw
R_bw = 0.1390096

sm_mean, sm_mfx = model.fit()
sm_mean2 = sm_mean[0:5]
sm_mfx = sm_mfx[0:5]
R_mean = 6.190486

sm_R2 = model.r_squared()
R_R2 = 0.1435323

npt.assert_allclose(sm_bw, R_bw, atol=1e-2)
npt.assert_allclose(sm_mean2, R_mean, atol=1e-2)
npt.assert_allclose(sm_R2, R_R2, atol=1e-2)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(italy_year, italy_gdp, 'o')
ax.plot(italy_year, sm_mean, '-')

plt.show()
