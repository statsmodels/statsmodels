"""
Created on Thu Jan  6 13:55:50 2022

Author: Josef Perktod
License: BSD-3
"""

import numpy as np

from statsmodels.tools.testing import Holder

hurdle_poisson = Holder()
# r library pscl, docvis data
# > mod = hurdle( docvis ~ aget + totchr, data=dt, zero.dist = "poisson")
hurdle_poisson.method = "BFGS"
hurdle_poisson.n = 3629
hurdle_poisson.df_null = 3627
hurdle_poisson.df_residual = 3623
hurdle_poisson.loglik = -13612.9091771797
hurdle_poisson.aic = 27237.81835436
hurdle_poisson.bic = 27274.9986288
hurdle_poisson.vcov = np.array([
    0.000239404800324688, -4.59559682721834e-05, -4.59865258972631e-05, 0, 0,
    0, -4.59559682721834e-05, 2.54346275490526e-05, -1.20164687288645e-06, 0,
    0, 0, -4.59865258972631e-05, -1.20164687288644e-06, 2.01936456643824e-05,
    0, 0, 0, 0, 0, 0, 0.00241829560973498, -0.000548499729173446,
    -0.000636055275016966, 0, 0, 0, -0.000548499729173446,
    0.000351548196602719, -6.30088654100178e-05, 0, 0, 0,
    -0.000636055275016966, -6.30088654100178e-05, 0.000562508220544602
    ]).reshape(6, 6, order="F")

hurdle_poisson.count = np.array([
    1.54175599063303, 0.0122763123129474, 0.209943725275436,
    0.0154727114729348, 0.00504327547820388, 0.00449373404468738,
    99.6435559035596, 2.43419427830254, 46.719214619218, 0,
    0.0149249819228085, 0
    ]).reshape(3, 4, order="F")

hurdle_poisson.zero = np.array([
    0.216740121452838, 0.0189277243223132, 0.386748883124962,
    0.0491761691242311, 0.0187496185721929, 0.0237172557549267,
    4.40742183282514, 1.00949916657955, 16.3066455546664,
    1.04608334100715e-05, 0.312735301122465, 8.85224303880408e-60
    ]).reshape(3, 4, order="F")

hurdle_poisson.params_table = np.concatenate((hurdle_poisson.zero,
                                              hurdle_poisson.count), axis=0)

# > dfm = data.frame(t(as.matrix(dtm)))  # at means
# > predict(mod, dfm)
hurdle_poisson.predict_mean = 6.530525236464
# > predict(mod, dfm, type="prob", at=c(0, 1, 2, 3))
hurdle_poisson.predict_prob = np.array([
    0.07266312350019, 0.005744258908698, 0.02020842274659, 0.04739576870239,
    ])
# > predict(mod, dfm, type="count")
hurdle_poisson.predict_mean_main = 7.036041748046
# > predict(mod, dfm, type="zero")
hurdle_poisson.predict_zero = 0.9281532813926

hurdle_l1 = Holder()
"""
# R pscl does not allow regularized fits, so we have to rely on internally
# generated benchmarks.
# Created on Mon Nov 24, 2025 with:
from statsmodels.discrete.truncated_model import HurdleCountModel
from statsmodels.sandbox.regression.tests.test_gmm_poisson import DATA

endog = DATA["docvis"]
exog_names = ["aget", "totchr", "const"]
exog = DATA[exog_names]

hurdle = HurdleCountModel(endog=endog, exog=exog, dist="poisson", zerodist="poisson")
hurdle_result = hurdle.fit_regularized(method="l1", alpha=1)
"""
hurdle_l1.params = [
        0.01919266, 0.38688315, 0.21549933, 0.01229817, 0.20997089, 1.54160776
]
hurdle_l1.conf_int = [
    [-0.01755779, 0.05594311],
    [0.34039707, 0.43336923],
    [0.11909076, 0.3119079],
    [np.nan, np.nan],
    [0.20107289, 0.21886889],
    [1.51681052, 1.56640499],
]
hurdle_l1.bse = [
    0.01875057, 0.02371782, 0.04918895, np.nan, 0.00453988, 0.01265188
]
hurdle_l1.nnz_params = 6
hurdle_l1.aic = 27237.819293111268
hurdle_l1.bic = 27274.999567554547
