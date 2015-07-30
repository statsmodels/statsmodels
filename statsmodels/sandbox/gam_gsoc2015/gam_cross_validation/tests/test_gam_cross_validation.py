__author__ = 'Luca Puggini'

import numpy as np
import os
import pandas as pd
from statsmodels.sandbox.gam_gsoc2015.gam import GLMGam, UnivariateGamPenalty


from statsmodels.sandbox.gam_gsoc2015.smooth_basis import UnivariateBSplines
from statsmodels.sandbox.gam_gsoc2015.gam_cross_validation.cross_validators import KFold
from statsmodels.sandbox.gam_gsoc2015.gam_cross_validation.gam_cross_validation import UnivariateGamCV, UnivariateGamCVPath
import matplotlib.pyplot as plt


# def sample_metric(y1, y2):
#     return np.linalg.norm(y1 - y2)/len(y1)


# n = 2000
# x = np.linspace(-1, 1, n)
# y = x*x*x - x*x + np.random.normal(0, .1, n)
#
# df = 10
# degree = 6
# univ_bsplines = UnivariateBSplines(x, df=df, degree=degree)
# gam = GLMGam
# kfolds = KFold(10, shuffle=True)
#
# # gam_cv_error = UnivariateGamCV(gam=gam, alpha=0, cost=sample_metric, univariate_smoother=univ_bsplines, y=y,
# #                                cv=kfolds).fit(method='nm', max_start_irls=0, disp=1, maxiter=5000, maxfun=5000)
# # print(gam_cv_error)
#
# alphas = np.linspace(.0, .5, 10)
# gam_cv_path = UnivariateGamCVPath(gam=gam, alphas=alphas, cost=sample_metric, univariate_smoother=univ_bsplines, y=y,
#                                   cv=kfolds)
#
# gam_cv_path.fit(method='nm', max_start_irls=0, disp=1, maxiter=5000, maxfun=5000)
#
# best_alpha = gam_cv_path.alpha_cv_
#
#
# gp = UnivariateGamPenalty(alpha=gam_cv_path.alpha_cv_, univariate_smoother=univ_bsplines)
# gam = GLMGam(y, univ_bsplines.basis_, penal=gp)
# res = gam.fit(method='nm', max_start_irls=0, disp=0, maxiter=5000, maxfun=5000)
# y_est = res.predict()
#
# plt.plot(y_est)
# plt.plot(y, '.')
# plt.show()
#
# gam_cv_path.plot_path()
# plt.show()
