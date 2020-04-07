# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:31:21 2020

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from statsmodels.tools.testing import Holder

"""
example from Kacker 2004, computed with R metafor

> y = c(61.0, 61.4 , 62.21, 62.3 , 62.34, 62.6 , 62.7 , 62.84, 65.9)
> v = c(0.2025, 1.2100, 0.0900, 0.2025, 0.3844, 0.5625, 0.0676, 0.0225, 1.8225)
> res = rma(y, v, data=dat, method="PM", control=list(tol=1e-9))
> convert_items(res, prefix="exk1_metafor.")

"""

exk1_metafor = Holder()
exk1_metafor.b = 62.4076199113286
exk1_metafor.beta = 62.4076199113286
exk1_metafor.se = 0.338030602684471
exk1_metafor.zval = 184.621213037276
exk1_metafor.pval = 0
exk1_metafor.ci_lb = 61.7450921043947
exk1_metafor.ci_ub = 63.0701477182625
exk1_metafor.vb = 0.114264688351227
exk1_metafor.tau2 = 0.705395309224248
exk1_metafor.se_tau2 = 0.51419109758052
exk1_metafor.tau2_f = 0.705395309224248
exk1_metafor.k = 9
exk1_metafor.k_f = 9
exk1_metafor.k_eff = 9
exk1_metafor.k_all = 9
exk1_metafor.p = 1
exk1_metafor.p_eff = 1
exk1_metafor.parms = 2
exk1_metafor.m = 1
exk1_metafor.QE = 24.801897741835
exk1_metafor.QEp = 0.00167935146372742
exk1_metafor.QM = 34084.9923033553
exk1_metafor.QMp = 0
exk1_metafor.I2 = 83.7218626490482
exk1_metafor.H2 = 6.14320900751909
exk1_metafor.yi = np.array([
    61, 61.4, 62.21, 62.3, 62.34, 62.6, 62.7, 62.84, 65.9
    ])
exk1_metafor.vi = np.array([
    0.2025, 1.21, 0.09, 0.2025, 0.3844, 0.5625, 0.0676, 0.0225, 1.8225
    ])
exk1_metafor.X = np.array([
    1, 1, 1, 1, 1, 1, 1, 1, 1
    ]).reshape(9, 1, order='F')

exk1_metafor.yi_f = np.array([
    61, 61.4, 62.21, 62.3, 62.34, 62.6, 62.7, 62.84, 65.9
    ])
exk1_metafor.vi_f = np.array([
    0.2025, 1.21, 0.09, 0.2025, 0.3844, 0.5625, 0.0676, 0.0225, 1.8225
    ])
exk1_metafor.X_f = np.array([
    1, 1, 1, 1, 1, 1, 1, 1, 1
    ]).reshape(9, 1, order='F')

exk1_metafor.M = np.array([
    0.907895309224248, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.91539530922425, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0.795395309224248, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.907895309224248, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.08979530922425, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1.26789530922425, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.772995309224248, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.727895309224248, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 2.52789530922425
    ]).reshape(9, 9, order='F')

exk1_metafor.ids = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9
    ])
exk1_metafor.slab = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9
    ])
exk1_metafor.measure = 'GEN'
exk1_metafor.method = 'PM'
exk1_metafor.test = 'z'
exk1_metafor.s2w = 1
exk1_metafor.btt = 1
exk1_metafor.digits = np.array([
    4, 4, 4, 4, 4, 4, 4, 4, 4
    ])
exk1_metafor.level = 0.05
exk1_metafor.add = 0.5
exk1_metafor.to = 'only0'
exk1_metafor.fit_stats = np.array([
    -12.722152033808, 21.73438033144, 29.4443040676159, 29.8387532222884,
    31.4443040676159, -11.7892200590463, 23.5784401180925, 27.5784401180925,
    27.7373232014522, 29.9784401180925
    ]).reshape(5, 2, order='F')

exk1_metafor.model = 'rma.uni'


# > res = rma(y, v, data=dat, method="DL", control=list(tol=1e-9))
# > convert_items(res, prefix="exk1_dl.")

exk1_dl = Holder()
exk1_dl.b = 62.3901386044504
exk1_dl.beta = 62.3901386044504
exk1_dl.se = 0.245749668040304
exk1_dl.zval = 253.876797075543
exk1_dl.pval = 0
exk1_dl.ci_lb = 61.9084781058787
exk1_dl.ci_ub = 62.8717991030221
exk1_dl.vb = 0.0603928993419195
exk1_dl.tau2 = 0.288049246973751
exk1_dl.se_tau2 = 0.269366223207558
exk1_dl.tau2_f = 0.288049246973751
exk1_dl.k = 9
exk1_dl.k_f = 9
exk1_dl.k_eff = 9
exk1_dl.k_all = 9
exk1_dl.p = 1
exk1_dl.p_eff = 1
exk1_dl.parms = 2
exk1_dl.m = 1
exk1_dl.QE = 24.801897741835
exk1_dl.QEp = 0.00167935146372742
exk1_dl.QM = 64453.4280933367
exk1_dl.QMp = 0
exk1_dl.I2 = 67.744403741711
exk1_dl.H2 = 3.10023721772938


# > res = rma(y, v, data=dat, method="DL", test="knha", control=list(tol=1e-9))
# > convert_items(res, prefix="exk1_dl_hksj.")

exk1_dl_hksj = Holder()
exk1_dl_hksj.b = 62.3901386044504
exk1_dl_hksj.beta = 62.3901386044504
exk1_dl_hksj.se = 0.29477605699879
exk1_dl_hksj.zval = 211.652666908108
exk1_dl_hksj.pval = 2.77938607433693e-16
exk1_dl_hksj.ci_lb = 61.710383798052
exk1_dl_hksj.ci_ub = 63.0698934108488
exk1_dl_hksj.vb = 0.0868929237797541
exk1_dl_hksj.tau2 = 0.288049246973751
exk1_dl_hksj.se_tau2 = 0.269366223207558
exk1_dl_hksj.tau2_f = 0.288049246973751
exk1_dl_hksj.k = 9
exk1_dl_hksj.k_f = 9
exk1_dl_hksj.k_eff = 9
exk1_dl_hksj.k_all = 9
exk1_dl_hksj.p = 1
exk1_dl_hksj.p_eff = 1
exk1_dl_hksj.parms = 2
exk1_dl_hksj.m = 1
exk1_dl_hksj.QE = 24.801897741835
exk1_dl_hksj.QEp = 0.00167935146372742
exk1_dl_hksj.QM = 44796.8514093144
exk1_dl_hksj.QMp = 2.77938607433693e-16
exk1_dl_hksj.I2 = 67.744403741711
exk1_dl_hksj.H2 = 3.10023721772938



# > res = rma(y, v, data=dat, method="FE", control=list(tol=1e-9))
# > convert_items(res, prefix="exk1_fe.")

exk1_fe = Holder()
exk1_fe.b = 62.5833970939982
exk1_fe.beta = 62.5833970939982
exk1_fe.se = 0.107845705498231
exk1_fe.zval = 580.304953311515
exk1_fe.pval = 0
exk1_fe.ci_lb = 62.3720233953344
exk1_fe.ci_ub = 62.7947707926621
exk1_fe.vb = 0.0116306961944112
exk1_fe.tau2 = 0
exk1_fe.tau2_f = 0
exk1_fe.k = 9
exk1_fe.k_f = 9
exk1_fe.k_eff = 9
exk1_fe.k_all = 9
exk1_fe.p = 1
exk1_fe.p_eff = 1
exk1_fe.parms = 1
exk1_fe.m = 1
exk1_fe.QE = 24.801897741835
exk1_fe.QEp = 0.00167935146372742
exk1_fe.QM = 336753.838837879
exk1_fe.QMp = 0
exk1_fe.I2 = 67.744403741711
exk1_fe.H2 = 3.10023721772938



# > res = rma(y, v, data=dat, method="FE", test="knha", control=list(tol=1e-9))
# Warning message:
# In rma(y, v, data = dat, method = "FE", test = "knha", control = list(tol = 1e-09)) :
#  Knapp & Hartung method is not meant to be used in the context of FE models.
# > convert_items(res, prefix="exk1_fe_hksj.")

exk1_fe_hksj = Holder()
exk1_fe_hksj.b = 62.5833970939982
exk1_fe_hksj.beta = 62.5833970939982
exk1_fe_hksj.se = 0.189889223522271
exk1_fe_hksj.zval = 329.57845597098
exk1_fe_hksj.pval = 8.04326466920145e-18
exk1_fe_hksj.ci_lb = 62.1455117593252
exk1_fe_hksj.ci_ub = 63.0212824286713
exk1_fe_hksj.vb = 0.0360579172098909
exk1_fe_hksj.tau2 = 0
exk1_fe_hksj.tau2_f = 0
exk1_fe_hksj.k = 9
exk1_fe_hksj.k_f = 9
exk1_fe_hksj.k_eff = 9
exk1_fe_hksj.k_all = 9
exk1_fe_hksj.p = 1
exk1_fe_hksj.p_eff = 1
exk1_fe_hksj.parms = 1
exk1_fe_hksj.m = 1
exk1_fe_hksj.QE = 24.801897741835
exk1_fe_hksj.QEp = 0.00167935146372742
exk1_fe_hksj.QM = 108621.958640215
exk1_fe_hksj.QMp = 8.04326466920145e-18
exk1_fe_hksj.I2 = 67.744403741711
exk1_fe_hksj.H2 = 3.10023721772938
