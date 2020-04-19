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
# In rma(y, v, data = dat, method = "FE", test = "knha",
#        control = list(tol = 1e-09)) :
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


# effect size for proportions, metafor `escalc` function

# > library(metafor)
# > dat <- dat.fine1993
# > dat_or <- escalc(measure="OR", ai=e2i, n1i=nei, ci=c2i, n2i=nci, data=dat,
#                    var.names=c("y2i","v2i"))
# > r = dat_or[c("y2i", "v2i")]
# > cat_items(r)
y_or = np.array([
    0.13613217432458, 0.768370601797533, 0.374938517449009, 1.65822807660353,
    0.784954729813068, 0.361663949151077, 0.575364144903562,
    0.250542525502324, 0.650587566141149, 0.0918075492531228,
    0.273865253802803, 0.485755524477543, 0.182321556793955,
    0.980829253011726, 1.31218638896617, -0.259511195485084, 0.138402322859119
    ])
v_or = np.array([
    0.399242424242424, 0.244867149758454, 0.152761481951271, 0.463095238095238,
    0.189078465394255, 0.0689052107900588, 0.240651709401709,
    0.142027027027027, 0.280657748049052, 0.210140736456526,
    0.0373104717196078, 0.0427774287950624, 0.194901960784314,
    0.509259259259259, 1.39835164835165, 0.365873015873016, 0.108630952380952
    ])

# > dat_rr <- escalc(measure="RR", ai=e2i, n1i=nei, ci=c2i, n2i=nci, data=dat,
#                    var.names=c("y2i","v2i"))
# > r = dat_rr[c("y2i", "v2i")]
# > cat_items(r)
y_rr = np.array([
    0.0595920972022457, 0.434452644981417, 0.279313822781264,
    0.934309237376833, 0.389960921572199, 0.219327702635984,
    0.328504066972036, 0.106179852041229, 0.28594445255324,
    0.0540672212702757, 0.164912297594691, 0.300079561474504,
    0.0813456394539525, 0.693147180559945, 0.177206456127184,
    -0.131336002061087, 0.0622131845015728
    ])
v_rr = np.array([
    0.0761562998405104, 0.080905695611578, 0.0856909430438842,
    0.175974025974026, 0.0551968864468864, 0.0267002515563729,
    0.074017094017094, 0.0257850995555914, 0.0590338164251208,
    0.073266499582289, 0.0137191240428942, 0.0179386112192693,
    0.0400361415752742, 0.3, 0.0213675213675214, 0.0922402159244264,
    0.021962676962677
    ])

# > dat_rd <- escalc(measure="RD", ai=e2i, n1i=nei, ci=c2i, n2i=nci, data=dat,
#                    var.names=c("y2i","v2i"))
# > r = dat_rd[c("y2i", "v2i")]
# > cat_items(r)
y_rd = np.array([
    0.0334928229665072, 0.186554621848739, 0.071078431372549,
    0.386363636363636, 0.19375, 0.0860946401581211, 0.14, 0.0611028315946349,
    0.158888888888889, 0.0222222222222222, 0.0655096935584741,
    0.114173373020248, 0.045021186440678, 0.2, 0.150793650793651,
    -0.0647773279352226, 0.0342342342342342
    ])
v_rd = np.array([
    0.0240995805934916, 0.0137648162576944, 0.00539777447807907,
    0.0198934072126221, 0.0109664132254464, 0.00376813659489987,
    0.0142233846153846, 0.00842011053321928, 0.0163926076817558,
    0.0122782676856751, 0.00211164860232433, 0.00219739135615223,
    0.0119206723560942, 0.016, 0.014339804116826, 0.0226799351233969,
    0.00663520262409963
    ])


# > dat_as <- escalc(measure="AS", ai=e2i, n1i=nei, ci=c2i, n2i=nci, data=dat,
#                    var.names=c("y2i","v2i"))
# > r = dat_as[c("y2i", "v2i")]
# > cat_items(r)
y_as = np.array([
    0.0337617513001424, 0.189280827304914, 0.0815955178338458,
    0.399912703180945, 0.194987153482868, 0.0882233598093272,
    0.141897054604164, 0.0618635353537276, 0.160745373792417,
    0.0225840453649413, 0.0669694915300637, 0.117733830714136,
    0.0452997410410423, 0.221071594001477, 0.220332915310739,
    -0.0648275244591966, 0.0344168494848509
    ])
v_as = np.array([
    0.0245215311004785, 0.0144957983193277, 0.00714869281045752,
    0.0238636363636364, 0.0113839285714286, 0.00402569468666434,
    0.0146153846153846, 0.00864381520119225, 0.0169444444444444,
    0.0126984126984127, 0.0022181832395247, 0.00242071803917245,
    0.0120497881355932, 0.0222222222222222, 0.0317460317460317,
    0.0227732793522267, 0.00671171171171171
    ])

eff_prop1 = Holder(y_rd=y_rd, v_rd=v_rd, y_rr=y_rr, v_rr=v_rr,
                   y_or=y_or, v_or=v_or, y_as=y_as, v_as=v_as)
