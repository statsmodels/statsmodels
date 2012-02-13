# -*- coding: utf-8 -*-
"""

Created on Sat Dec 17 08:39:16 2011

Author: Josef Perktold
"""
import numpy as np
from numpy.testing import assert_almost_equal

from scikits.statsmodels.regression.linear_model import OLS, GLSAR
from scikits.statsmodels.tools.tools import add_constant
import scikits.statsmodels.sandbox.panel.sandwich_covariance as sw
#import scikits.statsmodels.sandbox.panel.sandwich_covariance_generic as swg

def test_cov_cluster_2groups():
    #comparing cluster robust standard errors to Peterson
    #requires Petersen's test_data
    #http://www.kellogg.northwestern.edu/faculty/petersen/htm/papers/se/test_data.txt
    pet = np.genfromtxt("test_data.txt")
    endog = pet[:,-1]
    group = pet[:,0].astype(int)
    time = pet[:,1].astype(int)
    exog = sm.add_constant(pet[:,2], prepend=True)
    res = sm.OLS(endog, exog).fit()

    cov01, covg, covt = sw.cov_cluster_2groups(res, group, group2=time)

    #Reference number from Petersen
    #http://www.kellogg.northwestern.edu/faculty/petersen/htm/papers/se/test_data.htm

    bse_petw = [0.0284, 0.0284]
    bse_pet0 = [0.0670, 0.0506]
    bse_pet1 = [0.0234, 0.0334]  #year
    bse_pet01 = [0.0651, 0.0536]  #firm and year
    bse_0 = sw.se_cov(covg)
    bse_1 = sw.se_cov(covt)
    bse_01 = sw.se_cov(cov01)
    print res.HC0_se, bse_petw - res.HC0_se
    print bse_0, bse_0 - bse_pet0
    print bse_1, bse_1 - bse_pet1
    print bse_01, bse_01 - bse_pet01
    assert_almost_equal(bse_petw, res.HC0_se, decimal=4)
    assert_almost_equal(bse_0, bse_pet0, decimal=4)
    assert_almost_equal(bse_1, bse_pet1, decimal=4)
    assert_almost_equal(bse_01, bse_pet01, decimal=4)

def test_hac_simple():

    from scikits.statsmodels.datasets import macrodata
    d2 = macrodata.load().data
    g_gdp = 400*np.diff(np.log(d2['realgdp']))
    g_inv = 400*np.diff(np.log(d2['realinv']))
    exogg = add_constant(np.c_[g_gdp, d2['realint'][:-1]],prepend=True)
    res_olsg = OLS(g_inv, exogg).fit()



    #> NeweyWest(fm, lag = 4, prewhite = FALSE, sandwich = TRUE, verbose=TRUE, adjust=TRUE)
    #Lag truncation parameter chosen: 4
    #                     (Intercept)                   ggdp                  lint
    cov1_r = [[  1.40643899878678802, -0.3180328707083329709, -0.060621111216488610],
             [ -0.31803287070833292,  0.1097308348999818661,  0.000395311760301478],
             [ -0.06062111121648865,  0.0003953117603014895,  0.087511528912470993]]

    #> NeweyWest(fm, lag = 4, prewhite = FALSE, sandwich = TRUE, verbose=TRUE, adjust=FALSE)
    #Lag truncation parameter chosen: 4
    #                    (Intercept)                  ggdp                  lint
    cov2_r = [[ 1.3855512908840137, -0.313309610252268500, -0.059720797683570477],
             [ -0.3133096102522685,  0.108101169035130618,  0.000389440793564339],
             [ -0.0597207976835705,  0.000389440793564336,  0.086211852740503622]]

    cov1, se1 = sw.cov_hac_simple(res_olsg, nlags=4, use_correction=True)
    cov2, se2 = sw.cov_hac_simple(res_olsg, nlags=4, use_correction=False)
    assert_almost_equal(cov1, cov1_r, decimal=14)
    assert_almost_equal(cov2, cov2_r, decimal=14)

if __name__ == '__main__':
    test_hac_simple()
