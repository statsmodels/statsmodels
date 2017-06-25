# -*- coding: utf-8 -*-
"""

Created on Fri Aug 31 00:03:21 2012

Author: Josef Perktold
"""

import numpy as np
from statsmodels.tsa.johansen import coint_johansen

if __name__ == '__main__':

    np.random.seed(9642567)
    nobs = 500
    fact = np.cumsum(0.2 + np.random.randn(nobs, 4),0)

    xx = np.random.randn(nobs+2, 6)
    xx = xx[2:] + 0.6 * xx[1:-1] + 0.25 * xx[:-2]
    xx[:,:2] += fact[:,0][:,None]
    #xx[:,2:3] += fact[:,1][:,None]
    xx[:,2:4] += fact[:,1][:,None]
    xx[:,4:] += fact[:,-2:]

    p, k = 1, 2

    result = coint_johansen(xx, p, k)

    print(result.lr1 > result.cvt.T)
    print(result.lr2 > result.cvm.T)
    print(np.round(result.evec,4))
    print(result.eig)


    #I guess test statistic looks good, but
    #print np.round(result.evec,4)
    #looks strange, I don't see the interpretation
    #changed the DGP: I had some I(0) not integrated series included
    #      now all series are I(1) in the DGP
    # -> evec looks better now

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax0 = fig.add_subplot(3,1,1)
    pp = ax0.plot(xx)
    ax1 = fig.add_subplot(3,1,2)
    pp = ax1.plot(result.r0t, '.')
    ax2 = fig.add_subplot(3,1,3)
    pp = ax2.plot(result.rkt)
    plt.show()

    import os
    import statsmodels.tsa.tests
    test_path = os.path.dirname(os.path.abspath(statsmodels.tsa.tests.__file__))
    dta = np.genfromtxt(open(test_path + "/results/test_coint.csv", "rb"))

    #dta = np.genfromtxt(r"E:\Josef\eclipsegworkspace\matlab\jplv\coint\test.dat")
    res = coint_johansen(dta, 1, 2)
    print('\ntrace')
    print(np.column_stack((res.lr1, res.cvt)))
    print('max eval')
    print(np.column_stack((res.lr2, res.cvm)))

    ev = np.array([0.01102517075074406, -0.2185481584930077, 0.04565819524210763, -0.06556394587400775, 0.04711496306104131, -0.1500111976629196, 0.03775327003706507, 0.03479475877437702, 0.007517888890275335, -0.2014629352546497, 0.01526001455616041, 0.0707900418057458, -0.002388919695513273, 0.04486516694838273, -0.02936314422571188, 0.009900554050392113, 0.02846074144367176, 0.02021385478834498, -0.04276914888645468, 0.1738024290422287, 0.07821155002012749, -0.1066523077111768, -0.3011042488399306, 0.04965189679477353, 0.07141291326159237, -0.01406702689857725, -0.07842109866080313, -0.04773566072362181, -0.04768640728128824, -0.04428737926285261, 0.4143225656833862, 0.04512787132114879, -0.06817130121837202, 0.2246249779872569, -0.009356548567565763, 0.006685350535849125, -0.02040894506833539, 0.008131690308487425, -0.2503209797396666, 0.01560186979508953, 0.03327070126502506, -0.263036624535624, -0.04669882107497259, 0.0146457545413255, 0.01408691619062709, 0.1004753600191269, -0.02239205763487946, -0.02169291468272568, 0.08782313160608619, -0.07696508791577318, 0.008925177304198475, -0.06230900392092828, -0.01548907461158638, 0.04574831652028973, -0.2972228156126774, 0.003469819004961912, -0.001868995544352928, 0.05993345996347871, 0.01213394328069316, 0.02096614212178651, -0.08624395993789938, 0.02108183181049973, -0.08470307289295617, -5.135072530480897e-005])
    print(np.max(np.abs(ev - res.evec.ravel('C'))))

    res09 = coint_johansen(dta, 0, 9)
    #fprintf(1, '%18.16g, ', r1)
    res1_m = np.array([241.985452556075,  166.4781461662553,  110.3298006342814,  70.79801574443575,  44.90887371527634,  27.22385073668511,  11.74205493173769,  3.295435325623445,           169.0618,           133.7852,           102.4674,            75.1027,            51.6492,            32.0645,            16.1619,             2.7055,           175.1584,            139.278,           107.3429,  79.34220000000001,            55.2459,            35.0116,            18.3985,             3.8415,           187.1891,           150.0778,           116.9829,            87.7748,            62.5202,            41.0815,            23.1485,             6.6349])
    #r2 = [res.lr2 res.cvm]
    res2_m = np.array([75.50730638981975,  56.14834553197396,   39.5317848898456,   25.8891420291594,  17.68502297859124,  15.48179580494741,  8.446619606114249,  3.295435325623445,            52.5858,            46.5583,            40.5244,            34.4202,            28.2398,            21.8731,            15.0006,             2.7055,            55.7302,            49.5875,            43.4183,            37.1646,            30.8151,            24.2522,            17.1481,             3.8415,            62.1741,            55.8171,            49.4095,            42.8612,             36.193,            29.2631,            21.7465,             6.6349])

    resm18 = coint_johansen(dta, -1, 8)
