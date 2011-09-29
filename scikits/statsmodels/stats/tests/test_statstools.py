

import numpy as np
from numpy.testing import assert_almost_equal
from scikits.statsmodels.stats.stattools import (omni_normtest, jarque_bera,
                            durbin_watson)
from scikits.statsmodels.stats.adnorm import ad_normal


#a random array, rounded to 4 decimals
x = np.array([-0.1184, -1.3403,  0.0063, -0.612 , -0.3869, -0.2313, -2.8485,
       -0.2167,  0.4153,  1.8492, -0.3706,  0.9726, -0.1501, -0.0337,
       -1.4423,  1.2489,  0.9182, -0.2331, -0.6182,  0.183 ])


def test_durbin_watson():
    #benchmark values from R car::durbinWatsonTest(x)
    #library("car")
    #> durbinWatsonTest(x)
    #[1] 1.95298958377419
    #> durbinWatsonTest(x**2)
    #[1] 1.848802400319998
    #> durbinWatsonTest(x[2:20]+0.5*x[1:19])
    #[1] 1.09897993228779
    #> durbinWatsonTest(x[2:20]+0.8*x[1:19])
    #[1] 0.937241876707273
    #> durbinWatsonTest(x[2:20]+0.9*x[1:19])
    #[1] 0.921488912587806
    st_R = 1.95298958377419
    assert_almost_equal(durbin_watson(x), st_R, 14)

    st_R = 1.848802400319998
    assert_almost_equal(durbin_watson(x**2), st_R, 15)

    st_R = 1.09897993228779
    assert_almost_equal(durbin_watson(x[1:]+0.5*x[:-1]), st_R, 15)

    st_R = 0.937241876707273
    assert_almost_equal(durbin_watson(x[1:]+0.8*x[:-1]), st_R, 15)

    st_R = 0.921488912587806
    assert_almost_equal(durbin_watson(x[1:]+0.9*x[:-1]), st_R, 15)


def test_omni_normtest():
    #tests against R fBasics
    from scipy import stats
    st_pv_R = np.array(
              [[3.994138321207883, -1.129304302161460,  1.648881473704978],
               [0.1357325110375005, 0.2587694866795507, 0.0991719192710234]])

    nt = omni_normtest(x)
    assert_almost_equal(nt, st_pv_R[:,0], 14)

    st = stats.skewtest(x)
    assert_almost_equal(st, st_pv_R[:,1], 15)

    kt = stats.kurtosistest(x)
    assert_almost_equal(kt, st_pv_R[:,2], 15)

    st_pv_R = np.array(
              [[34.523210399523926,  4.429509162503833,  3.860396220444025],
               [3.186985686465249e-08, 9.444780064482572e-06, 1.132033129378485e-04]])

    x2 = x**2
    nt = omni_normtest(x2)
    assert_almost_equal(nt, st_pv_R[:,0], 14)

    st = stats.skewtest(x2)
    assert_almost_equal(st, st_pv_R[:,1], 15)

    kt = stats.kurtosistest(x2)
    assert_almost_equal(kt, st_pv_R[:,2], 15)

def test_jarque_bera():
    #tests against R fBasics
    st_pv_R = np.array([1.9662677226861689, 0.3741367669648314])
    jb = jarque_bera(x)[:2]
    assert_almost_equal(jb, st_pv_R, 14)

    st_pv_R = np.array([78.329987305556, 0.000000000000])
    jb = jarque_bera(x**2)[:2]
    assert_almost_equal(jb, st_pv_R, 13)

    st_pv_R = np.array([5.7135750796706670, 0.0574530296971343])
    jb = jarque_bera(np.log(x**2))[:2]
    assert_almost_equal(jb, st_pv_R, 14)

    st_pv_R = np.array([2.6489315748495761, 0.2659449923067881])
    jb = jarque_bera(np.exp(-x**2))[:2]
    assert_almost_equal(jb, st_pv_R, 15)

def test_shapiro():
    #tests against R fBasics
    #testing scipy.stats
    from scipy.stats import shapiro

    st_pv_R = np.array([0.939984787255526, 0.239621898000460])
    sh = shapiro(x)
    assert_almost_equal(sh, st_pv_R, 6)

    #st is ok -7.15e-06, pval agrees at -3.05e-10
    st_pv_R = np.array([5.799574255943298e-01, 1.838456834681376e-06 * 1e4])
    sh = shapiro(x**2)*np.array([1,1e4])
    assert_almost_equal(sh, st_pv_R, 6)

    st_pv_R = np.array([0.91730442643165588, 0.08793704167882448 ])
    sh = shapiro(np.log(x**2))
    assert_almost_equal(sh, st_pv_R, 6)

    #diff is [  9.38773155e-07,   5.48221246e-08]
    st_pv_R = np.array([0.818361863493919373, 0.001644620895206969 ])
    sh = shapiro(np.exp(-x**2))
    assert_almost_equal(sh, st_pv_R, 6)

def test_adnorm():
    #tests against R fBasics
    st_pv = []
    st_pv_R = np.array([0.5867235358882148, 0.1115380760041617])
    ad = ad_normal(x)
    assert_almost_equal(ad, st_pv_R, 14)
    st_pv.append(st_pv_R)

    st_pv_R = np.array([2.976266267594575e+00, 8.753003709960645e-08])
    ad = ad_normal(x**2)
    assert_almost_equal(ad, st_pv_R, 13)
    st_pv.append(st_pv_R)

    st_pv_R = np.array([0.4892557856308528, 0.1968040759316307])
    ad = ad_normal(np.log(x**2))
    assert_almost_equal(ad, st_pv_R, 14)
    st_pv.append(st_pv_R)

    st_pv_R = np.array([1.4599014654282669312, 0.0006380009232897535])
    ad = ad_normal(np.exp(-x**2))
    assert_almost_equal(ad, st_pv_R, 15)
    st_pv.append(st_pv_R)

    ad = ad_normal(np.column_stack((x,x**2, np.log(x**2),np.exp(-x**2))).T,
                   axis=1)
    assert_almost_equal(ad, np.column_stack(st_pv), 14)


if __name__ == '__main__':
    test_durbin_watson()
    test_omni_normtest()
    test_jarque_bera()
    test_shapiro()
    test_adnorm()