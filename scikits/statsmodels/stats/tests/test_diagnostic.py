# -*- coding: utf-8 -*-
"""Tests for Regression Diagnostics and Specification Tests

Created on Thu Feb 09 13:19:47 2012

Author: Josef Perktold
License: BSD-3

currently all tests are against R

"""
import os

import numpy as np

from numpy.testing import assert_almost_equal, assert_equal, assert_approx_equal

from scikits.statsmodels.regression.linear_model import OLS, GLSAR
from scikits.statsmodels.tools.tools import add_constant
from scikits.statsmodels.datasets import macrodata

import scikits.statsmodels.sandbox.panel.sandwich_covariance as sw
import scikits.statsmodels.stats.diagnostic as smsdia
#import scikits.statsmodels.sandbox.stats.diagnostic as smsdia
import local_scripts.outliers_influence as oi

cur_dir = os.path.abspath(os.path.dirname(__file__))

def compare_t_est(sp, sp_dict, decimal=(14, 14)):
    assert_almost_equal(sp[0], sp_dict['statistic'], decimal=decimal[0])
    assert_almost_equal(sp[1], sp_dict['pvalue'], decimal=decimal[1])


def notyet_atst():
    d = macrodata.load().data

    realinv = d['realinv']
    realgdp = d['realgdp']
    realint = d['realint']
    endog = realinv
    exog = add_constant(np.c_[realgdp, realint],prepend=True)
    res_ols1 = OLS(endog, exog).fit()

    #growth rates
    gs_l_realinv = 400 * np.diff(np.log(d['realinv']))
    gs_l_realgdp = 400 * np.diff(np.log(d['realgdp']))
    lint = d['realint'][:-1]
    tbilrate = d['tbilrate'][:-1]

    endogg = gs_l_realinv
    exogg = add_constant(np.c_[gs_l_realgdp, lint], prepend=True)
    exogg2 = add_constant(np.c_[gs_l_realgdp, tbilrate], prepend=True)

    res_ols = OLS(endogg, exogg).fit()
    res_ols2 = OLS(endogg, exogg2).fit()

    #the following were done accidentally with res_ols1 in R, with original Greene data

    params = np.array([-272.3986041341653, 0.1779455206941112, 0.2149432424658157])
    cov_hac_4 = np.array([1321.569466333051, -0.2318836566017612, 37.01280466875694, -0.2318836566017614, 4.602339488102263e-05, -0.0104687835998635, 37.012804668757, -0.0104687835998635, 21.16037144168061]).reshape(3,3, order='F')
    cov_hac_10 = np.array([2027.356101193361, -0.3507514463299015, 54.81079621448568, -0.350751446329901, 6.953380432635583e-05, -0.01268990195095196, 54.81079621448564, -0.01268990195095195, 22.92512402151113]).reshape(3,3, order='F')

    #goldfeld-quandt
    het_gq_greater = dict(statistic=13.20512768685082, df1=99, df2=98,
                          pvalue=1.246141976112324e-30, distr='f')
    het_gq_less = dict(statistic=13.20512768685082, df1=99, df2=98, pvalue=1.)
    het_gq_2sided = dict(statistic=13.20512768685082, df1=99, df2=98,
                          pvalue=1.246141976112324e-30, distr='f')

    #goldfeld-quandt, fraction = 0.5
    het_gq_greater_2 = dict(statistic=87.1328934692124, df1=48, df2=47,
                          pvalue=2.154956842194898e-33, distr='f')

    gq = smsdia.het_goldfeldquandt(endog, exog, split=0.5)
    compare_t_est(gq, het_gq_greater, decimal=(13, 14))
    assert_equal(gq[-1], 'increasing')


    harvey_collier = dict(stat=2.28042114041313, df=199,
                          pvalue=0.02364236161988260, distr='t')
    #hc = harvtest(fm, order.by=ggdp , data = list())
    harvey_collier_2 = dict(stat=0.7516918462158783, df=199,
                          pvalue=0.4531244858006127, distr='t')

    ##################################



class TestDiagnosticG(object):

    def __init__(self):
        d = macrodata.load().data
        #growth rates
        gs_l_realinv = 400 * np.diff(np.log(d['realinv']))
        gs_l_realgdp = 400 * np.diff(np.log(d['realgdp']))
        lint = d['realint'][:-1]
        tbilrate = d['tbilrate'][:-1]

        endogg = gs_l_realinv
        exogg = add_constant(np.c_[gs_l_realgdp, lint], prepend=True)
        exogg2 = add_constant(np.c_[gs_l_realgdp, tbilrate], prepend=True)
        exogg3 = add_constant(np.c_[gs_l_realgdp], prepend=True)

        res_ols = OLS(endogg, exogg).fit()
        res_ols2 = OLS(endogg, exogg2).fit()

        res_ols3 = OLS(endogg, exogg3).fit()

        self.res = res_ols
        self.res2 = res_ols2
        self.res3 = res_ols3
        self.endog = self.res.model.endog
        self.exog = self.res.model.exog

    def test_basic(self):
        #mainly to check I got the right regression
        #> mkarray(fm$coefficients, "params")
        params = np.array([-9.48167277465485, 4.3742216647032, -0.613996969478989])

        assert_almost_equal(self.res.params, params, decimal=14)

    def test_hac(self):
        res = self.res
        #> nw = NeweyWest(fm, lag = 4, prewhite = FALSE, verbose=TRUE)
        #> nw2 = NeweyWest(fm, lag=10, prewhite = FALSE, verbose=TRUE)

        #> mkarray(nw, "cov_hac_4")
        cov_hac_4 = np.array([1.385551290884014, -0.3133096102522685, -0.0597207976835705, -0.3133096102522685, 0.1081011690351306, 0.000389440793564336, -0.0597207976835705, 0.000389440793564339, 0.0862118527405036]).reshape(3,3, order='F')

        #> mkarray(nw2, "cov_hac_10")
        cov_hac_10 = np.array([1.257386180080192, -0.2871560199899846, -0.03958300024627573, -0.2871560199899845, 0.1049107028987101, 0.0003896205316866944, -0.03958300024627578, 0.0003896205316866961, 0.0985539340694839]).reshape(3,3, order='F')

        cov, bse_hac = sw.cov_hac_simple(res, nlags=4, use_correction=False)
        assert_almost_equal(cov, cov_hac_4, decimal=14)
        assert_almost_equal(bse_hac, np.sqrt(np.diag(cov)), decimal=14)

        cov, bse_hac = sw.cov_hac_simple(res, nlags=10, use_correction=False)
        assert_almost_equal(cov, cov_hac_10, decimal=14)
        assert_almost_equal(bse_hac, np.sqrt(np.diag(cov)), decimal=14)


    def test_het_goldfeldquandt(self):
        #TODO: test options missing

        #> gq = gqtest(fm, alternative='greater')
        #> mkhtest_f(gq, 'het_gq_greater', 'f')
        het_gq_greater = dict(statistic=0.5313259064778423, pvalue=0.9990217851193723, parameters=(98, 98), distr='f')

        #> gq = gqtest(fm, alternative='less')
        #> mkhtest_f(gq, 'het_gq_less', 'f')
        het_gq_less = dict(statistic=0.5313259064778423, pvalue=0.000978214880627621, parameters=(98, 98), distr='f')

        #> gq = gqtest(fm, alternative='two.sided')
        #> mkhtest_f(gq, 'het_gq_two_sided', 'f')
        het_gq_two_sided = dict(statistic=0.5313259064778423, pvalue=0.001956429761255241, parameters=(98, 98), distr='f')


        #> gq = gqtest(fm, fraction=0.1, alternative='two.sided')
        #> mkhtest_f(gq, 'het_gq_two_sided_01', 'f')
        het_gq_two_sided_01 = dict(statistic=0.5006976835928314, pvalue=0.001387126702579789, parameters=(88, 87), distr='f')

        #> gq = gqtest(fm, fraction=0.5, alternative='two.sided')
        #> mkhtest_f(gq, 'het_gq_two_sided_05', 'f')
        het_gq_two_sided_05 = dict(statistic=0.434815645134117, pvalue=0.004799321242905568, parameters=(48, 47), distr='f')

        endogg, exogg = self.endog, self.exog
        #tests
        gq = smsdia.het_goldfeldquandt(endogg, exogg, split=0.5)
        compare_t_est(gq, het_gq_greater, decimal=(14, 14))
        assert_equal(gq[-1], 'increasing')
        #TODO other options ???

    def test_het_breush_pagan(self):
        res = self.res

        bptest = dict(statistic=0.709924388395087, pvalue=0.701199952134347,
                      parameters=(2,), distr='f')

        bp = smsdia.het_breushpagan(res.resid, res.model.exog)
        compare_t_est(bp, bptest, decimal=(13, 13))



    def test_het_white(self):
        res = self.res

        #TODO: regressiontest compare with Greene or Gretl or Stata
        hw = smsdia.het_white(res.resid, res.model.exog)
        hw_values = (33.503722896538441, 2.9887960597830259e-06,
                     7.7945101228430946, 1.0354575277704231e-06)
        assert_almost_equal(hw, hw_values)


    def test_acorr_breush_godfrey(self):
        res = self.res

        #bgf = bgtest(fm, order = 4, type="F")
        breushgodfrey_f = dict(statistic=1.179280833676792, pvalue=0.321197487261203, parameters=(4,195,), distr='f')

        #> bgc = bgtest(fm, order = 4, type="Chisq")
        #> mkhtest(bgc, "breushpagan_c", "chi2")
        breushgodfrey_c = dict(statistic=4.771042651230007, pvalue=0.3116067133066697, parameters=(4,), distr='chi2')

        bg = smsdia.acorr_breush_godfrey(res, nlags=4)
        bg_r = [breushgodfrey_f['statistic'], breushgodfrey_f['pvalue'],
                breushgodfrey_c['statistic'], breushgodfrey_c['pvalue']]
        assert_almost_equal(bg, bg_r, decimal=13)

    def test_acorr_ljung_box(self):
        res = self.res

        #> bt = Box.test(residuals(fm), lag=4, type = "Ljung-Box")
        #> mkhtest(bt, "ljung_box_4", "chi2")
        ljung_box_4 = dict(statistic=5.23587172795227, pvalue=0.263940335284713, parameters=(4,), distr='chi2')

        #> bt = Box.test(residuals(fm), lag=4, type = "Box-Pierce")
        #> mkhtest(bt, "ljung_box_bp_4", "chi2")
        ljung_box_bp_4 = dict(statistic=5.12462932741681, pvalue=0.2747471266820692, parameters=(4,), distr='chi2')

        #ddof correction for fitted parameters in ARMA(p,q) fitdf=p+q
        #> bt = Box.test(residuals(fm), lag=4, type = "Ljung-Box", fitdf=2)
        #> mkhtest(bt, "ljung_box_4df2", "chi2")
        ljung_box_4df2 = dict(statistic=5.23587172795227, pvalue=0.0729532930400377, parameters=(2,), distr='chi2')

        #> bt = Box.test(residuals(fm), lag=4, type = "Box-Pierce", fitdf=2)
        #> mkhtest(bt, "ljung_box_bp_4df2", "chi2")
        ljung_box_bp_4df2 = dict(statistic=5.12462932741681, pvalue=0.0771260128929921, parameters=(2,), distr='chi2')


        lb, lbpval, bp, bppval = smsdia.acorr_ljungbox(res.resid, 4, boxpierce=True)
        compare_t_est([lb[-1], lbpval[-1]], ljung_box_4, decimal=(13, 14))
        compare_t_est([bp[-1], bppval[-1]], ljung_box_bp_4, decimal=(13, 14))


    def test_harvey_collier(self):
        #> hc = harvtest(fm, order.by = NULL, data = list())
        #> mkhtest_f(hc, 'harvey_collier', 't')
        harvey_collier = dict(statistic=0.494432160939874, pvalue=0.6215491310408242, parameters=(198), distr='t')

        #> hc2 = harvtest(fm, order.by=ggdp , data = list())
        #> mkhtest_f(hc2, 'harvey_collier_2', 't')
        harvey_collier_2 = dict(statistic=1.42104628340473, pvalue=0.1568762892441689, parameters=(198), distr='t')

    def test_rainbow(self):
        #rainbow test
        #> rt = raintest(fm)
        #> mkhtest_f(rt, 'raintest', 'f')
        raintest = dict(statistic=0.6809600116739604, pvalue=0.971832843583418, parameters=(101, 98), distr='f')

        #> rt = raintest(fm, center=0.4)
        #> mkhtest_f(rt, 'raintest_center_04', 'f')
        raintest_center_04 = dict(statistic=0.682635074191527, pvalue=0.971040230422121, parameters=(101, 98), distr='f')

        #> rt = raintest(fm, fraction=0.4)
        #> mkhtest_f(rt, 'raintest_fraction_04', 'f')
        raintest_fraction_04 = dict(statistic=0.565551237772662, pvalue=0.997592305968473, parameters=(122, 77), distr='f')

        #> rt = raintest(fm, order.by=ggdp)
        #Warning message:
        #In if (order.by == "mahalanobis") { :
        #  the condition has length > 1 and only the first element will be used
        #> mkhtest_f(rt, 'raintest_order_gdp', 'f')
        raintest_order_gdp = dict(statistic=1.749346160513353, pvalue=0.002896131042494884, parameters=(101, 98), distr='f')



    def test_compare_lr(self):
        res = self.res
        res3 = self.res3 #nested within res
        #lrtest
        #lrt = lrtest(fm, fm2)
        #Model 1: ginv ~ ggdp + lint
        #Model 2: ginv ~ ggdp

        lrtest = dict(loglike1=-763.9752181602237, loglike2=-766.3091902020184,
                      chi2value=4.66794408358942, pvalue=0.03073069384028677,
                      df=(4,3,1))
        lrt = res.compare_lr_test(res3)
        assert_almost_equal(lrt[0], lrtest['chi2value'], decimal=14)
        assert_almost_equal(lrt[1], lrtest['pvalue'], decimal=14)

        waldtest = dict(fvalue=4.65216373312492, pvalue=0.03221346195239025,
                        df=(199,200,1))

        wt = res.compare_f_test(res3)
        assert_almost_equal(wt[0], waldtest['fvalue'], decimal=13)
        assert_almost_equal(wt[1], waldtest['pvalue'], decimal=14)


    def test_compare_nonnested(self):
        res = self.res
        res2 = self.res2
        #jt = jtest(fm, lm(ginv ~ ggdp + tbilrate))
        #Estimate         Std. Error  t value Pr(>|t|)
        jtest = [('M1 + fitted(M2)', 1.591505670785873, 0.7384552861695823,
                  2.155182176352370, 0.032354572525314450, '*'),
                 ('M2 + fitted(M1)', 1.305687653016899, 0.4808385176653064,
                  2.715438978051544, 0.007203854534057954, '**')]

        jt1 = smsdia.compare_j(res2, res)
        assert_almost_equal(jt1, jtest[0][3:5], decimal=13)

        jt2 = smsdia.compare_j(res, res2)
        assert_almost_equal(jt2, jtest[1][3:5], decimal=14)

        #Estimate        Std. Error  z value   Pr(>|z|)
        coxtest = [('fitted(M1) ~ M2', -0.782030488930356, 0.599696502782265,
                    -1.304043770977755, 1.922186587840554e-01, ' '),
                   ('fitted(M2) ~ M1', -2.248817107408537, 0.392656854330139,
                    -5.727181590258883, 1.021128495098556e-08, '***')]

        ct1 = smsdia.compare_cox(res, res2)
        assert_almost_equal(ct1, coxtest[0][3:5], decimal=13)

        ct2 = smsdia.compare_cox(res2, res)
        assert_almost_equal(ct2, coxtest[1][3:5], decimal=12) #TODO should be approx

        #     Res.Df Df       F    Pr(>F)
        encomptest = [('M1 vs. ME',    198, -1, 4.644810213266983,
                       0.032354572525313666, '*'),
                      ('M2 vs. ME',    198, -1, 7.373608843521585,
                       0.007203854534058054, '**')]

        # Estimate          Std. Error  t value
        petest = [('M1 + log(fit(M1))-fit(M2)', -229.281878354594596,
                    44.5087822087058598, -5.15139, 6.201281252449979e-07),
                  ('M2 + fit(M1)-exp(fit(M2))',  0.000634664704814,
                   0.0000462387010349, 13.72583, 1.319536115230356e-30)]

    def test_influence(self):
        res = self.res
        #this test is slow
        import json
        fp = file(os.path.join(cur_dir,"results/influence_lsdiag_R.json"))
        lsdiag = json.load(fp)

        #basic
        assert_almost_equal(lsdiag['cov.scaled'],
                            res.cov_params().ravel(), decimal=14)
        assert_almost_equal(lsdiag['cov.unscaled'],
                            res.normalized_cov_params.ravel(), decimal=14)

        infl = oi.Influence(res)

        c0, c1 = infl.cooks_distance() #TODO: what's c1


        assert_almost_equal(c0, lsdiag['cooks'], decimal=14)
        assert_almost_equal(infl.hat_matrix_diag, lsdiag['hat'], decimal=14)
        assert_almost_equal(infl.resid_studentized_internal,
                            lsdiag['std.res'], decimal=14)

        #slow:
        infl.get_all_obs()  #slow, nobs estimation loop
        dffits, dffth = infl.dffits
        assert_almost_equal(dffits, lsdiag['dfits'], decimal=14)
        assert_almost_equal(infl.resid_studentized_external,
                            lsdiag['stud.res'], decimal=14)

        import pandas
        fn = os.path.join(cur_dir,"results/influence_measures_R.csv")
        infl_r = pandas.read_csv(fn, index_col=0)
        conv = lambda s: 1 if s=='TRUE' else 0
        fn = os.path.join(cur_dir,"results/influence_measures_bool_R.csv")
        #not used yet:
        #infl_bool_r  = pandas.read_csv(fn, index_col=0,
        #                                converters=dict(zip(range(7),[conv]*7)))
        infl_r2 = np.asarray(infl_r)
        assert_almost_equal(infl.dfbetas, infl_r2[:,:3], decimal=13)
        assert_almost_equal(infl.cov_ratio, infl_r2[:,4], decimal=14)
        #duplicates
        assert_almost_equal(dffits, infl_r2[:,3], decimal=14)
        assert_almost_equal(c0, infl_r2[:,5], decimal=14)
        assert_almost_equal(infl.hat_matrix_diag, infl_r2[:,6], decimal=14)

        #Note: for dffits, R uses a threshold around 0.36, mine: dffits[1]=0.24373
        #TODO: finish and check thresholds and pvalues
        '''
        R has
        >>> np.nonzero(np.asarray(infl_bool_r["dffit"]))[0]
        array([  6,  26,  63,  76,  90, 199])
        >>> np.nonzero(np.asarray(infl_bool_r["cov.r"]))[0]
        array([  4,  26,  59,  61,  63,  72,  76,  84,  91,  92,  94,  95, 108,
               197, 198])
        >>> np.nonzero(np.asarray(infl_bool_r["hat"]))[0]
        array([ 62,  76,  84,  90,  91,  92,  95, 108, 197, 199])
        '''


def grangertest():
    #> gt = grangertest(ginv, ggdp, order=4)
    #> gt
    #Granger causality test
    #
    #Model 1: ggdp ~ Lags(ggdp, 1:4) + Lags(ginv, 1:4)
    #Model 2: ggdp ~ Lags(ggdp, 1:4)

    grangertest = dict(fvalue=1.589672703015157, pvalue=0.178717196987075,
                       df=(198,193))

if __name__ == '__main__':

    t = TestDiagnosticG()
    t.test_basic()
    t.test_hac()
    t.test_acorr_breush_godfrey()
    t.test_acorr_ljung_box()
    t.test_het_goldfeldquandt()
    t.test_het_breush_pagan()
    t.test_het_white()
    t.test_compare_lr()
    t.test_compare_nonnested()
    t.test_influence()


    ##################################################

    '''
    J test

    Model 1: ginv ~ ggdp + lint
    Model 2: ginv ~ ggdp + tbilrate
                             Estimate         Std. Error t value  Pr(>|t|)
    M1 + fitted(M2) 1.591505670785873 0.7384552861695823 2.15518 0.0323546 *
    M2 + fitted(M1) 1.305687653016899 0.4808385176653064 2.71544 0.0072039 **
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1


     = lm(ginv ~ ggdp + tbilrate)
    > ct = coxtest(fm, fm3)
    > ct
    Cox test

    Model 1: ginv ~ ggdp + lint
    Model 2: ginv ~ ggdp + tbilrate
                              Estimate        Std. Error  z value   Pr(>|z|)
    fitted(M1) ~ M2 -0.782030488930356 0.599696502782265 -1.30404    0.19222
    fitted(M2) ~ M1 -2.248817107408537 0.392656854330139 -5.72718 1.0211e-08 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1



    > et = encomptest(fm, fm3)
    > et
    Encompassing test

    Model 1: ginv ~ ggdp + lint
    Model 2: ginv ~ ggdp + tbilrate
    Model E: ginv ~ ggdp + lint + tbilrate
              Res.Df Df       F    Pr(>F)
    M1 vs. ME    198 -1 4.64481 0.0323546 *
    M2 vs. ME    198 -1 7.37361 0.0072039 **
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1


    > fm4 = lm(realinv ~ realgdp + realint, data=d)
    > fm5 = lm(log(realinv) ~ realgdp + realint, data=d)
    > pet = petest(fm4, fm5)
    > pet
    PE test

    Model 1: realinv ~ realgdp + realint
    Model 2: log(realinv) ~ realgdp + realint
                                          Estimate          Std. Error  t value
    M1 + log(fit(M1))-fit(M2) -229.281878354594596 44.5087822087058598 -5.15139
    M2 + fit(M1)-exp(fit(M2))    0.000634664704814  0.0000462387010349 13.72583
                                Pr(>|t|)
    M1 + log(fit(M1))-fit(M2) 6.2013e-07 ***
    M2 + fit(M1)-exp(fit(M2)) < 2.22e-16 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

    '''

    #using rjson
    ss = "{\"statistic\":{\"Rain\":0.68096001167396},\"parameter\":{\"df1\":101,\"df2\":98},\"method\":\"Rainbow test\",\"p.value\":0.971832843583418,\"data.name\":\"fm\"}"

    ss3 = "{\"ginv\":[32.0850725097671,-28.8524174970515,13.770044469268,41.0655072592576,-42.6775380666317,-2.39115167756623,-52.7408071519403,10.0976723012611,28.7335494851266,32.1810826407564,6.69484534501947,23.1642561190032,-3.88633920958092,7.093588456366,-13.6587917432752,21.6028387172781,5.78708081411143,12.8357363325183,4.89300036400664,16.1181501990484,-1.8433918226652,9.39284419804594,3.25084450533488,38.3515655587399,-0.024234967336767,14.0359172100776,1.83986397470051,32.4660756856062,-7.36621167887925,-3.98352312735533,1.91596014722144,-11.1051074039523,-17.4296786710773,11.3189841723685,8.56139518765211,8.61211733721525,15.8531235201718,-13.2009472214381,4.13356780882665,25.5201839627446,-3.19990113282351,9.14282422623884,-22.1454127099456,-12.7192408547653,1.25106869810239,6.77727290013515,-23.869938198828,48.8377989761169,12.2079478510621,5.22725663038059,-12.7135512230918,27.3316073510291,23.7640356682533,5.65261414904938,2.07023319518314,24.6254583061852,18.2752664507824,-15.9538338455341,15.0152293218021,-26.3740457915279,-1.98708930309124,-23.2245789684846,3.78448100766029,-77.26529074604,-14.137076870572,32.5156488854674,10.8461484432148,39.4141212662021,16.7053524560469,0.741638575388848,2.77107264940462,19.2098371845052,29.7785070619664,21.2252255900182,-11.4538781818649,7.85556110737353,26.5619924014697,12.1602416918925,8.99996699179262,-0.0649079303173039,-0.926451169587139,-7.863896769884,-7.51646702475064,-2.9098347567718,-37.8220533741935,-31.7113254419869,38.7303218825426,37.8972439798549,-19.5105317483051,24.0879899088874,-15.8210434562758,-44.0076623238397,-0.354807183549255,-4.69575982804109,-37.3071536042314,13.9946767866419,36.8543398124082,26.0310430548344,40.0443838741023,39.8171515512065,13.2639630238174,9.19125628052448,-6.61696531847866,-13.4103044308144,6.84573450007626,-4.4444380575694,15.340456168579,-0.822768500341198,-8.97700179255168,-12.7411113134229,0.619830215993034,12.0383947930378,0.559571875145437,0.403133287075974,30.0338734603184,-23.1972514495851,9.61395609821984,2.56138170627835,5.26273157319928,15.0421991379236,-4.70143815052744,-4.72877130074671,-4.12652112054346,3.91738348539867,0.113590185004497,-9.52786913874561,-26.131500422618,-16.6268668381139,-2.01620862760556,9.8365176682691,15.027360461287,-8.99282129356358,25.5173871452271,4.1138223025289,12.4639312967851,9.29056304122575,3.12864808949733,-0.281873416071221,20.5699630085054,16.8963278268702,22.6600115965759,-7.25655342769969,18.2266924024546,4.04506285020361,-10.9047350934212,-3.86889188942732,11.1072091505278,5.23514620253138,20.1993520039231,19.67762464018,-1.08891813474052,9.23157463944762,24.7593779201132,7.04903197510305,6.33786088203792,18.6520313265422,-4.81528906155972,11.3000078601196,12.6609570209034,12.4002281675001,-1.39891283632494,9.9166695085767,13.9496471431286,-5.62411514471748,26.7751315073362,-6.3060835692081,0.178927199590362,-21.7396004077642,-1.28554556764087,-8.52567358300433,-23.7462531264615,13.2124463303988,4.76702717061777,0.82795707723875,-0.289325352011716,-0.017207337387859,2.32224294823631,14.2594458500081,14.5275321735863,2.0829616426699,17.0067558540545,5.15274739102836,8.16136567328947,8.4086467395462,-7.22160074807334,4.38244526947109,14.0869811729768,5.78682489003093,-0.614056605305535,-5.6312350309728,-11.5887568524819,-6.20813540971454,5.51463357941238,0.790444512492883,-8.03119439504236,-7.71055600453074,-10.9741530673528,-7.13449201425718,-27.6658599310299,-70.2392796522993,-27.0245878530897,8.07889712568475],\"ggdp\":[9.97685232655492,-0.477180844267266,1.39781306174882,8.87607180571734,-1.87382131309342,0.653152075567931,-5.16254397843134,2.36903650217855,7.41381366155736,6.41265566472455,8.0611264187592,7.11090371473446,4.39220613999041,3.68162688535847,0.970807485697378,5.19408418140586,4.98113383725141,7.46161629563815,3.02954471573145,8.87834554753795,4.56796667131201,5.39871376248016,1.1073727906421,9.70588512243111,5.39076839409987,8.03610795228735,9.53582508980304,9.69976715854344,1.32933170316889,2.62212703077012,3.22769621016405,3.50829979953602,0.0832834078174471,3.17851555782482,3.04033488757867,8.15972336476563,6.73450024939868,2.72727507027568,1.72941400720745,6.25079729079374,1.16321830189321,2.52164869148714,-1.88303609423457,-0.627993585202802,0.724433946617609,3.54590891482971,-4.2643286787829,10.880867333789,2.26271556526356,3.18035470469482,1.10997483786193,7.08932578715746,9.37555954483997,3.81520568310307,6.53471698443937,10.1032169371535,4.60043911850576,-2.14001717703667,3.79729530212884,-3.52304550259959,1.02288496165244,-3.97467139249557,-1.57732737549168,-4.89516852505361,3.04529135482099,6.68122214463764,5.19138171508615,8.99124423366402,2.99691898909344,1.95468276377042,2.89415930958157,4.6164636819519,7.87129869492063,7.0904550211786,-0.0827683825527004,1.36349308111647,15.4341901615027,3.90246519599557,5.25577451704891,0.668397828090406,0.375316340387855,2.86496804817205,1.09905593580422,1.28646058383808,-8.28317263213734,-0.744103244767302,7.33072280372156,8.22672987175253,-3.2045610896084,4.83083127595876,-5.01436407907718,-6.61889345413513,2.16175565897103,-1.54509030742602,0.315641398081112,4.94420989463151,8.8909341154725,7.81112564220905,8.18398397702111,7.68406632404961,6.84710540103524,3.86860663375899,3.24323817115015,3.75696098789007,3.37248894740014,6.1999653458301,3.02437881061408,3.82522717672344,1.60367128300294,3.83807538039207,1.92871758404607,2.21157429594641,4.23116666404866,3.45421713165308,6.78631991073146,2.06374112701013,5.10460534247841,2.05983221782446,5.30574046743055,3.7379538215383,2.98186271679555,3.15986367293561,0.872172819367023,4.15701079147652,1.58659611889789,-0.00605493839174187,-3.5199880201489,-1.94240582541383,2.69064808374182,1.68145591887239,1.564976915855,4.36668389867307,4.22774210601702,4.10616714395218,4.18745143615382,0.734455191095407,2.55099889858315,2.10009647656761,5.2477867961926,3.87530138501333,5.43427952064874,2.56816158254694,4.417910997185,0.980096002734854,0.858929817061949,3.3477553849437,2.77919249393079,2.73068911930352,6.85604618640667,3.46434421208457,4.34267336221978,3.06468817605392,5.88704710323924,4.98920901548203,3.05702961670562,3.76095031449069,3.58080364870972,5.24334674617961,6.86462974780326,3.54751828700444,3.11466407461936,5.05457744331679,7.12077079585498,1.04419013967387,7.72743425032516,0.334293400077001,2.36000286586133,-1.32108535201283,2.61439508120205,-1.09816642434453,1.41030576149745,3.42079316823174,2.11680410082025,1.99384900540167,0.0825686154072969,1.61740712579004,3.1777421531288,6.64891923001178,3.58179907141434,2.80694428436377,2.83287617282042,2.92740925994579,3.45554626475959,3.97145786856541,1.70122853492671,3.02701553096441,2.06185991116854,5.21313018174325,1.43823575253137,0.106570492621216,2.91281802352614,1.19942384727665,3.16535966655778,3.532739124676,2.10060560571037,-0.729020191772634,1.44577151662588,-2.71254455660923,-5.52193189439407,-6.64479189689118,-0.740499056988142,2.74487503252345],\"lint\":[0,0.74,1.09,4.06,1.19,2.55,-0.34,1.08,2.77,0.81,1.52,1.8,0.47,2.65,0.67,2.08,2.38,0.29,2.6,1.06,3.38,2.57,2.25,1.71,2.65,1.3,3.04,1.46,-0.37,2.55,0.33,4.39,1.8,0.17,0.84,0.18,1.67,-0.28,0.65,1.34,-0.58,1.02,1.63,1.26,0.47,2.52,1.04,-0.18,1.65,-0.19,1.75,0.95,0.64,0.98,0.66,0.38,-3.28,2.64,-4.41,-2.71,-3.16,-1.96,-5.4,-3.11,0.22,-1.91,-0.34,-1.24,2.77,-1.09,-1.22,-0.92,-4.16,-0.24,0.59,-0.88,-1.24,-3.18,-2.01,0.76,-2.66,-4.07,-1.38,-2.68,-0.85,-0.42,0.3,3.11,5.32,4.69,6.36,7.07,10.42,1.58,5.65,8.77,4.56,4.66,5.01,3.76,4.76,6.85,6.37,5.87,3.36,4.56,4.17,2.01,10.95,3.13,2.76,1.1,0.97,1.79,1.99,2.29,1.64,2.07,2.52,3.72,2.44,3.63,4.88,1.01,3.44,2.76,-1.46,2.79,4.65,2.29,2.25,0.95,0.71,0.36,-0.44,0.02,0.13,1.08,-0.04,1.13,1.02,0.96,2,2.6,2.28,3.42,2.97,2.05,1.31,2.79,2,1.97,3.85,3.76,2.29,3.88,4.53,2.52,2.78,2.43,1.49,2.62,1.41,2.35,1.87,1.62,3.3,1.81,2.57,1.28,2.27,1.51,-1.84,0.14,-1.05,-1.88,-0.17,-0.13,-1.67,-2.11,-1.42,-2.41,-1.95,0.11,-1.46,1.16,-5.62,3.6,1.91,0.85,6.48,1.62,0.36,1.97,0.55,-3.37,-1.26,-6.79,4.33,8.91,-0.71,-3.19],\"tbilrate\":[2.82,3.08,3.82,4.33,3.5,2.68,2.36,2.29,2.37,2.29,2.32,2.6,2.73,2.78,2.78,2.87,2.9,3.03,3.38,3.52,3.51,3.47,3.53,3.76,3.93,3.84,3.93,4.35,4.62,4.65,5.23,5,4.22,3.78,4.42,4.9,5.18,5.5,5.21,5.85,6.08,6.49,7.02,7.64,6.76,6.66,6.15,4.86,3.65,4.76,4.7,3.87,3.55,3.86,4.47,5.09,5.98,7.19,8.06,7.68,7.8,7.89,8.16,6.96,5.53,5.57,6.27,5.26,4.91,5.28,5.05,4.57,4.6,5.06,5.82,6.2,6.34,6.72,7.64,9.02,9.42,9.3,10.49,11.94,13.75,7.9,10.34,14.75,13.95,15.33,14.58,11.33,12.95,11.97,8.1,7.96,8.22,8.69,8.99,8.89,9.43,9.94,10.19,8.14,8.25,7.17,7.13,7.14,6.56,6.06,5.31,5.44,5.61,5.67,6.19,5.76,5.76,6.48,7.22,8.03,8.67,8.15,7.76,7.65,7.8,7.7,7.33,6.67,5.83,5.54,5.18,4.14,3.88,3.5,2.97,3.12,2.92,3.02,3,3.05,3.48,4.2,4.68,5.53,5.72,5.52,5.32,5.17,4.91,5.09,5.04,4.99,5.1,5.01,5.02,5.11,5.02,4.98,4.49,4.38,4.39,4.54,4.75,5.2,5.63,5.81,6.07,5.7,4.39,3.54,2.72,1.74,1.75,1.7,1.61,1.2,1.14,0.96,0.94,0.9,0.94,1.21,1.63,2.2,2.69,3.01,3.52,4,4.51,4.82,4.9,4.92,4.95,4.72,4,3.01,1.56,1.74,1.17,0.12,0.22,0.18]}"

    print smsdia.acorr_breush_godfrey(res_ols, nlags=4)
