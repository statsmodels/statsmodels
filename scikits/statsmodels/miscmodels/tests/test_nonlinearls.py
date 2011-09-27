import sys
sys.path.insert(0, r'E:\Josef\eclipsegworkspace\statsmodels-git\statsmodels-josef')
import numpy as np
import scikits.statsmodels.api as sm
from scikits.statsmodels.miscmodels.nonlinls import NonlinearLS

from numpy.testing import assert_almost_equal, assert_

# Tests for linear case with weights against WLS

class Myfunc(NonlinearLS):

    def _predict(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog

        a, b = params
        return a + b*x

class Myfunc0(NonlinearLS):

    def _predict(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog

        b = params
        return b*x

class Myfunc3(NonlinearLS):

    def _predict(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        x0, x1 = x.T
        a, b, c = params
        return a + b*x0 + c*x1


class TestNonlinearLS(object):
    #summary method has problems
    #example has a 1d exog (nobs,)
    #bug in call to leastsq if exog is column array (nobs,1)

    def setup(self):
        x = np.arange(5.)
        y = np.array([1, -2, 1, -2, 1.])
        sigma = np.array([1,  2, 1,  2, 1.])

        mod = Myfunc(y, x, sigma=sigma**2)
        self.res = mod.fit(start_value=(0.042, 0.42))

        self.res2 = sm.WLS(y, sm.add_constant(x, prepend=True),
                           weights=1./sigma**2).fit()

    def test_basic(self):
        res = self.res
        res2 = self.res2

        print ''

        for att in ['params', 'bse', 'resid', 'wresid', 'tvalues', 'pvalues',
                    'fittedvalues', 'mse_resid', 'mse_total', 'mse_model',
                    'rsquared', 'rsquared_adj', 'df_resid', 'df_model',
                    'f_pvalue', 'ess', 'centered_tss', 'uncentered_tss']:
            #fail: 'mse_model', 'f_pvalue' #ok if assumes constant
            print 'testing', att
            assert_almost_equal(getattr(res, att), getattr(res2, att),
                                decimal=6)

        for att in ['wexog']:
            print 'testing', att
            assert_almost_equal(getattr(res.model, att), getattr(res2.model, att),
                                decimal=6)

        for att in ['cov_params']: #, 'predict']: #ENH/BUG
            print 'testing', att
            assert_almost_equal(getattr(res, att)(), getattr(res2, att)(),
                                decimal=6)

        for att in []: #['predict']:  #BUG
            print 'testing', att
            assert_almost_equal(getattr(res.model, att)(res.params),
                                getattr(res2.model, att)(res2.params),
                                decimal=6)

        print 'testing f_test'
        rmat = np.ones(len(res.params))
        assert_almost_equal(res.f_test(rmat).fvalue, res.f_test(rmat).fvalue,
                            decimal=6)
        assert_almost_equal(res.f_test(rmat).pvalue, res.f_test(rmat).pvalue,
                            decimal=6)

        print 'testing t_test'
        assert_almost_equal(res.t_test(rmat).tvalue, res.t_test(rmat).tvalue,
                            decimal=6)
        assert_almost_equal(res.t_test(rmat).pvalue, res.t_test(rmat).pvalue,
                            decimal=6)



class TestNonlinearLS1(TestNonlinearLS):
    #summary method has problems
    #example has a 1d exog (nobs,)
    #bug in call to leastsq if exog is column array (nobs,1)

    def setup(self):
        x = np.arange(5.)
        y = np.array([1, -2, 1, -2, 1.])
        sigma = np.array([1,  2, 1,  2, 1.])

        mod = Myfunc(y, x, sigma=sigma**2)
        self.res = mod.fit(start_value=(0.042, 0.42))

        self.res2 = sm.WLS(y, sm.add_constant(x, prepend=True),
                           weights=1./sigma**2).fit()

class TestNonlinearLS0(TestNonlinearLS):
    #summary method has problems
    #example has a 1d exog (nobs,)
    #bug in call to leastsq if exog is column array (nobs,1)

    def setup(self):
        x = np.arange(5.)
        y = np.array([1, -2, 1, -2, 1.])
        sigma = np.array([1,  2, 1,  2, 1.])

        mod = Myfunc0(y, x, sigma=sigma**2)
        self.res = mod.fit(start_value=([0.042])) #requires [0.042], needs len()

        self.res2 = sm.WLS(y, x,
                           weights=1./sigma**2).fit()

    def _est_summary(self):
        #this raises in scipy.stats, math domain error, sample too short ?
        print 'testing summary'
        txt = self.res.summary(yname='y', xname=['const', 'x0', 'x1'])
        txtw = self.res2.summary(yname='y', xname=['const', 'x0', 'x1'])
        assert_(txt[txt.find('#'):] == txtw[txtw.find('#'):])


class TestNonlinearLS2(TestNonlinearLS):
    #summary method has problems
    #example has a 1d exog (nobs,)
    #bug in call to leastsq if exog is column array (nobs,1)

    def setup(self):
        x = np.arange(5.).repeat(2)
        y = np.array([1, -2, 1, -2, 1.]).repeat(2)
        sigma = np.array([1,  2, 1,  2, 1.]).repeat(2)

        mod = Myfunc(y, x, sigma=sigma**2)
        self.res = mod.fit(start_value=(0.042, 0.42))

        self.res2 = sm.WLS(y, sm.add_constant(x, prepend=True),
                           weights=1./sigma**2).fit()

    def _est_summary(self):
        #this fails because of different almost zero, 1e-7 vs.1e-17
        print 'testing summary'
        print txt
        print txtw
        txt = self.res.summary(yname='y', xname=['const', 'x0'])
        txtw = self.res2.summary(yname='y', xname=['const', 'x0'])
        assert_(txt[txt.find('#'):] == txtw[txtw.find('#'):])


class TestNonlinearLS3(TestNonlinearLS):
    #summary method has problems
    #example has a 1d exog (nobs,)
    #bug in call to leastsq if exog is column array (nobs,1)

    def setup(self):
        x = np.arange(5.).repeat(2)
        x = np.column_stack((x,0.1*x**2))
        y = np.array([1, -2, 1, -2, 1.]).repeat(2)
        sigma = np.array([1,  2, 1,  2, 1.]).repeat(2)

        mod = Myfunc3(y, x, sigma=sigma**2)
        self.res = mod.fit(start_value=(0.042, 0.42, 0.2))

        self.res2 = sm.WLS(y, sm.add_constant(x, prepend=True),
                           weights=1./sigma**2).fit()

    def test_summary(self):
        #print 'testing summary'
        txt = self.res.summary(yname='y', xname=['const', 'x0', 'x1'])
        txtw = self.res2.summary(yname='y', xname=['const', 'x0', 'x1'])
        assert_(txt[txt.find('#'):] == txtw[txtw.find('#'):])


def print_summarydiff(res1, res2):
    txt1 = res1.summary(yname='y', xname=['const', 'x0', 'x1'][len(res1.params)])
    txt2 = res2.summary(yname='y', xname=['const', 'x0', 'x1'][len(res2.params)])
    for ci, (c,cw) in enumerate(zip(txt1[txt1.find('#'):], txt2[txt2.find('#'):])):
        if not c == cw: print ci,c
    return txt1, txt2

if __name__ == '__main__':
    tt = TestNonlinearLS1()
    tt.setup()
    tt.test_basic()
    tt = TestNonlinearLS0()
    tt.setup()
    tt.test_basic()
    tt0 = tt
    tt = TestNonlinearLS2()
    tt.setup()
    tt.test_basic()
    txt1, txt2 = print_summarydiff(tt.res, tt.res2)
    '''
    >>> txt1[txt1.find('#'):][440:465]
    '                1.148e-07'
    >>> txt2[txt2.find('#'):][440:465]
    '               -1.388e-17'
    '''
    #tt.test_summary()   #not a good example for summary
    tt = TestNonlinearLS3()
    tt.setup()
    tt.test_basic()
    tt.test_summary()
