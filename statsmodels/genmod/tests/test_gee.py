"""
Test functions for GEE

Most comparisons are to R.  The statmodels GEE implementation should
generally agree with the R GEE implementation for the independence and
exchangeable correlation structures.  For other correlation structures
the details of the correlation estimation differ among implementations
and the results will not agree exactly.
"""

import numpy as np
import os
from numpy.testing import assert_almost_equal
from statsmodels.genmod.generalized_estimating_equations import GEE,\
    gee_setup_ordinal,gee_ordinal_starting_values, GEEMargins,\
    gee_setup_nominal
from statsmodels.genmod.families import Gaussian,Binomial,Poisson
from statsmodels.genmod.dependence_structures import Exchangeable,\
    Independence,GlobalOddsRatio,Autoregressive,Nested
import pandas as pd
import statsmodels.formula.api as sm

def load_data(fname, icept=True):
    """
    Load a data set from the results directory.  The data set should
    be a CSV file with the following format:

    Column 0: Group indicator
    Column 1: endog variable
    Columns 2-end: exog variables

    If `icept` is True, an intercept is prepended to the exog
    variables.
    """

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    Z = np.genfromtxt(os.path.join(cur_dir, 'results', fname),
                      delimiter=",")

    group = Z[:,0]
    endog = Z[:,1]
    exog = Z[:,2:]

    if icept:
        exog = np.concatenate((np.ones((exog.shape[0],1)), exog),
                              axis=1)

    return endog,exog,group


class TestGEE(object):


    def test_margins(self):

        n = 300
        exog = np.random.normal(size=(n, 4))
        exog[:,0] = 1
        exog[:,1] = 1*(exog[:,2] < 0)

        group = np.kron(np.arange(n/4), np.ones(4))
        time = np.zeros((n, 1))

        beta = np.r_[0, 1, -1, 0.5]
        lpr = np.dot(exog, beta)
        prob = 1 / (1 + np.exp(-lpr))

        endog = 1*(np.random.uniform(size=n) < prob)

        fa = Binomial()
        ex = Exchangeable()

        md = GEE(endog, exog, group, time, fa, ex)
        mdf = md.fit()

        marg = GEEMargins(mdf, ())
        marg.summary()
        # Nothing to compare to


    def test_logistic(self):
        """
        R code to for comparing results:

        library(gee)
        Z = read.csv("results/gee_logistic_1.csv", header=FALSE)
        Y = Z[,2]
        Id = Z[,1]
        X1 = Z[,3]
        X2 = Z[,4]
        X3 = Z[,5]

        mi = gee(Y ~ X1 + X2 + X3, id=Id, family=binomial,
                 corstr="independence")
        smi = summary(mi)
        u = coefficients(smi)
        cfi = paste(u[,1], collapse=",")
        sei = paste(u[,4], collapse=",")

        me = gee(Y ~ X1 + X2 + X3, id=Id, family=binomial,
                 corstr="exchangeable")
        sme = summary(me)
        u = coefficients(sme)
        cfe = paste(u[,1], collapse=",")
        see = paste(u[,4], collapse=",")

        ma = gee(Y ~ X1 + X2 + X3, id=Id, family=binomial,
                 corstr="AR-M")
        sma = summary(ma)
        u = coefficients(sma)
        cfa = paste(u[,1], collapse=",")
        sea = paste(u[,4], collapse=",")

        sprintf("cf = [[%s],[%s],[%s]]", cfi, cfe, cfa)
        sprintf("se = [[%s],[%s],[%s]]", sei, see, sea)
        """

        endog,exog,group = load_data("gee_logistic_1.csv")

        # Time values for the autoregressive model
        T = np.zeros(len(endog))
        idx = set(group)
        for ii in idx:
            jj = np.flatnonzero(group == ii)
            T[jj] = range(len(jj))

        family = Binomial()
        ve = Exchangeable()
        vi = Independence()
        va = Autoregressive()

        cf = [[0.0167272965285882,1.13038654425893,
               -1.86896345082962,1.09397608331333],
              [0.0178982283915449,1.13118798191788,
               -1.86133518416017,1.08944256230299],
              [0.0109621937947958,1.13226505028438,
               -1.88278757333046,1.09954623769449]]
        se = [[0.127291720283049,0.166725808326067,
               0.192430061340865,0.173141068839597],
              [0.127045031730155,0.165470678232842,
               0.192052750030501,0.173174779369249],
              [0.127240302296444,0.170554083928117,
               0.191045527104503,0.169776150974586]]

        for j,v in enumerate((vi,ve,va)):
            md = GEE(endog, exog, group, T, family, v)
            mdf = md.fit()
            if id(v) != id(va):
                assert_almost_equal(mdf.params, cf[j], decimal=6)
                assert_almost_equal(mdf.standard_errors, se[j], decimal=6)

        # Test with formulas
        D = np.concatenate((endog[:,None], group[:,None], exog[:,1:]), axis=1)
        D = pd.DataFrame(D)
        D.columns = ["Y","Id",] + ["X%d" % (k+1) for k in range(exog.shape[1]-1)]
        for j,v in enumerate((vi,ve)):
             md = GEE.from_formula("Y ~ X1 + X2 + X3", D, None, groups=D.loc[:,"Id"],
                                   family=family, varstruct=v)
             mdf = md.fit()
             assert_almost_equal(mdf.params, cf[j], decimal=6)
             assert_almost_equal(mdf.standard_errors, se[j], decimal=6)

        # Check for run-time exceptions in summary
        print mdf.summary()


    def test_post_estimation(self):

        family = Gaussian()
        endog,exog,group = load_data("gee_linear_1.csv")

        ve = Exchangeable()

        md = GEE(endog, exog, group, None, family, ve)
        mdf = md.fit()

        assert_almost_equal(np.dot(exog, mdf.params), mdf.fittedvalues)
        assert_almost_equal(endog - np.dot(exog, mdf.params), mdf.resid)



    def test_linear(self):
        """
        R code for comparing Gaussian GEE:

        library(gee)

        Z = read.csv("results/gee_linear_1.csv", header=FALSE)
        Y = Z[,2]
        Id = Z[,1]
        X1 = Z[,3]
        X2 = Z[,4]
        X3 = Z[,5]
        mi = gee(Y ~ X1 + X2 + X3, id=Id, family=gaussian,
                 corstr="independence", tol=1e-8, maxit=100)
        smi = summary(mi)
        u = coefficients(smi)

        cfi = paste(u[,1], collapse=",")
        sei = paste(u[,4], collapse=",")

        me = gee(Y ~ X1 + X2 + X3, id=Id, family=gaussian,
                 corstr="exchangeable", tol=1e-8, maxit=100)
        sme = summary(me)
        u = coefficients(sme)

        cfe = paste(u[,1], collapse=",")
        see = paste(u[,4], collapse=",")

        sprintf("cf = [[%s],[%s]]", cfi, cfe)
        sprintf("se = [[%s],[%s]]", sei, see)
        """

        family = Gaussian()

        endog,exog,group = load_data("gee_linear_1.csv")

        vi = Independence()
        ve = Exchangeable()

        cf = [[-0.01850226507491,0.81436304278962,
                -1.56167635393184,0.794239361055003],
              [-0.0182920577154767,0.814898414022467,
                -1.56194040106201,0.793499517527478]]
        se = [[0.0440733554189401,0.0479993639119261,
               0.0496045952071308,0.0479467597161284],
              [0.0440369906460754,0.0480069787567662,
               0.049519758758187,0.0479760443027526]]

        for j,v in enumerate((vi,ve)):
            md = GEE(endog, exog, group, None, family, v)
            mdf = md.fit()
            assert_almost_equal(mdf.params, cf[j], decimal=10)
            assert_almost_equal(mdf.standard_errors, se[j],
                                decimal=10)

        # Test with formulas
        D = np.concatenate((endog[:,None], group[:,None], exog[:,1:]),
                           axis=1)
        D = pd.DataFrame(D)
        D.columns = ["Y","Id",] + ["X%d" % (k+1)
                                   for k in range(exog.shape[1]-1)]
        for j,v in enumerate((vi,ve)):
            md = GEE.from_formula("Y ~ X1 + X2 + X3", D, None,
                                  groups=D.loc[:,"Id"],
                                  family=family, varstruct=v)
            mdf = md.fit()
            assert_almost_equal(mdf.params, cf[j], decimal=10)
            assert_almost_equal(mdf.standard_errors, se[j],
                                decimal=10)


    def test_linear_constrained(self):

        family = Gaussian()

        exog = np.random.normal(size=(300,4))
        exog[:,0] = 1
        endog = np.dot(exog, np.r_[1, 1, 0, 0.2]) +\
            np.random.normal(size=300)
        group = np.kron(np.arange(100), np.r_[1,1,1])

        vi = Independence()
        ve = Exchangeable()

        L = np.r_[[[0, 0, 0, 1]]]
        R = np.r_[0,]

        for j,v in enumerate((vi,ve)):
            md = GEE(endog, exog, group, None, family, v,
                     constraint=(L,R))
            mdf = md.fit()
            assert_almost_equal(mdf.params[3], 0, decimal=10)


    def test_nested_linear(self):

        family = Gaussian()

        endog,exog,group = load_data("gee_nested_linear_1.csv")

        group_n = []
        for i in range(300):
            group_n.extend([0,]*5)
            group_n.extend([1,]*5)
        group_n = np.array(group_n)

        ne = Nested(group_n)

        md = GEE(endog, exog, group_n, None, family, ne)
        mdf = md.fit()
        ## Nothing to compare to


    def test_ordinal(self):

        family = Binomial()

        endog_orig, exog_orig, groups = load_data("gee_ordinal_1.csv",
                                                  icept=False)

        data = np.concatenate((endog_orig[:,None], exog_orig,
                               groups[:,None]), axis=1)

        # Recode as cumulative indicators
        endog, exog, intercepts, nlevel = gee_setup_ordinal(data, 0)

        exog1 = np.concatenate((intercepts, exog), axis=1)
        groups = exog1[:,-1]
        exog1 = exog1[:,0:-1]

        v = GlobalOddsRatio(nlevel, "ordinal")

        beta = gee_ordinal_starting_values(endog_orig,
                                           exog_orig.shape[1])

        md = GEE(endog, exog1, groups, None, family, v)
        mdf = md.fit(starting_params = beta)
        # Nothing to compare to...
        #assert_almost_equal(md.params, cf[j], decimal=2)
        #assert_almost_equal(mdf.standard_errors, se[j], decimal=2)


    def test_nominal(self):

        family = Binomial()

        endog_orig, exog_orig, groups = load_data("gee_nominal_1.csv",
                                                  icept=False)

        data = np.concatenate((endog_orig[:,None], exog_orig,
                               groups[:,None]), axis=1)

        # Recode as indicators
        endog, exog, exog_ne, nlevel = gee_setup_nominal(data, 0,
                                                         [4,])

        groups = exog_ne[:,0]

        v = Independence()
        md = GEE(endog, exog, groups, None, family, v)
        mdf1 = md.fit()

        v = GlobalOddsRatio(nlevel, "nominal")
        md = GEE(endog, exog, groups, None, family, v)
        mdf = md.fit(starting_params=np.r_[0,1,-1,0,-1,1])
        # Nothing to compare to...
        #assert_almost_equal(md.params, cf[j], decimal=2)
        #assert_almost_equal(mdf.standard_errors, se[j], decimal=2)


    def test_ordinal_pandas(self):

        family = Binomial()

        endog_orig, exog_orig, groups = load_data("gee_ordinal_1.csv",
                                                 icept=False)

        data = np.concatenate((endog_orig[:,None], exog_orig,
                               groups[:,None]), axis=1)
        data = pd.DataFrame(data)
        data.columns = ["endog", "x1", "x2", "x3", "x4", "x5",
                        "group"]

        # Recode as cumulative indicators
        endog, exog, intercepts, nlevel = \
            gee_setup_ordinal(data, "endog")

        exog1 = np.concatenate((intercepts, exog), axis=1)
        groups = exog1[:,-1]
        exog1 = exog1[:,0:-1]

        v = GlobalOddsRatio(nlevel, "ordinal")

        beta = gee_ordinal_starting_values(endog_orig,
                                           exog_orig.shape[1])

        md = GEE(endog, exog1, groups, None, family, v)
        mdf = md.fit(starting_params = beta)
        # Nothing to compare to...
        #assert_almost_equal(md.params, cf[j], decimal=2)
        #assert_almost_equal(mdf.standard_errors, se[j], decimal=2)



    def test_poisson(self):
        """
        poisson

        library(gee)
        Z = read.csv("results/gee_poisson_1.csv", header=FALSE)
        Y = Z[,2]
        Id = Z[,1]
        X1 = Z[,3]
        X2 = Z[,4]
        X3 = Z[,5]
        X4 = Z[,6]
        X5 = Z[,7]

        mi = gee(Y ~ X1 + X2 + X3 + X4 + X5, id=Id, family=poisson,
                corstr="independence", scale.fix=TRUE)
        smi = summary(mi)
        u = coefficients(smi)
        cfi = paste(u[,1], collapse=",")
        sei = paste(u[,4], collapse=",")

        me = gee(Y ~ X1 + X2 + X3 + X4 + X5, id=Id, family=poisson,
                corstr="exchangeable", scale.fix=TRUE)
        sme = summary(me)

        u = coefficients(sme)
        cfe = paste(u[,1], collapse=",")
        see = paste(u[,4], collapse=",")

        sprintf("cf = [[%s],[%s]]", cfi, cfe)
        sprintf("se = [[%s],[%s]]", sei, see)
        """

        family = Poisson()

        endog,exog,group_n = load_data("gee_poisson_1.csv")

        vi = Independence()
        ve = Exchangeable()

        cf = [[-0.0364450410793481,-0.0543209391301178,
                0.0156642711741052,0.57628591338724,
                -0.00465659951186211,-0.477093153099256],
              [-0.0315615554826533,-0.0562589480840004,
                0.0178419412298561,0.571512795340481,
                -0.00363255566297332,-0.475971696727736]]
        se = [[0.0611309237214186,0.0390680524493108,
               0.0334234174505518,0.0366860768962715,
               0.0304758505008105,0.0316348058881079],
              [0.0610840153582275,0.0376887268649102,
               0.0325168379415177,0.0369786751362213,
               0.0296141014225009,0.0306115470200955]]

        for j,v in enumerate((vi,ve)):
            md = GEE(endog, exog, group_n, None, family, v)
            mdf = md.fit()
            assert_almost_equal(mdf.params, cf[j], decimal=5)
            assert_almost_equal(mdf.standard_errors, se[j], decimal=6)

        # Test with formulas
        D = np.concatenate((endog[:,None], group_n[:,None],
                            exog[:,1:]), axis=1)
        D = pd.DataFrame(D)
        D.columns = ["Y","Id",] + ["X%d" % (k+1)
                                   for k in range(exog.shape[1]-1)]
        for j,v in enumerate((vi,ve)):
             md = GEE.from_formula("Y ~ X1 + X2 + X3 + X4 + X5", D,
                                   None, groups=D.loc[:,"Id"],
                                   family=family, varstruct=v)
             mdf = md.fit()
             assert_almost_equal(mdf.params, cf[j], decimal=5)
             assert_almost_equal(mdf.standard_errors, se[j],
                                 decimal=6)


    def test_compare_OLS(self):
        """
        Gaussian GEE with independence correlation should agree
        exactly with OLS.
        """

        vs = Independence()
        family = Gaussian()

        Y = np.random.normal(size=100)
        X1 = np.random.normal(size=100)
        X2 = np.random.normal(size=100)
        X3 = np.random.normal(size=100)
        groups = np.kron(range(20), np.ones(5))

        D = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3})

        md = GEE.from_formula("Y ~ X1 + X2 + X3", D, None,
                              groups=groups, family=family,
                              varstruct=vs)
        mdf = md.fit()

        ols = sm.ols("Y ~ X1 + X2 + X3", data=D).fit()

        assert_almost_equal(ols.params.values, mdf.params, decimal=10)

        naive_tvalues = mdf.params / \
            np.sqrt(np.diag(mdf.naive_covariance))
        assert_almost_equal(naive_tvalues, ols.tvalues, decimal=10)


    def test_compare_logit(self):

        vs = Independence()
        family = Binomial()

        Y = 1*(np.random.normal(size=100) < 0)
        X1 = np.random.normal(size=100)
        X2 = np.random.normal(size=100)
        X3 = np.random.normal(size=100)
        groups = np.random.randint(0, 4, size=100)

        D = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3})

        md = GEE.from_formula("Y ~ X1 + X2 + X3", D, None, groups=groups,
                               family=family, varstruct=vs).fit()

        sml = sm.logit("Y ~ X1 + X2 + X3", data=D).fit()

        assert_almost_equal(sml.params.values, md.params, decimal=10)


    def test_compare_poisson(self):

        vs = Independence()
        family = Poisson()

        Y = np.ceil(-np.log(np.random.uniform(size=100)))
        X1 = np.random.normal(size=100)
        X2 = np.random.normal(size=100)
        X3 = np.random.normal(size=100)
        groups = np.random.randint(0, 4, size=100)

        D = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3})

        md = GEE.from_formula("Y ~ X1 + X2 + X3", D, None, groups=groups,
                               family=family, varstruct=vs).fit()

        sml = sm.poisson("Y ~ X1 + X2 + X3", data=D).fit()

        assert_almost_equal(sml.params.values, md.params, decimal=10)


if  __name__=="__main__":

    import nose

    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],
                   exit=False)

