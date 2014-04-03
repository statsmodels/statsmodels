"""
Test functions for GEE

External comparisons are to R.  The statmodels GEE implementation
should generally agree with the R GEE implementation for the
independence and exchangeable correlation structures.  For other
correlation structures, the details of the correlation estimation
differ among implementations and the results will not agree exactly.
"""

import numpy as np
import os
from numpy.testing import assert_almost_equal
from statsmodels.genmod.generalized_estimating_equations import GEE,\
    GEEMargins, Multinomial
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


    # This is in the release announcement for version 0.6.
    def test_poisson_epil(self):

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        fname = os.path.join(cur_dir, "results", "epil.csv")
        data = pd.read_csv(fname)

        fam = Poisson()
        ind = Independence()
        md1 = GEE.from_formula("y ~ age + trt + base", data,
                               groups=data["subject"], cov_struct=ind,
                               family=fam)
        mdf1 = md1.fit()

        # Coefficients should agree with GLM
        from statsmodels.genmod.generalized_linear_model import GLM
        from statsmodels.genmod import families

        md2 = GLM.from_formula("y ~ age + trt + base", data,
                               family=families.Poisson())
        mdf2 = md2.fit(scale="X2")

        assert_almost_equal(mdf1.params, mdf2.params, decimal=6)
        assert_almost_equal(mdf1.scale, mdf2.scale, decimal=6)



    # TODO: why does this test fail?
    def t_est_missing(self):

        Y = np.random.normal(size=100)
        X1 = np.random.normal(size=100)
        X2 = np.random.normal(size=100)
        X3 = np.random.normal(size=100)
        groups = np.kron(range(20), np.ones(5))

        Y[0] = np.nan
        Y[5:7] = np.nan
        X2[10:12] = np.nan

        D = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3,
                          "groups": groups})

        md = GEE.from_formula("Y ~ X1 + X2 + X3", D, None,
                              groups=D["groups"], missing='drop')
        mdf = md.fit()

        assert(len(md.endog) == 95)
        assert(md.exog.shape) == (95,4)

    def test_default_time(self):
        """
        Check that the time defaults work correctly.
        """

        endog,exog,group = load_data("gee_logistic_1.csv")

        # Time values for the autoregressive model
        T = np.zeros(len(endog))
        idx = set(group)
        for ii in idx:
            jj = np.flatnonzero(group == ii)
            T[jj] = range(len(jj))

        family = Binomial()
        va = Autoregressive()


        md1 = GEE(endog, exog, group, family=family, cov_struct=va)
        mdf1 = md1.fit()

        md2 = GEE(endog, exog, group, time=T, family=family,
                  cov_struct=va)
        mdf2 = md2.fit()

        assert_almost_equal(mdf1.params, mdf2.params, decimal=6)
        assert_almost_equal(mdf1.standard_errors(),
                            mdf2.standard_errors(), decimal=6)



    def test_logistic(self):
        """
        R code for comparing results:

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

        # From R gee
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
                assert_almost_equal(mdf.standard_errors(), se[j],
                                    decimal=6)

        # Test with formulas
        D = np.concatenate((endog[:,None], group[:,None], exog[:,1:]),
                           axis=1)
        D = pd.DataFrame(D)
        D.columns = ["Y","Id",] + ["X%d" % (k+1)
                                   for k in range(exog.shape[1]-1)]
        for j,v in enumerate((vi,ve)):
             md = GEE.from_formula("Y ~ X1 + X2 + X3", D, None,
                                   groups=D.loc[:,"Id"],
                                   family=family, cov_struct=v)
             mdf = md.fit()
             assert_almost_equal(mdf.params, cf[j], decimal=6)
             assert_almost_equal(mdf.standard_errors(), se[j],
                                 decimal=6)

        # Check for run-time exceptions in summary
        # print mdf.summary()


    def test_post_estimation(self):

        family = Gaussian()
        endog,exog,group = load_data("gee_linear_1.csv")

        ve = Exchangeable()

        md = GEE(endog, exog, group, None, family, ve)
        mdf = md.fit()

        assert_almost_equal(np.dot(exog, mdf.params),
                            mdf.fittedvalues)
        assert_almost_equal(endog - np.dot(exog, mdf.params),
                            mdf.resid)



    def test_linear(self):
        """
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

        # From R gee
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
            assert_almost_equal(mdf.standard_errors(), se[j],
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
                                  family=family, cov_struct=v)
            mdf = md.fit()
            assert_almost_equal(mdf.params, cf[j], decimal=10)
            assert_almost_equal(mdf.standard_errors(), se[j],
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
        for i in range(endog.shape[0]//10):
            group_n.extend([0,]*5)
            group_n.extend([1,]*5)
        group_n = np.array(group_n)[:,None]

        dp = Independence()
        md = GEE(endog, exog, group, None, family, dp)
        mdf1 = md.fit()

        # From statsmodels.GEE (not an independent test)
        cf = np.r_[-0.1671073 ,  1.00467426, -2.01723004,  0.97297106]
        se = np.r_[0.08629606,  0.04058653,  0.04067038,  0.03777989]
        assert_almost_equal(mdf1.params, cf, decimal=6)
        assert_almost_equal(mdf1.standard_errors(), se,
                            decimal=6)

        ne = Nested()
        md = GEE(endog, exog, group, None, family, ne,
                 dep_data=group_n)
        mdf2 = md.fit(start_params=mdf1.params)

        # From statsmodels.GEE (not an independent test)
        cf = np.r_[-0.16655319,  1.02183688, -2.00858719,  1.00101969]
        se = np.r_[0.08632616,  0.02913582,  0.03114428,  0.02893991]
        assert_almost_equal(mdf2.params, cf, decimal=6)
        assert_almost_equal(mdf2.standard_errors(), se,
                            decimal=6)


    def test_ordinal(self):

        family = Binomial()

        endog, exog, groups = load_data("gee_ordinal_1.csv",
                                        icept=False)

        v = GlobalOddsRatio("ordinal")

        md = GEE(endog, exog, groups, None, family, v)
        md.setup_ordinal()
        mdf = md.fit()

        cf = np.r_[1.09238131, 0.02148193, -0.39879146, -0.01855666,
                   0.02983409, 1.18123172,  0.01845318, -1.10233886]
        se = np.r_[0.10878752,  0.10326078,  0.11171241, 0.05488705,
                   0.05995019, 0.0916574,  0.05951445,  0.08539281]

        assert_almost_equal(mdf.params, cf, decimal=5)
        assert_almost_equal(mdf.bse, se, decimal=5)


    def test_nominal(self):

        family = Multinomial(3)

        endog, exog, groups = load_data("gee_nominal_1.csv",
                                        icept=False)

        # Test with independence correlation
        v = Independence()
        md = GEE(endog, exog, groups, None, family, v)
        md.setup_nominal()
        mdf1 = md.fit()

        # From statsmodels.GEE (not an independent test)
        cf1 = np.r_[0.44944752,  0.45569985, -0.92007064, -0.46766728]
        se1 = np.r_[0.09801821,  0.07718842,  0.13229421,  0.08544553]
        assert_almost_equal(mdf1.params, cf1, decimal=5)
        assert_almost_equal(mdf1.standard_errors(), se1, decimal=5)

        # Test with global odds ratio dependence
        v = GlobalOddsRatio("nominal")
        md = GEE(endog, exog, groups, None, family, v)
        md.setup_nominal()
        mdf2 = md.fit(start_params=mdf1.params)

        # From statsmodels.GEE (not an independent test)
        cf2 = np.r_[0.45397549,  0.42278345, -0.91997131, -0.50115943]
        se2 = np.r_[0.09646057,  0.07405713,  0.1324629 ,  0.09025019]
        assert_almost_equal(mdf2.params, cf2, decimal=5)
        assert_almost_equal(mdf2.standard_errors(), se2, decimal=5)



    def test_poisson(self):
        """
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

        # From R gee
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
            assert_almost_equal(mdf.standard_errors(), se[j],
                                decimal=6)

        # Test with formulas
        D = np.concatenate((endog[:,None], group_n[:,None],
                            exog[:,1:]), axis=1)
        D = pd.DataFrame(D)
        D.columns = ["Y","Id",] + ["X%d" % (k+1)
                                   for k in range(exog.shape[1]-1)]
        for j,v in enumerate((vi,ve)):
             md = GEE.from_formula("Y ~ X1 + X2 + X3 + X4 + X5", D,
                                   None, groups=D.loc[:,"Id"],
                                   family=family, cov_struct=v)
             mdf = md.fit()
             assert_almost_equal(mdf.params, cf[j], decimal=5)
             assert_almost_equal(mdf.standard_errors(), se[j],
                                 decimal=6)
             # print mdf.params


    def test_compare_OLS(self):
        """
        Gaussian GEE with independence correlation should agree
        exactly with OLS for parameter estimates and standard errors
        derived from the naive covariance estimate.
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
                              cov_struct=vs)
        mdf = md.fit()

        ols = sm.ols("Y ~ X1 + X2 + X3", data=D).fit()

        assert_almost_equal(ols.params.values, mdf.params, decimal=10)

        se = mdf.standard_errors(covariance_type="naive")
        assert_almost_equal(ols.bse, se, decimal=10)

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
                               family=family, cov_struct=vs).fit()

        sml = sm.logit("Y ~ X1 + X2 + X3", data=D).fit(disp=False)

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
                               family=family, cov_struct=vs).fit()

        sml = sm.poisson("Y ~ X1 + X2 + X3", data=D).fit(disp=False)

        assert_almost_equal(sml.params.values, md.params, decimal=10)


if  __name__=="__main__":

    import nose

    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],
                   exit=False)
