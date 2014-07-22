"""
Test functions for GEE

External comparisons are to R.  The statmodels GEE implementation
should generally agree with the R GEE implementation for the
independence and exchangeable correlation structures.  For other
correlation structures, the details of the correlation estimation
differ among implementations and the results will not agree exactly.
"""

from statsmodels.compat import lrange
import numpy as np
import os
from numpy.testing import (assert_almost_equal, assert_array_less,
                           assert_equal)
from statsmodels.genmod.generalized_estimating_equations import (GEE,
     OrdinalGEE, NominalGEE, GEEMargins, Multinomial,
     NominalGEEResults, OrdinalGEEResults)
from statsmodels.genmod.families import Gaussian, Binomial, Poisson
from statsmodels.genmod.dependence_structures import (Exchangeable,
    Independence, GlobalOddsRatio, Autoregressive, Nested)
import pandas as pd
import statsmodels.formula.api as sm
from scipy.stats.distributions import norm

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
        mod1 = GEE.from_formula("y ~ age + trt + base", data["subject"],
                                data, cov_struct=ind, family=fam)
        rslt1 = mod1.fit()

        # Coefficients should agree with GLM
        from statsmodels.genmod.generalized_linear_model import GLM
        from statsmodels.genmod import families

        mod2 = GLM.from_formula("y ~ age + trt + base", data,
                               family=families.Poisson())
        rslt2 = mod2.fit(scale="X2")

        assert_almost_equal(rslt1.params, rslt2.params, decimal=6)
        assert_almost_equal(rslt1.scale, rslt2.scale, decimal=6)



    # TODO: why does this test fail?
    def t_est_missing(self):

        Y = np.random.normal(size=100)
        X1 = np.random.normal(size=100)
        X2 = np.random.normal(size=100)
        X3 = np.random.normal(size=100)
        groups = np.kron(lrange(20), np.ones(5))

        Y[0] = np.nan
        Y[5:7] = np.nan
        X2[10:12] = np.nan

        D = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3,
                          "groups": groups})

        md = GEE.from_formula("Y ~ X1 + X2 + X3", D["groups"],
                              missing='drop')
        mdf = md.fit()

        assert_almost_equal(len(md.endog), 95)
        assert_almost_equal(np.asarray(md.exog.shape), np.r_[95, 4])

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
            T[jj] = lrange(len(jj))

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
            T[jj] = lrange(len(jj))

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
             md = GEE.from_formula("Y ~ X1 + X2 + X3", "Id", D,
                                   family=family, cov_struct=v)
             mdf = md.fit()
             assert_almost_equal(mdf.params, cf[j], decimal=6)
             assert_almost_equal(mdf.standard_errors(), se[j],
                                 decimal=6)

        # Check for run-time exceptions in summary
        # print(mdf.summary())

    def test_autoregressive(self):

        dep_params_true = [0, 0.589208623896, 0.559823804948]

        params_true = [[1.08043787, 1.12709319, 0.90133927],
                       [0.9613677, 1.05826987, 0.90832055],
                       [1.05370439, 0.96084864, 0.93923374]]

        np.random.seed(342837482)

        num_group = 100
        ar_param = 0.5
        k = 3

        ga = Gaussian()

        for gsize in 1,2,3:

            ix = np.arange(gsize)[:,None] - np.arange(gsize)[None,:]
            ix = np.abs(ix)
            cmat = ar_param ** ix
            cmat_r = np.linalg.cholesky(cmat)

            endog = []
            exog = []
            groups = []
            for i in range(num_group):
                x = np.random.normal(size=(gsize,k))
                exog.append(x)
                expval = x.sum(1)
                errors = np.dot(cmat_r, np.random.normal(size=gsize))
                endog.append(expval + errors)
                groups.append(i*np.ones(gsize))

            endog = np.concatenate(endog)
            groups = np.concatenate(groups)
            exog = np.concatenate(exog, axis=0)

            ar = Autoregressive()
            md = GEE(endog, exog, groups, family=ga, cov_struct = ar)
            mdf = md.fit()
            assert_almost_equal(ar.dep_params, dep_params_true[gsize-1])
            assert_almost_equal(mdf.params, params_true[gsize-1])

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

    def test_scoretest(self):
        # Regression tests

        np.random.seed(6432)
        n = 200 # Must be divisible by 4
        exog = np.random.normal(size=(n, 4))
        endog = exog[:, 0] + exog[:, 1] + exog[:, 2]
        endog += 3*np.random.normal(size=n)
        group = np.kron(np.arange(n/4), np.ones(4))

        # Test under the null.
        L = np.array([[1., -1, 0, 0]])
        R = np.array([0.,])
        family = Gaussian()
        va = Independence()
        mod1 = GEE(endog, exog, group, family=family,
                  cov_struct=va, constraint=(L, R))
        rslt1 = mod1.fit()
        assert_almost_equal(mod1.score_test_results["statistic"],
                            1.08126334)
        assert_almost_equal(mod1.score_test_results["p-value"],
                            0.2984151086)

        # Test under the alternative.
        L = np.array([[1., -1, 0, 0]])
        R = np.array([1.0,])
        family = Gaussian()
        va = Independence()
        mod2 = GEE(endog, exog, group, family=family,
                   cov_struct=va, constraint=(L, R))
        rslt2 = mod2.fit()
        assert_almost_equal(mod2.score_test_results["statistic"],
                            3.491110965)
        assert_almost_equal(mod2.score_test_results["p-value"],
                            0.0616991659)

        # Compare to Wald tests
        exog = np.random.normal(size=(n, 2))
        L = np.array([[1, -1]])
        R = np.array([0.])
        f = np.r_[1, -1]
        for i in range(10):
            endog = exog[:, 0] + (0.5 + i/10.)*exog[:, 1] +\
                    np.random.normal(size=n)
            family = Gaussian()
            va = Independence()
            mod0 = GEE(endog, exog, group, family=family,
                       cov_struct=va)
            rslt0 = mod0.fit()
            family = Gaussian()
            va = Independence()
            mod1 = GEE(endog, exog, group, family=family,
                       cov_struct=va, constraint=(L, R))
            rslt1 = mod1.fit()
            se = np.sqrt(np.dot(f, np.dot(rslt0.cov_params(), f)))
            wald_z = np.dot(f, rslt0.params) / se
            wald_p = 2*norm.cdf(-np.abs(wald_z))
            score_p = mod1.score_test_results["p-value"]
            assert_array_less(np.abs(wald_p - score_p), 0.02)


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

        for j,v in enumerate((vi, ve)):
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
            md = GEE.from_formula("Y ~ X1 + X2 + X3", "Id", D,
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

        endog, exog, group = load_data("gee_nested_linear_1.csv")

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

        va = GlobalOddsRatio("ordinal")

        mod = OrdinalGEE(endog, exog, groups, None, family, va)
        rslt = mod.fit()

        # Regression test
        cf = np.r_[1.09250002, 0.0217443 , -0.39851092, -0.01812116,
                   0.03023969, 1.18258516, 0.01803453, -1.10203381]
        assert_almost_equal(rslt.params, cf, decimal=5)

        # Regression test
        se = np.r_[0.10883461, 0.10330197, 0.11177088, 0.05486569,
                   0.05997153, 0.09168148, 0.05953324, 0.0853862]
        assert_almost_equal(rslt.bse, se, decimal=5)

        # Check that we get the correct results type
        assert_equal(type(rslt), OrdinalGEEResults)

    def test_nominal(self):

        family = Multinomial(3)

        endog, exog, groups = load_data("gee_nominal_1.csv",
                                        icept=False)

        # Test with independence correlation
        va = Independence()
        mod1 = NominalGEE(endog, exog, groups, None, family, va)
        rslt1 = mod1.fit()

        # Regression test
        cf1 = np.r_[0.44944752,  0.45569985, -0.92007064, -0.46766728]
        se1 = np.r_[0.09801821,  0.07718842,  0.13229421,  0.08544553]
        assert_almost_equal(rslt1.params, cf1, decimal=5)
        assert_almost_equal(rslt1.standard_errors(), se1, decimal=5)

        # Test with global odds ratio dependence
        va = GlobalOddsRatio("nominal")
        mod2 = NominalGEE(endog, exog, groups, None, family, va)
        rslt2 = mod2.fit(start_params=rslt1.params)

        # Regression test
        cf2 = np.r_[0.45448248,  0.41945568, -0.92008924, -0.50485758]
        se2 = np.r_[0.09632274,  0.07433944,  0.13264646,  0.0911768]
        assert_almost_equal(rslt2.params, cf2, decimal=5)
        assert_almost_equal(rslt2.standard_errors(), se2, decimal=5)

        # Make sure we get the correct results type
        assert_equal(type(rslt1), NominalGEEResults)


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
             md = GEE.from_formula("Y ~ X1 + X2 + X3 + X4 + X5", "Id",
                                   D, family=family, cov_struct=v)
             mdf = md.fit()
             assert_almost_equal(mdf.params, cf[j], decimal=5)
             assert_almost_equal(mdf.standard_errors(), se[j],
                                 decimal=6)
             # print(mdf.params)


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
        groups = np.kron(lrange(20), np.ones(5))

        D = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3})

        md = GEE.from_formula("Y ~ X1 + X2 + X3", groups, D,
                              family=family, cov_struct=vs)
        mdf = md.fit()

        ols = sm.ols("Y ~ X1 + X2 + X3", data=D).fit()

        assert_almost_equal(ols.params.values, mdf.params, decimal=10)

        se = mdf.standard_errors(covariance_type="naive")
        assert_almost_equal(ols.bse, se, decimal=10)

        naive_tvalues = mdf.params / \
            np.sqrt(np.diag(mdf.naive_covariance))
        assert_almost_equal(naive_tvalues, ols.tvalues, decimal=10)

    def test_formulas(self):
        """
        Check formulas, especially passing groups and time as either
        variable names or arrays.
        """

        n = 100
        Y = np.random.normal(size=n)
        X1 = np.random.normal(size=n)
        mat = np.concatenate((np.ones((n,1)), X1[:, None]), axis=1)
        Time = np.random.uniform(size=n)
        groups = np.kron(lrange(20), np.ones(5))

        data = pd.DataFrame({"Y": Y, "X1": X1, "Time": Time, "groups": groups})

        va = Autoregressive()
        family = Gaussian()

        mod1 = GEE(Y, mat, groups, time=Time, family=family,
                   cov_struct=va)
        rslt1 = mod1.fit()

        mod2 = GEE.from_formula("Y ~ X1", groups, data, time=Time,
                                family=family, cov_struct=va)
        rslt2 = mod2.fit()

        mod3 = GEE.from_formula("Y ~ X1", groups, data, time="Time",
                                family=family, cov_struct=va)
        rslt3 = mod3.fit()

        mod4 = GEE.from_formula("Y ~ X1", "groups", data, time=Time,
                                family=family, cov_struct=va)
        rslt4 = mod4.fit()

        mod5 = GEE.from_formula("Y ~ X1", "groups", data, time="Time",
                                family=family, cov_struct=va)
        rslt5 = mod5.fit()

        assert_almost_equal(rslt1.params, rslt2.params, decimal=8)
        assert_almost_equal(rslt1.params, rslt3.params, decimal=8)
        assert_almost_equal(rslt1.params, rslt4.params, decimal=8)
        assert_almost_equal(rslt1.params, rslt5.params, decimal=8)

    def test_compare_logit(self):

        vs = Independence()
        family = Binomial()

        Y = 1*(np.random.normal(size=100) < 0)
        X1 = np.random.normal(size=100)
        X2 = np.random.normal(size=100)
        X3 = np.random.normal(size=100)
        groups = np.random.randint(0, 4, size=100)

        D = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3})

        mod1 = GEE.from_formula("Y ~ X1 + X2 + X3", groups, D,
                                family=family, cov_struct=vs)
        rslt1 = mod1.fit()

        mod2 = sm.logit("Y ~ X1 + X2 + X3", data=D)
        rslt2 = mod2.fit()

        assert_almost_equal(rslt1.params, rslt2.params, decimal=10)


    def test_compare_poisson(self):

        vs = Independence()
        family = Poisson()

        Y = np.ceil(-np.log(np.random.uniform(size=100)))
        X1 = np.random.normal(size=100)
        X2 = np.random.normal(size=100)
        X3 = np.random.normal(size=100)
        groups = np.random.randint(0, 4, size=100)

        D = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3})

        mod1 = GEE.from_formula("Y ~ X1 + X2 + X3", groups, D,
                                family=family, cov_struct=vs)
        rslt1 = mod1.fit()

        mod2 = sm.poisson("Y ~ X1 + X2 + X3", data=D)
        rslt2 = mod2.fit(disp=False)

        assert_almost_equal(rslt1.params, rslt2.params, decimal=10)


    def test_sensitivity(self):

        va = Exchangeable()
        family = Gaussian()

        n = 100
        Y = np.random.normal(size=n)
        X1 = np.random.normal(size=n)
        X2 = np.random.normal(size=n)
        groups = np.kron(np.arange(50), np.r_[1, 1])

        D = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2})

        mod = GEE.from_formula("Y ~ X1 + X2", groups, D,
                               family=family, cov_struct=va)
        rslt = mod.fit()
        ps = rslt.params_sensitivity(0, 0.5, 2)
        assert_almost_equal(len(ps), 2)
        assert_almost_equal([x.cov_struct.dep_params for x in ps],
                            [0.0, 0.5])

        # Regression test
        assert_almost_equal([x.params[0] for x in ps],
                            np.r_[-0.1256575, -0.126747036])

if  __name__=="__main__":

    import nose

    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],
                   exit=False)
