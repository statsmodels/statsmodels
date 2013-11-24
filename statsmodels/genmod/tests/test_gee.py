"""
Test functions for GEE
"""

import numpy as np
from numpy.testing import assert_almost_equal
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Gaussian,Binomial,Poisson
from statsmodels.genmod.dependence_structures import Exchangeable,Independence,\
    GlobalOddsRatio,Autoregressive,Nested
import pandas as pd

def load_data(fname, icept=True):

    Z = np.loadtxt(fname, delimiter=",")

    Id = Z[:,0]
    Y = Z[:,1]
    X = Z[:,2:]

    if icept:
        X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)

    return Y,X,Id


class TestGEE(object):


    def test_logistic(self):
        """
        logistic

        library(gee)
        Z = read.csv("data/gee_logistic_1.csv", header=FALSE)
        Y = Z[,2]
        Id = Z[,1]
        X1 = Z[,3]
        X2 = Z[,4]
        X3 = Z[,5]

        mi = gee(Y ~ X1 + X2 + X3, id=Id, family=binomial, corstr="independence")
        smi = summary(mi)
        u = coefficients(smi)
        cfi = paste(u[,1], collapse=",")
        sei = paste(u[,4], collapse=",")

        me = gee(Y ~ X1 + X2 + X3, id=Id, family=binomial, corstr="exchangeable")
        sme = summary(me)
        u = coefficients(sme)
        cfe = paste(u[,1], collapse=",")
        see = paste(u[,4], collapse=",")

        ma = gee(Y ~ X1 + X2 + X3, id=Id, family=binomial, corstr="AR-M")
        sma = summary(ma)
        u = coefficients(sma)
        cfa = paste(u[,1], collapse=",")
        sea = paste(u[,4], collapse=",")

        sprintf("cf = [[%s],[%s],[%s]]", cfi, cfe, cfa)
        sprintf("se = [[%s],[%s],[%s]]", sei, see, sea)
        """

        Y,X,Id = load_data("data/gee_logistic_1.csv")

        # Time values for the autoregressive model
        T = np.zeros(len(Y))
        idx = set(Id)
        for ii in idx:
            jj = np.flatnonzero(Id == ii)
            T[jj] = range(len(jj))

        family = Binomial()
        ve = Exchangeable()
        vi = Independence()
        va = Autoregressive()

        cf = [[0.161734060032208,1.12611789573132,-1.97324634010308,0.966502770589527],
              [0.159868996418795,1.12859602923397,-1.97524775767612,0.958142284106185]]

        se = [[0.0812181805908476,0.0933962273608725,0.122192175318107,0.100816619280202],
              [0.0816175453351956,0.0928973822942355,0.121459304850799,0.100993351847033]]

        for j,v in enumerate((vi,ve,va)):
            md = GEE(Y, X, Id, T, family, v)
            mdf = md.fit()
            if id(v) != id(va):
                assert_almost_equal(mdf.params, cf[j], decimal=6)
                assert_almost_equal(mdf.standard_errors, se[j], decimal=6)

        # Test with formulas
        D = np.concatenate((Y[:,None], Id[:,None], X[:,1:]), axis=1)
        D = pd.DataFrame(D)
        D.columns = ["Y","Id",] + ["X%d" % (k+1) for k in range(X.shape[1]-1)]
        for j,v in enumerate((vi,ve)):
             md = GEE.from_formula("Y ~ X1 + X2 + X3", "Id", D, None, family, v)
             mdf = md.fit()
             assert_almost_equal(mdf.params, cf[j], decimal=6)
             assert_almost_equal(mdf.standard_errors, se[j], decimal=6)



    def test_linear(self):
        """
        linear
        
        library(gee)

        Z = read.csv("gee_linear_1.csv", header=FALSE)        
        Y = Z[,2]                                                                                                                  
        Id = Z[,1]
        X1 = Z[,3]
        X2 = Z[,4]
        X3 = Z[,5]
        mi = gee(Y ~ X1 + X2 + X3, id=Id, family=gaussian, corstr="independence",
                tol=1e-8, maxit=100)
        smi = summary(mi)
        u = coefficients(smi)
          
        cfi = paste(u[,1], collapse=",")
        sei = paste(u[,4], collapse=",")

        me = gee(Y ~ X1 + X2 + X3, id=Id, family=gaussian, corstr="exchangeable", 
                tol=1e-8, maxit=100)          
        sme = summary(me)
        u = coefficients(sme)

        cfe = paste(u[,1], collapse=",")
        see = paste(u[,4], collapse=",")

        sprintf("cf = [[%s],[%s]]", cfi, cfe) 
        sprintf("se = [[%s],[%s]]", sei, see)                                        
        """

        family = Gaussian()

        Y,X,Id = load_data("data/gee_linear_1.csv")

        vi = Independence()
        ve = Exchangeable()

        cf = [[0.00515978834534064,0.78615903847622,-1.57628929834004,0.782486240348685],
              [0.00516507033680904,0.786253541786879,-1.57666801099155,0.781741984193051]]
        se = [[0.025720523853008,0.0303348838938358,0.0371658992200722,0.0301352423377647],
              [0.025701817387204,0.0303307060257735,0.0371977050322601,0.0301218562204013]] 

        for j,v in enumerate((vi,ve)):
            md = GEE(Y, X, Id, None, family, v)
            mdf = md.fit()
            assert_almost_equal(mdf.params, cf[j], decimal=10)
            assert_almost_equal(mdf.standard_errors, se[j], decimal=10)

        # Test with formulas
        D = np.concatenate((Y[:,None], Id[:,None], X[:,1:]), axis=1)
        D = pd.DataFrame(D)
        D.columns = ["Y","Id",] + ["X%d" % (k+1) for k in range(X.shape[1]-1)]
        for j,v in enumerate((vi,ve)):
             md = GEE.from_formula("Y ~ X1 + X2 + X3", "Id", D, None, family, v)
             mdf = md.fit()
             assert_almost_equal(mdf.params, cf[j], decimal=10)
             assert_almost_equal(mdf.standard_errors, se[j], decimal=10)



    def test_nested_linear(self):

        family = Gaussian()

        Y,X,Id = load_data("data/gee_nested_linear_1.csv")

        Idn = []
        for i in range(300):
            Idn.extend([0,]*5)
            Idn.extend([1,]*5)
        Idn = np.array(Idn)

        ne = Nested(Idn)

        md = GEE(Y, X, Id, None, family, ne)
        mdf = md.fit()
        ## Nothing to compare to


    def test_ordinal(self):

        family = Binomial()

        Y,X,Id = load_data("data/gee_ordinal_1.csv", icept=False)

        v = GlobalOddsRatio()

        md = GEE(Y, X, Id, None, family, v, ytype="ordinal")
        mdf = md.fit()
        # Nothing to compare to...
        #assert_almost_equal(md.params, cf[j], decimal=2)
        #assert_almost_equal(mdf.standard_errors, se[j], decimal=2)


    def test_poisson(self):
        """
        poisson

        library(gee)
        Z = read.csv("gee_poisson_1.csv", header=FALSE)
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

        Y,X,Id = load_data("data/gee_poisson_1.csv")

        vi = Independence()
        ve = Exchangeable()

        cf = [[-0.0146481939473855,-0.00354936927720112,0.00373735567047755,0.50536434354091,0.00536672970672592,-0.506763623216482],
              [-0.0146390416486013,-0.00378457467315029,0.00359526175252784,0.505218312342825,0.00520243210015778,-0.506959420331449]]
        se = [[0.0180718833872629,0.00804583519493001,0.00932754357592236,0.00859676512232225,0.00917599454216625,0.00903356938618812],
              [0.0180852155632977,0.00805161458483081,0.00933886210442408,0.00862255601233811,0.00917229773191988,0.00904411930948212]]

        for j,v in enumerate((vi,ve)):
            md = GEE(Y, X, Id, None, family, v)
            mdf = md.fit()
            assert_almost_equal(mdf.params, cf[j], decimal=5)
            assert_almost_equal(mdf.standard_errors, se[j], decimal=6)

        # Test with formulas
        D = np.concatenate((Y[:,None], Id[:,None], X[:,1:]), axis=1)
        D = pd.DataFrame(D)
        D.columns = ["Y","Id",] + ["X%d" % (k+1) for k in range(X.shape[1]-1)]
        for j,v in enumerate((vi,ve)):
             md = GEE.from_formula("Y ~ X1 + X2 + X3 + X4 + X5", "Id", D, None, family, v)
             mdf = md.fit()
             assert_almost_equal(mdf.params, cf[j], decimal=5)
             assert_almost_equal(mdf.standard_errors, se[j], decimal=6)


if  __name__=="__main__":

    import nose

    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],
                   exit=False)

