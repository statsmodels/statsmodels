import numpy as np
import pandas as pd
from statsmodels.sandbox.mice import mice
import statsmodels.api as sm
import os

def load_data():
    """
    Load a data set from the results directory, generated by R mice routine.
    """

    params = pd.io.parsers.read_csv("params.csv")
    params.columns = ['int', 'x2', 'x3']
    se = pd.io.parsers.read_csv("cov.csv")
    se.columns = ['int', 'x2', 'x3']
    data = pd.io.parsers.read_csv("missingdata.csv")
    data.columns = ['x1', 'x2', 'x3']

    return params,se,data

class TestMice(object):
    def __init__(self):
        self.formula = "X2~X3+X4"

    def test_get_data_from_formula(self):
        np.random.seed(1325)
        data = np.random.normal(size=(10,4))
        data[8:, 1] = np.nan
        df = pd.DataFrame(data, columns=["X1", "X2", "X3", "X4"])
        imp_dat = mice.ImputedData(df)
        endog_obs, exog_obs, exog_miss = imp_dat.get_data_from_formula(
                                                            self.formula)
        endog_obs, exog_obs, exog_miss = imp_dat.get_data_from_formula(
                                                            self.formula)
        endog_obs = np.asarray(endog_obs).flatten()
        exog_obs = np.asarray(exog_obs)[:,1:]
        exog_miss = np.asarray(exog_miss)[:,1:]
        test_exog_obs = data[0:8,2:]
        test_exog_miss = data[-2:,2:]
        test_endog_obs = data[0:8,1]
        np.testing.assert_almost_equal(exog_obs, test_exog_obs)
        np.testing.assert_almost_equal(exog_miss, test_exog_miss)
        np.testing.assert_almost_equal(endog_obs, test_endog_obs)

    def test_store_changes(self):
        np.random.seed(1325)
        data = np.random.normal(size=(10,4))
        data[8:, 1] = np.nan
        df = pd.DataFrame(data, columns=["X1", "X2", "X3", "X4"])
        imp_dat = mice.ImputedData(df)
        imp_dat.store_changes("X2", [0] * 2)
        test_data = np.asarray(imp_dat.data["X2"][8:])
        np.testing.assert_almost_equal(test_data, np.asarray([0., 0.]))

    def test_perturb_params(self):
        np.random.seed(1325)
        data = np.random.normal(size=(10,4))
        data[8:, 1] = np.nan
        df = pd.DataFrame(data, columns=["X1", "X2", "X3", "X4"])
        params_test = np.asarray([-0.06040184,  0.40924707, -0.65996571])
        scale_test = 1.0
        md = sm.OLS.from_formula(self.formula, df)
        mdf = md.fit()
        imputer = mice.Imputer(self.formula, sm.OLS, mice.ImputedData(df))
        params, scale_per = imputer.perturb_params(mdf)
        params = np.asarray(params)
        np.testing.assert_almost_equal(params, params_test)
        np.testing.assert_almost_equal(scale_per, scale_test)

    def test_impute_asymptotic_bayes(self):
        np.random.seed(1325)
        data = np.random.normal(size=(10,4))
        data[8:, 1] = np.nan
        df = pd.DataFrame(data, columns=["X1", "X2", "X3", "X4"])
        imputer = mice.Imputer(self.formula, sm.OLS, mice.ImputedData(df))
        imputer.impute_asymptotic_bayes()
        np.testing.assert_almost_equal(np.asarray(imputer.data.data['X2'][8:]),
                                       np.asarray([-0.39097484, -0.31759086]))

    def test_impute_pmm(self):
        np.random.seed(1325)
        data = np.random.normal(size=(10,4))
        data[8:, 1] = np.nan
        df = pd.DataFrame(data, columns=["X1", "X2", "X3", "X4"])
        imputer = mice.Imputer(self.formula, sm.OLS, mice.ImputedData(df))
        imputer.impute_pmm()
        np.testing.assert_almost_equal(np.asarray(imputer.data.data['X2'][8:]),
                                       np.asarray([-0.77954822, -0.77954822]))

    def test_combine(self):
        np.random.seed(1325)
        data = np.random.normal(size=(10,4))
        data[8:, 1] = np.nan
        df = pd.DataFrame(data, columns=["X1", "X2", "X3", "X4"])
        impdata = mice.ImputedData(df)
        m = impdata.new_imputer("X2", scale_method="perturb_chi2")
        impcomb = mice.MICE("X2 ~ X1 + X3", sm.OLS, [m])
        implist = impcomb.run(method="pmm")
        p1 = impcomb.combine(implist)
        np.testing.assert_almost_equal(p1.params, np.asarray([0.30651575,
                                                              0.2264856 ,
                                                              0.03370901]))
        np.testing.assert_almost_equal(p1.scale, 0.64051079487931539)
        np.testing.assert_almost_equal(p1.normalized_cov_params, np.asarray([
       [ 0.1273622 ,  0.04854805, -0.00280011],
       [ 0.04854805,  0.10090235, -0.01022825],
       [-0.00280011, -0.01022825,  0.10273376]]))

    def test_overall(self):
        """
        R code used for comparison:

        N<-250;
        x1<-rbinom(N,1,prob=.4)  #draw from a binomial dist with probability=.4
        x2<-rnorm(N,0,1)         #draw from a normal dist with mean=0, sd=1
        x3<-rnorm(N,-10,1)
        y<--1+1*x1-1*x2+1*x3+rnorm(N,0,1)  #simulate linear regression data with a normal error (sd=1)

        #Generate MAR data

        alpha.1<-exp(16+2*y-x2)/(1+exp(16+2*y-x2));
        alpha.2<-exp(3.5+.7*y)/(1+exp(3.5+.7*y));
        alpha.3<-exp(-13-1.2*y-x1)/(1+exp(-13-1.2*y-x1));

        r.x1.mar<-rbinom(N,1,prob=alpha.1)
        r.x2.mar<-rbinom(N,1,prob=alpha.2)
        r.x3.mar<-rbinom(N,1,prob=alpha.3)
        x1.mar<-x1*(1-r.x1.mar)+r.x1.mar*99999  #x1.mar=x1 if not missing, 99999 if missing
        x2.mar<-x2*(1-r.x2.mar)+r.x2.mar*99999
        x3.mar<-x3*(1-r.x3.mar)+r.x3.mar*99999
        x1.mar[x1.mar==99999]=NA  #change 99999 to NA (R's notation for missing)
        x2.mar[x2.mar==99999]=NA
        x3.mar[x3.mar==99999]=NA

        require(mice)
        data = as.data.frame(cbind(x1.mar,x2.mar,x3.mar))
        data$x1.mar = as.factor(data$x1.mar)
        nrep = 500
        params = array(0, nrep)
        imp_pmm = mice(data,method="pmm", maxit=50)
        pooled = pool(with(imp_pmm,glm(x1.mar~x2.mar+x3.mar,family=binomial)))
        summary(pooled)

        setwd("C:/Users/Frank/Dropbox/statsmodels/statsmodels/sandbox/mice/tests")
        write.csv(cbind(pooled$u[1:5], pooled$u[21:25], pooled$u[41:45]), "cov.csv", row.names=FALSE)
        write.csv(pooled$qhat, "params.csv", row.names=FALSE)
        write.csv(data, "missingdata.csv", row.names=FALSE)
        """
        params,se,data = load_data()
        r_pooled_se = np.asarray(np.mean(se) + (1 + 1/5) * np.std(params))
        r_pooled_params = np.asarray(np.mean(params))
#        cur_dir = os.getcwd()
#        fn = os.path.join(cur_dir,"missingdata.csv")
#        data = pd.read_csv(fn)
#        data.columns = ['x1','x2','x3']
        impdata = mice.ImputedData(data)
        m1 = impdata.new_imputer("x2")
        m2 = impdata.new_imputer("x3")
        m3 = impdata.new_imputer("x1", model_class=sm.Logit)
        impcomb = mice.MICE("x1 ~ x2 + x3", sm.Logit,[m1,m2,m3])
        implist = impcomb.run(method="pmm")
        p1 = impcomb.combine(implist)
        np.testing.assert_allclose(p1.params, r_pooled_params, rtol=0.5)
        np.testing.assert_allclose(np.sqrt(np.diag(p1.normalized_cov_params)), r_pooled_se, rtol=1)

if  __name__=="__main__":

    import nose

    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],
                   exit=False)
