import numpy as np
import pandas as pd
from statsmodels.sandbox.mice import mice
import statsmodels.api as sm

class TestMice(object):
    def __init__(self):
        self.formula = "X2~X3+X4"

    def test_get_data_from_formula(self):
        np.random.seed(1325)
        data = np.random.normal(size=(10,4))
        data[8:, 1] = np.nan
        df = pd.DataFrame(data, columns=["X1", "X2", "X3", "X4"])
        imp_dat = mice.ImputedData(df)
        endog_obs, exog_obs, exog_miss = imp_dat.get_data_from_formula(self.formula)
        endog_obs, exog_obs, exog_miss = imp_dat.get_data_from_formula(self.formula)
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
        imp_dat.store_changes([0] * 2, "X2")
        test_data = np.asarray(imp_dat.data["X2"][8:])
        np.testing.assert_almost_equal(test_data, np.asarray([0., 0.]))

    def test_perturb_params(self):
        np.random.seed(1325)
        data = np.random.normal(size=(10,4))
        data[8:, 1] = np.nan
        df = pd.DataFrame(data, columns=["X1", "X2", "X3", "X4"])
        params_test = np.asarray([-0.06642686, 0.36131348, -0.69498469])
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
        np.testing.assert_almost_equal(np.asarray(imputer.data.data['X2'][8:]),np.asarray([-0.82292821, -0.22632992]))
        
    def test_impute_pmm(self):
        np.random.seed(1325)
        data = np.random.normal(size=(10,4))
        data[8:, 1] = np.nan
        df = pd.DataFrame(data, columns=["X1", "X2", "X3", "X4"])
        imputer = mice.Imputer(self.formula, sm.OLS, mice.ImputedData(df))
        imputer.impute_pmm()
        np.testing.assert_almost_equal(np.asarray(imputer.data.data['X2'][8:]),np.asarray([-0.77954822, -0.77954822]))
        
if  __name__=="__main__":

    import nose

    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],
                   exit=False)
