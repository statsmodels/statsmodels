import numpy as np
import pandas as pd
from statsmodels.sandbox.mice import mice
#import patsy
#from numpy.testing import assert_almost_equal

class TestMice(object):

    def test_get_data_from_formula(self):
        formula = "X2~X3+X4"
        np.random.seed(1325)
        data = np.random.normal(size=(10,4))
        data[8:, 1] = np.nan
        df = pd.DataFrame(data, columns=["X1", "X2", "X3", "X4"])
        imp_dat = mice.ImputedData(df)
        endog_obs, exog_obs, exog_miss = imp_dat.get_data_from_formula(formula)
        endog_obs, exog_obs, exog_miss = imp_dat.get_data_from_formula(formula)
        endog_obs = np.asarray(endog_obs).flatten()
        exog_obs = np.asarray(exog_obs)[:,1:]
        exog_miss = np.asarray(exog_miss)[:,1:]
        #test = patsy.dmatrices(formula, df)
        test_exog_obs = data[0:8,2:]
        test_exog_miss = data[-2:,2:]
        test_endog_obs = data[0:8,1]
        np.testing.assert_almost_equal(exog_obs, test_exog_obs)
        np.testing.assert_almost_equal(exog_miss, test_exog_miss)
        np.testing.assert_almost_equal(endog_obs, test_endog_obs)

if  __name__=="__main__":

    import nose

    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],
                   exit=False)
