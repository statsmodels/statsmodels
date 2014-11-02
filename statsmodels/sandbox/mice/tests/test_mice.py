import numpy as np
import pandas as pd
from statsmodels.sandbox.mice import mice
import statsmodels.api as sm

def gendat():
    """
    Create a data set with missing values.
    """

    np.random.seed(34243)

    n = 500
    p = 5

    exog = np.random.normal(size=(n, p))
    exog[:, 0] = exog[:, 1] - exog[:, 2] + 2*exog[:, 4]
    exog[:, 0] += np.random.normal(size=n)

    endog = exog.sum(1) + np.random.normal(size=n)

    df = pd.DataFrame(exog)
    df.columns = ["x%d" % k for k in range(1, p+1)]

    df["y"] = endog

    df.x1[0:60] = np.nan
    df.x2[0:40] = np.nan
    df.y[30:100] = np.nan

    return df

class TestMice(object):

    def test_MICE_data_default(self):
        """
        Test MICE_data with all defaults.
        """

        df = gendat()

        imp_data = mice.MICE_data(df)

        for k in range(3):
            imp_data.update_all()

    def test_MICE_data(self):
        """
        Test MICE_data with specified options.
        """

        df = gendat()

        for perturbation_method in "gaussian", "boot":
            for scale_method in "fix", "perturb_chi2":

                if perturbation_method == "boot" and scale_method != "fix":
                    continue

                imp_data = mice.MICE_data(df,
                             perturbation_method=perturbation_method)

                for k in range(3):
                    imp_data.update_all()


    def test_MICE(self):
        """
        Test MICE with specified options.
        """

        df = gendat()

        imp_data = mice.MICE_data(df)
        mi = mice.MICE("y ~ x1 + x2 + x1:x2", sm.OLS, imp_data)
        mi.burnin(1)
        mi.run(2, 1)
        mi.combine()



if  __name__=="__main__":

    import nose

    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],
                   exit=False)
