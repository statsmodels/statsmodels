
from statsmodels.compat.collections import OrderedDict
import numpy as np

import statsmodels.api as sm
from statsmodels.discrete.dcm_mxlogit import MXLogit#, CLogitResults

from statsmodels.discrete.tests.results.results_dcm_clogit \
    import Travelmodechoice





if __name__ == "__main__":

    DEBUG = 0

    print('Example:')

    # Loading data as pandas object
    data = sm.datasets.modechoice.load_pandas()
    data.endog[:5]
    data.exog[:5]
    data.exog['Intercept'] = 1  # include an intercept
    y, X = data.endog, data.exog

    # Set up model
    # Names of the variables for the utility function for each alternative
    # variables with common coefficients have to be first in each array
    # the order should be the same as sequence in data
    # ie: row1 -- air data, row2 -- train data, row3 -- bus data, ...

    V = OrderedDict((
        ('air',   ['gc', 'ttme', 'Intercept', 'hinc']),
        ('train', ['gc', 'ttme', 'Intercept']),
        ('bus',   ['gc', 'ttme', 'Intercept']),
        ('car',   ['gc', 'ttme']))
        )

    # Number of common coefficients
    ncommon = 2

    # Random coefficients and distributions (By now, only Normal)
    # TODO: add Uniform, Triangular and Log-nomal

    NORMAL = ['gc']

    # Number of draws used for simulation
    # eg: 50 for draf model, 1000 for final model
    draws = 100

    # Describe model
    ref_level = 'car'
    name_intercept = 'Intercept'
    mxlogit_mod = MXLogit(endog_data = y, exog_data = X,  V = V,
                          NORMAL = NORMAL, draws = draws,
                          ncommon = ncommon, ref_level = ref_level,
                          name_intercept = name_intercept)

    # Fit model
#    mxlogit_res = mxlogit_mod.fit(method = "bfgs", disp = 1)
#    print mxlogit_res.params
#    print mxlogit_res.llf

    # Summarize model
    # TODO means and standard errors for mixed logit parameters
    print(mxlogit_mod.summary())
