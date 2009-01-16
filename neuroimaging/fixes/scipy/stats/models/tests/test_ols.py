''' Test results of OLS against R '''


import numpy as np
import numpy.testing as nptest
import nose.tools

import neuroimaging.fixes.scipy.stats.models as SSM

from exampledata import y, x

def assert_model_similar(res1, res2):
    ''' Test if models have similar parameters '''
    nptest.assert_almost_equal(res1.beta, res2.beta, 4)
    nptest.assert_almost_equal(res1.resid, res2.resid, 4)
    nptest.assert_almost_equal(res1.predict, res2.predict, 4)
    nptest.assert_almost_equal(res1.df_resid, res2.df_resid, 4)

def check_model_class(model_class, r_model_type):
    results = model_class(x).fit(y)
    r_results = WrappedRModel(y, x, r_model_type)
    r_results.assert_similar(results)

@nptest.dec.knownfailureif(True)
def test_using_rpy():
    """
    this test fails because the glm results don't agree with the ols and rlm
    results
    """
    try:
        from rpy import r
        from rmodelwrap import RModel

        # Test OLS
        ols_res = SSM.regression.OLSModel(x).fit(y)
        rlm_res = RModel(y, x, r.lm)
        yield assert_model_similar, ols_res, rlm_res

        glm_res = SSM.glm(x).fit(y)
        yield assert_model_similar, glm_res, rlm_res
    except ImportError:
        yield nose.tools.assert_true, True
