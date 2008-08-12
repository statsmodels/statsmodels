''' Test results of OLS against R '''

from numpy.testing import *

import numpy as np

import neuroimaging.fixes.scipy.stats.models as SSM

from exampledata import y, x

def assert_model_similar(res1, res2):
    ''' Test if models have similar parameters '''
    assert np.allclose(res1.beta, res2.beta)
    assert np.allclose(res1.resid, res2.resid)
    assert np.allclose(res1.predict, res2.predict)
    assert np.allclose(res1.df_resid, res2.df_resid)

# FIXME: TypeError: test_model_class() takes exactly 2 arguments (0 given)
@dec.skipknownfailure
def test_model_class(model_class, r_model_type):
    results = model_class(x).fit(y)
    r_results = WrappedRModel(y, x, r_model_type)
    r_results.assert_similar(results)

# FIXME: ImportError: No module named rpy
@dec.skipknownfailure
def test_using_rpy():
    from rpy import r
    from rmodelwrap import RModel
    # Test OLS
    ols_res = SSM.regression.ols_model(x).fit(y)
    rlm_res = RModel(y, x, r.lm)
    assert_model_similar(ols_res, rlm_res)

# FIXME: NameError: global name 'rlm_res' is not defined
@dec.skipknownfailure
def test1():
    # Standard GLM
    glm_res = SSM.glm(x).fit(y)
    assert_model_similar(glm_res, rlm_res)
    # Which should be same as each other
    assert_model_similar(ols_res, glm_res)
