#!/bin/env python
''' Test results of OLS against R '''

import numpy as N

from rpy import r

import neuroimaging.fixes.scipy.stats.models as SSM

from rmodelwrap import RModel

from exampledata import y, x

def assert_model_similar(res1, res2):
    ''' Test if models have similar parameters '''
    assert N.allclose(res1.beta, res2.beta)
    assert N.allclose(res1.resid, res2.resid)
    assert N.allclose(res1.predict, res2.predict)
    assert N.allclose(res1.df_resid, res2.df_resid)

def test_model_class(model_class, r_model_type):
    results = model_class(x).fit(y)
    r_results = WrappedRModel(y, x, r_model_type)
    r_results.assert_similar(results)

# Test OLS
ols_res = SSM.regression.ols_model(x).fit(y)
rlm_res = RModel(y, x, r.lm)
assert_model_similar(ols_res, rlm_res)
# Standard GLM
glm_res = SSM.glm(x).fit(y)
assert_model_similar(glm_res, rlm_res)
# Which should be same as each other
assert_model_similar(ols_res, glm_res)
