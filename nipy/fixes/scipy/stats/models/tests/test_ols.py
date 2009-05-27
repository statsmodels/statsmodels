''' Test results of OLS against R '''


import numpy as np
import numpy.testing as nptest
import nose.tools

import nipy.fixes.scipy.stats.models as SSM
#from nipy.fixes.scipy.stats.models.formula import Term, I


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

def test_longley():
    '''
    Test OLS accuracy with Longley (1967) data
    '''

    from exampledata import longley
    y,x = longley()
    nist_long = (-3482258.63459582,15.0618722713733,-0.358191792925910E-01,
                 -2.02022980381683, -1.03322686717359, -0.511041056535807E-01,
                 1829.15146461355)
    nist_long_bse=(890420.383607373,84.9149257747669,0.334910077722432E-01,
                   0.488399681651699,0.214274163161675,0.226073200069370,
                   455.478499142212)
    x = np.hstack((np.ones((len(x), 1)), x))  # A constant is not added by default
    res = SSM.regression.OLSModel(x).fit(y)
    nptest.assert_almost_equal(res.beta, nist_long, 4)
    nptest.assert_almost_equal(np.diag(np.sqrt(res.cov_beta())),nist_long_bse)
    nptest.assert_almost_equal(res.scale, 92936.0061673238, 6)
    nptest.assert_almost_equal(res.Rsq(), 0.995479004577296, 12)

def test_wampler():
    nist_wamp1=(1.00000000000000,1.00000000000000,1.00000000000000,
                1.00000000000000,1.00000000000000,1.00000000000000)
    x=np.arange(21,dtype=float)[:,np.newaxis]
    p=np.poly1d([1,1,1,1,1,1])
    y=np.polyval(p,x).reshape(len(x))
    x=np.hstack((np.ones((len(x), 1)), x, x**2, x**3, x**4, x**5))
    res = SSM.regression.OLSModel(x).fit(y)
    nptest.assert_almost_equal(res.beta,nist_wamp1)
##  include next three wampler sets

##  the precision appears to be machine specific,
##      so default to 4 decimal places
##  check scipy test suite



if __name__=="__main__":
    nptest.run_module_suite()





