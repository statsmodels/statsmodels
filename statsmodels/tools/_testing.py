"""Testing helper functions

Warning: current status experimental, mostly copy paste

Warning: these functions will be changed without warning as the need
during refactoring arises.

The first group of functions provide consistency checks

"""

import numpy as np
from numpy.testing import assert_allclose, assert_
from nose import SkipTest

# the following are copied from
# statsmodels.base.tests.test_generic_methods.CheckGenericMixin
# and only adjusted to work as standalone functions

def check_ttest_tvalues(results):
    # test that t_test has same results a params, bse, tvalues, ...
    res = results
    mat = np.eye(len(res.params))
    tt = res.t_test(mat)

    assert_allclose(tt.effect, res.params, rtol=1e-12)
    # TODO: tt.sd and tt.tvalue are 2d also for single regressor, squeeze
    assert_allclose(np.squeeze(tt.sd), res.bse, rtol=1e-10)
    assert_allclose(np.squeeze(tt.tvalue), res.tvalues, rtol=1e-12)
    assert_allclose(tt.pvalue, res.pvalues, rtol=5e-10)
    assert_allclose(tt.conf_int(), res.conf_int(), rtol=1e-10)

    # test params table frame returned by t_test
    table_res = np.column_stack((res.params, res.bse, res.tvalues,
                                res.pvalues, res.conf_int()))
    table1 = np.column_stack((tt.effect, tt.sd, tt.tvalue, tt.pvalue,
                             tt.conf_int()))
    table2 = tt.summary_frame().values
    assert_allclose(table2, table_res, rtol=1e-12)

    # move this to test_attributes ?
    assert_(hasattr(res, 'use_t'))

    tt = res.t_test(mat[0])
    tt.summary()   # smoke test for #1323
    assert_allclose(tt.pvalue, res.pvalues[0], rtol=5e-10)


def check_ftest_pvalues(results):
    res = results
    use_t = res.use_t
    k_vars = len(res.params)
    # check default use_t
    pvals = [res.wald_test(np.eye(k_vars)[k], use_f=use_t).pvalue
                                               for k in range(k_vars)]
    assert_allclose(pvals, res.pvalues, rtol=5e-10, atol=1e-25)

    # sutomatic use_f based on results class use_t
    pvals = [res.wald_test(np.eye(k_vars)[k]).pvalue
                                               for k in range(k_vars)]
    assert_allclose(pvals, res.pvalues, rtol=5e-10, atol=1e-25)

    # label for pvalues in summary
    string_use_t = 'P>|z|' if use_t is False else 'P>|t|'
    summ = str(res.summary())
    assert_(string_use_t in summ)

    # try except for models that don't have summary2
    try:
        summ2 = str(res.summary2())
    except AttributeError:
        summ2 = None
    if summ2 is not None:
        assert_(string_use_t in summ2)


# TODO The following is not (yet) guaranteed across models
#@knownfailureif(True)
def check_fitted(results):
    # ignore wrapper for isinstance check
    from statsmodels.genmod.generalized_linear_model import GLMResults
    from statsmodels.discrete.discrete_model import DiscreteResults
    # FIXME: work around GEE has no wrapper
    if hasattr(results, '_results'):
        results = results._results
    else:
        results = results
    if (isinstance(results, GLMResults) or
        isinstance(results, DiscreteResults)):
        raise SkipTest

    res = results
    fitted = res.fittedvalues
    assert_allclose(res.model.endog - fitted, res.resid, rtol=1e-12)
    assert_allclose(fitted, res.predict(), rtol=1e-12)

def check_predict_types(results):
    res = results
    # squeeze to make 1d for single regressor test case
    p_exog = np.squeeze(np.asarray(res.model.exog[:2]))

    # ignore wrapper for isinstance check
    from statsmodels.genmod.generalized_linear_model import GLMResults
    from statsmodels.discrete.discrete_model import DiscreteResults

    # FIXME: work around GEE has no wrapper
    if hasattr(results, '_results'):
        results = results._results
    else:
        results = results

    if (isinstance(results, GLMResults) or
        isinstance(results, DiscreteResults)):
        # SMOKE test only  TODO
        res.predict(p_exog)
        res.predict(p_exog.tolist())
        res.predict(p_exog[0].tolist())
    else:
        fitted = res.fittedvalues[:2]
        assert_allclose(fitted, res.predict(p_exog), rtol=1e-12)
        # this needs reshape to column-vector:
        assert_allclose(fitted, res.predict(np.squeeze(p_exog).tolist()),
                        rtol=1e-12)
        # only one prediction:
        assert_allclose(fitted[:1], res.predict(p_exog[0].tolist()),
                        rtol=1e-12)
        assert_allclose(fitted[:1], res.predict(p_exog[0]),
                        rtol=1e-12)

        # predict doesn't preserve DataFrame, e.g. dot converts to ndarray
        #import pandas
        #predicted = res.predict(pandas.DataFrame(p_exog))
        #assert_(isinstance(predicted, pandas.DataFrame))
        #assert_allclose(predicted, fitted, rtol=1e-12)

