from numpy.testing import assert_
from statsmodels.base.model import (_fit_mle_newton, _fit_mle_nm,
                                    _fit_mle_bfgs, _fit_mle_cg,
                                    _fit_mle_ncg, _fit_mle_powell)

fit_funcs = {
    'newton': _fit_mle_newton,
    'nm': _fit_mle_nm,  # Nelder-Mead
    'bfgs': _fit_mle_bfgs,
    'cg': _fit_mle_cg,
    'ncg': _fit_mle_ncg,
    'powell': _fit_mle_powell
            }


def dummy_func(x):
    return x**2

def dummy_score(x):
    return 2*x

def dummy_hess(x):
    return [[2]]

def test_full_output_false():
    # just a smoke test

    # newton needs f, score, start, fargs, kwargs
    # bfgs needs f, score start, fargs, kwargs
    # nm needs ""
    # cg ""
    # ncg ""
    # powell ""
    for method in fit_funcs:
        func = fit_funcs[method]
        if method == "newton":
            xopts, retvals = func(dummy_func, dummy_score, [1], (), {},
                    hess=dummy_hess, full_output=False, disp=0)

        else:
            xopts, retvals = func(dummy_func, dummy_score, [1], (), {},
                full_output=False, disp=0)
        assert_(xopts == None)
        if method == "powell":
            #NOTE: I think I reported this? Might be version/optimize API
            # dependent
            assert_(retvals.shape == () and retvals.size == 1)
        else:
            assert_(len(retvals)==1)
