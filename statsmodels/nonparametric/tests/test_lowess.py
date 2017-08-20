'''
Lowess testing suite.

Expected outcomes are generated by R's lowess function given the same
arguments. The R script test_lowess_r_outputs.R can be used to
generate the expected outcomes.

The delta tests utilize Silverman's motorcycle collision data,
available in R's MASS package.
'''

import os
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_raises,
                           assert_equal)
#import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

# Number of decimals to test equality with.
# The default is 7.
curdir = os.path.dirname(os.path.abspath(__file__))
rpath = os.path.join(curdir, 'results')


class TestLowess(object):

    def test_import(self):
        #this doesn't work
        #from statsmodels.api.nonparametric import lowess as lowess1
        import statsmodels.api as sm
        lowess1 = sm.nonparametric.lowess
        assert_(lowess is lowess1)

    def test_flat(self):
        test_data = {
            'x': np.arange(20), 'y': np.zeros(20), 'out': np.zeros(20)}
        expected_lowess = np.array([test_data['x'], test_data['out']]).T
        actual_lowess = lowess(test_data['y'], test_data['x'])
        assert_almost_equal(expected_lowess, actual_lowess, 7)

    def test_range(self):
        test_data = {
            'x': np.arange(20), 'y': np.arange(20), 'out': np.arange(20)}
        expected_lowess = np.array([test_data['x'], test_data['out']]).T
        actual_lowess = lowess(test_data['y'], test_data['x'])
        assert_almost_equal(expected_lowess, actual_lowess, 7)

    def test_all(self):
        def generate(name, fname,
                     x='x', y='y', out='out', kwargs={}, decimal=7):
            data = np.genfromtxt(
                os.path.join(rpath, fname), delimiter=',', names=True)
            assert_almost_equal.description = name
            if callable(kwargs):
                kwargs = kwargs(data)
            result = lowess(data[y], data[x], **kwargs)
            expect = np.array([data[x], data[out]]).T
            return assert_almost_equal, result, expect, decimal

        yield generate('test_simple', 'test_lowess_simple.csv')
        yield generate('test_iter_0', 'test_lowess_iter.csv', out='out_0',
                       kwargs={'it': 0})
        yield generate('test_iter_0', 'test_lowess_iter.csv', out='out_3',
                       kwargs={'it': 3})
        yield generate('test_frac_2_3', 'test_lowess_frac.csv', out='out_2_3',
                       kwargs={'frac': 2. / 3})
        yield generate('test_frac_1_5', 'test_lowess_frac.csv', out='out_1_5',
                       kwargs={'frac': 1. / 5})
        yield generate('test_delta_0', 'test_lowess_delta.csv', out='out_0',
                       kwargs={'frac': 0.1})
        yield generate('test_delta_Rdef', 'test_lowess_delta.csv', out='out_Rdef',
                       kwargs=lambda data: {'frac': .1,
                                            'delta': .01 * data['x'].ptp()})
        yield generate('test_delta_1', 'test_lowess_delta.csv', out='out_1',
                       kwargs={'frac': 0.1, 'delta': 1 + 1e-10}, decimal=10)

    def test_options(self):
        rfile = os.path.join(rpath, 'test_lowess_simple.csv')
        test_data = np.genfromtxt(open(rfile, 'rb'),
                                  delimiter = ',', names = True)
        y, x = test_data['y'], test_data['x']
        res1_fitted = test_data['out']
        expected_lowess = np.array([test_data['x'], test_data['out']]).T

        # check skip sorting
        actual_lowess1 = lowess(y, x, is_sorted=True)
        assert_almost_equal(actual_lowess1, expected_lowess, decimal=13)

        # check skip missing
        actual_lowess = lowess(y, x, is_sorted=True, missing='none')
        assert_almost_equal(actual_lowess, actual_lowess1, decimal=13)

        # check order/index, returns yfitted only
        actual_lowess = lowess(y[::-1], x[::-1], return_sorted=False)
        assert_almost_equal(actual_lowess, actual_lowess1[::-1, 1], decimal=13)

        # check returns yfitted only
        actual_lowess = lowess(y, x, return_sorted=False, missing='none',
                               is_sorted=True)
        assert_almost_equal(actual_lowess, actual_lowess1[:, 1], decimal=13)

        # check integer input
        actual_lowess = lowess(np.round(y).astype(int), x, is_sorted=True)
        actual_lowess1 = lowess(np.round(y), x, is_sorted=True)
        assert_almost_equal(actual_lowess, actual_lowess1, decimal=13)
        assert_(actual_lowess.dtype is np.dtype(float))
        # this will also have duplicate x
        actual_lowess = lowess(y, np.round(x).astype(int), is_sorted=True)
        actual_lowess1 = lowess(y, np.round(x), is_sorted=True)
        assert_almost_equal(actual_lowess, actual_lowess1, decimal=13)
        assert_(actual_lowess.dtype is np.dtype(float))

        # check with nans,  this changes the arrays
        y[[5, 6]] = np.nan
        x[3] = np.nan
        mask_valid = np.isfinite(x) & np.isfinite(y)
        #actual_lowess1[[3, 5, 6], 1] = np.nan
        actual_lowess = lowess(y, x, is_sorted=True)
        actual_lowess1 = lowess(y[mask_valid], x[mask_valid], is_sorted=True)
        assert_almost_equal(actual_lowess, actual_lowess1, decimal=13)
        assert_raises(ValueError, lowess, y, x, missing='raise')

        perm_idx = np.arange(len(x))
        np.random.shuffle(perm_idx)
        yperm = y[perm_idx]
        xperm = x[perm_idx]
        actual_lowess2 = lowess(yperm, xperm, is_sorted=False)
        assert_almost_equal(actual_lowess, actual_lowess2, decimal=13)

        actual_lowess3 = lowess(yperm, xperm, is_sorted=False,
                                return_sorted=False)
        mask_valid = np.isfinite(xperm) & np.isfinite(yperm)
        assert_equal(np.isnan(actual_lowess3), ~mask_valid)
        # get valid sorted smoothed y from actual_lowess3
        sort_idx = np.argsort(xperm)
        yhat = actual_lowess3[sort_idx]
        yhat = yhat[np.isfinite(yhat)]
        assert_almost_equal(yhat, actual_lowess2[:,1], decimal=13)


def test_returns_inputs():
    # see 1960
    y = [0] * 10 + [1] * 10
    x = np.arange(20)
    result = lowess(y, x, frac=.4)
    assert_almost_equal(result, np.column_stack((x, y)))

if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-vvs', '-x', '--pdb'])
