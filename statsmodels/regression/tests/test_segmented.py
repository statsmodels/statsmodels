# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 21:32:33 2015

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from statsmodels.regression.linear_model import OLS
from statsmodels.base._segmented import Segmented, segmented


class _Bunch(object):
    """Storage class"""
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


class CheckSegmented(object):
    pass  # nothing has been moved up here yet


def compare_results_segmented(res, expected):
    # Note: 2 base columns: constant and linear x
    assert_equal(res.model.exog.shape[1] - 2, len(expected.knot_locations))
    assert_allclose(res.ssr, expected.ssr, atol=1e-3, rtol=1e-2)
    # TODO: ignore knot sequence for now
    assert_allclose(np.sort(res.knot_locations), np.sort(expected.knot_locations),
                    atol=1e-3, rtol=1e-2)


class TestSegmentedOLS(CheckSegmented):

    @classmethod
    def setup_class(cls):
        nobs = 500
        sig_e = 0.1

        np.random.seed(9999)

        x01 = np.sort(np.random.uniform(-1.99, 0.9, size=nobs))
        x0z = np.abs(np.exp(x01 + 0.5) % 2 - 1)
        y_true = x0z #exog0.dot(beta)
        y = y_true + sig_e * np.random.randn(nobs)
        cls.mod_base0 = OLS(y, np.ones(nobs))
        #res_oracle = OLS(y, exog0).fit()

        cls.x0 = x01
        cls.nobs = nobs
        # numbers regression test, visual inspection
        # TODO/Warning: knots are not sorted
        cls.result_expected3_it = _Bunch(ssr=5.286451,
                                          knot_locations=np.array([ 0.605425,
                                                            -0.428502,  0.205006]))
        cls.result_expected3_it1 = _Bunch(ssr=5.99222103,
                                         knot_locations=np.array([ 0.534805,
                                                -0.471663,  0.228336]))

    def test_seg3_brent(self):
        x0 = self.x0
        endog = self.mod_base0.endog
        nobs = self.nobs

        #res_fitted2 = segmented(mod_base2, 1, k_segments=1)

        q = np.percentile(x0, [25, 60, 85])

        mod_base2 = OLS(endog, np.column_stack((np.ones(nobs), x0,
                                                np.maximum(x0 - q[0], 0),
                                                np.maximum(x0 - q[1], 0),
                                                np.maximum(x0 - q[2], 0))))

        #res_base = mod_base2.fit()

        seg = Segmented(mod_base2, x0, [2, 3, 4])
        q = np.percentile(x0, [10, 25, 60, 85, 90])
        seg._fit_all(q, maxiter=1)
        res_fitted2 = seg.get_results()
        compare_results_segmented(res_fitted2, self.result_expected3_it1)

        seg = Segmented(mod_base2, x0, [2, 3, 4])
        q = np.percentile(x0, [10, 25, 60, 85, 90])
        seg._fit_all(q, maxiter=10)
        res_fitted_it2 = seg.get_results()
        # this converges to a bad local minimum
        # regression test
        result_expected3_bad = _Bunch(ssr=12.65665,
                                      knot_locations=np.array([ 0.534805,
                                                -0.471663,  0.228336]))
        #compare_results_segmented(res_fitted_it2, result_expected3_bad)
        compare_results_segmented(res_fitted_it2, self.result_expected3_it)


    def test_from_model3(self):
        x0 = self.x0

        seg = Segmented.from_model(self.mod_base0, x0, k_knots=3, degree=1)
        #q = np.percentile(x0, [10, 25, 60, 85, 90])
        assert self.mod_base0.exog.shape[1] == 1 # check we still have original
        seg_bounds0 = seg.bounds.copy()
        seg_exog_var0 = seg.exog.var(0)
        seg._fit_all(maxiter=10)
        res_fitted = seg.get_results()
        compare_results_segmented(res_fitted, self.result_expected3_it)


    def test_add3(self):
        # add knot from 2 to 3
        x0 = self.x0
        seg = Segmented.from_model(self.mod_base0, x0, k_knots=2, degree=1)
        seg._fit_all(maxiter=10)
        res_fitted = seg.get_results()
        seg_p1, r = seg.add_knot(maxiter=10)
        res_fitted_p1 = seg_p1.get_results()

        compare_results_segmented(res_fitted_p1, self.result_expected3_it)

    def test_add1_3(self):
        # add knots twice to get to three, start with one knot
        x0 = self.x0
        seg = Segmented.from_model(self.mod_base0, x0, k_knots=1, degree=1)
        seg._fit_all(maxiter=10)
        res_fitted = seg.get_results()
        seg_p1, r = seg.add_knot(maxiter=10)
        res_fitted_p1 = seg_p1.get_results()

        seg_p2, r = seg_p1.add_knot(maxiter=10)
        res_fitted_p2 = seg_p2.get_results()

        compare_results_segmented(res_fitted_p2, self.result_expected3_it)


class TestSegmentedOLSOracle(CheckSegmented):


    def test_oracle(cls):
        nobs = 500
        sig_e = 0.05

        bp_true = -0.5
        beta1 = 0.1
        beta_diff = 0.2
        for bp_true in [-0.5, 0, 0.5]:
            for beta_diff in [0.4, 0.5, 1]:

                beta = [1, beta1, beta1 + beta_diff]

                np.random.seed(9999)
                x0 = np.sort(np.random.uniform(-2, 2, size=nobs))
                exog1 = np.column_stack((np.ones(nobs), x0,
                                         np.maximum(x0 - bp_true, 0)))
                y_true = exog1.dot(beta)
                y = y_true + sig_e * np.random.randn(nobs)
                mod_base0 = OLS(y, np.ones(nobs))
                res_oracle = OLS(y, exog1).fit()

                seg = Segmented.from_model(mod_base0, x0, k_knots=1, degree=1)
                seg._fit_all(maxiter=10)
                res_fitted = seg.get_results()
                assert_allclose(res_fitted.ssr, res_oracle.ssr, rtol=1e-2)
                # knot location difference looks large ?
                assert_allclose(res_fitted.knot_locations, bp_true,
                                rtol=1e-3, atol=0.2)

                assert_allclose(res_fitted.params, res_oracle.params, rtol=0.1)
                assert_allclose(res_fitted.bse, res_oracle.bse, rtol=0.03)
                # pvalues are tiny, not informative
                assert_allclose(res_fitted.pvalues, res_oracle.pvalues,
                                rtol=1e-3, atol=1e-15)

                # test original function
                mod_base1 = OLS(y, exog1.copy())
                mod_base1.exog[:, -1] += 0.1   # mess up the column we estimate
                res_best = segmented(mod_base1, 1)

                assert_allclose(res_best.ssr, res_oracle.ssr, rtol=1e-2)
                # knot location difference looks large ?
                assert_allclose(res_best.knot_location, bp_true,
                                rtol=1e-3, atol=0.2)

                assert_allclose(res_best.params, res_oracle.params, rtol=0.1)
                assert_allclose(res_fitted.bse, res_oracle.bse, rtol=0.03)
                # pvalues are tiny, not informative
                assert_allclose(res_best.pvalues, res_oracle.pvalues,
                                rtol=1e-3, atol=1e-15)

                # test source target indices
                exog10 = np.column_stack((np.ones(nobs),
                                          np.maximum(x0 - bp_true, 0),
                                          x0))
                mod_base1 = OLS(y, exog10)
                mod_base1.exog[:, 1] += 0.1   # mess up the column we estimate
                res_best2 = segmented(mod_base1, 2, 1)
                assert_allclose(res_best2.ssr, res_best.ssr, rtol=1e-8)
