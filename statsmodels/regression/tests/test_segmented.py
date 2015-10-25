# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 21:32:33 2015

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from statsmodels.regression.linear_model import OLS
from statsmodels.base._segmented import Segmented


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
        compare_results_segmented(res_fitted_it2, self.result_expected3_it)


        seg_p1, r = seg.add_knot(maxiter=3)
        res_fitted_p1 = seg_p1.get_results()


    def test_from_model3(self):
        x0 = self.x0

        seg = Segmented.from_model(self.mod_base0, x0, k_knots=3, degree=1)
        #q = np.percentile(x0, [10, 25, 60, 85, 90])
        seg_bounds0 = seg.bounds.copy()
        seg_exog_var0 = seg.exog.var(0)
        seg._fit_all(maxiter=10)
        res_fitted = seg.get_results()
        compare_results_segmented(res_fitted, self.result_expected3_it)

    def test_add3(self):
        x0 = self.x0
        seg = Segmented.from_model(self.mod_base0, x0, k_knots=2, degree=1)
        seg._fit_all(maxiter=10)
        res_fitted = seg.get_results()
        seg_p1, r = seg.add_knot(maxiter=10)
        res_fitted_p1 = seg_p1.get_results()

        compare_results_segmented(res_fitted_p1, self.result_expected3_it)
