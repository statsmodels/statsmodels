import numpy as np
import numpy.random
from numpy.testing import assert_almost_equal, assert_equal

import statsmodels.stats.contrast as smc
from statsmodels.stats.contrast import Contrast


class TestContrast:
    @classmethod
    def setup_class(cls):
        rs = numpy.random.RandomState(54321)
        cls.X = rs.standard_normal((40, 10))

    def test_contrast1(self):
        term = np.column_stack((self.X[:, 0], self.X[:, 2]))
        c = Contrast(term, self.X)
        test_contrast = [[1] + [0] * 9, [0] * 2 + [1] + [0] * 7]
        assert_almost_equal(test_contrast, c.contrast_matrix)

    def test_contrast2(self):
        zero = np.zeros((40,))
        term = np.column_stack((zero, self.X[:, 2]))
        c = Contrast(term, self.X)
        test_contrast = [0] * 2 + [1] + [0] * 7
        assert_almost_equal(test_contrast, c.contrast_matrix)

    def test_contrast3(self):
        rs = np.random.RandomState(5432111)
        P = np.dot(self.X, np.linalg.pinv(self.X))
        resid = np.identity(40) - P
        noise = np.dot(resid, rs.standard_normal((40, 5)))
        term = np.column_stack((noise, self.X[:, 2]))
        c = Contrast(term, self.X)
        assert_equal(c.contrast_matrix.shape, (10,))

    # TODO: this should actually test the value of the contrast, not only its dimension

    def test_estimable(self):
        X2 = np.column_stack((self.X, self.X[:, 5]))
        Contrast(self.X[:, 5], X2)
        # TODO: I do not think this should be estimable?  isestimable correct?


def test_constraints():
    cm_ = np.eye(4, 3, k=-1)
    cpairs = np.array(
        [
            [+1.0, +0.0, 0.0],
            [+0.0, +1.0, 0.0],
            [+0.0, +0.0, 1.0],
            [-1.0, +1.0, 0.0],
            [-1.0, +0.0, 1.0],
            [+0.0, -1.0, 1.0],
        ]
    )
    c0 = smc._constraints_factor(cm_)
    assert_equal(c0, cpairs)

    c1 = smc._contrast_pairs(3, 4, 0)
    assert_equal(c1, cpairs)

    # embedded
    cpairs2 = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
        ]
    )

    c0 = smc._constraints_factor(cm_, k_params=6, idx_start=1)
    assert_equal(c0, cpairs2)

    c1 = smc._contrast_pairs(6, 4, 1)  # k_params, k_level, idx_start
    assert_equal(c1, cpairs2)

    # "pw" and "pairs" are documented aliases for "pairwise"
    assert_equal(smc._constraints_factor(cm_, comparison="pw"), cpairs)
    assert_equal(smc._constraints_factor(cm_, comparison="pairs"), cpairs)

    import pytest

    with pytest.raises(ValueError, match="comparison"):
        smc._constraints_factor(cm_, comparison="not-a-comparison")


def test_contrast_results_str_repr():
    import statsmodels.api as sm

    rs = numpy.random.RandomState(918273)
    n = 60
    exog = sm.add_constant(rs.standard_normal((n, 2)))
    endog = exog @ [1.0, 0.5, -0.5] + rs.standard_normal(n)
    res = sm.OLS(endog, exog).fit()

    tt = res.t_test(np.eye(3))
    text = str(tt)
    assert repr(tt) == str(type(tt)) + "\n" + text
    for val in tt.effect:
        assert f"{val:0.4f}"[:6] in text

    ft = res.f_test(np.eye(3)[1:])
    text_f = str(ft)
    assert repr(ft) == str(type(ft)) + "\n" + text_f
    assert "F test" in text_f


def test_wald_test_results_summary_frame_str_repr():
    import statsmodels.api as sm

    rs = numpy.random.RandomState(918274)
    n = 80
    exog = sm.add_constant(rs.standard_normal((n, 4)))
    endog = exog @ [1.0, 0.5, -0.5, 0.2, 0.1] + rs.standard_normal(n)
    res = sm.OLS(endog, exog).fit()

    wa = res.wald_test_terms(skip_single=False, scalar=True)
    assert isinstance(wa, smc.WaldTestResults)

    frame = wa.summary_frame()
    assert wa.dframe is frame
    assert wa.summary_frame() is frame  # cached, not recomputed

    text = str(wa)
    assert text == frame.to_string()
    assert repr(wa) == str(type(wa)) + "\n" + text
    for col in wa.col_names:
        assert col in frame.columns


def test_wald_test_results_direct_construction():
    import pytest
    from scipy import stats as sp_stats

    stat_chi2 = np.array([3.5, 8.2, 1.1])
    wa_chi2 = smc.WaldTestResults(stat_chi2, "chi2", dist_args=(4,))
    assert_almost_equal(wa_chi2.df_constraints, 4)
    assert_almost_equal(wa_chi2.pvalues, sp_stats.chi2.sf(stat_chi2, 4))

    stat_f = np.array([2.1, 5.4])
    wa_f = smc.WaldTestResults(stat_f, "F", dist_args=(3, 40))
    assert_almost_equal(wa_f.df_constraints, 3)
    assert_almost_equal(wa_f.df_denom, 40)
    assert_almost_equal(wa_f.pvalues, sp_stats.f.sf(stat_f, 3, 40))

    with pytest.raises(ValueError, match="only F and chi2"):
        smc.WaldTestResults(stat_chi2, "t", dist_args=(4,))
