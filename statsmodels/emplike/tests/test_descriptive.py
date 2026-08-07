import warnings

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from statsmodels.datasets import star98
from statsmodels.emplike.descriptive import DescStat, EmpLikeTestResult

from .results.el_results import DescStatRes


class GenRes:
    """
    Reads in the data and creates class instance to be tested
    """

    @classmethod
    def setup_class(cls):
        data = star98.load()
        data.exog = np.asarray(data.exog)
        desc_stat_data = data.exog[:50, 5]
        mv_desc_stat_data = data.exog[:50, 5:7]  # mv = multivariate
        cls.res1 = DescStat(desc_stat_data)
        cls.res2 = DescStatRes()
        cls.mvres1 = DescStat(mv_desc_stat_data)


class TestDescriptiveStatistics(GenRes):
    @classmethod
    def setup_class(cls):
        super().setup_class()

    def test_test_mean(self):
        res = self.res1.test_mean(14, use_namedtuple=True)
        assert_almost_equal(res[:2], self.res2.test_mean_14, 4)

    def test_test_mean_weights(self):
        assert_almost_equal(
            self.res1.test_mean(14, return_weights=1)[2], self.res2.test_mean_weights, 4
        )

    def test_ci_mean(self):
        assert_almost_equal(self.res1.ci_mean(), self.res2.ci_mean, 4)

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_test_var(self):
        res = self.res1.test_var(3, use_namedtuple=True)
        assert_almost_equal(res[:2], self.res2.test_var_3, 4)

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_test_var_weights(self):
        assert_almost_equal(
            self.res1.test_var(3, return_weights=1)[2], self.res2.test_var_weights, 4
        )

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_ci_var(self):
        assert_almost_equal(self.res1.ci_var(), self.res2.ci_var, 4)

    def test_mv_test_mean(self):
        assert_almost_equal(
            self.mvres1.mv_test_mean(np.array([14, 56]), use_namedtuple=True)[:2],
            self.res2.mv_test_mean,
            4,
        )

    def test_mv_test_mean_weights(self):
        assert_almost_equal(
            self.mvres1.mv_test_mean(np.array([14, 56]), return_weights=1)[2],
            self.res2.mv_test_mean_wts,
            4,
        )

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_test_skew(self):
        res = self.res1.test_skew(0, use_namedtuple=True)
        assert_almost_equal(res[:2], self.res2.test_skew, 4)

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_ci_skew(self):
        # This will be tested in a round about way since MATLAB fails when
        # computing CI with multiple nuisance parameters.  The process is:
        #
        # (1) Get CI for skewness from ci.skew()
        # (2) In MATLAB test the hypotheis that skew=results of test_skew.
        # (3) If p-value approx .05, test confirmed
        skew_ci = self.res1.ci_skew()
        lower_lim = skew_ci[0]
        upper_lim = skew_ci[1]
        ul_pval = self.res1.test_skew(lower_lim, use_namedtuple=True).pvalue
        ll_pval = self.res1.test_skew(upper_lim, use_namedtuple=True).pvalue
        assert_almost_equal(ul_pval, 0.050000, 4)
        assert_almost_equal(ll_pval, 0.050000, 4)

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_ci_skew_weights(self):
        assert_almost_equal(
            self.res1.test_skew(0, return_weights=1)[2], self.res2.test_skew_wts, 4
        )

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_test_kurt(self):
        res = self.res1.test_kurt(0, use_namedtuple=True)
        assert_almost_equal(res[:2], self.res2.test_kurt_0, 4)

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_ci_kurt(self):
        # Same strategy for skewness CI
        kurt_ci = self.res1.ci_kurt(upper_bound=0.5, lower_bound=-1.5)
        lower_lim = kurt_ci[0]
        upper_lim = kurt_ci[1]
        ul_pval = self.res1.test_kurt(upper_lim, use_namedtuple=True).pvalue
        ll_pval = self.res1.test_kurt(lower_lim, use_namedtuple=True).pvalue
        assert_almost_equal(ul_pval, 0.050000, 4)
        assert_almost_equal(ll_pval, 0.050000, 4)

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_joint_skew_kurt(self):
        assert_almost_equal(
            self.res1.test_joint_skew_kurt(0, 0, use_namedtuple=True)[:2],
            self.res2.test_joint_skew_kurt,
            4,
        )

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_test_corr(self):
        res = self.mvres1.test_corr(0.5, use_namedtuple=True)
        assert_almost_equal(res[:2], self.res2.test_corr, 4)

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_ci_corr(self):
        corr_ci = self.mvres1.ci_corr()
        lower_lim = corr_ci[0]
        upper_lim = corr_ci[1]
        ul_pval = self.mvres1.test_corr(upper_lim, use_namedtuple=True).pvalue
        ll_pval = self.mvres1.test_corr(lower_lim, use_namedtuple=True).pvalue
        assert_almost_equal(ul_pval, 0.050000, 4)
        assert_almost_equal(ll_pval, 0.050000, 4)

    @pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
    def test_test_corr_weights(self):
        assert_almost_equal(
            self.mvres1.test_corr(0.5, return_weights=1)[2],
            self.res2.test_corr_weights,
            4,
        )

    @pytest.mark.parametrize(
        "endog",
        [
            np.array([]),
            np.zeros((2, 2, 2)),
        ],
    )
    def test_descstat_invalid_input(self, endog):
        with pytest.raises(ValueError):
            DescStat(endog)


@pytest.mark.parametrize(
    ("attr", "args"),
    [
        ("test_mean", (14,)),
        ("test_var", (3,)),
        ("test_skew", (0,)),
        ("test_kurt", (0,)),
        ("test_joint_skew_kurt", (0, 0)),
    ],
)
@pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
def test_univariate_namedtuple(attr, args):
    data = np.asarray(star98.load().exog)[:50, 5]
    meth = getattr(DescStat(data), attr)

    # return_weights=True already yields three values, so the NamedTuple is
    # adopted silently
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        full = meth(*args, return_weights=True)
    assert isinstance(full, EmpLikeTestResult)
    assert len(full) == 3

    # the two-value path still warns and still returns a plain tuple
    with pytest.warns(FutureWarning, match=attr):
        legacy = meth(*args)
    assert not isinstance(legacy, EmpLikeTestResult)
    assert len(legacy) == 2
    assert_almost_equal(legacy, full[:2], 10)

    # opting in or out is silent either way
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        opted_in = meth(*args, use_namedtuple=True)
        opted_out = meth(*args, use_namedtuple=False)
    assert isinstance(opted_in, EmpLikeTestResult)
    assert opted_in.weights is not None
    assert len(opted_out) == 2


@pytest.mark.parametrize(
    ("attr", "args"),
    [
        ("mv_test_mean", (np.array([14, 56]),)),
        ("test_corr", (0.5,)),
    ],
)
@pytest.mark.thread_unsafe("calculation sets attributes and is not thread safe")
def test_multivariate_namedtuple(attr, args):
    data = np.asarray(star98.load().exog)[:50, 5:7]
    meth = getattr(DescStat(data), attr)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        full = meth(*args, return_weights=True)
    assert isinstance(full, EmpLikeTestResult)
    assert len(full) == 3

    with pytest.warns(FutureWarning, match=attr):
        legacy = meth(*args)
    assert len(legacy) == 2
    assert_almost_equal(legacy, full[:2], 10)
