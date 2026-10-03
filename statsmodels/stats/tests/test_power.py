# pylint: disable=W0231, W0142
"""Tests for statistical power calculations

Note:
    tests for chisquare power are in test_gof.py

Created on Sat Mar 09 08:44:49 2013

Author: Josef Perktold
"""

from statsmodels.compat.platform import PLATFORM_OSX, PLATFORM_WIN

import copy
import warnings

import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
)
import pytest
from scipy import integrate, stats

import statsmodels.stats.power as smp
from statsmodels.stats.tests.test_weightstats import Holder
from statsmodels.tools.sm_exceptions import HypothesisTestWarning

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


class CheckPowerMixin:

    def test_power(self):
        # test against R results
        kwds = copy.copy(self.kwds)
        del kwds["power"]
        kwds.update(self.kwds_extra)
        if hasattr(self, "decimal"):
            decimal = self.decimal
        else:
            decimal = 6
        res1 = self.cls()
        assert_almost_equal(res1.power(**kwds), self.res2.power, decimal=decimal)

    # @pytest.mark.xfail(strict=True)
    def test_positional(self):

        res1 = self.cls()

        kwds = copy.copy(self.kwds)
        del kwds["power"]
        kwds.update(self.kwds_extra)

        # positional args
        if hasattr(self, "args_names"):
            args_names = self.args_names
        else:
            nobs_ = "nobs" if "nobs" in kwds else "nobs1"
            args_names = ["effect_size", nobs_, "alpha"]

        # pop positional args
        args = [kwds.pop(arg) for arg in args_names]

        if hasattr(self, "decimal"):
            decimal = self.decimal
        else:
            decimal = 6

        res = res1.power(*args, **kwds)
        assert_almost_equal(res, self.res2.power, decimal=decimal)

    def test_roots(self):
        kwds = copy.copy(self.kwds)
        kwds.update(self.kwds_extra)
        # kwds_extra are used as argument, but not as target for root
        for key in self.kwds:
            if key == "alpha":
                if (PLATFORM_WIN and isinstance(self, TestTTPowerOneS1)) or (
                    PLATFORM_OSX
                    and isinstance(self, (TestTTPowerOneS1, TestTTPowerTwoS1))
                ):
                    pytest.xfail(
                        f"alpha test failing test {self.__class__} for key {key} on "
                        f"recent SciPy on Windows and Darwin"
                    )

            value = kwds[key]
            kwds[key] = None
            try:
                result = self.cls().solve_power(**kwds)
            except Exception as exc:
                msg = (
                    f"class: {self.__class__.__name__}, key: {key}, "
                    f"value: {value}, kwds: {kwds}"
                )
                raise AssertionError(msg) from exc
            assert_allclose(result, value, rtol=0.001, err_msg=key + " failed")
            # yield can be used to investigate specific errors
            # yield assert_allclose, result, value, 0.001, 0, key+' failed'
            kwds[key] = value  # reset dict

    @pytest.mark.thread_unsafe(reason="Uses matplotlib")
    @pytest.mark.matplotlib
    def test_power_plot(self, close_figures):
        if self.cls in [smp.FTestPower, smp.FTestPowerF2]:
            pytest.skip("skip FTestPower plot_power")
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        fig = self.cls().plot_power(
            dep_var="nobs",
            nobs=np.arange(2, 100),
            effect_size=np.array([0.1, 0.2, 0.3, 0.5, 1]),
            # alternative='larger',
            ax=ax,
            title="Power of t-Test",
            **self.kwds_extra,
        )
        ax = fig.add_subplot(2, 1, 2)
        self.cls().plot_power(
            dep_var="es",
            nobs=np.array([10, 20, 30, 50, 70, 100]),
            effect_size=np.linspace(0.01, 2, 51),
            # alternative='larger',
            ax=ax,
            title="",
            **self.kwds_extra,
        )


# ''' test cases
# one sample
#               two-sided one-sided
# large power     OneS1      OneS3
# small power     OneS2      OneS4
#
# two sample
#               two-sided one-sided
# large power     TwoS1       TwoS3
# small power     TwoS2       TwoS4
# small p, ratio  TwoS4       TwoS5
# '''


class TestTTPowerOneS1(CheckPowerMixin):

    @classmethod
    def setup_class(cls):

        # > p = pwr.t.test(d=1,n=30,sig.level=0.05,type="two.sample",alternative="two.sided")
        # > cat_items(p, prefix='tt_power2_1.')
        res2 = Holder()
        res2.n = 30
        res2.d = 1
        res2.sig_level = 0.05
        res2.power = 0.9995636009612725
        res2.alternative = "two.sided"
        res2.note = "NULL"
        res2.method = "One-sample t test power calculation"

        cls.res2 = res2
        cls.kwds = {
            "effect_size": res2.d,
            "nobs": res2.n,
            "alpha": res2.sig_level,
            "power": res2.power,
        }
        cls.kwds_extra = {}
        cls.cls = smp.TTestPower


class TestTTPowerOneS2(CheckPowerMixin):
    # case with small power

    @classmethod
    def setup_class(cls):

        res2 = Holder()
        # > p = pwr.t.test(d=0.2,n=20,sig.level=0.05,type="one.sample",alternative="two.sided")
        # > cat_items(p, "res2.")
        res2.n = 20
        res2.d = 0.2
        res2.sig_level = 0.05
        res2.power = 0.1359562887679666
        res2.alternative = "two.sided"
        res2.note = """NULL"""
        res2.method = "One-sample t test power calculation"

        cls.res2 = res2
        cls.kwds = {
            "effect_size": res2.d,
            "nobs": res2.n,
            "alpha": res2.sig_level,
            "power": res2.power,
        }
        cls.kwds_extra = {}
        cls.cls = smp.TTestPower


class TestTTPowerOneS3(CheckPowerMixin):

    @classmethod
    def setup_class(cls):

        res2 = Holder()
        # > p = pwr.t.test(d=1,n=30,sig.level=0.05,type="one.sample",alternative="greater")
        # > cat_items(p, prefix='tt_power1_1g.')
        res2.n = 30
        res2.d = 1
        res2.sig_level = 0.05
        res2.power = 0.999892010204909
        res2.alternative = "greater"
        res2.note = "NULL"
        res2.method = "One-sample t test power calculation"

        cls.res2 = res2
        cls.kwds = {
            "effect_size": res2.d,
            "nobs": res2.n,
            "alpha": res2.sig_level,
            "power": res2.power,
        }
        cls.kwds_extra = {"alternative": "larger"}
        cls.cls = smp.TTestPower


class TestTTPowerOneS4(CheckPowerMixin):

    @classmethod
    def setup_class(cls):

        res2 = Holder()
        # > p = pwr.t.test(d=0.05,n=20,sig.level=0.05,type="one.sample",alternative="greater")
        # > cat_items(p, "res2.")
        res2.n = 20
        res2.d = 0.05
        res2.sig_level = 0.05
        res2.power = 0.0764888785042198
        res2.alternative = "greater"
        res2.note = """NULL"""
        res2.method = "One-sample t test power calculation"

        cls.res2 = res2
        cls.kwds = {
            "effect_size": res2.d,
            "nobs": res2.n,
            "alpha": res2.sig_level,
            "power": res2.power,
        }
        cls.kwds_extra = {"alternative": "larger"}
        cls.cls = smp.TTestPower


class TestTTPowerOneS5(CheckPowerMixin):
    # case one-sided less, not implemented yet

    @classmethod
    def setup_class(cls):

        res2 = Holder()
        # > p = pwr.t.test(d=0.2,n=20,sig.level=0.05,type="one.sample",alternative="less")
        # > cat_items(p, "res2.")
        res2.n = 20
        res2.d = 0.2
        res2.sig_level = 0.05
        res2.power = 0.006063932667926375
        res2.alternative = "less"
        res2.note = """NULL"""
        res2.method = "One-sample t test power calculation"

        cls.res2 = res2
        cls.kwds = {
            "effect_size": res2.d,
            "nobs": res2.n,
            "alpha": res2.sig_level,
            "power": res2.power,
        }
        cls.kwds_extra = {"alternative": "smaller"}
        cls.cls = smp.TTestPower


class TestTTPowerOneS6(CheckPowerMixin):
    # case one-sided less, negative effect size, not implemented yet

    @classmethod
    def setup_class(cls):

        res2 = Holder()
        # > p = pwr.t.test(d=-0.2,n=20,sig.level=0.05,type="one.sample",alternative="less")
        # > cat_items(p, "res2.")
        res2.n = 20
        res2.d = -0.2
        res2.sig_level = 0.05
        res2.power = 0.21707518167191
        res2.alternative = "less"
        res2.note = """NULL"""
        res2.method = "One-sample t test power calculation"

        cls.res2 = res2
        cls.kwds = {
            "effect_size": res2.d,
            "nobs": res2.n,
            "alpha": res2.sig_level,
            "power": res2.power,
        }
        cls.kwds_extra = {"alternative": "smaller"}
        cls.cls = smp.TTestPower


class TestTTPowerTwoS1(CheckPowerMixin):

    @classmethod
    def setup_class(cls):

        # > p = pwr.t.test(d=1,n=30,sig.level=0.05,type="two.sample",alternative="two.sided")
        # > cat_items(p, prefix='tt_power2_1.')
        res2 = Holder()
        res2.n = 30
        res2.d = 1
        res2.sig_level = 0.05
        res2.power = 0.967708258242517
        res2.alternative = "two.sided"
        res2.note = "n is number in *each* group"
        res2.method = "Two-sample t test power calculation"

        cls.res2 = res2
        cls.kwds = {
            "effect_size": res2.d,
            "nobs1": res2.n,
            "alpha": res2.sig_level,
            "power": res2.power,
            "ratio": 1,
        }
        cls.kwds_extra = {}
        cls.cls = smp.TTestIndPower


class TestTTPowerTwoS2(CheckPowerMixin):

    @classmethod
    def setup_class(cls):

        res2 = Holder()
        # > p = pwr.t.test(d=0.1,n=20,sig.level=0.05,type="two.sample",alternative="two.sided")
        # > cat_items(p, "res2.")
        res2.n = 20
        res2.d = 0.1
        res2.sig_level = 0.05
        res2.power = 0.06095912465411235
        res2.alternative = "two.sided"
        res2.note = "n is number in *each* group"
        res2.method = "Two-sample t test power calculation"

        cls.res2 = res2
        cls.kwds = {
            "effect_size": res2.d,
            "nobs1": res2.n,
            "alpha": res2.sig_level,
            "power": res2.power,
            "ratio": 1,
        }
        cls.kwds_extra = {}
        cls.cls = smp.TTestIndPower


class TestTTPowerTwoS3(CheckPowerMixin):

    @classmethod
    def setup_class(cls):

        res2 = Holder()
        # > p = pwr.t.test(d=1,n=30,sig.level=0.05,type="two.sample",alternative="greater")
        # > cat_items(p, prefix='tt_power2_1g.')
        res2.n = 30
        res2.d = 1
        res2.sig_level = 0.05
        res2.power = 0.985459690251624
        res2.alternative = "greater"
        res2.note = "n is number in *each* group"
        res2.method = "Two-sample t test power calculation"

        cls.res2 = res2
        cls.kwds = {
            "effect_size": res2.d,
            "nobs1": res2.n,
            "alpha": res2.sig_level,
            "power": res2.power,
            "ratio": 1,
        }
        cls.kwds_extra = {"alternative": "larger"}
        cls.cls = smp.TTestIndPower


class TestTTPowerTwoS4(CheckPowerMixin):
    # case with small power

    @classmethod
    def setup_class(cls):

        res2 = Holder()
        # > p = pwr.t.test(d=0.01,n=30,sig.level=0.05,type="two.sample",alternative="greater")
        # > cat_items(p, "res2.")
        res2.n = 30
        res2.d = 0.01
        res2.sig_level = 0.05
        res2.power = 0.0540740302835667
        res2.alternative = "greater"
        res2.note = "n is number in *each* group"
        res2.method = "Two-sample t test power calculation"

        cls.res2 = res2
        cls.kwds = {
            "effect_size": res2.d,
            "nobs1": res2.n,
            "alpha": res2.sig_level,
            "power": res2.power,
        }
        cls.kwds_extra = {"alternative": "larger"}
        cls.cls = smp.TTestIndPower


class TestTTPowerTwoS5(CheckPowerMixin):
    # case with unequal n, ratio>1

    @classmethod
    def setup_class(cls):

        res2 = Holder()
        # > p = pwr.t2n.test(d=0.1,n1=20, n2=30,sig.level=0.05,alternative="two.sided")
        # > cat_items(p, "res2.")
        res2.n1 = 20
        res2.n2 = 30
        res2.d = 0.1
        res2.sig_level = 0.05
        res2.power = 0.0633081832564667
        res2.alternative = "two.sided"
        res2.method = "t test power calculation"

        cls.res2 = res2
        cls.kwds = {
            "effect_size": res2.d,
            "nobs1": res2.n1,
            "alpha": res2.sig_level,
            "power": res2.power,
            "ratio": 1.5,
        }
        cls.kwds_extra = {"alternative": "two-sided"}
        cls.cls = smp.TTestIndPower


class TestTTPowerTwoS6(CheckPowerMixin):
    # case with unequal n, ratio>1

    @classmethod
    def setup_class(cls):

        res2 = Holder()
        # > p = pwr.t2n.test(d=0.1,n1=20, n2=30,sig.level=0.05,alternative="greater")
        # > cat_items(p, "res2.")
        res2.n1 = 20
        res2.n2 = 30
        res2.d = 0.1
        res2.sig_level = 0.05
        res2.power = 0.09623589080917805
        res2.alternative = "greater"
        res2.method = "t test power calculation"

        cls.res2 = res2
        cls.kwds = {
            "effect_size": res2.d,
            "nobs1": res2.n1,
            "alpha": res2.sig_level,
            "power": res2.power,
            "ratio": 1.5,
        }
        cls.kwds_extra = {"alternative": "larger"}
        cls.cls = smp.TTestIndPower


def test_normal_power_explicit():
    # a few initial test cases for NormalIndPower
    d = 0.3
    nobs = 80
    res1 = smp.normal_power(d, nobs / 2.0, 0.05)
    res2 = smp.NormalIndPower().power(d, nobs, 0.05)
    res3 = smp.NormalIndPower().solve_power(
        effect_size=0.3, nobs1=80, alpha=0.05, power=None
    )
    res_R = 0.475100870572638
    assert_almost_equal(res1, res_R, decimal=13)
    assert_almost_equal(res2, res_R, decimal=13)
    assert_almost_equal(res3, res_R, decimal=13)

    norm_pow = smp.normal_power(-0.01, nobs / 2.0, 0.05)
    norm_pow_R = 0.05045832927039234
    # value from R: >pwr.2p.test(h=0.01,n=80,sig.level=0.05,alternative="two.sided")
    assert_almost_equal(norm_pow, norm_pow_R, decimal=11)

    norm_pow = smp.NormalIndPower().power(0.01, nobs, 0.05, alternative="larger")
    norm_pow_R = 0.056869534873146124
    # value from R: >pwr.2p.test(h=0.01,n=80,sig.level=0.05,alternative="greater")
    assert_almost_equal(norm_pow, norm_pow_R, decimal=11)

    # Note: negative effect size is same as switching one-sided alternative
    # TODO: should I switch to larger/smaller instead of "one-sided" options
    norm_pow = smp.NormalIndPower().power(-0.01, nobs, 0.05, alternative="larger")
    norm_pow_R = 0.0438089705093578
    # value from R: >pwr.2p.test(h=0.01,n=80,sig.level=0.05,alternative="less")
    assert_almost_equal(norm_pow, norm_pow_R, decimal=11)


class TestNormalIndPower1(CheckPowerMixin):

    @classmethod
    def setup_class(cls):
        # > example from above
        # results copied not directly from R
        res2 = Holder()
        res2.n = 80
        res2.d = 0.3
        res2.sig_level = 0.05
        res2.power = 0.475100870572638
        res2.alternative = "two.sided"
        res2.note = "NULL"
        res2.method = "two sample power calculation"

        cls.res2 = res2
        cls.kwds = {
            "effect_size": res2.d,
            "nobs1": res2.n,
            "alpha": res2.sig_level,
            "power": res2.power,
            "ratio": 1,
        }
        cls.kwds_extra = {}
        cls.cls = smp.NormalIndPower


class TestNormalIndPower2(CheckPowerMixin):

    @classmethod
    def setup_class(cls):
        res2 = Holder()
        # > np = pwr.2p.test(h=0.01,n=80,sig.level=0.05,alternative="less")
        # > cat_items(np, "res2.")
        res2.h = 0.01
        res2.n = 80
        res2.sig_level = 0.05
        res2.power = 0.0438089705093578
        res2.alternative = "less"
        res2.method = (
            "Difference of proportion power calculation for binomial distribution "
            "(arcsine transformation)"
        )
        res2.note = "same sample sizes"

        cls.res2 = res2
        cls.kwds = {
            "effect_size": res2.h,
            "nobs1": res2.n,
            "alpha": res2.sig_level,
            "power": res2.power,
            "ratio": 1,
        }
        cls.kwds_extra = {"alternative": "smaller"}
        cls.cls = smp.NormalIndPower


class TestNormalIndPower_onesamp1(CheckPowerMixin):

    @classmethod
    def setup_class(cls):
        # forcing one-sample by using ratio=0
        # > example from above
        # results copied not directly from R
        res2 = Holder()
        res2.n = 40
        res2.d = 0.3
        res2.sig_level = 0.05
        res2.power = 0.475100870572638
        res2.alternative = "two.sided"
        res2.note = "NULL"
        res2.method = "two sample power calculation"

        cls.res2 = res2
        cls.kwds = {
            "effect_size": res2.d,
            "nobs1": res2.n,
            "alpha": res2.sig_level,
            "power": res2.power,
        }
        # keyword for which we do not look for root:
        cls.kwds_extra = {"ratio": 0}

        cls.cls = smp.NormalIndPower


class TestNormalIndPower_onesamp2(CheckPowerMixin):
    # Note: same power as two sample case with twice as many observations

    @classmethod
    def setup_class(cls):
        # forcing one-sample by using ratio=0
        res2 = Holder()
        # > np = pwr.norm.test(d=0.01,n=40,sig.level=0.05,alternative="less")
        # > cat_items(np, "res2.")
        res2.d = 0.01
        res2.n = 40
        res2.sig_level = 0.05
        res2.power = 0.0438089705093578
        res2.alternative = "less"
        res2.method = (
            "Mean power calculation for normal distribution with known variance"
        )

        cls.res2 = res2
        cls.kwds = {
            "effect_size": res2.d,
            "nobs1": res2.n,
            "alpha": res2.sig_level,
            "power": res2.power,
        }
        # keyword for which we do not look for root:
        cls.kwds_extra = {"ratio": 0, "alternative": "smaller"}

        cls.cls = smp.NormalIndPower


class TestChisquarePower(CheckPowerMixin):

    @classmethod
    def setup_class(cls):
        # one example from test_gof, results_power
        res2 = Holder()
        res2.w = 0.1
        res2.N = 5
        res2.df = 4
        res2.sig_level = 0.05
        res2.power = 0.05246644635810126
        res2.method = "Chi squared power calculation"
        res2.note = "N is the number of observations"

        cls.res2 = res2
        cls.kwds = {
            "effect_size": res2.w,
            "nobs": res2.N,
            "alpha": res2.sig_level,
            "power": res2.power,
        }
        # keyword for which we do not look for root:
        # solving for n_bins does not work, will not be used in regular usage
        cls.kwds_extra = {"n_bins": res2.df + 1}

        cls.cls = smp.GofChisquarePower

    def test_positional(self):

        res1 = self.cls()
        args_names = ["effect_size", "nobs", "alpha", "n_bins"]
        kwds = copy.copy(self.kwds)
        del kwds["power"]
        kwds.update(self.kwds_extra)
        args = [kwds[arg] for arg in args_names]
        if hasattr(self, "decimal"):
            decimal = self.decimal  # pylint: disable-msg=E1101
        else:
            decimal = 6
        assert_almost_equal(res1.power(*args), self.res2.power, decimal=decimal)


def test_ftest_power():
    # equivalence ftest, ttest

    for alpha in [0.01, 0.05, 0.1, 0.20, 0.50]:
        res0 = smp.ttest_power(0.01, 200, alpha)
        res1 = smp.ftest_power(0.01, 199, 1, alpha=alpha, ncc=0)
        assert_almost_equal(res1, res0, decimal=6)

    # example from Gplus documentation F-test ANOVA
    # Total sample size:200
    # Effect size "f":0.25
    # Beta/alpha ratio:1
    # Result:
    # Alpha:0.1592
    # Power (1-beta):0.8408
    # Critical F:1.4762
    # Lambda: 12.50000
    res1 = smp.ftest_anova_power(0.25, 200, 0.1592, k_groups=10)
    res0 = 0.8408
    assert_almost_equal(res1, res0, decimal=4)

    # TODO: no class yet
    # examples against R::pwr
    res2 = Holder()
    # > rf = pwr.f2.test(u=5, v=199, f2=0.1**2, sig.level=0.01)
    # > cat_items(rf, "res2.")
    res2.u = 5
    res2.v = 199
    res2.f2 = 0.01
    res2.sig_level = 0.01
    res2.power = 0.0494137732920332
    res2.method = "Multiple regression power calculation"

    res1 = smp.ftest_power(
        np.sqrt(res2.f2), res2.v, res2.u, alpha=res2.sig_level, ncc=1
    )
    assert_almost_equal(res1, res2.power, decimal=5)

    res2 = Holder()
    # > rf = pwr.f2.test(u=5, v=199, f2=0.3**2, sig.level=0.01)
    # > cat_items(rf, "res2.")
    res2.u = 5
    res2.v = 199
    res2.f2 = 0.09
    res2.sig_level = 0.01
    res2.power = 0.7967191006290872
    res2.method = "Multiple regression power calculation"

    res1 = smp.ftest_power(
        np.sqrt(res2.f2), res2.v, res2.u, alpha=res2.sig_level, ncc=1
    )
    assert_almost_equal(res1, res2.power, decimal=5)

    res2 = Holder()
    # > rf = pwr.f2.test(u=5, v=19, f2=0.3**2, sig.level=0.1)
    # > cat_items(rf, "res2.")
    res2.u = 5
    res2.v = 19
    res2.f2 = 0.09
    res2.sig_level = 0.1
    res2.power = 0.235454222377575
    res2.method = "Multiple regression power calculation"

    res1 = smp.ftest_power(
        np.sqrt(res2.f2), res2.v, res2.u, alpha=res2.sig_level, ncc=1
    )
    assert_almost_equal(res1, res2.power, decimal=5)


# class based version of two above test for Ftest
class TestFtestAnovaPower(CheckPowerMixin):

    @classmethod
    def setup_class(cls):
        res2 = Holder()
        # example from Gplus documentation F-test ANOVA
        # Total sample size:200
        # Effect size "f":0.25
        # Beta/alpha ratio:1
        # Result:
        # Alpha:0.1592
        # Power (1-beta):0.8408
        # Critical F:1.4762
        # Lambda: 12.50000
        # converted to res2 by hand
        res2.f = 0.25
        res2.n = 200
        res2.k = 10
        res2.alpha = 0.1592
        res2.power = 0.8408
        res2.method = "Multiple regression power calculation"

        cls.res2 = res2
        cls.kwds = {
            "effect_size": res2.f,
            "nobs": res2.n,
            "alpha": res2.alpha,
            "power": res2.power,
        }
        # keyword for which we do not look for root:
        # solving for n_bins does not work, will not be used in regular usage
        cls.kwds_extra = {"k_groups": res2.k}  # rootfinding does not work
        # cls.args_names = ['effect_size','nobs', 'alpha']#, 'k_groups']
        cls.cls = smp.FTestAnovaPower
        # precision for test_power
        cls.decimal = 4


class TestFtestPower(CheckPowerMixin):

    @classmethod
    def setup_class(cls):
        res2 = Holder()
        # > rf = pwr.f2.test(u=5, v=19, f2=0.3**2, sig.level=0.1)
        # > cat_items(rf, "res2.")
        res2.u = 5
        res2.v = 19
        res2.f2 = 0.09
        res2.sig_level = 0.1
        res2.power = 0.235454222377575
        res2.method = "Multiple regression power calculation"

        cls.res2 = res2
        cls.kwds = {
            "effect_size": np.sqrt(res2.f2),
            "df_num": res2.v,
            "df_denom": res2.u,
            "alpha": res2.sig_level,
            "power": res2.power,
        }
        # keyword for which we do not look for root:
        # solving for n_bins does not work, will not be used in regular usage
        cls.kwds_extra = {}
        cls.args_names = ["effect_size", "df_num", "df_denom", "alpha"]
        cls.cls = smp.FTestPower
        # precision for test_power
        cls.decimal = 5

    def test_kwargs(self):

        with pytest.warns(UserWarning, match="nobs is not use"):
            smp.FTestPower().solve_power(
                effect_size=0.3, alpha=0.1, power=0.9, df_denom=2, nobs=None
            )

        with pytest.raises(ValueError):
            smp.FTestPower().solve_power(
                effect_size=0.3, alpha=0.1, power=0.9, df_denom=2, junk=3
            )


class TestFtestPowerF2(CheckPowerMixin):

    @classmethod
    def setup_class(cls):
        res2 = Holder()
        # > rf = pwr.f2.test(u=5, v=19, f2=0.3**2, sig.level=0.1)
        # > cat_items(rf, "res2.")
        res2.u = 5
        res2.v = 19
        res2.f2 = 0.09
        res2.sig_level = 0.1
        res2.power = 0.235454222377575
        res2.method = "Multiple regression power calculation"

        cls.res2 = res2
        cls.kwds = {
            "effect_size": res2.f2,
            "df_num": res2.u,
            "df_denom": res2.v,
            "alpha": res2.sig_level,
            "power": res2.power,
        }
        # keyword for which we do not look for root:
        # solving for n_bins does not work, will not be used in regular usage
        cls.kwds_extra = {}
        cls.args_names = ["effect_size", "df_num", "df_denom", "alpha"]
        cls.cls = smp.FTestPowerF2
        # precision for test_power
        cls.decimal = 5


def test_power_solver():
    # messing up the solver to trigger backup

    nip = smp.NormalIndPower()

    # check result
    es0 = 0.1
    pow_ = nip.solve_power(
        es0, nobs1=1600, alpha=0.01, power=None, ratio=1, alternative="larger"
    )
    # value is regression test
    assert_almost_equal(pow_, 0.69219411243824214, decimal=5)
    es = nip.solve_power(
        None, nobs1=1600, alpha=0.01, power=pow_, ratio=1, alternative="larger"
    )
    assert_almost_equal(es, es0, decimal=4)
    assert_equal(nip.cache_fit_res[0], 1)
    assert_equal(len(nip.cache_fit_res), 2)

    # cause first optimizer to fail
    nip.start_bqexp["effect_size"] = {"upp": -10, "low": -20}
    nip.start_ttp["effect_size"] = 0.14
    es = nip.solve_power(
        None, nobs1=1600, alpha=0.01, power=pow_, ratio=1, alternative="larger"
    )
    assert_almost_equal(es, es0, decimal=4)
    assert_equal(nip.cache_fit_res[0], 1)
    assert_equal(len(nip.cache_fit_res), 3, err_msg=repr(nip.cache_fit_res))

    nip.start_ttp["effect_size"] = np.nan
    es = nip.solve_power(
        None, nobs1=1600, alpha=0.01, power=pow_, ratio=1, alternative="larger"
    )
    assert_almost_equal(es, es0, decimal=4)
    assert_equal(nip.cache_fit_res[0], 1)
    assert_equal(len(nip.cache_fit_res), 4)

    # Test our edge-case where effect_size = 0
    es = nip.solve_power(nobs1=1600, alpha=0.01, effect_size=0, power=None)
    assert_almost_equal(es, 0.01)

    # I let this case fail, could be fixed for some statistical tests
    # (we should not get here in the first place)
    # effect size is negative, but last stage brentq uses [1e-8, 1-1e-8]
    with pytest.raises(ValueError):
        nip.solve_power(
            None,
            nobs1=1600,
            alpha=0.01,
            power=0.005,
            ratio=1,
            alternative="larger",
        )

    with pytest.warns(HypothesisTestWarning):
        with pytest.raises(ValueError):
            nip.solve_power(
                nobs1=None,
                effect_size=0,
                alpha=0.01,
                power=0.005,
                ratio=1,
                alternative="larger",
            )


def test_solve_power_no_solution_returns_nan():
    # GH#9378: when the power equation has no solution the root finder
    # cannot converge. Previously solve_power still returned the last value
    # the solver evaluated -- a bracket bound such as 10 -- which
    # masqueraded as a valid sample size. It should return nan instead,
    # while still warning that it failed to converge.
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    tt = smp.TTestPower()

    # 'smaller' alternative with a positive effect size and a target power
    # below alpha is not intercepted by the up-front sign check, but the
    # target exceeds the attainable maximum (about 0.018 at nobs=2), so
    # the root finder fails.
    with pytest.warns(
        ConvergenceWarning,
        match=r"last value evaluated by the root finder was \[?\d",
    ):
        val = tt.solve_power(
            effect_size=0.5,
            nobs=None,
            alpha=0.05,
            power=0.03,
            alternative="smaller",
        )
    assert np.isnan(val)
    assert_equal(tt.cache_fit_res[0], 0)

    # a solvable case is unaffected and still returns a finite sample size
    val = tt.solve_power(
        effect_size=0.5,
        nobs=None,
        alpha=0.05,
        power=0.8,
        alternative="larger",
    )
    assert np.isfinite(val)


def test_solve_power_impossible_one_sided_raises():
    # GH#9378: a one-sided alternative with an effect size of the opposite
    # sign keeps the attained power below alpha for any sample size, so
    # solving for a sample size with power >= alpha is impossible. Such
    # requests are intercepted up front with an informative error instead
    # of a ConvergenceWarning and nan from the root finder.
    tt = smp.TTestPower()
    ttind = smp.TTestIndPower()
    nip = smp.NormalIndPower()

    match = "No solution exists"
    with pytest.raises(ValueError, match=match):
        tt.solve_power(
            effect_size=0.5,
            nobs=None,
            alpha=0.05,
            power=0.8,
            alternative="smaller",
        )
    with pytest.raises(ValueError, match=match):
        tt.solve_power(
            effect_size=-0.5,
            nobs=None,
            alpha=0.05,
            power=0.8,
            alternative="larger",
        )
    with pytest.raises(ValueError, match=match):
        ttind.solve_power(
            effect_size=0.5,
            nobs1=None,
            alpha=0.05,
            power=0.8,
            ratio=1,
            alternative="smaller",
        )
    with pytest.raises(ValueError, match=match):
        nip.solve_power(
            effect_size=0.5,
            nobs1=None,
            alpha=0.05,
            power=0.8,
            ratio=1,
            alternative="smaller",
        )
    with pytest.raises(ValueError, match=match):
        ttind.solve_power(
            effect_size=0.5,
            nobs1=10,
            alpha=0.05,
            power=0.8,
            ratio=None,
            alternative="smaller",
        )

    # matching signs still solve
    res = tt.solve_power(
        effect_size=0.5,
        nobs=None,
        alpha=0.05,
        power=0.8,
        alternative="larger",
    )
    assert_almost_equal(res, 26.1375, decimal=3)
    res = tt.solve_power(
        effect_size=-0.5, nobs=None, alpha=0.05, power=0.8,
        alternative="smaller",
    )
    assert_almost_equal(res, 26.1375, decimal=3)

    # a wrong-signed effect size with target power below alpha can have a
    # valid solution and is not intercepted
    res = tt.solve_power(
        effect_size=0.5, nobs=None, alpha=0.05, power=0.01,
        alternative="smaller",
    )
    roundtrip = tt.power(
        effect_size=0.5, nobs=res, alpha=0.05, alternative="smaller"
    )
    assert_almost_equal(roundtrip, 0.01, decimal=6)

    # solving for other parameters is not affected by the sign check
    res = tt.solve_power(
        effect_size=0.5, nobs=25, alpha=None, power=0.8,
        alternative="smaller",
    )
    assert np.isfinite(res)
    res = tt.solve_power(
        effect_size=None, nobs=25, alpha=0.05, power=0.8,
        alternative="smaller",
    )
    assert res < 0


# TODO: can something useful be made from this?
@pytest.mark.xfail(reason="Known failure on modern SciPy >= 0.10", strict=True)
def test_power_solver_warn():
    # messing up the solver to trigger warning
    # I wrote this with scipy 0.9,
    # convergence behavior of scipy 0.11 is different,
    # fails at a different case, but is successful where it failed before

    pow_ = 0.69219411243824214  # from previous function
    nip = smp.NormalIndPower()
    # using nobs, has one backup (fsolve)
    nip.start_bqexp["nobs1"] = {"upp": 50, "low": -20}
    val = nip.solve_power(
        0.1, nobs1=None, alpha=0.01, power=pow_, ratio=1, alternative="larger"
    )

    assert_almost_equal(val, 1600, decimal=4)
    assert_equal(nip.cache_fit_res[0], 1)
    assert_equal(len(nip.cache_fit_res), 3)

    # case that has convergence failure, and should warn
    nip.start_ttp["nobs1"] = np.nan

    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    with pytest.warns(ConvergenceWarning):
        nip.solve_power(
            0.1, nobs1=None, alpha=0.01, power=pow_, ratio=1, alternative="larger"
        )
    # this converges with scipy 0.11  ???
    # nip.solve_power(0.1, nobs1=None, alpha=0.01, power=pow_, ratio=1, alternative='larger')

    with warnings.catch_warnings():  # python >= 2.6
        warnings.simplefilter("ignore")
        nip.solve_power(
            0.1, nobs1=None, alpha=0.01, power=pow_, ratio=1, alternative="larger"
        )
        assert_equal(nip.cache_fit_res[0], 0)
        assert_equal(len(nip.cache_fit_res), 3)


def test_normal_sample_size_one_tail():
    # Test that using default value of std_alternative does not raise an
    # exception. A power of 0.8 and alpha of 0.05 were chosen to reflect
    # commonly used values in hypothesis testing. Difference in means and
    # standard deviation of null population were chosen somewhat arbitrarily --
    # there's nothing special about those values. Return value doesn't matter
    # for this "test", so long as an exception is not raised.
    smp.normal_sample_size_one_tail(5, 0.8, 0.05, 2, std_alternative=None)

    # Test that zero is returned in the correct elements if power is less
    # than alpha.
    alphas = np.asarray([0.01, 0.05, 0.1, 0.5, 0.8])
    powers = np.asarray([0.99, 0.95, 0.9, 0.5, 0.2])
    # zero_mask = np.where(alphas - powers > 0, 0.0, alphas - powers)
    nobs_with_zeros = smp.normal_sample_size_one_tail(5, powers, alphas, 2, 2)
    # check_nans = np.isnan(zero_mask) == np.isnan(nobs_with_nans)
    assert_array_equal(nobs_with_zeros[powers <= alphas], 0)


@pytest.mark.parametrize(
    "power_func",
    [
        lambda alternative: smp.ttest_power(0.5, 20, 0.05, alternative=alternative),
        lambda alternative: smp.normal_power(0.5, 20, 0.05, alternative=alternative),
        lambda alternative: smp.normal_power_het(0.5, 20, 0.05, alternative=alternative),
    ],
    ids=["ttest_power", "normal_power", "normal_power_het"],
)
def test_alternative_deprecated_alias(power_func):
    # the undocumented "2s" short form still works but warns, and is
    # equivalent to spelling out "two-sided"
    with pytest.warns(FutureWarning, match="is a deprecated alias"):
        power_alias = power_func("2s")
    power_canonical = power_func("two-sided")
    assert power_alias == power_canonical

    with pytest.raises(ValueError, match="alternative must be one of"):
        power_func("bogus")


def _ttest_power_het_integral(diff, nobs, alpha, s0, s1, df, alternative):
    # Conditional on V=v, T1 > c iff
    # Z > (c*s0*sqrt(v/df) - diff*sqrt(nobs))/s1. Integrate this normal
    # probability against chi-square(df), without any noncentral-t routines.
    tail_alpha = alpha / 2 if alternative == "two-sided" else alpha
    lower = stats.t.ppf(tail_alpha, df)
    upper = stats.t.isf(tail_alpha, df)

    def integrand(v):
        scale = s0 * np.sqrt(v / df)
        shift = diff * np.sqrt(nobs)
        probability = 0.0
        if alternative != "smaller":
            probability += stats.norm.sf((upper * scale - shift) / s1)
        if alternative != "larger":
            probability += stats.norm.cdf((lower * scale - shift) / s1)
        return probability * stats.chi2.pdf(v, df)

    result, error = integrate.quad(integrand, 0, np.inf, epsabs=1e-11, epsrel=1e-11)
    assert error < 1e-10
    return result


@pytest.mark.parametrize("alternative", ["larger", "smaller", "two-sided"])
@pytest.mark.parametrize("scales", [(1.0, 2.0), (2.0, 1.0)])
@pytest.mark.parametrize(
    "diff,nobs,alpha,df",
    [(0.3, 10, 0.05, 9), (-0.2, 30, 0.1, 12.5), (0.0, 50, 0.01, 49)],
)
def test_ttest_power_het_integral(alternative, scales, diff, nobs, alpha, df):
    s0, s1 = scales
    expected = _ttest_power_het_integral(diff, nobs, alpha, s0, s1, df, alternative)
    actual = smp.ttest_power_het(diff, nobs, alpha, s0, s1, alternative, df=df)
    assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("alternative", ["larger", "smaller", "two-sided"])
@pytest.mark.parametrize("diff", [-0.3, 0.0, 0.3])
@pytest.mark.parametrize("df", [None, 12.5])
def test_ttest_power_het_equal_scales(alternative, diff, df):
    expected = smp.ttest_power(diff / 2, 30, 0.05, df=df, alternative=alternative)
    for s1 in (None, 2.0):
        actual = smp.ttest_power_het(diff, 30, 0.05, 2, s1, alternative, df=df)
        assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("alternative", ["larger", "smaller", "two-sided"])
@pytest.mark.parametrize("scales", [(1.0, 2.0), (2.0, 1.0)])
def test_ttest_power_het_normal_limit(alternative, scales):
    s0, s1 = scales
    diff = np.array([-0.3, 0, 0.3])
    expected = smp.normal_power_het(diff, 30, 0.05, s0, s1, alternative)
    actual = smp.ttest_power_het(diff, 30, 0.05, s0, s1, alternative, df=1e7)
    # Allow for finite-df corrections at fixed noncentrality.
    assert_allclose(actual, expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize("alternative", ["larger", "smaller", "two-sided"])
def test_ttest_power_het_broadcast_and_scale(alternative):
    diff = np.array([[-0.2], [0.3]])
    nobs = np.array([10, 30, 50])
    alpha = np.array([0.01, 0.05, 0.1])
    s0, s1 = np.array([1, 2, 1]), np.array([2, 1, 1])
    actual = smp.ttest_power_het(diff, nobs, alpha, s0, s1, alternative)
    expected = np.array(
        [
            [
                _ttest_power_het_integral(d, n, a, x, y, n - 1, alternative)
                for n, a, x, y in zip(nobs, alpha, s0, s1, strict=True)
            ]
            for d in diff[:, 0]
        ]
    )
    # SciPy 1.14's CDFLIB nctdtr differs from quadrature by up to 8e-10
    # in the small one-sided probabilities here, including on CPython.
    assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)
    explicit = smp.ttest_power_het(diff, nobs, alpha, s0, s1, alternative, df=nobs - 1)
    assert_array_equal(actual, explicit)
    scaled = smp.ttest_power_het(7 * diff, nobs, alpha, 7 * s0, 7 * s1, alternative)
    assert_allclose(actual, scaled, rtol=1e-12, atol=1e-15)


def test_ttest_power_het_symmetry():
    diff = np.array([0.1, 0.3, 0.5])
    for s0, s1 in [(1, 2), (2, 1)]:
        upper = smp.ttest_power_het(diff, 20, 0.05, s0, s1, "larger")
        lower = smp.ttest_power_het(-diff, 20, 0.05, s0, s1, "smaller")
        assert_allclose(upper, lower, rtol=1e-6, atol=1e-10)
        positive = smp.ttest_power_het(diff, 20, 0.05, s0, s1)
        negative = smp.ttest_power_het(-diff, 20, 0.05, s0, s1)
        assert_allclose(positive, negative, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("name", ["nobs", "df", "std_null", "std_alternative"])
@pytest.mark.parametrize("value", [0, -1, np.inf])
def test_ttest_power_het_invalid_scale_or_df(name, value):
    kwds = dict(diff=0.3, nobs=20, alpha=0.05)
    kwds[name] = value
    with pytest.raises(ValueError, match=f"{name} must be positive and finite"):
        smp.ttest_power_het(**kwds)


def test_ttest_power_het_invalid_alpha_or_alternative():
    for alpha in (0, 1, -0.1, 1.1):
        with pytest.raises(ValueError, match="alpha must be between"):
            smp.ttest_power_het(0.3, 20, alpha)
    with pytest.raises(ValueError, match="alternative must be one of"):
        smp.ttest_power_het(0.3, 20, 0.05, alternative="bogus")


@pytest.mark.parametrize(
    "name", ["diff", "nobs", "alpha", "std_null", "std_alternative", "df"]
)
def test_ttest_power_het_nan(name):
    kwds = dict(diff=0.3, nobs=20, alpha=0.05, std_null=1, std_alternative=2, df=19)
    expected = smp.ttest_power_het(**kwds)
    kwds[name] = [kwds[name], np.nan]
    actual = smp.ttest_power_het(**kwds)
    assert_allclose(actual, [expected, np.nan])
