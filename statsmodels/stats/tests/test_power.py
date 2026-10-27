# pylint: disable=W0231, W0142
"""Tests for statistical power calculations

Note:
    tests for chisquare power are in test_gof.py

Created on Sat Mar 09 08:44:49 2013

Author: Josef Perktold
"""

from statsmodels.compat.pandas import PD_LT_2
from statsmodels.compat.platform import PLATFORM_OSX, PLATFORM_WIN
from statsmodels.compat.python import PYTHON_IMPL_WASM
from statsmodels.compat.scipy import SP_LT_115

import copy

import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
)
import pandas as pd
import pytest

import statsmodels.stats.power as smp
from statsmodels.stats.tests.test_weightstats import Holder
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HypothesisTestWarning
from statsmodels.tools.testing import warns_or_none

# The noncentral t distribution of SciPy on WASM is accurate to about 1e-6
# (relative), which is far from the 1e-12 of the other platforms
NCT_RTOL = 1e-5 if (PYTHON_IMPL_WASM or SP_LT_115) else 1e-7
POWER_RTOL = 1e-6 if (PYTHON_IMPL_WASM or SP_LT_115) else 1e-9

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
    with pytest.warns(ConvergenceWarning):
        es = nip.solve_power(
            None,
            nobs1=1600,
            alpha=0.01,
            power=0.005,
            ratio=1,
            alternative="larger",
        )
    assert np.isnan(es)

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


@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_solve_power_alpha_search_leaves_unit_interval(alpha):
    # The first root finder does not converge for these problems, and the
    # fallback fsolve evaluates the power at alpha < 0. The check of alpha then
    # raised, and the solve failed although the problem has a solution. The
    # solution is exact to 1e-13 with scipy 1.15, 1.18 and the development
    # version. (A problem like effect_size=0.1, nobs=1000, alpha=0.01 is only
    # solved to a relative error of 1e-4 with scipy 1.15.)
    es, nobs = 0.3, 300
    power = smp.TTestPower().power(es, nobs, alpha)
    solved = smp.TTestPower().solve_power(
        effect_size=es, nobs=nobs, alpha=None, power=power
    )
    assert_allclose(solved, alpha, rtol=1e-3)


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
        effect_size=-0.5,
        nobs=None,
        alpha=0.05,
        power=0.8,
        alternative="smaller",
    )
    assert_almost_equal(res, 26.1375, decimal=3)

    # a wrong-signed effect size with target power below alpha can have a
    # valid solution and is not intercepted
    res = tt.solve_power(
        effect_size=0.5,
        nobs=None,
        alpha=0.05,
        power=0.01,
        alternative="smaller",
    )
    roundtrip = tt.power(effect_size=0.5, nobs=res, alpha=0.05, alternative="smaller")
    assert_almost_equal(roundtrip, 0.01, decimal=6)

    # solving for other parameters is not affected by the sign check
    res = tt.solve_power(
        effect_size=0.5,
        nobs=25,
        alpha=None,
        power=0.8,
        alternative="smaller",
    )
    assert np.isfinite(res)
    res = tt.solve_power(
        effect_size=None,
        nobs=25,
        alpha=0.05,
        power=0.8,
        alternative="smaller",
    )
    assert res < 0


# TODO: can something useful be made from this?
def test_power_solver_warn():
    # messing up the solver to trigger warning
    # I wrote this with scipy 0.9,
    # convergence behavior of scipy 0.11 is different,
    # fails at a different case, but is successful where it failed before

    pow_ = 0.69219411243824214  # from previous function
    nip = smp.NormalIndPower()
    # using nobs, has one backup (fsolve)
    nip.start_bqexp["nobs1"] = {"upp": 50, "low": -20}
    # the solver evaluates the power at the negative lower bound of nobs
    with warns_or_none(RuntimeWarning, match="invalid value encountered in sqrt"):
        val = nip.solve_power(
            0.1, nobs1=None, alpha=0.01, power=pow_, ratio=1, alternative="larger"
        )

    assert_almost_equal(val, 1600, decimal=4)
    assert_equal(nip.cache_fit_res[0], 1)
    assert_equal(len(nip.cache_fit_res), 3)

    # case that has convergence failure, and should warn
    nip.start_ttp["nobs1"] = np.nan

    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    with (
        pytest.warns(ConvergenceWarning),
        warns_or_none(RuntimeWarning, match="invalid value encountered in sqrt"),
    ):
        nip.solve_power(
            0.1, nobs1=None, alpha=0.01, power=pow_, ratio=1, alternative="larger"
        )
    # this converges with scipy 0.11  ???
    # nip.solve_power(0.1, nobs1=None, alpha=0.01, power=pow_, ratio=1, alternative='larger')

    with (
        pytest.warns(ConvergenceWarning),
        warns_or_none(RuntimeWarning, match="invalid value encountered in sqrt"),
    ):
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
        lambda alternative: smp.normal_power_het(
            0.5, 20, 0.05, alternative=alternative
        ),
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


@pytest.mark.parametrize("alternative", ["two-sided", "larger", "smaller"])
@pytest.mark.parametrize("nobs", [5, 20, 50.5, 200])
def test_ttest_ind_power_ratio_zero_is_one_sample(alternative, nobs):
    # ratio=0 is the convention for a one-sample t-test with nobs1
    # observations; it used to raise a ZeroDivisionError for Python scalars and
    # to return alpha with a RuntimeWarning for numpy scalars
    ind = smp.TTestIndPower()
    one = smp.TTestPower()
    effect_size = -0.5 if alternative == "smaller" else 0.5
    p_zero = ind.power(effect_size, nobs, 0.05, ratio=0, alternative=alternative)
    p_one = one.power(effect_size, nobs, 0.05, alternative=alternative)
    assert_allclose(p_zero, p_one, rtol=1e-12)
    assert np.ndim(p_zero) == 0
    # df is honored, and defaults to nobs1 - 1
    p_df = ind.power(
        effect_size, nobs, 0.05, ratio=0, df=nobs - 1, alternative=alternative
    )
    assert_allclose(p_df, p_zero, rtol=1e-12)
    p_df = ind.power(effect_size, nobs, 0.05, ratio=0, df=7, alternative=alternative)
    expected = smp.ttest_power(effect_size, nobs, 0.05, df=7, alternative=alternative)
    assert_allclose(p_df, expected, rtol=1e-12)
    assert not np.isclose(p_df, p_zero, rtol=1e-6)


@pytest.mark.parametrize(
    "effect_size, nobs1, alpha, alternative, power_r",
    [
        (0.5, 50, 0.05, "two-sided", 0.933897581262),
        (0.2, 5, 0.05, "two-sided", 0.0642695740480347),
        (0.2, 5, 0.01, "two-sided", 0.0135358237849607),
        (0.5, 20, 0.05, "larger", 0.6951493382),
        (0.5, 50.5, 0.05, "two-sided", 0.9361781095),
    ],
)
def test_ttest_ind_power_ratio_zero_r(effect_size, nobs1, alpha, alternative, power_r):
    # R 4.6.1, package pwr 1.3-0:
    # pwr.t.test(n = nobs1, d = effect_size, sig.level = alpha,
    #            type = "one.sample", alternative = c("two.sided", "greater"))$power
    # R has no equivalent of ratio=0 in the two-sample functions
    res = smp.TTestIndPower().power(
        effect_size, nobs1, alpha, ratio=0, alternative=alternative
    )
    # the noncentral t distribution of SciPy on WASM is accurate to about 1e-8
    assert_allclose(res, power_r, rtol=1e-7)


def test_ttest_ind_power_ratio_zero_is_not_the_limit():
    # ratio=0 is a convention: for ratio -> 0 the second sample vanishes and
    # the power goes to alpha, it does not go to the one-sample power
    ind = smp.TTestIndPower()
    assert_allclose(ind.power(0.5, 50, 0.05, ratio=1e-9), 0.05, atol=1e-6)
    assert ind.power(0.5, 50, 0.05, ratio=0) > 0.9


@pytest.mark.parametrize("alternative", ["two-sided", "larger", "smaller"])
def test_ttest_ind_solve_power_ratio_zero(alternative):
    ind = smp.TTestIndPower()
    one = smp.TTestPower()
    sign = -1 if alternative == "smaller" else 1
    kwds = {"ratio": 0, "alternative": alternative}

    nobs1 = ind.solve_power(sign * 0.5, None, 0.05, 0.8, **kwds)
    nobs = one.solve_power(sign * 0.5, None, 0.05, 0.8, alternative=alternative)
    assert_allclose(nobs1, nobs, rtol=1e-5)
    assert_allclose(ind.power(sign * 0.5, nobs1, 0.05, **kwds), 0.8, rtol=1e-5)

    effect_size = ind.solve_power(None, 30, 0.05, 0.8, **kwds)
    expected = one.solve_power(None, 30, 0.05, 0.8, alternative=alternative)
    assert_allclose(effect_size, expected, rtol=1e-5)
    assert_allclose(ind.power(effect_size, 30, 0.05, **kwds), 0.8, rtol=1e-5)

    alpha = ind.solve_power(sign * 0.5, 30, None, 0.8, **kwds)
    expected = one.solve_power(sign * 0.5, 30, None, 0.8, alternative=alternative)
    assert_allclose(alpha, expected, rtol=1e-5)
    assert_allclose(ind.power(sign * 0.5, 30, alpha, **kwds), 0.8, rtol=1e-5)

    # solving for the power does not depend on the root finding
    power = ind.solve_power(sign * 0.5, 30, 0.05, None, **kwds)
    assert_allclose(power, ind.power(sign * 0.5, 30, 0.05, **kwds), rtol=1e-12)


# R 4.6.1, package pwr 1.3-0, effect size 0.5, nobs1 = 50, alpha = 0.05; the
# first element is for ratio = 0, the others for ratio = 0.5 and 1:
# pwr.t.test(n = 50, d = 0.5, sig.level = 0.05, type = "one.sample",
#            alternative = alt)$power
# pwr.t2n.test(n1 = 50, n2 = c(25, 50), d = 0.5, sig.level = 0.05,
#              alternative = alt)$power
# with alt in c("two.sided", "greater", "less")
POWER_RATIO_R = {
    "two-sided": [0.933897581262, 0.521721300128, 0.696893405534],
    "larger": [0.967206745826, 0.647066651831, 0.798936164150],
    "smaller": [1.42685133153e-07, 1.22510474520e-04, 1.83145096917e-05],
}
RATIOS = np.array([0.0, 0.5, 1.0])


@pytest.mark.parametrize("alternative", ["two-sided", "larger", "smaller"])
def test_ttest_ind_power_ratio_array(alternative):
    ind = smp.TTestIndPower()
    kwds = {"alternative": alternative}
    res = ind.power(0.5, 50, 0.05, ratio=RATIOS, **kwds)
    assert isinstance(res, np.ndarray)
    assert res.shape == (3,)
    # the noncentral t distribution of SciPy on WASM is accurate to about 1e-8
    # (absolute), which matters for the tiny powers of the other alternative
    assert_allclose(res, POWER_RATIO_R[alternative], rtol=1e-6, atol=1e-8)
    # same as the scalar calls
    expected = [ind.power(0.5, 50, 0.05, ratio=r, **kwds) for r in RATIOS]
    assert_allclose(res, expected, rtol=1e-12)
    # a list works too, and df is applied to all elements
    assert_allclose(ind.power(0.5, 50, 0.05, ratio=list(RATIOS), **kwds), res)
    res_df = ind.power(0.5, 50, 0.05, ratio=RATIOS, df=30, **kwds)
    expected = [ind.power(0.5, 50, 0.05, ratio=r, df=30, **kwds) for r in RATIOS]
    assert_allclose(res_df, expected, rtol=1e-12)


def test_ttest_ind_power_ratio_array_positive():
    # array ratio without zero works as before the ratio=0 support
    ind = smp.TTestIndPower()
    ratios = np.array([0.5, 1.0, 2.0])
    res = ind.power(0.5, 50, 0.05, ratio=ratios)
    # R 4.6.1, pwr 1.3-0: pwr.t2n.test(n1 = 50, n2 = c(25, 50, 100), d = 0.5)
    # the noncentral t distribution of SciPy on WASM is accurate to about 1e-8
    assert_allclose(res, [0.521721300128, 0.696893405534, 0.818063361118], rtol=1e-7)
    assert_array_equal(res, [ind.power(0.5, 50, 0.05, ratio=r) for r in ratios])


def test_ttest_ind_power_ratio_broadcast():
    ind = smp.TTestIndPower()
    ratios = np.array([0.0, 1.0, 2.0])
    effect_size = np.array([[0.2], [0.5]])
    res = ind.power(effect_size, 50, 0.05, ratio=ratios)
    assert res.shape == (2, 3)
    for i, es in enumerate([0.2, 0.5]):
        expected = [ind.power(es, 50, 0.05, ratio=r) for r in ratios]
        assert_allclose(res[i], expected, rtol=1e-12)

    nobs1 = np.array([20, 50, 100])
    res = ind.power(0.5, nobs1, 0.05, ratio=ratios)
    assert res.shape == (3,)
    expected = [
        ind.power(0.5, n, 0.05, ratio=r) for n, r in zip(nobs1, ratios, strict=True)
    ]
    assert_allclose(res, expected, rtol=1e-12)


def test_ttest_ind_power_ratio_series():
    ind = smp.TTestIndPower()
    ratios = pd.Series(RATIOS, index=list("abc"))
    res = ind.power(0.5, 50, 0.05, ratio=ratios)
    assert_allclose(res, POWER_RATIO_R["two-sided"], rtol=1e-6)


@pytest.mark.skipif(
    PD_LT_2, reason="pandas 1.x cannot apply a SciPy ufunc to a Series and an array"
)
def test_ttest_ind_power_ratio_series_result():
    ind = smp.TTestIndPower()
    ratios = pd.Series(RATIOS, index=list("abc"))
    # without a zero the Series input keeps working as before, with a Series
    # as result
    res = ind.power(0.5, 50, 0.05, ratio=ratios.iloc[1:])
    assert isinstance(res, pd.Series)
    assert_allclose(res.to_numpy(), POWER_RATIO_R["two-sided"][1:], rtol=1e-6)
    assert list(res.index) == ["b", "c"]


@pytest.mark.parametrize("bad", [-1, -1e-10, np.nan, [0.5, -1.0], [1.0, np.nan]])
def test_ttest_ind_ratio_invalid(bad):
    ind = smp.TTestIndPower()
    with pytest.raises(ValueError, match="ratio must be non-negative"):
        ind.power(0.5, 50, 0.05, ratio=bad)
    with pytest.raises(ValueError, match="ratio must be non-negative"):
        ind.power(0.5, 50, 0.05, ratio=np.asarray(bad))
    # solve_power validates before the root finding, which would swallow the
    # error and return nan with a ConvergenceWarning
    for args in [
        (0.5, None, 0.05, 0.8),
        (None, 50, 0.05, 0.8),
        (0.5, 50, None, 0.8),
        (0.5, 50, 0.05, None),
    ]:
        with pytest.raises(ValueError, match="ratio must be non-negative"):
            ind.solve_power(*args, ratio=bad)


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (smp.TTestPower, {"effect_size": 0.3, "nobs": 50}),
        (smp.NormalIndPower, {"effect_size": 0.3, "nobs1": 50}),
        (smp.FTestPower, {"effect_size": 0.3, "df_num": 3, "df_denom": 57}),
        (smp.FTestPowerF2, {"effect_size": 0.3, "df_num": 3, "df_denom": 57}),
    ],
)
def test_power_class_invalid_inputs_raises(cls, kwargs):
    # alpha outside (0, 1) previously returned power > 1 or nan silently
    with pytest.raises(ValueError, match="alpha must be in the range"):
        cls().power(alpha=2, **kwargs)
    if cls in (smp.FTestPower, smp.FTestPowerF2):
        with pytest.raises(ValueError, match="df_num and df_denom must be positive"):
            cls().power(effect_size=0.3, df_num=-1, df_denom=57, alpha=0.05)
        with pytest.raises(ValueError, match="df_num and df_denom must be positive"):
            cls().power(effect_size=0.3, df_num=3, df_denom=0, alpha=0.05)
    p = cls().power(alpha=0.05, **kwargs)
    assert np.isfinite(p)
    assert 0 < p < 1


POWER_ALPHA = [
    (smp.TTestPower().power, {"effect_size": 0.3, "nobs": 50}),
    (smp.TTestIndPower().power, {"effect_size": 0.3, "nobs1": 50}),
    (smp.NormalIndPower().power, {"effect_size": 0.3, "nobs1": 50}),
    (smp.FTestPower().power, {"effect_size": 0.3, "df_num": 3, "df_denom": 57}),
    (smp.FTestPowerF2().power, {"effect_size": 0.3, "df_num": 3, "df_denom": 57}),
    (smp.FTestAnovaPower().power, {"effect_size": 0.3, "nobs": 60, "k_groups": 3}),
    (smp.ttest_power, {"effect_size": 0.3, "nobs": 50}),
    (smp.normal_power, {"effect_size": 0.3, "nobs": 50}),
    (smp.normal_power_het, {"diff": 0.3, "nobs": 50}),
    (smp.ftest_power, {"effect_size": 0.3, "df2": 57, "df1": 3}),
    (smp.ftest_power_f2, {"effect_size": 0.3, "df_num": 3, "df_denom": 57}),
    (smp.ftest_anova_power, {"effect_size": 0.3, "nobs": 60}),
]
POWER_ALPHA_IDS = [
    "TTestPower",
    "TTestIndPower",
    "NormalIndPower",
    "FTestPower",
    "FTestPowerF2",
    "FTestAnovaPower",
    "ttest_power",
    "normal_power",
    "normal_power_het",
    "ftest_power",
    "ftest_power_f2",
    "ftest_anova_power",
]


@pytest.mark.parametrize("func, kwargs", POWER_ALPHA, ids=POWER_ALPHA_IDS)
def test_power_alpha_range(func, kwargs):
    # alpha outside (0, 1) was only checked by four of the classes: 0 and 1
    # gave a power of 0 and 1, alpha above 1 a power above 1 and a negative
    # alpha nan
    for alpha in [0, 1, 1.5, -0.1, np.nan, [0.05, 1.5], [np.nan, 0.05]]:
        with pytest.raises(ValueError, match="alpha must be in the range"):
            func(alpha=alpha, **kwargs)
    power = func(alpha=np.array([0.01, 0.05, 0.1]), **kwargs)
    assert power.shape == (3,)
    assert np.all(np.diff(power) > 0)
    assert_allclose(power[1], func(alpha=0.05, **kwargs), rtol=1e-12)


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (smp.TTestPower, {"effect_size": 0.5, "nobs": 30, "power": 0.8}),
        (smp.TTestIndPower, {"effect_size": 0.5, "nobs1": 30, "power": 0.8}),
        (smp.NormalIndPower, {"effect_size": 0.5, "nobs1": 30, "power": 0.8}),
        (
            smp.FTestPower,
            {"effect_size": 0.3, "df_num": 3, "df_denom": 57, "power": 0.5},
        ),
        (
            smp.FTestPowerF2,
            {"effect_size": 0.1, "df_num": 3, "df_denom": 57, "power": 0.5},
        ),
        (smp.FTestAnovaPower, {"effect_size": 0.25, "nobs": 60, "power": 0.8}),
    ],
)
def test_solve_power_for_alpha(cls, kwargs):
    # solving for alpha evaluates the power function while searching, and
    # the check of alpha must not stand in the way
    alpha = cls().solve_power(alpha=None, **kwargs)
    assert 0 < alpha < 1
    kwds = dict(kwargs)
    power = kwds.pop("power")
    assert_allclose(cls().power(alpha=alpha, **kwds), power, rtol=1e-5)


def test_ttest_power_far_tail_not_nan():
    # special.nctdtr can return nan far in the tails (scipy gh-25470);
    # the power of an ordinary test must not become nan
    pow_ = smp.TTestPower().power(0.3, 1000, 0.05)
    assert np.isfinite(pow_)
    assert_allclose(pow_, 1, atol=1e-10)
    # R 4.6.1, pwr 1.3-0, pwr.t.test(n=1237, d=0.5, type="two.sample")$power
    pow_ = smp.TTestIndPower().power(0.5, 1237, 0.05)
    assert_allclose(pow_, 1, atol=1e-10)


def test_ttest_power_small_alpha_not_nan():
    # both the cdf and its reflection were nan in the lower tail for alpha from
    # about 1e-12 to 1e-5 in this example (previously 536 of 2998 alphas)
    alpha = np.logspace(-12, -0.001, 2998)
    power = np.array([smp.TTestPower().power(1.0, 50, a) for a in alpha])
    assert np.all(np.isfinite(power))
    assert np.all(np.diff(power) > -1e-12)
    # R 4.6.1, pwr 1.3-0
    # pwr.t.test(n=50, d=1, sig.level=alpha, type="one.sample")$power
    alpha = [1e-12, 1e-10, 1e-8, 1e-6, 1e-5]
    expected = [
        0.0405096719210527,
        0.204604203411695,
        0.573291656181119,
        0.905732544135957,
        0.973852570784929,
    ]
    power = [smp.TTestPower().power(1.0, 50, a) for a in alpha]
    assert_allclose(power, expected, rtol=POWER_RTOL)


@pytest.mark.parametrize("nobs1", [685, 1237, 1833])
def test_solve_effect_size_smaller(nobs1):
    # by symmetry the 'smaller' solution is minus the 'larger' solution
    p = smp.TTestIndPower()
    kwds = {"nobs1": nobs1, "alpha": 0.05, "power": 0.8}
    es_larger = p.solve_power(alternative="larger", **kwds)
    es_smaller = p.solve_power(alternative="smaller", **kwds)
    assert_allclose(es_smaller, -es_larger, rtol=1e-5)


class _NanPower(smp.TTestIndPower):
    """Power function that is nan for effect sizes above ``nan_above``"""

    nan_above = 0.4

    def power(
        self, effect_size, nobs1, alpha, ratio=1, df=None, alternative="two-sided"
    ):
        if effect_size > self.nan_above:
            return np.nan
        return super().power(
            effect_size, nobs1, alpha, ratio=ratio, df=df, alternative=alternative
        )


def test_solve_power_rejects_jump_as_root():
    # a power function that is nan on part of the bracket must not produce
    # a "converged" root at the nan boundary
    kwds = {"nobs1": 1237, "alpha": 0.05, "power": 0.8, "alternative": "smaller"}
    with pytest.warns(ConvergenceWarning):
        es = _NanPower().solve_power(**kwds)
    assert np.isnan(es)


def test_solve_power_nan_region_with_root():
    # same as above, but the nan region leaves the root of the equation
    # R 4.6.1, pwr 1.3-0, solution for 'larger'
    # pwr.t.test(n=1237, power=0.8, sig.level=0.05, d=NULL, type="two.sample",
    #            alternative="greater")$d = 0.100007522036773
    class NanPowerFar(_NanPower):
        nan_above = 2.0

    kwds = {"nobs1": 1237, "alpha": 0.05, "power": 0.8, "alternative": "smaller"}
    es = NanPowerFar().solve_power(**kwds)
    assert_allclose(es, -0.100007522036773, atol=1e-5)


@pytest.mark.parametrize(
    "effect_size, nobs, expected",
    [
        (1.0, 50, 1.55870183959859e-07),
        (0.5, 200, 3.23943954424975e-09),
    ],
)
def test_solve_power_alpha_small_root(effect_size, nobs, expected):
    # the roots are far below the default x tolerance of brentq
    # R 4.6.1, pwr 1.3-0, uniroot (tol=1e-14) on log(sig.level) of
    # pwr.t.test(n=nobs, d=effect_size, sig.level=exp(log_alpha),
    #            type="one.sample")$power - 0.8
    alpha = smp.TTestPower().solve_power(
        effect_size=effect_size, nobs=nobs, power=0.8, alpha=None
    )
    assert_allclose(alpha, expected, rtol=1e-3)


def test_solve_power_alpha_no_root_nan():
    # The power of the 'smaller' test for a positive effect size stays below
    # 0.5 in the whole search range of alpha, so there is no solution.
    # R 4.6.1, pwr 1.3-0, pwr.t.test(n=1500, d=0.2, sig.level=0.999999,
    #                                type="one.sample", alternative="less")
    # has power 0.0015
    with pytest.warns(ConvergenceWarning):
        alpha = smp.TTestPower().solve_power(
            effect_size=0.2, nobs=1500, power=0.5, alpha=None, alternative="smaller"
        )
    assert np.isnan(alpha)


@pytest.mark.parametrize(
    "x, expected",
    [
        (-1.96, 1.391293061255217e-30),
        (1.96, 2.737100334771259e-14),
        ([-1.96, 1.96], [1.391293061255217e-30, 2.737100334771259e-14]),
    ],
)
def test_nct_cdf_tail_fallback_scalar_and_array(x, expected):
    # Fixed references from conditional-normal / chi-squared integration
    # (60 and 85 decimal digits) avoid platform-dependent SciPy NaNs.
    # The absolute tolerance covers cancellation in the reflected tail.
    assert_allclose(smp.nct_cdf(x, 999, 0.3 * np.sqrt(1000)), expected, atol=1e-12)


def test_nct_cdf_sf_far_tail_limit():
    # CDFLIB returns nan for the cdf and for its reflection here. The exact
    # probabilities (mpmath integration over the chi-squared density at 50
    # digits) are 8.6e-35 and 2.3e-38; R's pt() is only accurate to about
    # 1e-13 in this tail, pt(-9.52, 49, ncp=sqrt(50)) = 6.7e-14.
    nc = np.sqrt(50)
    assert_allclose(smp.nct_cdf(-7.75, 49, nc), 0, atol=1e-12)
    assert_allclose(smp.nct_cdf(-9.52, 49, nc), 0, atol=1e-12)
    assert_allclose(smp.nct_cdf(9.52, 49, -nc), 1, atol=1e-12)
    assert_allclose(smp.nct_sf(9.52, 49, -nc), 0, atol=1e-12)
    assert_allclose(smp.nct_sf(-9.52, 49, nc), 1, atol=1e-12)
    # an array with nan and finite entries, the finite entries are unchanged;
    # exact cdf(5; 49, sqrt(50)) = 0.030847613410683136
    res = smp.nct_cdf(np.array([-9.52, 5.0]), 49, nc)
    assert_allclose(res, [0, 0.030847613410683136], rtol=NCT_RTOL, atol=1e-12)
    # invalid arguments stay nan
    assert np.isnan(smp.nct_cdf(np.nan, 49, nc))
    assert np.isnan(smp.nct_cdf(-9.52, -1, nc))


def test_nct_cdf_sf_keep_pandas_type():
    nc = np.sqrt(50)
    x = pd.Series([-9.52, 5.0], index=["a", "b"])
    for func in (smp.nct_cdf, smp.nct_sf):
        # with and without nan from CDFLIB in the input
        for xs in (x, x.iloc[1:]):
            res = func(xs, 49, nc)
            assert isinstance(res, pd.Series)
            assert_array_equal(res.index, xs.index)
    assert_allclose(
        smp.nct_cdf(x, 49, nc), [0, 0.030847613410683136], rtol=NCT_RTOL, atol=1e-12
    )


def test_nct_sf_upper_tail_relative_accuracy():
    # exact value from mpmath (50 digits) as the integral over the
    # chi-squared density of Phi(nc - x * sqrt(v / df)); 1 - cdf loses all but
    # about 6 digits here
    nc = 0.2 * np.sqrt(30)
    atol = 1e-10 if PYTHON_IMPL_WASM or SP_LT_115 else 0
    assert_allclose(
        smp.nct_sf(12.0, 29, nc), 7.6017385833974256e-11, rtol=1e-10, atol=atol
    )
