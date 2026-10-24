"""

Created on Wed Mar 28 15:34:18 2012

Author: Josef Perktold
"""

from statsmodels.compat.python import asbytes

from io import BytesIO
import warnings

import numpy as np
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_almost_equal,
    assert_equal,
)
import pandas as pd
import pytest
from scipy import stats

from statsmodels.sandbox.stats.multicomp import _smm_ppf, _smm_sf
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multicomp import MultiComparison, pairwise_tukeyhsd, tukeyhsd

ss = """\
  43.9  1   1
  39.0  1   2
  46.7  1   3
  43.8  1   4
  44.2  1   5
  47.7  1   6
  43.6  1   7
  38.9  1   8
  43.6  1   9
  40.0  1  10
  89.8  2   1
  87.1  2   2
  92.7  2   3
  90.6  2   4
  87.7  2   5
  92.4  2   6
  86.1  2   7
  88.1  2   8
  90.8  2   9
  89.1  2  10
  68.4  3   1
  69.3  3   2
  68.5  3   3
  66.4  3   4
  70.0  3   5
  68.1  3   6
  70.6  3   7
  65.2  3   8
  63.8  3   9
  69.2  3  10
  36.2  4   1
  45.2  4   2
  40.7  4   3
  40.5  4   4
  39.3  4   5
  40.3  4   6
  43.2  4   7
  38.7  4   8
  40.9  4   9
  39.7  4  10"""

# idx   Treatment StressReduction
ss2 = """\
1     mental               2
2     mental               2
3     mental               3
4     mental               4
5     mental               4
6     mental               5
7     mental               3
8     mental               4
9     mental               4
10    mental               4
11  physical               4
12  physical               4
13  physical               3
14  physical               5
15  physical               4
16  physical               1
17  physical               1
18  physical               2
19  physical               3
20  physical               3
21   medical               1
22   medical               2
23   medical               2
24   medical               2
25   medical               3
26   medical               2
27   medical               3
28   medical               1
29   medical               3
30   medical               1"""

ss3 = """\
1 24.5
1 23.5
1 26.4
1 27.1
1 29.9
2 28.4
2 34.2
2 29.5
2 32.2
2 30.1
3 26.1
3 28.3
3 24.3
3 26.2
3 27.8"""

ss5 = """\
2 - 3\t4.340\t0.691\t7.989\t***
2 - 1\t4.600\t0.951\t8.249\t***
3 - 2\t-4.340\t-7.989\t-0.691\t***
3 - 1\t0.260\t-3.389\t3.909\t-
1 - 2\t-4.600\t-8.249\t-0.951\t***
1 - 3\t-0.260\t-3.909\t3.389\t-
"""

# result in R: library(rstatix)
# games_howell_test(df, StressReduction ~ Treatment, conf.level = 0.99)
ss2_unequal = """\
1\tStressReduction\tmedical\tmental\t1.8888888888888888\t0.7123347940930316\t3.0654429836847461\t0.000196\t***
2\tStressReduction\tmedical\tphysical\t0.8888888888888888\t-0.8105797509636128\t2.5883575287413905\t0.206000\tns
3\tStressReduction\tmental\tphysical\t-1.0000000000000000\t-2.6647460755237473\t0.6647460755237473\t0.127000\tns
"""

cylinders = np.array(
    [
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        4,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        4,
        6,
        6,
        6,
        4,
        4,
        4,
        4,
        4,
        4,
        6,
        8,
        8,
        8,
        8,
        4,
        4,
        4,
        4,
        8,
        8,
        8,
        8,
        6,
        6,
        6,
        6,
        4,
        4,
        4,
        4,
        6,
        6,
        6,
        6,
        4,
        4,
        4,
        4,
        4,
        8,
        4,
        6,
        6,
        8,
        8,
        8,
        8,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        6,
        6,
        4,
        6,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
    ]
)
cyl_labels = np.array(
    [
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "France",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "Japan",
        "USA",
        "USA",
        "USA",
        "Japan",
        "Germany",
        "France",
        "Germany",
        "Sweden",
        "Germany",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "Germany",
        "USA",
        "USA",
        "France",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "Germany",
        "Japan",
        "USA",
        "USA",
        "USA",
        "USA",
        "Germany",
        "Japan",
        "Japan",
        "USA",
        "Sweden",
        "USA",
        "France",
        "Japan",
        "Germany",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "USA",
        "Germany",
        "Japan",
        "Japan",
        "USA",
        "USA",
        "Japan",
        "Japan",
        "Japan",
        "Japan",
        "Japan",
        "Japan",
        "USA",
        "USA",
        "USA",
        "USA",
        "Japan",
        "USA",
        "USA",
        "USA",
        "Germany",
        "USA",
        "USA",
        "USA",
    ]
)

# accommodate recfromtxt for python 3.2, requires bytes
ss = asbytes(ss)
ss2 = asbytes(ss2)
ss3 = asbytes(ss3)
ss5 = asbytes(ss5)
ss2_unequal = asbytes(ss2_unequal)

dta = pd.read_csv(BytesIO(ss), sep=r"\s+", header=None, engine="python")
dta.columns = "Rust", "Brand", "Replication"
dta2 = pd.read_csv(BytesIO(ss2), sep=r"\s+", header=None, engine="python")
dta2.columns = "idx", "Treatment", "StressReduction"
dta2["Treatment"] = dta2["Treatment"].map(lambda v: v.encode("utf-8"))
dta3 = pd.read_csv(BytesIO(ss3), sep=r"\s+", header=None, engine="python")
dta3.columns = ["Brand", "Relief"]
dta5 = pd.read_csv(BytesIO(ss5), sep=r"\t", header=None, engine="python")
dta5.columns = ["pair", "mean", "lower", "upper", "sig"]
for col in ("pair", "sig"):
    dta5[col] = dta5[col].map(lambda v: v.encode("utf-8"))
sas_ = dta5.iloc[[1, 3, 2]]
games_howell_r_result = pd.read_csv(
    BytesIO(ss2_unequal), sep=r"\t", header=None, engine="python"
)
games_howell_r_result.columns = [
    "idx",
    "y",
    "group1",
    "group2",
    "meandiff",
    "lower",
    "upper",
    "pvalue",
    "sig",
]
for col in ("y", "group1", "group2", "sig"):
    games_howell_r_result[col] = games_howell_r_result[col].map(
        lambda v: v.encode("utf-8")
    )


def get_thsd(mci, alpha=0.05):
    var_ = np.var(mci.groupstats.groupdemean(), ddof=len(mci.groupsunique))
    means = mci.groupstats.groupmean
    nobs = mci.groupstats.groupnobs
    resi = tukeyhsd(
        means,
        nobs,
        var_,
        df=None,
        alpha=alpha,
        q_crit=qsturng(1 - alpha, len(means), (nobs - 1).sum()),
    )
    # print resi[4]
    var2 = (mci.groupstats.groupvarwithin() * (nobs - 1.0)).sum() / (nobs - 1.0).sum()
    # print nobs, (nobs - 1).sum()
    # print mci.groupstats.groupvarwithin()
    assert_almost_equal(var_, var2, decimal=14)
    return resi


class CheckTuckeyHSDMixin:

    @classmethod
    def setup_class_(cls):
        cls.mc = MultiComparison(cls.endog, cls.groups)
        if hasattr(cls, "use_var"):
            cls.res = cls.mc.tukeyhsd(alpha=cls.alpha, use_var=cls.use_var)
        else:
            cls.res = cls.mc.tukeyhsd(alpha=cls.alpha)

    def test_multicomptukey(self):
        assert_almost_equal(self.res.meandiffs, self.meandiff2, decimal=14)
        assert_almost_equal(self.res.confint, self.confint2, decimal=2)
        assert_equal(self.res.reject, self.reject2)

    def test_group_tukey(self):
        if hasattr(self, "use_var") and self.use_var == "unequal":
            # in unequal variance case, we feed groupvarwithin, no need to test total variance
            return
        res_t = get_thsd(self.mc, alpha=self.alpha)
        assert_almost_equal(res_t[4], self.confint2, decimal=2)

    def test_shortcut_function(self):
        # check wrapper function
        if hasattr(self, "use_var"):
            res = pairwise_tukeyhsd(
                self.endog, self.groups, alpha=self.alpha, use_var=self.use_var
            )
        else:
            res = pairwise_tukeyhsd(self.endog, self.groups, alpha=self.alpha)
        assert_almost_equal(res.confint, self.res.confint, decimal=14)

    @pytest.mark.smoke
    @pytest.mark.thread_unsafe(reason="Uses matplotlib")
    @pytest.mark.matplotlib
    def test_plot_simultaneous_ci(self, close_figures):
        self.res._simultaneous_ci()
        reference = self.res.groupsunique[1]
        self.res.plot_simultaneous(comparison_name=reference)


class TestTuckeyHSD2(CheckTuckeyHSDMixin):

    @classmethod
    def setup_class(cls):
        # balanced case
        cls.endog = dta2["StressReduction"]
        cls.groups = dta2["Treatment"]
        cls.alpha = 0.05
        cls.setup_class_()  # in super

        # from R
        tukeyhsd2s = np.array(
            [
                1.5,
                1,
                -0.5,
                0.3214915,
                -0.1785085,
                -1.678509,
                2.678509,
                2.178509,
                0.6785085,
                0.01056279,
                0.1079035,
                0.5513904,
            ]
        ).reshape(3, 4, order="F")
        cls.meandiff2 = tukeyhsd2s[:, 0]
        cls.confint2 = tukeyhsd2s[:, 1:3]
        pvals = tukeyhsd2s[:, 3]
        cls.reject2 = pvals < 0.05

    def test_table_names_default_group_order(self):
        t = self.res._results_table
        # if the group_order parameter is not used, the groups should
        # be reported in alphabetical order
        expected_order = [
            (b"medical", b"mental"),
            (b"medical", b"physical"),
            (b"mental", b"physical"),
        ]
        for i in range(1, 4):
            first_group = t[i][0].data
            second_group = t[i][1].data
            assert_((first_group, second_group) == expected_order[i - 1])

    def test_table_names_custom_group_order(self):
        # if the group_order parameter is used, the groups should
        # be reported in the specified order
        mc = MultiComparison(
            self.endog, self.groups, group_order=[b"physical", b"medical", b"mental"]
        )
        res = mc.tukeyhsd(alpha=self.alpha)
        # print(res)
        t = res._results_table
        expected_order = [
            (b"physical", b"medical"),
            (b"physical", b"mental"),
            (b"medical", b"mental"),
        ]
        for i in range(1, 4):
            first_group = t[i][0].data
            second_group = t[i][1].data
            assert_((first_group, second_group) == expected_order[i - 1])

        frame = res.summary_frame()
        assert_equal(frame["p-adj"], res.pvalues)
        assert_equal(frame["meandiff"], res.meandiffs)
        # Why are we working with binary strings, old time numpy?
        group_t = [b"medical", b"mental", b"mental"]
        group_c = [b"physical", b"physical", b"medical"]
        assert frame["group_t"].to_list() == group_t
        assert frame["group_c"].to_list() == group_c


class TestTuckeyHSD2Pandas(TestTuckeyHSD2):

    @classmethod
    def setup_class(cls):
        super().setup_class()

        cls.endog = pd.Series(cls.endog)
        # we are working with bytes on python 3, not with strings in this case
        cls.groups = pd.Series(cls.groups, dtype=object)

    def test_incorrect_output(self):
        # too few groups
        with pytest.raises(ValueError):
            MultiComparison(np.array([1] * 10), [1, 2] * 4)
        # too many groups
        with pytest.raises(ValueError):
            MultiComparison(np.array([1] * 10), [1, 2] * 6)
        # just one group
        with pytest.raises(ValueError):
            MultiComparison(np.array([1] * 10), [1] * 10)

        # group_order does not select all observations, only one group left
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(ValueError):
                MultiComparison(np.array([1] * 10), [1, 2] * 5, group_order=[1])

        # group_order does not select all observations,
        # we do tukey_hsd with reduced set of observations
        data = np.arange(15)
        groups = np.repeat([1, 2, 3], 5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mod1 = MultiComparison(np.array(data), groups, group_order=[1, 2])
            assert_equal(len(w), 1)
            assert issubclass(w[0].category, UserWarning)

        res1 = mod1.tukeyhsd(alpha=0.01)
        mod2 = MultiComparison(np.array(data[:10]), groups[:10])
        res2 = mod2.tukeyhsd(alpha=0.01)

        attributes = [
            "confint",
            "data",
            "df_total",
            "groups",
            "groupsunique",
            "meandiffs",
            "q_crit",
            "reject",
            "reject2",
            "std_pairs",
            "variance",
        ]
        for att in attributes:
            err_msg = att + "failed"
            assert_allclose(
                getattr(res1, att), getattr(res2, att), rtol=1e-14, err_msg=err_msg
            )

        attributes = [
            "data",
            "datali",
            "groupintlab",
            "groups",
            "groupsunique",
            "ngroups",
            "nobs",
            "pairindices",
        ]
        for att in attributes:
            err_msg = att + "failed"
            assert_allclose(
                getattr(mod1, att), getattr(mod2, att), rtol=1e-14, err_msg=err_msg
            )


class TestTuckeyHSD2s(CheckTuckeyHSDMixin):
    @classmethod
    def setup_class(cls):
        # unbalanced case
        cls.endog = dta2["StressReduction"][3:29]
        cls.groups = dta2["Treatment"][3:29]
        cls.alpha = 0.01
        cls.setup_class_()

        # from R
        tukeyhsd2s = np.array(
            [
                1.8888888888888889,
                0.888888888888889,
                -1,
                0.2658549,
                -0.5908785,
                -2.587133,
                3.511923,
                2.368656,
                0.5871331,
                0.002837638,
                0.150456,
                0.1266072,
            ]
        ).reshape(3, 4, order="F")
        cls.meandiff2 = tukeyhsd2s[:, 0]
        cls.confint2 = tukeyhsd2s[:, 1:3]
        pvals = tukeyhsd2s[:, 3]
        cls.reject2 = pvals < 0.01


class TestTukeyHSD2sUnequal(CheckTuckeyHSDMixin):

    @classmethod
    def setup_class(cls):
        # Games-Howell test
        cls.endog = dta2["StressReduction"][3:29]
        cls.groups = dta2["Treatment"][3:29]
        cls.alpha = 0.01
        cls.use_var = "unequal"
        cls.setup_class_()

        # from R: library(rstatix)
        cls.meandiff2 = games_howell_r_result["meandiff"]
        cls.confint2 = (
            games_howell_r_result[["lower", "upper"]]
            .astype(float)
            .values.reshape((3, 2))
        )
        cls.reject2 = games_howell_r_result["sig"] == asbytes("***")


class TestTuckeyHSD3(CheckTuckeyHSDMixin):

    @classmethod
    def setup_class(cls):
        # SAS case
        cls.endog = dta3["Relief"]
        cls.groups = dta3["Brand"]
        cls.alpha = 0.05
        cls.setup_class_()
        # super(cls, cls).setup_class_()
        # CheckTuckeyHSD.setup_class_()
        cls.meandiff2 = sas_["mean"]
        cls.confint2 = sas_[["lower", "upper"]].astype(float).values.reshape((3, 2))
        cls.reject2 = sas_["sig"] == asbytes("***")


class TestTuckeyHSD4(CheckTuckeyHSDMixin):

    @classmethod
    def setup_class(cls):
        # unbalanced case verified in Matlab
        cls.endog = cylinders
        cls.groups = cyl_labels
        cls.alpha = 0.05
        cls.setup_class_()
        cls.res._simultaneous_ci()

        # from Matlab
        cls.halfwidth2 = np.array(
            [
                1.5228335685980883,
                0.9794949704444682,
                0.78673802805533644,
                2.3321237694566364,
                0.57355135882752939,
            ]
        )
        cls.meandiff2 = np.array(
            [
                0.22222222222222232,
                0.13333333333333375,
                0.0,
                2.2898550724637685,
                -0.088888888888888573,
                -0.22222222222222232,
                2.0676328502415462,
                -0.13333333333333375,
                2.1565217391304348,
                2.2898550724637685,
            ]
        )
        cls.confint2 = np.array(
            [
                -2.32022210717,
                2.76466655161,
                -2.247517583,
                2.51418424967,
                -3.66405224956,
                3.66405224956,
                0.113960166573,
                4.46574997835,
                -1.87278583908,
                1.6950080613,
                -3.529655688,
                3.08521124356,
                0.568180988881,
                3.5670847116,
                -3.31822643175,
                3.05155976508,
                0.951206924521,
                3.36183655374,
                -0.74487911754,
                5.32458926247,
            ]
        ).reshape(10, 2)
        cls.reject2 = np.array(
            [False, False, False, True, False, False, True, False, True, False]
        )

    def test_hochberg_intervals(self):
        assert_almost_equal(self.res.halfwidths, self.halfwidth2, 4)


@pytest.mark.smoke
@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
def test_plot(close_figures):
    # SMOKE test
    cylinders_adj = cylinders.astype(float)
    # avoid zero division, zero within variance in France and Sweden
    cylinders_adj[[10, 28]] += 0.05
    alpha = 0.05
    mc = MultiComparison(cylinders_adj, cyl_labels)
    resth = mc.tukeyhsd(alpha=alpha, use_var="equal")
    resgh = mc.tukeyhsd(alpha=alpha, use_var="unequal")
    resth.plot_simultaneous()
    resgh.plot_simultaneous()


def test_tukeyhsd_invalid_use_var_raises():
    cylinders_adj = cylinders.astype(float)
    cylinders_adj[[10, 28]] += 0.05
    mc = MultiComparison(cylinders_adj, cyl_labels)
    with pytest.raises(ValueError, match="use_var"):
        mc.tukeyhsd(use_var="not-a-use-var")


# R 4.5.3, studentized maximum modulus survival function by integration
# psmm_sf <- function(q, k, df) {
#   tail <- function(s) -expm1(k * log1p(-2 * pnorm(q * s, lower.tail = FALSE)))
#   f <- function(s) tail(s) * dchisq(df * s^2, df) * 2 * df * s
#   integrate(f, 0, Inf, rel.tol = 1e-13, subdivisions = 1000L)$value
# }
# for (df in c(2, 5, 30, 300, 5000)) for (q in c(1.5, 4)) for (k in c(3, 21))
#   cat(sprintf("%g %g %g %.15g\n", q, k, df, psmm_sf(q, k, df)))
smm_sf_r = [
    (1.5, 3, 2, 0.508810867799348),
    (1.5, 21, 2, 0.846709245672448),
    (4, 3, 2, 0.117766849906116),
    (4, 21, 2, 0.261671024330229),
    (1.5, 3, 5, 0.431129024650245),
    (1.5, 21, 5, 0.878607390565468),
    (4, 3, 5, 0.027143612822165),
    (4, 21, 5, 0.104411622893358),
    (1.5, 3, 30, 0.366070706664739),
    (1.5, 21, 30, 0.931414448492557),
    (4, 3, 30, 0.0011416160848127),
    (4, 21, 30, 0.00776118838471536),
    (1.5, 3, 300, 0.351377929932101),
    (1.5, 21, 300, 0.948625408146656),
    (4, 3, 300, 0.000239490720530816),
    (4, 21, 300, 0.00167458420323826),
    (1.5, 3, 5000, 0.349773087975445),
    (1.5, 21, 5000, 0.950670939209224),
    (4, 3, 5000, 0.000192758805816826),
    (4, 21, 5000, 0.00134850917116419),
]


@pytest.mark.parametrize("q, k, df, expected", smm_sf_r)
def test_smm_sf_r(q, k, df, expected):
    assert_allclose(_smm_sf(q, k, df), expected, rtol=1e-9)


def test_smm_sf_pmvt():
    # R 4.5.3, mvtnorm 1.4.2, multivariate t with uncorrelated components
    # pmvt(lower = rep(-q, k), upper = rep(q, k), df = df, corr = diag(k),
    #      algorithm = GenzBretz(maxpts = 2e6, abseps = 1e-9, releps = 0))
    q = np.array([2.5, 3.0, 2.0, 3.5, 4.2])
    k = [3, 6, 1, 10, 15]
    df = [10, 5, 7, 20, 3]
    cdf = [0.9136858617, 0.8713328011, 0.9143806714, 0.9787528378, 0.8583708778]
    for i in range(len(q)):
        assert_allclose(1 - _smm_sf(q[i], k[i], df[i]), cdf[i], atol=1e-6)


def test_smm_special_cases():
    q = np.array([0.5, 2.0, 5.0, 9.0])
    df = np.array([3.0, 12.7, 40.0, 1000.0])
    # one variable is the absolute value of a t random variable
    assert_allclose(_smm_sf(q, 1, df), 2 * stats.t.sf(q, df), rtol=1e-9)
    # infinite df, independent standard normal variables
    expected = (1 - 2 * stats.norm.sf(q)) ** 4
    assert_allclose(1 - _smm_sf(q, 4, np.inf), expected, rtol=1e-12)
    # in the extreme tail the Bonferroni bound is tight
    assert_allclose(_smm_sf(8.0, 21, 5000), 21 * 2 * stats.t.sf(8.0, 5000), rtol=1e-6)
    assert_equal(_smm_sf(0.0, 3, 10), 1.0)
    assert np.isnan(_smm_sf(np.nan, 3, 10))
    # quantile function inverts the survival function
    qcrit = _smm_ppf(0.95, 6, df)
    assert_allclose(_smm_sf(qcrit, 6, df), 0.05, rtol=1e-8)
    assert_allclose(_smm_ppf(0.95, 1, df), stats.t.isf(0.025, df), rtol=1e-12)


class TestDunnettT3:
    # R 4.5.3, data are dta2.iloc[3:29], x is StressReduction, g Treatment
    # ni <- tapply(x, g, length); xi <- tapply(x, g, mean)
    # s2i <- tapply(x, g, var); m <- 3
    # for pairs (mental, medical), (physical, medical), (physical, mental)
    #   A <- s2i[i] / ni[i] + s2i[j] / ni[j]
    #   t <- (xi[i] - xi[j]) / sqrt(A)
    #   df <- A^2 / (s2i[i]^2 / (ni[i]^2 * (ni[i] - 1))
    #                + s2i[j]^2 / (ni[j]^2 * (ni[j] - 1)))
    #   p <- psmm_sf(abs(t), m, df)
    #   q <- uniroot(function(q) psmm_sf(q, m, df) - alpha, c(0.5, 50),
    #                tol = 1e-13)$root
    #   ci <- xi[i] - xi[j] + c(-1, 1) * q * sqrt(A)
    meandiffs = [1.88888888888889, 0.888888888888889, -1]
    tvalues = [5.55747210368222, 1.79334335864888, -2.10632849249823]
    df = [13.9847203830091, 14.7647853847229, 13.0614653807399]
    pvalues = [0.000210597265165326, 0.244853628908946, 0.149387917798748]
    q_crit = {
        0.05: [2.6914129859905, 2.67393276823228, 2.71506658039917],
        0.01: [3.51700190538227, 3.48172827852556, 3.56505594289967],
    }
    confint = {
        0.05: [
            [0.974124048003703, 2.80365372977407],
            [-0.436473103757241, 2.21425088153502],
            [-2.28900434574616, 0.28900434574616],
        ],
        0.01: [
            [0.693520617606698, 3.08425716017108],
            [-0.836865170413678, 2.61464294819146],
            [-2.69254508762368, 0.692545087623679],
        ],
    }

    @classmethod
    def setup_class(cls):
        data = dta2.iloc[3:29]
        cls.mc = MultiComparison(data["StressReduction"], data["Treatment"])

    @pytest.mark.parametrize("alpha", [0.05, 0.01])
    def test_r(self, alpha):
        res = self.mc.dunnett_t3(alpha=alpha)
        assert_allclose(res.meandiffs, self.meandiffs, rtol=1e-12)
        assert_allclose(res.meandiffs / res.std_pairs, self.tvalues, rtol=1e-12)
        assert_allclose(res.df_total, self.df, rtol=1e-12)
        assert_allclose(res.pvalues, self.pvalues, rtol=1e-8)
        assert_allclose(res.q_crit, self.q_crit[alpha], rtol=1e-8)
        assert_allclose(res.confint, self.confint[alpha], rtol=1e-8)
        assert_equal(res.reject, [True, False, False])
        assert_equal(res.reject, res.pvalues < alpha)
        assert_equal(res.alpha, alpha)

    def test_pmcmrplus(self):
        # R 4.5.3, PMCMRplus 1.9.12, set.seed(1); dunnettT3Test(x, g)
        # PMCMRplus rounds the degrees of freedom and pmvt has an absolute
        # error of about 1e-3 with the default settings
        pvalues = [0.000162464294426234, 0.244207785973616920, 0.149537580996924]
        res = self.mc.dunnett_t3()
        tvalues = res.meandiffs / res.std_pairs
        pval_round = _smm_sf(np.abs(tvalues), 3, np.round(res.df_total))
        assert_allclose(pval_round, pvalues, atol=1e-3)

    def test_summary(self):
        res = self.mc.dunnett_t3()
        assert "Dunnett T3, FWER=0.05" in str(res.summary())
        frame = res.summary_frame()
        assert_equal(frame["group_t"].tolist(), [b"mental", b"physical", b"physical"])
        assert_equal(frame["group_c"].tolist(), [b"medical", b"medical", b"mental"])
        assert_allclose(frame["p-adj"], self.pvalues, rtol=1e-8)

    def test_alpha_error(self):
        with pytest.raises(ValueError, match="alpha"):
            self.mc.dunnett_t3(alpha=1.5)


def test_dunnett_t3_zero_variance():
    # France and Sweden have zero variance, their comparison is undefined
    mc = MultiComparison(cylinders.astype(float), cyl_labels)
    res = mc.dunnett_t3()
    frame = res.summary_frame()
    undefined = (frame["group_c"] == "France") & (frame["group_t"] == "Sweden")
    assert undefined.sum() == 1
    assert np.isnan(res.pvalues[undefined]).all()
    assert np.isnan(res.confint[undefined]).all()
    assert not res.reject[undefined].any()
    assert np.isfinite(res.pvalues[~undefined]).all()
    assert np.isfinite(res.confint[~undefined]).all()
