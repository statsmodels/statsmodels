from io import StringIO

import numpy as np
import pandas as pd
from pandas import read_csv
import pytest

from statsmodels.formula.api import ols, wls
from statsmodels.stats.anova import anova_lm
from statsmodels.tools.sm_exceptions import SingularMatrixWarning

kidney_table = StringIO("""Days      Duration Weight ID
    0.0      1      1      1
    2.0      1      1      2
    1.0      1      1      3
    3.0      1      1      4
    0.0      1      1      5
    2.0      1      1      6
    0.0      1      1      7
    5.0      1      1      8
    6.0      1      1      9
    8.0      1      1     10
    2.0      1      2      1
    4.0      1      2      2
    7.0      1      2      3
   12.0      1      2      4
   15.0      1      2      5
    4.0      1      2      6
    3.0      1      2      7
    1.0      1      2      8
    5.0      1      2      9
   20.0      1      2     10
   15.0      1      3      1
   10.0      1      3      2
    8.0      1      3      3
    5.0      1      3      4
   25.0      1      3      5
   16.0      1      3      6
    7.0      1      3      7
   30.0      1      3      8
    3.0      1      3      9
   27.0      1      3     10
    0.0      2      1      1
    1.0      2      1      2
    1.0      2      1      3
    0.0      2      1      4
    4.0      2      1      5
    2.0      2      1      6
    7.0      2      1      7
    4.0      2      1      8
    0.0      2      1      9
    3.0      2      1     10
    5.0      2      2      1
    3.0      2      2      2
    2.0      2      2      3
    0.0      2      2      4
    1.0      2      2      5
    1.0      2      2      6
    3.0      2      2      7
    6.0      2      2      8
    7.0      2      2      9
    9.0      2      2     10
   10.0      2      3      1
    8.0      2      3      2
   12.0      2      3      3
    3.0      2      3      4
    7.0      2      3      5
   15.0      2      3      6
    4.0      2      3      7
    9.0      2      3      8
    6.0      2      3      9
    1.0      2      3     10
""")

kidney_table.seek(0)
kidney_table = read_csv(kidney_table, sep=r"\s+", engine="python").astype(int)


class TestAnovaLM:
    @classmethod
    def setup_class(cls):
        # kidney data taken from JT's course
        # do not know the license

        cls.data = kidney_table
        cls.kidney_lm = ols(
            "np.log(Days+1) ~ C(Duration) * C(Weight)", data=cls.data
        ).fit()

    def test_results(self):
        Df = np.array([1, 2, 2, 54])
        sum_sq = np.array([2.339693, 16.97129, 0.6356584, 28.9892])
        # mean_sq = np.array([2.339693, 8.485645, 0.3178292, 0.536837])
        f_value = np.array([4.358293, 15.80674, 0.5920404, np.nan])
        pr_f = np.array([0.0415617, 3.944502e-06, 0.5567479, np.nan])

        results = anova_lm(self.kidney_lm)
        np.testing.assert_equal(results["df"].values, Df)
        np.testing.assert_almost_equal(results["sum_sq"].values, sum_sq, 4)
        np.testing.assert_almost_equal(results["F"].values, f_value, 4)
        np.testing.assert_almost_equal(results["PR(>F)"].values, pr_f)


class TestAnovaLMNoconstant:
    @classmethod
    def setup_class(cls):
        # kidney data taken from JT's course
        # do not know the license

        cls.data = kidney_table
        cls.kidney_lm = ols(
            "np.log(Days+1) ~ C(Duration) * C(Weight) - 1", data=cls.data
        ).fit()

    def test_results(self):
        Df = np.array([2, 2, 2, 54])
        sum_sq = np.array([158.6415227, 16.97129, 0.6356584, 28.9892])
        # mean_sq = np.array([79.3207613, 8.485645, 0.3178292, 0.536837])
        f_value = np.array([147.7557648, 15.80674, 0.5920404, np.nan])
        pr_f = np.array([1.262324e-22, 3.944502e-06, 0.5567479, np.nan])

        results = anova_lm(self.kidney_lm)
        np.testing.assert_equal(results["df"].values, Df)
        np.testing.assert_almost_equal(results["sum_sq"].values, sum_sq, 4)
        np.testing.assert_almost_equal(results["F"].values, f_value, 4)
        np.testing.assert_almost_equal(results["PR(>F)"].values, pr_f)

        # > sum2.lm = lm(logDays ~ Duration * Weight - 1, contrasts=list(Duration=contr.sum, Weight=contr.sum))
        # > anova.lm.sum2 <- anova(sum2.lm)
        # > anova.lm.sum2
        # Analysis of Variance Table
        #
        # Response: logDays
        #                 Df  Sum Sq Mean Sq F value    Pr(>F)
        # Duration         2 158.642  79.321 147.756 < 2.2e-16 ***
        # Weight           2  16.971   8.486  15.807 3.945e-06 ***
        # Duration:Weight  2   0.636   0.318   0.592    0.5567
        # Residuals       54  28.989   0.537
        # ---
        # Signif. codes:  0`***` 0.001`**` 0.01`*` 0.05`.` 0.1` ` 1


class TestAnovaLMCompare(TestAnovaLM):
    def test_results(self):
        new_model = ols("np.log(Days+1) ~ C(Duration) + C(Weight)", self.data).fit()
        results = anova_lm(new_model, self.kidney_lm)

        Res_Df = np.array([56, 54])
        RSS = np.array([29.62486, 28.9892])
        Df = np.array([0, 2])
        Sum_of_Sq = np.array([np.nan, 0.6356584])
        F = np.array([np.nan, 0.5920404])
        PrF = np.array([np.nan, 0.5567479])

        np.testing.assert_equal(results["df_resid"].values, Res_Df)
        np.testing.assert_almost_equal(results["ssr"].values, RSS, 4)
        np.testing.assert_almost_equal(results["df_diff"].values, Df)
        np.testing.assert_almost_equal(results["ss_diff"].values, Sum_of_Sq)
        np.testing.assert_almost_equal(results["F"].values, F)
        np.testing.assert_almost_equal(results["Pr(>F)"].values, PrF)


class TestAnovaLMCompareNoconstant(TestAnovaLM):
    def test_results(self):
        new_model = ols("np.log(Days+1) ~ C(Duration) + C(Weight) - 1", self.data).fit()
        results = anova_lm(new_model, self.kidney_lm)

        Res_Df = np.array([56, 54])
        RSS = np.array([29.62486, 28.9892])
        Df = np.array([0, 2])
        Sum_of_Sq = np.array([np.nan, 0.6356584])
        F = np.array([np.nan, 0.5920404])
        PrF = np.array([np.nan, 0.5567479])

        np.testing.assert_equal(results["df_resid"].values, Res_Df)
        np.testing.assert_almost_equal(results["ssr"].values, RSS, 4)
        np.testing.assert_almost_equal(results["df_diff"].values, Df)
        np.testing.assert_almost_equal(results["ss_diff"].values, Sum_of_Sq)
        np.testing.assert_almost_equal(results["F"].values, F)
        np.testing.assert_almost_equal(results["Pr(>F)"].values, PrF)


class TestAnova2(TestAnovaLM):
    # drop some observations to make an unbalanced, disproportionate panel
    # to make sure things are okay

    def test_results(self):
        data = self.data.drop([0, 1, 2])
        anova_ii = ols("np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)", data).fit()

        Sum_Sq = np.array([3.067066, 13.27205, 0.1905093, 27.60181])
        Df = np.array([1, 2, 2, 51])
        F_value = np.array([5.667033, 12.26141, 0.1760025, np.nan])
        PrF = np.array([0.02106078, 4.487909e-05, 0.8391231, np.nan])

        results = anova_lm(anova_ii, typ="II")
        np.testing.assert_equal(results["df"].values, Df)
        np.testing.assert_almost_equal(results["sum_sq"].values, Sum_Sq, 4)
        np.testing.assert_almost_equal(results["F"].values, F_value, 4)
        np.testing.assert_almost_equal(results["PR(>F)"].values, PrF)


class TestAnova2Noconstant(TestAnovaLM):
    # drop some observations to make an unbalanced, disproportionate panel
    # to make sure things are okay

    def test_results(self):
        data = self.data.drop([0, 1, 2])
        anova_ii = ols(
            "np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum) - 1", data
        ).fit()

        Sum_Sq = np.array([154.7131692, 13.27205, 0.1905093, 27.60181])
        Df = np.array([2, 2, 2, 51])
        F_value = np.array([142.9321191, 12.26141, 0.1760025, np.nan])
        PrF = np.array([1.238624e-21, 4.487909e-05, 0.8391231, np.nan])

        results = anova_lm(anova_ii, typ="II")
        np.testing.assert_equal(results["df"].values, Df)
        np.testing.assert_almost_equal(results["sum_sq"].values, Sum_Sq, 4)
        np.testing.assert_almost_equal(results["F"].values, F_value, 4)
        np.testing.assert_almost_equal(results["PR(>F)"].values, PrF)

        # > sum2.lm.dropped <- lm(logDays ~ Duration * Weight - 1,  dta.dropped,
        #                 contrasts=list(Duration=contr.sum, Weight=contr.sum))
        # > anova.ii.dropped2 <- Anova(sum2.lm.dropped, type='II')
        # > anova.ii.dropped2
        # Anova Table (Type II tests)
        #
        # Response: logDays
        #                  Sum Sq Df F value    Pr(>F)
        # Duration        154.713  2 142.932 < 2.2e-16 ***
        # Weight           13.272  2  12.261 4.488e-05 ***
        # Duration:Weight   0.191  2   0.176    0.8391
        # Residuals        27.602 51


class TestAnova2HC0(TestAnovaLM):
    # NOTE: R does not return SSq with robust covariance. Why?
    # drop some observations to make an unbalanced, disproportionate panel
    # to make sure things are okay

    def test_results(self):
        data = self.data.drop([0, 1, 2])
        anova_ii = ols("np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)", data).fit()

        # Sum_Sq = np.array([
        #      151.4065, 2.904723, 13.45718, 0.1905093, 27.60181
        #     ])

        Df = np.array([1, 2, 2, 51])
        F = np.array([6.972744, 13.7804, 0.1709936, np.nan])
        PrF = np.array([0.01095599, 1.641682e-05, 0.8433081, np.nan])

        results = anova_lm(anova_ii, typ="II", robust="hc0")
        np.testing.assert_equal(results["df"].values, Df)
        # np.testing.assert_almost_equal(results['sum_sq'].values, Sum_Sq, 4)

        np.testing.assert_almost_equal(results["F"].values, F, 4)
        np.testing.assert_almost_equal(results["PR(>F)"].values, PrF)


class TestAnova2HC1(TestAnovaLM):
    # drop some observations to make an unbalanced, disproportionate panel
    # to make sure things are okay

    def test_results(self):
        data = self.data.drop([0, 1, 2])
        anova_ii = ols("np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)", data).fit()

        # Sum_Sq = np.array([
        #      151.4065, 2.904723, 13.45718, 0.1905093, 27.60181
        #     ])

        Df = np.array([1, 2, 2, 51])
        F = np.array([6.238771, 12.32983, 0.1529943, np.nan])
        PrF = np.array([0.01576555, 4.285456e-05, 0.858527, np.nan])

        results = anova_lm(anova_ii, typ="II", robust="hc1")
        np.testing.assert_equal(results["df"].values, Df)
        # np.testing.assert_almost_equal(results['sum_sq'].values, Sum_Sq, 4)

        np.testing.assert_almost_equal(results["F"].values, F, 4)
        np.testing.assert_almost_equal(results["PR(>F)"].values, PrF)


class TestAnova2HC2(TestAnovaLM):
    # drop some observations to make an unbalanced, disproportionate panel
    # to make sure things are okay

    def test_results(self):
        data = self.data.drop([0, 1, 2])
        anova_ii = ols("np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)", data).fit()

        # Sum_Sq = np.array([
        #      151.4065, 2.904723, 13.45718, 0.1905093, 27.60181
        #     ])

        Df = np.array([1, 2, 2, 51])
        F = np.array([6.267499, 12.25354, 0.1501224, np.nan])
        PrF = np.array([0.01554009, 4.511826e-05, 0.8609815, np.nan])

        results = anova_lm(anova_ii, typ="II", robust="hc2")
        np.testing.assert_equal(results["df"].values, Df)
        # np.testing.assert_almost_equal(results['sum_sq'].values, Sum_Sq, 4)

        np.testing.assert_almost_equal(results["F"].values, F, 4)
        np.testing.assert_almost_equal(results["PR(>F)"].values, PrF)


class TestAnova2HC3(TestAnovaLM):
    # drop some observations to make an unbalanced, disproportionate panel
    # to make sure things are okay

    def test_results(self):
        data = self.data.drop([0, 1, 2])
        anova_ii = ols("np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)", data).fit()

        # Sum_Sq = np.array([
        #      151.4065, 2.904723, 13.45718, 0.1905093, 27.60181
        #     ])

        Df = np.array([1, 2, 2, 51])
        F = np.array([5.633786, 10.89842, 0.1317223, np.nan])
        PrF = np.array([0.02142223, 0.0001145965, 0.8768817, np.nan])

        results = anova_lm(anova_ii, typ="II", robust="hc3")
        np.testing.assert_equal(results["df"].values, Df)
        # np.testing.assert_almost_equal(results['sum_sq'].values, Sum_Sq, 4)

        np.testing.assert_almost_equal(results["F"].values, F, 4)
        np.testing.assert_almost_equal(results["PR(>F)"].values, PrF)


class TestAnova3(TestAnovaLM):
    # drop some observations to make an unbalanced, disproportionate panel
    # to make sure things are okay

    def test_results(self):
        data = self.data.drop([0, 1, 2])
        anova_iii = ols("np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)", data).fit()

        Sum_Sq = np.array([151.4065, 2.904723, 13.45718, 0.1905093, 27.60181])
        Df = np.array([1, 1, 2, 2, 51])
        F_value = np.array([279.7545, 5.367071, 12.43245, 0.1760025, np.nan])
        PrF = np.array([2.379855e-22, 0.02457384, 3.999431e-05, 0.8391231, np.nan])

        results = anova_lm(anova_iii, typ="III")
        np.testing.assert_equal(results["df"].values, Df)
        np.testing.assert_almost_equal(results["sum_sq"].values, Sum_Sq, 4)
        np.testing.assert_almost_equal(results["F"].values, F_value, 4)
        np.testing.assert_almost_equal(results["PR(>F)"].values, PrF)


class TestAnova3HC0(TestAnovaLM):
    # NOTE: R does not return SSq with robust covariance. Why?
    # drop some observations to make an unbalanced, disproportionate panel
    # to make sure things are okay

    def test_results(self):
        data = self.data.drop([0, 1, 2])
        anova_iii = ols("np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)", data).fit()

        # Sum_Sq = np.array([
        #      151.4065, 2.904723, 13.45718, 0.1905093, 27.60181
        #     ])

        Df = np.array([1, 1, 2, 2, 51])
        F = np.array([298.3404, 5.723638, 13.76069, 0.1709936, np.nan])
        PrF = np.array([5.876255e-23, 0.02046031, 1.662826e-05, 0.8433081, np.nan])

        results = anova_lm(anova_iii, typ="III", robust="hc0")
        np.testing.assert_equal(results["df"].values, Df)
        # np.testing.assert_almost_equal(results['sum_sq'].values, Sum_Sq, 4)

        np.testing.assert_almost_equal(results["F"].values, F, 4)
        np.testing.assert_almost_equal(results["PR(>F)"].values, PrF)


class TestAnova3HC1(TestAnovaLM):
    # drop some observations to make an unbalanced, disproportionate panel
    # to make sure things are okay

    def test_results(self):
        data = self.data.drop([0, 1, 2])
        anova_iii = ols("np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)", data).fit()

        # Sum_Sq = np.array([
        #      151.4065, 2.904723, 13.45718, 0.1905093, 27.60181
        #     ])

        Df = np.array([1, 1, 2, 2, 51])
        F = np.array([266.9361, 5.12115, 12.3122, 0.1529943, np.nan])
        PrF = np.array([6.54355e-22, 0.02792296, 4.336712e-05, 0.858527, np.nan])

        results = anova_lm(anova_iii, typ="III", robust="hc1")
        np.testing.assert_equal(results["df"].values, Df)
        # np.testing.assert_almost_equal(results['sum_sq'].values, Sum_Sq, 4)

        np.testing.assert_almost_equal(results["F"].values, F, 4)
        np.testing.assert_almost_equal(results["PR(>F)"].values, PrF)


class TestAnova3HC2(TestAnovaLM):
    # drop some observations to make an unbalanced, disproportionate panel
    # to make sure things are okay

    def test_results(self):
        data = self.data.drop([0, 1, 2])
        anova_iii = ols("np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)", data).fit()

        # Sum_Sq = np.array([
        #      151.4065, 2.904723, 13.45718, 0.1905093, 27.60181
        #     ])

        Df = np.array([1, 1, 2, 2, 51])
        F = np.array([264.5137, 5.074677, 12.19158, 0.1501224, np.nan])
        PrF = np.array([7.958286e-22, 0.02860926, 4.704831e-05, 0.8609815, np.nan])

        results = anova_lm(anova_iii, typ="III", robust="hc2")
        np.testing.assert_equal(results["df"].values, Df)
        # np.testing.assert_almost_equal(results['sum_sq'].values, Sum_Sq, 4)

        np.testing.assert_almost_equal(results["F"].values, F, 4)
        np.testing.assert_almost_equal(results["PR(>F)"].values, PrF)


class TestAnova3HC3(TestAnovaLM):
    # drop some observations to make an unbalanced, disproportionate panel
    # to make sure things are okay

    def test_results(self):
        data = self.data.drop([0, 1, 2])
        anova_iii = ols("np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)", data).fit()

        # Sum_Sq = np.array([
        #      151.4065, 2.904723, 13.45718, 0.1905093, 27.60181
        #     ])

        Df = np.array([1, 1, 2, 2, 51])
        F = np.array([234.4026, 4.496996, 10.79903, 0.1317223, np.nan])
        PrF = np.array([1.037224e-20, 0.03883841, 0.0001228716, 0.8768817, np.nan])

        results = anova_lm(anova_iii, typ="III", robust="hc3")
        np.testing.assert_equal(results["df"].values, Df)
        # np.testing.assert_almost_equal(results['sum_sq'].values, Sum_Sq, 4)

        np.testing.assert_almost_equal(results["F"].values, F, 4)
        np.testing.assert_almost_equal(results["PR(>F)"].values, PrF)


def test_anova_lm_model_order_error():
    # Models passed in decreasing order of complexity should raise,
    # rather than silently producing NaN via negative df_diff.
    rs = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "y": rs.randn(50),
            "x1": rs.randn(50),
            "x2": rs.randn(50),
        }
    )

    model_small = ols("y ~ x1", data=df).fit()
    model_large = ols("y ~ x1 + x2", data=df).fit()

    error_msg = "Models must be passed in order of increasing complexity"
    with pytest.raises(ValueError, match=error_msg):
        anova_lm(model_large, model_small)


def test_anova_lm_model_order_correct():
    # Correct order should not raise and should produce a real p-value.
    rs = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "y": rs.randn(50),
            "x1": rs.randn(50),
            "x2": rs.randn(50),
        }
    )

    model_small = ols("y ~ x1", data=df).fit()
    model_large = ols("y ~ x1 + x2", data=df).fit()

    result = anova_lm(model_small, model_large)
    assert not np.isnan(result["Pr(>F)"].iloc[-1])


ANOVA_TYPES = [
    (1, "I"),
    ("I", "I"),
    (2, "II"),
    ("II", "II"),
    (3, "III"),
    ("III", "III"),
]


def fit_rank_deficient(model_func, formula, data, **kwargs):
    with pytest.warns(SingularMatrixWarning):
        return model_func(formula, data=data, **kwargs).fit()


def nested_data(label_within):
    # Factor B has nB levels in each of the nA levels of factor A. The labels
    # of B are reused in every level of A if label_within is True and are
    # unique to the level of A that contains them otherwise.
    nA, nB, nobs = 3, 4, 5
    a = [f"A{i}" for i in range(nA) for _ in range(nB * nobs)]
    if label_within:
        b = [f"B{j}" for _ in range(nA) for j in range(nB) for _ in range(nobs)]
    else:
        b = [f"B{i}{j}" for i in range(nA) for j in range(nB) for _ in range(nobs)]
    rng = np.random.default_rng(8506)
    data = pd.DataFrame({"A": a, "B": b, "y": rng.standard_normal(len(a))})
    return data.astype({"A": "category", "B": "category"})


@pytest.mark.parametrize("robust", [None, "hc3"])
@pytest.mark.parametrize("typ, label", ANOVA_TYPES)
@pytest.mark.parametrize("formula", ["y ~ x + duplicate", "y ~ x + zero"])
def test_rank_deficient(formula, typ, label, robust):
    # The duplicate or the zero column would be assigned an arbitrary sum of
    # squares, for Type I because the effects of the QR decomposition are
    # mapped back to the terms, and for Type II and III because the
    # hypothesis on the redundant column is not testable.
    data = pd.DataFrame(
        {
            "y": [0, 1, 0, -1] * 3,
            "x": np.arange(12),
            "duplicate": np.arange(12),
            "zero": np.zeros(12),
        }
    )
    res = fit_rank_deficient(ols, formula, data)
    msg = f"Type {label} ANOVA requires a full-rank design matrix.*3 columns and rank 2"
    with pytest.raises(ValueError, match=msg):
        anova_lm(res, typ=typ, robust=robust)


@pytest.mark.parametrize("typ, label", ANOVA_TYPES)
def test_more_columns_than_observations(typ, label):
    data = pd.DataFrame({"y": [1.0, 2.0, 4.0], "x": [0.0, 1.0, 2.0]})
    res = fit_rank_deficient(ols, "y ~ x + I(x**2) + I(x**3)", data)
    with pytest.raises(ValueError, match=f"Type {label} ANOVA requires a full-rank"):
        anova_lm(res, typ=typ)


@pytest.mark.parametrize("typ, label", ANOVA_TYPES)
@pytest.mark.parametrize("formula", ["y ~ C(a) * C(b)", "y ~ C(a):C(b)"])
def test_empty_cell(formula, typ, label):
    # No observation in the cell (a="u", b="q"), so one interaction column is
    # identically zero. This used to give a negative sum of squares for Type
    # II and too many degrees of freedom for the interaction.
    rng = np.random.default_rng(5398)
    cells = [("u", "p"), ("v", "p"), ("v", "q"), ("w", "p"), ("w", "q")]
    data = pd.DataFrame(
        {
            "a": [a for a, _ in cells for _ in range(4)],
            "b": [b for _, b in cells for _ in range(4)],
            "y": rng.standard_normal(20),
        }
    )
    res = fit_rank_deficient(ols, formula, data)
    with pytest.raises(ValueError, match=f"Type {label} ANOVA requires a full-rank"):
        anova_lm(res, typ=typ)


@pytest.mark.parametrize("typ, label", ANOVA_TYPES)
def test_nested_factor_crossed_with_unique_labels(typ, label):
    # GH 8506. If the labels of B are unique to a level of A, A + A:B contains
    # the interactions of B with the levels of A that do not contain it.
    data = nested_data(label_within=False)
    res = fit_rank_deficient(ols, "y ~ A + A:B", data)
    assert res.model.exog.shape[1] == 36
    with pytest.raises(ValueError, match=f"Type {label} ANOVA requires a full-rank"):
        anova_lm(res, typ=typ)


@pytest.mark.parametrize("typ", [1, 2, 3])
def test_nested_factor_labeled_within(typ):
    # Reusing the labels of B in every level of A gives a full-rank design.
    data = nested_data(label_within=True)
    res = ols("y ~ A/B", data=data).fit()
    table = anova_lm(res, typ=typ)
    assert table.loc["Residual", "df"] == 48
    # The sum of squares of the last term is the same for all types and
    # is the reduction in the residual sum of squares from adding A:B to A.
    ssr_a = ols("y ~ A", data=data).fit().ssr
    ssr_nested = ols("y ~ A + A:B", data=data).fit().ssr
    assert table.loc["A:B", "df"] == 9
    np.testing.assert_allclose(table.loc["A:B", "sum_sq"], ssr_a - ssr_nested)
    if typ in (1, 2):
        # A is not adjusted for A:B
        ssr_const = ols("y ~ 1", data=data).fit().ssr
        assert table.loc["A", "df"] == 2
        np.testing.assert_allclose(table.loc["A", "sum_sq"], ssr_const - ssr_a)


@pytest.mark.parametrize("typ, label", ANOVA_TYPES)
def test_zero_weights_rank_deficient(typ, label):
    # The design matrix has full column rank, but all of the observations of
    # level "u" have zero weight, so the weighted design matrix does not.
    rng = np.random.default_rng(10340)
    data = pd.DataFrame(
        {
            "a": np.repeat(["u", "v", "w"], 8),
            "b": np.tile(["p", "q"], 12),
            "y": rng.standard_normal(24),
        }
    )
    weights = np.where(data["a"] == "u", 0.0, 1.0)
    res = fit_rank_deficient(wls, "y ~ C(a) + C(b)", data, weights=weights)
    assert np.linalg.matrix_rank(res.model.exog) == res.model.exog.shape[1]
    with pytest.raises(ValueError, match=f"Type {label} ANOVA requires a full-rank"):
        anova_lm(res, typ=typ)


@pytest.mark.parametrize("typ", [1, "I", 2, "II", 3, "III"])
def test_full_rank_control(typ):
    # Dropping the redundant column leaves the fit unchanged and the table
    # of the model without it is available.
    data = pd.DataFrame(
        {
            "y": [0, 1, 0, -1] * 3,
            "x": np.arange(12),
            "duplicate": np.arange(12),
        }
    )
    res = ols("y ~ x", data=data).fit()
    table = anova_lm(res, typ=typ)
    pvalue = "PR(>F)"
    np.testing.assert_allclose(table.loc["x", pvalue], res.pvalues["x"])
    np.testing.assert_allclose(table.loc["Residual", "sum_sq"], res.ssr)
    assert table.loc["Residual", "df"] == res.df_resid
    redundant = fit_rank_deficient(ols, "y ~ x + duplicate", data)
    np.testing.assert_allclose(redundant.fittedvalues, res.fittedvalues, atol=1e-12)
