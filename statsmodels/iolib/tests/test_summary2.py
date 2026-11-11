from statsmodels.compat.scipy import SP_LT_116

import warnings

import numpy as np
from numpy.testing import assert_equal
import pandas as pd
import pytest

from statsmodels.iolib.summary2 import summary_col
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


def _rstrip_txt(txt):
    return "\n".join([line.rstrip() for line in txt.split("\n")])


class TestSummaryLatex:

    def test_summarycol(self):
        # Test for latex output of summary_col object
        desired = r"""
\begin{table}
\caption{}
\label{}
\begin{center}
\begin{tabular}{lll}
\hline
               & y I      & y II      \\
\hline
const          & 7.7500   & 12.4231   \\
               & (1.1058) & (3.1872)  \\
x1             & -0.7500  & -1.5769   \\
               & (0.2368) & (0.6826)  \\
R-squared      & 0.7697   & 0.6401    \\
R-squared Adj. & 0.6930   & 0.5202    \\
\hline
\end{tabular}
\end{center}
\end{table}
\bigskip
Standard errors in parentheses.
"""
        x = [1, 5, 7, 3, 5]
        x = add_constant(x)
        y1 = [6, 4, 2, 7, 4]
        y2 = [8, 5, 0, 12, 4]
        reg1 = OLS(y1, x).fit()
        reg2 = OLS(y2, x).fit()
        actual = summary_col([reg1, reg2]).as_latex()
        actual = f"\n{actual}\n"
        assert_equal(desired, actual)

    def test_summarycol_float_format(self):
        # Test for latex output of summary_col object
        desired = r"""
==========================
                y I   y II
--------------------------
const          7.7   12.4
               (1.1) (3.2)
x1             -0.7  -1.6
               (0.2) (0.7)
R-squared      0.8   0.6
R-squared Adj. 0.7   0.5
==========================
Standard errors in
parentheses.
"""
        x = [1, 5, 7, 3, 5]
        x = add_constant(x)
        y1 = [6, 4, 2, 7, 4]
        y2 = [8, 5, 0, 12, 4]
        reg1 = OLS(y1, x).fit()
        reg2 = OLS(y2, x).fit()
        actual = summary_col([reg1, reg2], float_format="%0.1f").as_text()
        actual = f"{actual}\n"

        starred = summary_col([reg1, reg2], stars=True, float_format="%0.1f")
        assert "7.7***" in str(starred)
        assert "12.4**" in str(starred)
        assert "12.4***" not in str(starred)

        assert_equal(_rstrip_txt(actual), desired)

    def test_summarycol_drop_omitted(self):
        # gh-3702
        x = [1, 5, 7, 3, 5]
        x = add_constant(x)
        x2 = np.concatenate([x, np.array([[3], [9], [-1], [4], [0]])], 1)
        y1 = [6, 4, 2, 7, 4]
        y2 = [8, 5, 0, 12, 4]
        reg1 = OLS(y1, x).fit()
        reg2 = OLS(y2, x2).fit()
        actual = summary_col(
            [reg1, reg2], regressor_order=["const", "x1"], drop_omitted=True
        )
        assert "x2" not in str(actual)
        actual = summary_col([reg1, reg2], regressor_order=["x1"], drop_omitted=False)
        assert "const" in str(actual)
        assert "x2" in str(actual)

    def test_summary_col_ordering_preserved(self):
        # gh-3767
        x = [1, 5, 7, 3, 5]
        x = add_constant(x)
        x2 = np.concatenate([x, np.array([[3], [9], [-1], [4], [0]])], 1)
        x2 = pd.DataFrame(x2, columns=["const", "b", "a"])
        y1 = [6, 4, 2, 7, 4]
        y2 = [8, 5, 0, 12, 4]
        reg1 = OLS(y1, x2).fit()
        reg2 = OLS(y2, x2).fit()

        info_dict = {
            "R2": lambda x: f"{int(x.rsquared):.3f}",
            "N": lambda x: f"{int(x.nobs):d}",
        }
        original = actual = summary_col([reg1, reg2], float_format="%0.4f")
        actual = summary_col(
            [reg1, reg2],
            regressor_order=["a", "b"],
            float_format="%0.4f",
            info_dict=info_dict,
        )
        variables = ("const", "b", "a")
        for line in str(original).split("\n"):
            for variable in variables:
                if line.startswith(variable):
                    assert line in str(actual)

    def test__repr_latex_(self):
        desired = r"""
\begin{table}
\caption{}
\label{}
\begin{center}
\begin{tabular}{lll}
\hline
               & y I      & y II      \\
\hline
const          & 7.7500   & 12.4231   \\
               & (1.1058) & (3.1872)  \\
x1             & -0.7500  & -1.5769   \\
               & (0.2368) & (0.6826)  \\
R-squared      & 0.7697   & 0.6401    \\
R-squared Adj. & 0.6930   & 0.5202    \\
\hline
\end{tabular}
\end{center}
\end{table}
\bigskip
Standard errors in parentheses.
"""
        x = [1, 5, 7, 3, 5]
        x = add_constant(x)
        y1 = [6, 4, 2, 7, 4]
        y2 = [8, 5, 0, 12, 4]
        reg1 = OLS(y1, x).fit()
        reg2 = OLS(y2, x).fit()

        actual = summary_col([reg1, reg2])._repr_latex_()
        actual = f"\n{actual}\n"
        assert_equal(actual, desired)

    def test_OLSsummary(self):
        # Test that latex output of regular OLS output still contains
        # multiple tables

        x = [1, 5, 7, 3, 5]
        x = add_constant(x)
        y1 = [6, 4, 2, 7, 4]
        reg1 = OLS(y1, x).fit()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actual = reg1.summary().as_latex()
        string_to_find = r"""\end{tabular}
\begin{tabular}"""
        result = string_to_find in actual
        assert result is True

    def test_summarycol_fixed_effects(self):
        desired1 = r"""
======================================
               model_0 model_1 model_2
--------------------------------------
Intercept      1.35    1.32    1.48
               (0.19)  (0.42)  (0.73)
yrs_married    -0.03   -0.02   -0.02
               (0.00)  (0.00)  (0.00)
educ           -0.03   -0.02   -0.02
               (0.01)  (0.02)  (0.02)
occupation FE          Yes     Yes
religious FE           Yes     Yes
R-squared      0.01    0.02    0.03
R-squared Adj. 0.01    0.02    0.02
======================================
Standard errors in parentheses."""

        desired2 = r"""
========================================
                     mod0   mod1   mod2
----------------------------------------
Intercept           1.35   1.32   1.48
                    (0.19) (0.42) (0.73)
yrs_married         -0.03  -0.02  -0.02
                    (0.00) (0.00) (0.00)
educ                -0.03  -0.02  -0.02
                    (0.01) (0.02) (0.02)
C(religious)[T.2.0]        -0.46  -0.86
                           (0.08) (0.87)
C(religious)[T.3.0]        -0.66  -0.71
                           (0.08) (1.13)
C(religious)[T.4.0]        -0.91  -0.92
                           (0.11) (1.03)
occupation FE              Yes    Yes
R-squared           0.01   0.02   0.03
R-squared Adj.      0.01   0.02   0.02
========================================
Standard errors in parentheses."""

        from statsmodels.datasets.fair import load_pandas

        df_fair = load_pandas().data

        res0 = OLS.from_formula("affairs ~ yrs_married + educ", df_fair).fit()
        form1 = "affairs ~ yrs_married + educ + C(occupation) + C(religious)"
        res1 = OLS.from_formula(form1, df_fair).fit()
        form2 = "affairs ~ yrs_married + educ + C(occupation) : C(religious)"
        res2 = OLS.from_formula(form2, df_fair).fit()

        regressions = [res0, res1, res2]
        summary1 = summary_col(
            regressions,
            model_names=[f"model_{i}" for i, _ in enumerate(regressions)],
            fixed_effects=["occupation", "religious"],
            float_format="%0.2f",
        )

        assert_equal(_rstrip_txt(summary1.as_text()), desired1)

        summary2 = summary_col(
            regressions,
            model_names=[f"mod{i}" for i, _ in enumerate(regressions)],
            fixed_effects=["occupation"],
            float_format="%0.2f",
        )

        assert_equal(_rstrip_txt(summary2.as_text()), desired2)


def test_ols_summary_rsquared_label():
    # Check that the "uncentered" label is correctly added after rsquared
    x = [1, 5, 7, 3, 5, 2, 5, 3]
    y = [6, 4, 2, 7, 4, 9, 10, 2]
    reg_with_constant = OLS(y, add_constant(x)).fit()
    r2_str = "R-squared:"
    if SP_LT_116:
        with pytest.warns(UserWarning, match=r"(.*)kurtosistest(.*)"):
            assert r2_str in str(reg_with_constant.summary2())
        with pytest.warns(UserWarning, match=r"(.*)kurtosistest(.*)"):
            assert r2_str in str(reg_with_constant.summary())
    else:
        assert r2_str in str(reg_with_constant.summary2())
        assert r2_str in str(reg_with_constant.summary())

    reg_without_constant = OLS(y, x, hasconst=False).fit()
    r2_str = "R-squared (uncentered):"
    if SP_LT_116:
        with pytest.warns(UserWarning, match=r"(.*)kurtosistest(.*)"):
            assert r2_str in str(reg_without_constant.summary2())
        with pytest.warns(UserWarning, match=r"(.*)kurtosistest(.*)"):
            assert r2_str in str(reg_without_constant.summary())
    else:
        assert r2_str in str(reg_without_constant.summary2())
        assert r2_str in str(reg_without_constant.summary())


class TestSummaryLabels:
    """
    Test that the labels are correctly set in the summary table"""

    @classmethod
    def setup_class(cls):
        y = [1, 1, 4, 2] * 4
        x = add_constant([1, 2, 3, 4] * 4)
        cls.mod = OLS(endog=y, exog=x).fit()

    def test_summary_col_r2(
        self,
    ):
        # GH 6578
        table = summary_col(results=self.mod, include_r2=True)
        assert "R-squared  " in str(table)
        assert "R-squared Adj." in str(table)

    def test_absence_of_r2(
        self,
    ):
        table = summary_col(results=self.mod, include_r2=False)
        assert "R-squared" not in str(table)
        assert "R-squared Adj." not in str(table)


def test_summarycol_fixed_effects_before_info_dict():
    # gh-9282: the fixed-effects indicator rows belong with the
    # parameters, i.e. before the rows generated from info_dict
    from statsmodels.datasets.fair import load_pandas

    df_fair = load_pandas().data

    res0 = OLS.from_formula("affairs ~ yrs_married", df_fair).fit()
    form1 = "affairs ~ yrs_married + C(occupation)"
    res1 = OLS.from_formula(form1, df_fair).fit()

    summary = summary_col(
        [res0, res1],
        model_names=["ols0", "ols1"],
        float_format="%0.2f",
        info_dict={"N": lambda x: f"{int(x.nobs):d}"},
        fixed_effects=["occupation"],
    )

    assert list(summary.tables[0].index) == [
        "Intercept",
        "",
        "yrs_married",
        "",
        "occupation FE",
        "R-squared",
        "R-squared Adj.",
        "N",
    ]


def test_summarycol_parentheses_options():
    # gh-8996
    x = [1, 5, 7, 3, 5]
    x = add_constant(x)
    y = [6, 4, 2, 7, 4]
    reg = OLS(y, x).fit()

    # t-statistics
    s_t = summary_col([reg], parentheses="t", float_format="%.4f")
    assert s_t.extra_txt == ["t-statistics in parentheses."]
    assert f"({reg.tvalues[0]:.4f})" in str(s_t)
    assert f"({reg.bse[0]:.4f})" not in str(s_t)

    # z-statistics
    s_z = summary_col([reg], parentheses="z", float_format="%.4f")
    assert s_z.extra_txt == ["z-statistics in parentheses."]
    assert f"({reg.tvalues[0]:.4f})" in str(s_z)

    # stat
    s_stat = summary_col([reg], parentheses="stat", float_format="%.4f")
    assert s_stat.extra_txt == ["Test statistics in parentheses."]

    # p-values
    s_p = summary_col([reg], parentheses="p", float_format="%.4f")
    assert s_p.extra_txt == ["p-values in parentheses."]
    assert f"({reg.pvalues[0]:.4f})" in str(s_p)

    # None / False
    s_none = summary_col([reg], parentheses=None)
    assert s_none.extra_txt == []
    assert "(" not in str(s_none)
    assert list(s_none.tables[0].index) == ["const", "x1", "R-squared", "R-squared Adj."]

    s_false = summary_col([reg], parentheses=False)
    assert s_false.extra_txt == []
    assert list(s_false.tables[0].index) == ["const", "x1", "R-squared", "R-squared Adj."]

    with pytest.raises(ValueError, match="Unknown parentheses option"):
        summary_col([reg], parentheses="unsupported")


def test_summarycol_float_format_dict():
    # gh-8996
    x = [1, 5, 7, 3, 5]
    x = add_constant(x)
    y = [6, 4, 2, 7, 4]
    reg = OLS(y, x).fit()

    fmt = {"coeff": "%.2f", "parentheses": "%.3f", "r2": "%.1f"}
    s = summary_col([reg], float_format=fmt)
    # Coeff format: 7.75
    assert "7.75" in str(s)
    # SE format: 1.106
    assert "(1.106)" in str(s)
    # R2 format: 0.8
    assert "0.8" in str(s)

    with pytest.raises(TypeError, match="float_format must be a str or dict"):
        summary_col([reg], float_format=1234)


def test_summarycol_granular_r2_and_n():
    # gh-8996
    x = [1, 5, 7, 3, 5]
    x = add_constant(x)
    y = [6, 4, 2, 7, 4]
    reg = OLS(y, x).fit()

    # include_adj_r2=False
    s_no_adj = summary_col([reg], include_r2=True, include_adj_r2=False)
    assert "R-squared" in str(s_no_adj)
    assert "R-squared Adj." not in str(s_no_adj)

    # include_r2=False, include_adj_r2=True
    s_only_adj = summary_col([reg], include_r2=False, include_adj_r2=True)
    assert "R-squared Adj." in str(s_only_adj)
    assert "R-squared  " not in str(s_only_adj)

    # include_n=True
    s_n = summary_col([reg], include_n=True)
    assert "N" in s_n.tables[0].index
    assert s_n.tables[0].loc["N"].iloc[0] == "5"

    # include_n=True with custom N format in float_format dict
    s_n_fmt = summary_col([reg], include_n=True, float_format={"N": "%04d"})
    assert s_n_fmt.tables[0].loc["N"].iloc[0] == "0005"

    # include_n=True combined with info_dict
    s_n_info = summary_col(
        [reg],
        include_n=True,
        info_dict={"AIC": lambda m: f"{m.aic:.1f}"},
    )
    idx = list(s_n_info.tables[0].index)
    assert "N" in idx
    assert "AIC" in idx

