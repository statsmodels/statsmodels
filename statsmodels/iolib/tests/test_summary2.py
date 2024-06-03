import re
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal

from statsmodels.iolib.summary2 import summary_col
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


class TestSummaryLatex:

    def test_summarycol(self):
        # Test for latex output of summary_col object
        desired = r'''
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
'''
        x = [1, 5, 7, 3, 5]
        x = add_constant(x)
        y1 = [6, 4, 2, 7, 4]
        y2 = [8, 5, 0, 12, 4]
        reg1 = OLS(y1, x).fit()
        reg2 = OLS(y2, x).fit()
        actual = summary_col([reg1, reg2]).as_latex()
        actual = '\n%s\n' % actual
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
"""  # noqa:W291
        x = [1, 5, 7, 3, 5]
        x = add_constant(x)
        y1 = [6, 4, 2, 7, 4]
        y2 = [8, 5, 0, 12, 4]
        reg1 = OLS(y1, x).fit()
        reg2 = OLS(y2, x).fit()
        actual = summary_col([reg1, reg2], float_format='%0.1f').as_text()
        actual = '%s\n' % actual

        starred = summary_col([reg1, reg2], stars=True, float_format='%0.1f')
        assert "7.7***" in str(starred)
        assert "12.4**" in str(starred)
        assert "12.4***" not in str(starred)

        assert_equal(actual, desired)

    def test_summarycol_drop_omitted(self):
        # gh-3702
        x = [1, 5, 7, 3, 5]
        x = add_constant(x)
        x2 = np.concatenate([x, np.array([[3], [9], [-1], [4], [0]])], 1)
        y1 = [6, 4, 2, 7, 4]
        y2 = [8, 5, 0, 12, 4]
        reg1 = OLS(y1, x).fit()
        reg2 = OLS(y2, x2).fit()
        actual = summary_col([reg1, reg2], regressor_order=['const', 'x1'],
                             drop_omitted=True)
        assert 'x2' not in str(actual)
        actual = summary_col([reg1, reg2], regressor_order=['x1'],
                             drop_omitted=False)
        assert 'const' in str(actual)
        assert 'x2' in str(actual)

    def test_summary_col_ordering_preserved(self):
        # gh-3767
        x = [1, 5, 7, 3, 5]
        x = add_constant(x)
        x2 = np.concatenate([x, np.array([[3], [9], [-1], [4], [0]])], 1)
        x2 = pd.DataFrame(x2, columns=['const', 'b', 'a'])
        y1 = [6, 4, 2, 7, 4]
        y2 = [8, 5, 0, 12, 4]
        reg1 = OLS(y1, x2).fit()
        reg2 = OLS(y2, x2).fit()

        info_dict = {'R2': lambda x: f'{int(x.rsquared):.3f}',
                     'N': lambda x: f'{int(x.nobs):d}'}
        original = actual = summary_col([reg1, reg2], float_format='%0.4f')
        actual = summary_col([reg1, reg2], regressor_order=['a', 'b'],
                             float_format='%0.4f',
                             info_dict=info_dict)
        variables = ('const', 'b', 'a')
        for line in str(original).split('\n'):
            for variable in variables:
                if line.startswith(variable):
                    assert line in str(actual)

    def test__repr_latex_(self):
        desired = r'''
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
'''
        x = [1, 5, 7, 3, 5]
        x = add_constant(x)
        y1 = [6, 4, 2, 7, 4]
        y2 = [8, 5, 0, 12, 4]
        reg1 = OLS(y1, x).fit()
        reg2 = OLS(y2, x).fit()

        actual = summary_col([reg1, reg2])._repr_latex_()
        actual = '\n%s\n' % actual
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
        string_to_find = r'''\end{tabular}
\begin{tabular}'''
        result = string_to_find in actual
        assert (result is True)


def test_ols_summary_rsquared_label():
    # Check that the "uncentered" label is correctly added after rsquared
    x = [1, 5, 7, 3, 5, 2, 5, 3]
    y = [6, 4, 2, 7, 4, 9, 10, 2]
    reg_with_constant = OLS(y, add_constant(x)).fit()
    r2_str = 'R-squared:'
    with pytest.warns(UserWarning):
        assert r2_str in str(reg_with_constant.summary2())
    with pytest.warns(UserWarning):
        assert r2_str in str(reg_with_constant.summary())

    reg_without_constant = OLS(y, x, hasconst=False).fit()
    r2_str = 'R-squared (uncentered):'
    with pytest.warns(UserWarning):
        assert r2_str in str(reg_without_constant.summary2())
    with pytest.warns(UserWarning):
        assert r2_str in str(reg_without_constant.summary())


class TestSummaryLabels:
    """
    Test that the labels are correctly set in the summary table"""

    @classmethod
    def setup_class(cls):
        y = [1, 1, 4, 2] * 4
        x = add_constant([1, 2, 3, 4] * 4)
        cls.mod = OLS(endog=y, exog=x).fit()

    def test_summary_col_r2(self):
        # GH 6578
        table = summary_col(results=self.mod, include_r2=True)
        assert "R-squared  " in str(table)
        assert "R-squared Adj." in str(table)

    def test_absence_of_r2(self):
        table = summary_col(results=self.mod, include_r2=False)
        assert "R-squared" not in str(table)
        assert "R-squared Adj." not in str(table)

    @pytest.mark.parametrize(
        "coef_stat, expected_constant_stat, expected_x1_stat",
        [
            (None, 0.670820, 0.244949),  # default behavior
            ("std_error", 0.670820, 0.244949),
            ("t_stat", 0.745356, 2.449490),
            ("z_stat", 0.745356, 2.449490),
        ]
    )
    def test_coef_stats_parameter_return_correct_value(
        self, coef_stat, expected_constant_stat, expected_x1_stat
    ):
        if coef_stat is None:
            # Test the default behavior.
            # N.B.: we don't pass None to the coef_stats parameter
            table = summary_col(
                results=self.mod,
                float_format="%.5f",  # to avoid rounding errors
            )
        else:
            table = summary_col(
                results=self.mod,
                float_format="%.5f",  # to avoid rounding errors
                coef_stats=coef_stat
            )

        constant_stat = table.tables[0].iloc[1, 0]
        # check that the stat is in parentheses
        assert re.match(r"\(.*\)", constant_stat)
        constant_stat_value = float(constant_stat.strip("()"))
        assert np.isclose(constant_stat_value, expected_constant_stat)

        x1_stat = table.tables[0].iloc[3, 0]
        # check that the stat is in parentheses
        assert re.match(r"\(.*\)", x1_stat)
        x1_stat_value = float(x1_stat.strip("()"))
        assert np.isclose(x1_stat_value, expected_x1_stat)

    @pytest.mark.parametrize(
        "coef_stat, expected_footer",
        [
            ("std_error", "Standard errors in parentheses."),
            ("t_stat", "t-statistics in parentheses."),
            ("z_stat", "z-statistics in parentheses."),
        ]
    )
    def test_coef_stats_parameter_adds_correct_footer(
        self, coef_stat, expected_footer
    ):
        table = summary_col(results=self.mod, coef_stats=coef_stat)
        assert expected_footer in table.extra_txt

    @pytest.mark.parametrize(
        "coef_stat",
        ["", None]
    )
    def test_no_coef_stats_parameter_deletes_stats(self, coef_stat):
        table = summary_col(results=self.mod, coef_stats=coef_stat)
        # ensure the table has no stats index
        # stat index would be an empty string after 'const' and after 'x1'
        assert all(table.tables[0].index == [
            "const", "x1", "R-squared", "R-squared Adj."
        ])
        # ensure that the footer is not present
        assert "in parentheses." not in str(table)
        print(table)

    @pytest.mark.parametrize(
        "coef_stat",
        ["invalid", 42, True]
    )
    def test_invalid_coef_stats_parameter_raises_error(self, coef_stat):
        with pytest.raises(ValueError):
            summary_col(results=self.mod, coef_stats=coef_stat)
