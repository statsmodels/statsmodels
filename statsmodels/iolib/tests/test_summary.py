"""examples to check summary, not converted to tests yet"""

import numpy as np
from numpy.testing import assert_equal
import pytest

from statsmodels.datasets import macrodata
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


def test_escaped_variable_name():
    # Rename 'cpi' column to 'CPI_'
    data = macrodata.load().data
    data = data.rename(columns={"cpi": "CPI_"})

    mod = OLS.from_formula("CPI_ ~ 1 + np.log(realgdp)", data=data)
    res = mod.fit()
    assert "CPI\\_" in res.summary().as_latex()
    assert "CPI_" in res.summary().as_text()


def test_wrong_len_xname():
    rs = np.random.RandomState(8390293)
    y = rs.randn(100)
    x = rs.randn(100, 2)
    res = OLS(y, x).fit()
    with pytest.raises(ValueError):
        res.summary(xname=["x1"])
    with pytest.raises(ValueError):
        res.summary(xname=["x1", "x2", "x3"])


class TestSummaryLatex:
    def test__repr_latex_(self):
        desired = r"""
\begin{center}
\begin{tabular}{lcccccc}
\toprule
               & \textbf{coef} & \textbf{std err} & \textbf{t} & \textbf{P$> |$t$|$} & \textbf{[0.025} & \textbf{0.975]}  \\
\midrule
\textbf{const} &       7.2248  &        0.866     &     8.346  &         0.000        &        5.406    &        9.044     \\
\textbf{x1}    &      -0.6609  &        0.177     &    -3.736  &         0.002        &       -1.033    &       -0.289     \\
\bottomrule
\end{tabular}
\end{center}
"""
        x = [1, 5, 7, 3, 5, 5, 8, 3, 3, 4, 6, 4, 2, 7, 4, 2, 1, 9, 2, 6]
        x = add_constant(x)
        y = [6, 4, 2, 7, 4, 2, 1, 9, 2, 6, 1, 5, 7, 3, 5, 5, 8, 3, 3, 4]
        reg = OLS(y, x).fit()

        actual = reg.summary().tables[1]._repr_latex_()
        actual = f"\n{actual}\n"
        assert_equal(actual, desired)


def test_summary_as_csv_and_as_html():
    # as_csv()/as_html() are exported as part of iolib.summary.Summary but,
    # unlike as_text() and as_latex() above, had no test coverage at all.
    rs = np.random.RandomState(0)
    y = rs.standard_normal(50)
    x = add_constant(rs.standard_normal((50, 2)))
    res = OLS(y, x).fit()
    summary = res.summary()

    csv = summary.as_csv()
    assert "OLS Regression Results" in csv
    assert "R-squared" in csv
    # csv formatting replaces the fixed-width padding with comma separators
    assert csv.count(",") > summary.as_text().count(",")

    html = summary.as_html()
    assert html.startswith('<table class="simpletable">')
    assert "OLS Regression Results" in html
    assert "R-squared" in html
    assert html.count("<table") == len(summary.tables)
