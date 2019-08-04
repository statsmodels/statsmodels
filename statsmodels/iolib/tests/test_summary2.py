import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal

from statsmodels.iolib.summary2 import summary_col
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS

class TestSummaryLatex(object):

    def test_summarycol(self):
        # Test for latex output of summary_col object
        desired = r'''
\begin{table}
\caption{}
\begin{center}
\begin{tabular}{lcc}
\hline
          &   y I    &   y II    \\
\midrule
const     & 7.7500   & 12.4231   \\
          & (1.1058) & (3.1872)  \\
x1        & -0.7500  & -1.5769   \\
          & (0.2368) & (0.6826)  \\
R-squared & 0.6930   & 0.5202    \\
          & 0.7697   & 0.6401    \\
\hline
\end{tabular}
\end{center}
\end{table}
'''
        x = [1,5,7,3,5]
        x = add_constant(x)
        y1 = [6,4,2,7,4]
        y2 = [8,5,0,12,4]
        reg1 = OLS(y1,x).fit()
        reg2 = OLS(y2,x).fit()
        actual = summary_col([reg1,reg2]).as_latex()
        actual = '\n%s\n' % actual
        assert_equal(desired, actual)

    def test_summarycol_float_format(self):
        # Test for latex output of summary_col object
        desired = r"""
=====================
           y I   y II
---------------------
const     7.7   12.4 
          (1.1) (3.2)
x1        -0.7  -1.6 
          (0.2) (0.7)
R-squared 0.7   0.5  
          0.8   0.6  
=====================
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

        info_dict = {'R2': lambda x: '{:.3f}'.format(int(x.rsquared)),
                     'N': lambda x: '{0:d}'.format(int(x.nobs))}
        original = actual = summary_col([reg1, reg2], float_format='%0.4f')
        actual = summary_col([reg1, reg2], regressor_order=['a', 'b'],
                             float_format='%0.4f',
                             info_dict=info_dict)
        variables = ('const', 'b', 'a')
        for line in str(original).split('\n'):
            for variable in variables:
                if line.startswith(variable):
                    assert line in str(actual)

    def test_OLSsummary(self):
        # Test that latex output of regular OLS output still contains
        # multiple tables

        x = [1,5,7,3,5]
        x = add_constant(x)
        y1 = [6,4,2,7,4]
        reg1 = OLS(y1,x).fit()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actual = reg1.summary().as_latex()
        string_to_find = r'''\end{tabular}
\begin{tabular}'''
        result = string_to_find in actual
        assert(result is True)


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
