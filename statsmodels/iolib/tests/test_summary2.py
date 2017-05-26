from numpy.testing import assert_equal, assert_
from statsmodels.regression.linear_model import OLS, add_constant
from statsmodels.iolib.summary2 import summary_col


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
\midrule
const & 7.7500   & 12.4231   \\
      & (1.1058) & (3.1872)  \\
x1    & -0.7500  & -1.5769   \\
      & (0.2368) & (0.6826)  \\
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

    def test_OLSsummary(self):
        # Test that latex output of regular OLS output still contains
        # multiple tables

        x = [1,5,7,3,5]
        x = add_constant(x)
        y1 = [6,4,2,7,4]
        reg1 = OLS(y1,x).fit()
        actual = reg1.summary().as_latex()
        string_to_find = r'''\end{tabular}
\begin{tabular}'''
        result = string_to_find in actual
        assert(result is True)
