'''examples to check summary, not converted to tests yet


'''
from __future__ import print_function

import numpy as np  # noqa: F401
import pytest

from statsmodels.datasets import macrodata
from statsmodels.regression.linear_model import OLS


def test_escaped_variable_name():
    # Rename 'cpi' column to 'CPI_'
    data = macrodata.load(True).data
    data.rename(columns={'cpi': 'CPI_'}, inplace=True)

    mod = OLS.from_formula('CPI_ ~ 1 + np.log(realgdp)', data=data)
    res = mod.fit()
    assert 'CPI\\_' in res.summary().as_latex()
    assert 'CPI_' in res.summary().as_text()


def test_wrong_len_xname(reset_randomstate):
    y = np.random.randn(100)
    x = np.random.randn(100, 2)
    res = OLS(y, x).fit()
    with pytest.raises(ValueError):
        res.summary(xname=['x1'])
    with pytest.raises(ValueError):
        res.summary(xname=['x1', 'x2', 'x3'])


if __name__ == '__main__':

    from statsmodels.regression.tests.test_regression import TestOLS

    #def mytest():
    aregression = TestOLS()
    TestOLS.setup_class()
    results = aregression.res1
    r_summary = str(results.summary_old())
    print(r_summary)
    olsres = results

    print('\n\n')

    r_summary = str(results.summary())
    print(r_summary)
    print('\n\n')


    from statsmodels.discrete.tests.test_discrete import TestProbitNewton

    aregression = TestProbitNewton()
    TestProbitNewton.setup_class()
    results = aregression.res1
    r_summary = str(results.summary())
    print(r_summary)
    print('\n\n')

    probres = results

    from statsmodels.robust.tests.test_rlm import TestHampel

    aregression = TestHampel()
    #TestHampel.setup_class()
    results = aregression.res1
    r_summary = str(results.summary())
    print(r_summary)
    rlmres = results

    print('\n\n')

    from statsmodels.genmod.tests.test_glm import TestGlmBinomial

    aregression = TestGlmBinomial()
    #TestGlmBinomial.setup_class()
    results = aregression.res1
    r_summary = str(results.summary())
    print(r_summary)

    #print(results.summary2(return_fmt='latex'))
    #print(results.summary2(return_fmt='csv'))

    smry = olsres.summary()
    print(smry.as_csv())

#    import matplotlib.pyplot as plt
#    plt.plot(rlmres.model.endog,'o')
#    plt.plot(rlmres.fittedvalues,'-')
#
#    plt.show()
