'''examples to check summary, not converted to tests yet


'''
from __future__ import print_function

import numpy as np  # noqa: F401

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
