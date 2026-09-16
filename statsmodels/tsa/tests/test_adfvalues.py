import numpy as np
import pytest

from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp


@pytest.mark.parametrize("regression", ["c", "n", "ct", "ctt"])
def test_mackinnonp_valid_regression(regression):
    # smoke test: must not raise, and returns a probability
    pvalue = mackinnonp(-2.5, regression=regression)
    assert 0 <= pvalue <= 1


def test_mackinnonp_invalid_regression_raises():
    with pytest.raises(ValueError, match="regression"):
        mackinnonp(-2.5, regression="not-a-regression")


@pytest.mark.parametrize("regression", ["c", "n", "ct", "ctt"])
def test_mackinnoncrit_valid_regression(regression):
    crit = mackinnoncrit(regression=regression)
    assert np.asarray(crit).shape == (3,)


def test_mackinnoncrit_invalid_regression_raises():
    with pytest.raises(ValueError, match="regression"):
        mackinnoncrit(regression="not-a-regression")
