import warnings

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest

from statsmodels.datasets import star98
from statsmodels.emplike.elanova import ANOVA, ANOVAResult

from .results.el_results import ANOVAResults

DATA = np.asarray(star98.load().exog)[:30, 1:3]


def test_anova():
    res1 = ANOVA([DATA[:, 0], DATA[:, 1]])
    res2 = ANOVAResults()
    res = res1.compute_ANOVA(result_object=True)
    assert_almost_equal(res[:2], res2.compute_ANOVA[:2], 4)
    assert_almost_equal(res[2], res2.compute_ANOVA[2], 4)
    assert_almost_equal(
        res1.compute_ANOVA(return_weights=True)[3], res2.compute_ANOVA[3], 4
    )


def test_anova_namedtuple():
    res1 = ANOVA([DATA[:, 0], DATA[:, 1]])

    # return_weights=True already yields four values, so the NamedTuple is
    # adopted silently
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        full = res1.compute_ANOVA(return_weights=True)
    assert isinstance(full, ANOVAResult)
    assert_equal(len(full), 4)

    # the three-value path still warns and still returns a plain tuple
    with pytest.warns(FutureWarning, match="compute_ANOVA"):
        legacy = res1.compute_ANOVA()
    assert not isinstance(legacy, ANOVAResult)
    assert_equal(len(legacy), 3)
    assert_almost_equal(legacy, full[:3], 10)

    # opting in or out is silent either way
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        opted_in = res1.compute_ANOVA(result_object=True)
        opted_out = res1.compute_ANOVA(result_object=False)
    assert isinstance(opted_in, ANOVAResult)
    assert opted_in.weights is not None
    assert_equal(len(opted_out), 3)
