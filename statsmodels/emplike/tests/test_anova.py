import numpy as np
from numpy.testing import assert_almost_equal

from statsmodels.datasets import star98
from statsmodels.emplike.elanova import ANOVA

from .results.el_results import ANOVAResults

DATA = np.asarray(star98.load().exog)[:30, 1:3]


def test_anova():
    res1 = ANOVA([DATA[:, 0], DATA[:, 1]])
    res2 = ANOVAResults()
    assert_almost_equal(res1.compute_ANOVA()[:2], res2.compute_ANOVA[:2], 4)
    assert_almost_equal(res1.compute_ANOVA()[2], res2.compute_ANOVA[2], 4)
    assert_almost_equal(
        res1.compute_ANOVA(return_weights=True)[3], res2.compute_ANOVA[3], 4
    )
