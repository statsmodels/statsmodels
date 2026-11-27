"""

Created on Wed Mar 28 13:49:11 2012

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest
from scipy.stats import studentized_range

from statsmodels.sandbox.stats.multicomp import get_tukeyQcrit
from statsmodels.stats.libqsturng import psturng, qsturng


@pytest.mark.parametrize("alpha", [0.01, 0.05])
@pytest.mark.parametrize("k", np.arange(2, 11))
def test_qstrung(alpha, k):
    rows = [5,    6,    7,    8,    9,   10,   11,   12,   13,   14,   15,
            16,   17,   18,   19,   20,   24,   30,   40,   60,  120, 9999]

    c1 = get_tukeyQcrit(k, rows, alpha=alpha)
    c2 = qsturng(1 - alpha, k, rows)
    assert_almost_equal(c1, c2, decimal=2)
    # roundtrip
    assert_almost_equal(psturng(qsturng(1 - alpha, k, rows), k, rows),
                        alpha,
                        5)


class TestStudentizedRangeScipy:
    # GH 8185: psturng bounded the returned probability to
    # [0.001, 0.9] and qsturng rejected p outside [0.1, 0.999].
    # Both now delegate to scipy.stats.studentized_range.

    @pytest.mark.parametrize("r", [2, 5, 10])
    @pytest.mark.parametrize("v", [5, 25, 120])
    def test_psturng_matches_scipy_tails(self, r, v):
        # the q values cover probabilities below 0.001 and above 0.9,
        # which the previous implementation returned as 0.001 and 0.9
        q = np.array([0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0])
        assert_allclose(
            psturng(q, r, v), studentized_range.sf(q, r, v), rtol=1e-10
        )

    def test_psturng_not_bounded(self):
        # values at arguments that were previously clamped to the
        # bounds 0.001 and 0.9
        assert_allclose(psturng(8.0, 5, 25), 6.2879665e-05, rtol=1e-6)
        assert_allclose(psturng(0.5, 5, 25), 0.996425, rtol=1e-6)
        assert psturng(8.0, 5, 25) < 0.001
        assert psturng(0.5, 5, 25) > 0.9

    @pytest.mark.parametrize(
        "p", [0.001, 0.01, 0.1, 0.5, 0.9, 0.999, 0.9999]
    )
    def test_qsturng_matches_scipy(self, p):
        # p below 0.1 and above 0.999 raised a ValueError before
        assert_allclose(
            qsturng(p, 5, 25), studentized_range.ppf(p, 5, 25), rtol=1e-10
        )

    def test_roundtrip_extreme(self):
        for p in [0.001, 0.9999]:
            assert_allclose(
                psturng(qsturng(p, 5, 25), 5, 25), 1 - p, atol=1e-8
            )

    def test_invalid_arguments(self):
        with pytest.raises(ValueError, match="p must be between 0 and 1"):
            qsturng(1.5, 5, 25)
        with pytest.raises(ValueError, match="q should be >= 0"):
            psturng(-1.0, 5, 25)

    def test_small_v_restriction_kept(self):
        # for 1 <= v < 2, probabilities above 0.1 still raise,
        # as in previous versions
        with pytest.raises(ValueError, match="v must be >= 2"):
            psturng(0.5, 5, 1.5)
        with pytest.raises(ValueError, match="v must be >= 2"):
            qsturng(0.5, 5, 1.5)
        # but the 0.001 bound no longer applies for 1 <= v < 2
        assert_allclose(
            psturng(30.0, 5, 1.5),
            studentized_range.sf(30.0, 5, 1.5),
            rtol=1e-10,
        )
