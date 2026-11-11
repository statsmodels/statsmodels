import pytest

from statsmodels.stats._inference_tools import _mover_confint


def test_mover_confint_invalid_contrast_raises():
    with pytest.raises(ValueError, match="contrast"):
        _mover_confint(1.0, 2.0, (0.5, 1.5), (1.5, 2.5), contrast="not-a-contrast")


def test_mover_confint_contrasts():
    stat1, stat2 = 3.0, 5.0
    ci1, ci2 = (2.0, 4.0), (4.0, 6.0)

    low_diff, upp_diff = _mover_confint(stat1, stat2, ci1, ci2, contrast="diff")
    assert low_diff < stat1 - stat2 < upp_diff

    low_sum, upp_sum = _mover_confint(stat1, stat2, ci1, ci2, contrast="sum")
    assert low_sum < stat1 + stat2 < upp_sum

    low_ratio, upp_ratio = _mover_confint(stat1, stat2, ci1, ci2, contrast="ratio")
    assert low_ratio < stat1 / stat2 < upp_ratio
