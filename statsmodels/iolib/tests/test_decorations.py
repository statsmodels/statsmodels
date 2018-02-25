import numpy as np

from numpy.testing import assert_equal, assert_raises

from statsmodels.iolib.decorations import pvalue_to_stars


def test_pvalue_to_stars():
    assert_equal(pvalue_to_stars(0), '****')
    assert_equal(pvalue_to_stars(np.nextafter(0.0001, 1)), '***')

    assert_equal(pvalue_to_stars(0.001), '***')
    assert_equal(pvalue_to_stars(np.nextafter(0.001, 1)), '**')

    assert_equal(pvalue_to_stars(0.01), '**')
    assert_equal(pvalue_to_stars(np.nextafter(0.01, 1)), '*')

    assert_equal(pvalue_to_stars(0.05), '*')
    assert_equal(pvalue_to_stars(np.nextafter(0.05, 1)), '')

    assert_equal(pvalue_to_stars(1), '')

    assert_raises(ValueError, lambda: pvalue_to_stars(np.nextafter(0, -1)))
    assert_raises(ValueError, lambda: pvalue_to_stars(np.nextafter(1, 2)))
