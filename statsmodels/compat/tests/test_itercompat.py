"""

Created on Wed Feb 29 10:34:00 2012

Author: Josef Perktold
"""
from itertools import combinations, zip_longest

from numpy.testing import assert_

from statsmodels.compat import lrange


def test_zip_longest():
    lili = [
        ["a0", "b0", "c0", "d0"],
        ["a1", "b1", "c1"],
        ["a2", "b2", "c2", "d2"],
        ["a3", "b3", "c3", "d3"],
        ["a4", "b4"],
    ]

    transposed = [
        ("a0", "a1", "a2", "a3", "a4"),
        ("b0", "b1", "b2", "b3", "b4"),
        ("c0", "c1", "c2", "c3", None),
        ("d0", None, "d2", "d3", None),
    ]

    assert_(
        list(zip_longest(*lili)) == transposed,
        f"{zip_longest(*lili)!r} not equal {transposed!r}",
    )


def test_combinations():
    actual = list(combinations("ABCD", 2))
    desired = [
        ("A", "B"),
        ("A", "C"),
        ("A", "D"),
        ("B", "C"),
        ("B", "D"),
        ("C", "D"),
    ]
    assert_(actual == desired, f"{actual!r} not equal {desired!r}")

    actual = list(combinations(lrange(4), 3))
    desired = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    assert_(actual == desired, f"{actual!r} not equal {desired!r}")
