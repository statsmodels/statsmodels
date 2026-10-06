"""
Created on Jan. 26, 2025 4:17:49 p.m.

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from numpy.testing import assert_
from statsmodels.stats._multicomp_tools import group_pairwise


class CheckPairwiseGrouping(object):

    def test_basic(self):
        assert_(self.gr.groupings == self.res_groupings)
        assert_(self.gr.letter == self.res_letter)


class TestPairwiseGrouping1(CheckPairwiseGrouping):
    # example 1 from Piepho 2004

    @classmethod
    def setup_class(cls):
        edges = np.asarray([[1, 2], [1, 3], [1, 4], [2, 4]]) - 1
        cls.res_groupings = [frozenset({0}),
                             frozenset({1, 2}),
                             frozenset({2, 3})]
        cls.res_letter = ['a  ', ' b ', ' bc', '  c']

        cls.gr = group_pairwise(edges)


class TestPairwiseGrouping2(CheckPairwiseGrouping):
    # example 3 from Piepho 2004

    @classmethod
    def setup_class(cls):
        edges = np.asarray([[1, 7], [1, 8], [2, 4], [2, 5], [3, 5]]) - 1
        cls.res_groupings = [frozenset({0, 1, 2, 5}),
                             frozenset({0, 2, 3, 5}),
                             frozenset({0, 3, 4, 5}),
                             frozenset({1, 2, 5, 6, 7}),
                             frozenset({2, 3, 5, 6, 7}),
                             frozenset({3, 4, 5, 6, 7})]
        cls.res_letter = ['abc   ', 'a  d  ', 'ab de ', ' bc ef', '  c  f',
                          'abcdef', '   def', '   def']

        cls.gr = group_pairwise(edges)


class TestPairwiseGrouping3(CheckPairwiseGrouping):
    # example wheat from Piepho 2004

    @classmethod
    def setup_class(cls):
        edgesli = [[12, [5, 7, 10]],
                   [16, [12]],
                   [18, [3, 12]],
                   [20, [2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 16, 18]]
                  ]
        edges = np.asarray([[i, j] for i, row in edgesli for j in row]) - 1

        cls.res_groupings = [
                frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15,
                           16, 18}),
                frozenset({0, 1, 2, 3, 5, 7, 8, 10, 11, 12, 13, 14, 16, 18}),
                frozenset({0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16,
                           17, 18}),
                frozenset({0, 7, 11, 13, 14, 16, 18, 19})]
        cls.res_letter = [
            'abcd', 'abc ', 'ab  ', 'abc ', 'a c ', 'abc ', 'a c ', 'abcd',
            'abc ', 'a c ', 'abc ', ' b d', 'abc ', 'abcd', 'abcd', 'a c ',
            'abcd', '  c ', 'abcd', '   d']

        cls.gr = group_pairwise(edges)
