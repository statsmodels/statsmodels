# -*- coding: utf-8 -*-
import warnings

import pytest
from numpy.testing import assert_

from statsmodels.tools.decorators import deprecated_alias


class TestDeprecatedAlias(object):

    @classmethod
    def setup_class(cls):
        
        class Dummy(object):

            y = deprecated_alias('y', 'x', '0.11.0')

            def __init__(self, y):
                self.x = y

        cls.Dummy = Dummy

    def test_get(self):
        inst = self.Dummy(4)

        with warnings.catch_warnings(record=True) as record:
            assert inst.y == 4

        assert len(record) == 1, record
        assert 'is a deprecated alias' in str(record[0])

    def test_set(self):
        inst = self.Dummy(4)

        with warnings.catch_warnings(record=True) as record:
            inst.y = 5

        assert len(record) == 1, record
        assert 'is a deprecated alias' in str(record[0])
        assert inst.x == 5
