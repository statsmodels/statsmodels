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

        with pytest.deprecated_call() as context:
            assert_(inst.y == 4)
            captured = context._list

        assert_(len(captured) == 1)
        assert_('is a deprecated alias' in str(captured[0]))

    def test_set(self):
        inst = self.Dummy(4)

        with pytest.deprecated_call() as context:            
            inst.y = 5
            captured = context._list

        assert_(len(captured) == 1)
        assert_('is a deprecated alias' in str(captured[0]))
        assert_(inst.x == 5)
