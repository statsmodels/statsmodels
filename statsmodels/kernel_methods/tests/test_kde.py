from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .. import kde, kde_methods, kernels
from ...tools.testing import assert_equal, assert_raises


class TestConstruction(object):
    def test_simple(self):
        kde.KDE([1, 2, 3])

    def test_long(self):
        kde.KDE([1, 2, 3], lower=1, upper=2, adjust=[3, 2, 1], weights=4.,
                bandwidth=0.5)

    def test_fit(self):
        k = kde.KDE([1, 2, 3], bandwidth=0.5)
        m = k.fit(lower=0, upper=2)
        assert_equal(m.lower, 0.)
        assert_equal(m.upper, 2.)

    def test_bad_init(self):
        assert_raises(TypeError, kde.KDE, [1, 2, 3], ndim=3)

    def test_bad_fit(self):
        k = kde.KDE([1, 2, 3], bandwidth=0.5)
        assert_raises(AttributeError, k.fit, ndim=3)

    def change_kde(self, name, value):
        k = kde.KDE([1, 2, 3])
        getattr(k, "set_"+name)(value)
        assert_equal(getattr(k, name), value)

    def test_setter(self):
        attrs = dict(lower=0,
                     upper=3,
                     exog=np.arange(4, dtype=float).reshape(4, 1),
                     axis_type='o',
                     method=kde_methods.Cyclic1D(),
                     weights=[3., 2., 1.],
                     adjust=3.,
                     bandwidth=0.7,
                     kernel=kernels.tricube)
        for a in attrs:
            yield self.change_kde, a, attrs[a]

    def test_bad_setter(self):
        k = kde.KDE([1, 2, 3])
        with assert_raises(AttributeError):
            k.set_ndim(3)
