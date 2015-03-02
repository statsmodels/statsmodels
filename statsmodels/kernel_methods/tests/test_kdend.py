from __future__ import division, absolute_import, print_function

from .. import kde1d_methods, bandwidths
import numpy as np
from numpy.random import randn
from scipy import integrate
from . import _kde_utils
from nose.plugins.attrib import attr

@attr('nonparametric')
class TestBasic(object):
    @classmethod
    def setUpClass(cls):
        cls.ratios = np.array([1., 2., 5.])
        d = randn(500, 2)
        cls.vs = (cls.ratios[:, np.newaxis, np.newaxis] *
                  np.concatenate(3 * [d[np.newaxis]], axis=0))
