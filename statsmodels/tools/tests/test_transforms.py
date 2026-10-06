# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 18:46:18 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from numpy.testing import assert_allclose
from statsmodels.tools import numdiff as nd
from statsmodels.tools.transforms import (
        SinhArcsinh, Sinh, BirnbaumSaunders, TMI, _TMI1, RtoInterval)


class CheckConsistency(object):

    def test_transform(self):
        tr = self.transf
        x = self.x
        z0 = self.z
        params = self.params

        # test round tripp
        z = tr.transform(x, params)
        xr = tr.inverse(z, params)
        assert_allclose(x, xr, rtol=1e-13, atol=1e-20)
        # the following currently only tests that we get twice the same
        # need reference or regression test numbers in self.z
        assert_allclose(z, z0, rtol=1e-13, atol=1e-20)

    def test_deriv(self):
        tr = self.transf
        x = self.x
        z = self.z
        params = self.params

        dx1 = tr.deriv(x, params)
        dx2 = 1 / tr.deriv_inverse(z, params)
        assert_allclose(dx1, dx2, rtol=1e-13, atol=1e-20)

        dz1 = tr.deriv_inverse(z, params)
        dz2 = 1 / tr.deriv(x, params)
        assert_allclose(dz1, dz2, rtol=1e-13, atol=1e-20)

        dx = [np.squeeze(nd.approx_fprime(np.asarray([xi]),
                                          tr.transform, args=(params,)))
              for xi in x]
        assert_allclose(dx1, dx, rtol=1e-6, atol=1e-10)

    def test_deriv2(self):
        tr = self.transf
        x = self.x
        z = self.z
        params = self.params

        dx1 = tr.deriv2(x, params)
        dx2 = -tr.deriv2_inverse(z, params) / tr.deriv_inverse(z, params)**3
        assert_allclose(dx1, dx2, rtol=1e-13, atol=1e-20)

        dz1 = tr.deriv2_inverse(z, params)
        dz2 = -tr.deriv2(x, params) / tr.deriv(x, params)**3
        assert_allclose(dz1, dz2, rtol=1e-13, atol=1e-20)

        dx = [np.squeeze(nd.approx_hess(np.asarray([xi]),
                                        tr.transform, args=(params,)))
              for xi in x]
        assert_allclose(dx1, dx, rtol=1e-4, atol=1e-10)


class TestSinhArcsinh(CheckConsistency):

    @classmethod
    def setup_class(cls):
        cls.params = (0.1, 1)
        cls.transf = SinhArcsinh()
        cls.x = np.linspace(-5, 5, 5)
        # consitency check, not verified
        cls.z = cls.transf.transform(cls.x, cls.params)


class TestSinh(CheckConsistency):

    @classmethod
    def setup_class(cls):
        cls.params = (0.1, 1)
        cls.transf = Sinh()
        cls.x = np.linspace(-5, 5, 5)
        # consitency check, not verified
        cls.z = cls.transf.transform(cls.x, cls.params)


class TestBirnbaumSaunders(CheckConsistency):

    @classmethod
    def setup_class(cls):
        cls.params = (0.1, 1)
        cls.transf = BirnbaumSaunders()
        cls.x = np.linspace(-5, 5, 5)
        # consitency check, not verified
        cls.z = cls.transf.transform(cls.x, cls.params)


class TestTMI(CheckConsistency):

    @classmethod
    def setup_class(cls):
        cls.params = (0.1, 1)
        cls.transf = TMI()
        cls.x = np.linspace(-5, 5, 5)
        # consitency check, not verified
        cls.z = cls.transf.transform(cls.x, cls.params)


class TestTMI1(CheckConsistency):

    @classmethod
    def setup_class(cls):
        cls.params = (0.1, 1)
        cls.transf = _TMI1()
        cls.x = np.linspace(-5, 5, 5)
        # consitency check, not verified
        cls.z = cls.transf.transform(cls.x, cls.params)


class TestRtoInterval(CheckConsistency):

    @classmethod
    def setup_class(cls):
        cls.params = (0.1, 1)
        cls.transf = RtoInterval()
        # TODO: we avoid having zero in x because of numerical problems
        # it should work better at and around zero when we have
        # taylor series expansion
        # currently numerical derivatives are not reliable at around zero
        cls.x = np.linspace(-5, 6, 5)
        # consitency check, not verified
        cls.z = cls.transf.transform(cls.x, cls.params)
