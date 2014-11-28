import nose.tools as nt
import numpy as np
import numpy.testing as nptest

import statsmodels.api as sm
from statsmodels.graphics._probscale import (
    _sig_figs,
    _get_probs,
    ProbFormatter,
    ProbTransform,
    ProbScale,
    InvertedProbTransform
)
from scipy import stats


class test__get_probs(object):
    @nt.nottest
    def compare_probs(self, N, expected):
        probs = _get_probs(N)
        nptest.assert_array_almost_equal(probs, expected, decimal=6)

    def test_Order1(self):
        N = 7
        expected = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
        self.compare_probs(N, expected)

    def test_Order2(self):
        N = 77
        expected = np.array([1, 2, 5, 10, 20, 30, 40, 50,
                             60, 70, 80, 90, 95, 98, 99], dtype=float)
        self.compare_probs(N, expected)

    def test_Order3(self):
        N = 378
        expected = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 40, 50,
                             60, 70, 80, 90, 95, 98, 99, 99.5, 99.8, 99.9])
        self.compare_probs(N, expected)

    def test_Order6(self):
        N = 4.5e6
        expected = np.array([
            1.0000e-05, 1.00000e-04, 1.000000e-03,
            1.0000e-02, 2.00000e-02, 5.000000e-02,
            1.0000e-01, 2.00000e-01, 5.000000e-01,
            1.0000e+00, 2.00000e+00, 5.000000e+00,
            1.0000e+01, 2.00000e+01, 3.000000e+01,
            4.0000e+01, 5.00000e+01, 6.000000e+01,
            7.0000e+01, 8.00000e+01, 9.000000e+01,
            9.5000e+01, 9.80000e+01, 9.900000e+01,
            9.9500e+01, 9.98000e+01, 9.990000e+01,
            9.9950e+01, 9.99800e+01, 9.999000e+01,
            9.9999e+01, 9.99999e+01, 9.999999e+01
        ])
        self.compare_probs(N, expected)


class base__sig_figs_Mixin(object):
    def teardown(self):
        pass

    def test_baseline(self):
        nt.assert_equal(_sig_figs(self.x, 3), self.known_3)
        nt.assert_equal(_sig_figs(self.x, 4), self.known_4)

    def test_trailing_zeros(self):
        nt.assert_equal(_sig_figs(self.x, 8), self.known_8)

    @nptest.raises(ValueError)
    def test_sigFigs_zero_n(self):
        _sig_figs(self.x, 0)

    @nptest.raises(ValueError)
    def test_sigFigs_negative_n(self):
        _sig_figs(self.x, -1)


class test__sig_figs_gt1(base__sig_figs_Mixin):
    def setup(self):
        self.x = 1234.56
        self.known_3 = '1,230'
        self.known_4 = '1,235'
        self.known_8 = '1,234.5600'
        self.known_exp3 = '1.23e+08'
        self.factor = 10**5


class test__sig_figs_lt1(base__sig_figs_Mixin):
    def setup(self):
        self.x = 0.123456
        self.known_3 = '0.123'
        self.known_4 = '0.1235'
        self.known_8 = '0.12345600'
        self.known_exp3 = '1.23e-07'
        self.factor = 10**-6


class test_ProbFormatter(object):
    def setup(self):
        self.pf = ProbFormatter()

    @nt.nottest
    def compare_formats(self, val, known):
        nt.assert_equal(self.pf(val), known)

    def test_lt001(self):
        self.compare_formats(0.0002, '0.0002')
        self.compare_formats(0.00078, '0.0008')

    def test_lt1(self):
        self.compare_formats(0.1, '0.1')
        self.compare_formats(0.512,'0.5')

    def test_lt10(self):
        self.compare_formats(3, '3')
        self.compare_formats(9, '9')

    def test_lt99(self):
        self.compare_formats(24, '24')
        self.compare_formats(98, '98')
        self.compare_formats(99, '99')

    def test_gt99(self):
        self.compare_formats(99.9, '99.9')
        self.compare_formats(99.9995, '99.9995')
        self.compare_formats(99.9981, '99.998')


class base_transformMixin(object):
    def test_transform_non_affine_array(self):
        nptest.assert_array_almost_equal(
            self.transform.transform_non_affine(self.array),
            self.known_transformed_array,
            decimal=4
        )

    def test_transform_non_affine_scalar(self):
        nt.assert_equal(
            self.transform.transform_non_affine(self.scalar),
            self.known_transformed_scalar
        )

    def test_inverted(self):
        transinv = self.transform.inverted()
        nt.assert_equal(type(transinv), type(self.known_inverted))
        nt.assert_equal(transinv.dist, self.known_inverted.dist)


class test_ProbTransform_Norm(base_transformMixin):
    def setup(self):
        self.dist = stats.norm
        self.transform = ProbTransform(self.dist)
        self.array = _get_probs(8)
        self.known_transformed_array = np.array([
            -1.28155157, -0.84162123, -0.52440051, -0.2533471 ,  0.,
            0.2533471 ,  0.52440051,  0.84162123,  1.28155157
        ])
        self.scalar = 50.0
        self.known_transformed_scalar = 0.0
        self.known_inverted = InvertedProbTransform(self.dist)


class test_InvertedProbTransform_Norm(base_transformMixin):
    def setup(self):
        self.dist = stats.norm
        self.transform = InvertedProbTransform(self.dist)
        self.array = np.arange(-3, 3.1, 0.5)
        self.known_transformed_array = np.array([
             0.1349898 ,   0.62096653,   2.27501319,   6.68072013,
            15.86552539,  30.85375387,  50.        ,  69.14624613,
            84.13447461,  93.31927987,  97.72498681,  99.37903347,  99.8650102
        ])
        self.scalar = 0.
        self.known_transformed_scalar = 50.
        self.known_inverted = ProbTransform(self.dist)

