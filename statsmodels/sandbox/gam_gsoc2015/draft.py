import numpy as np
import matplotlib.pyplot as plt
from statsmodels.sandbox.gam_gsoc2015.gam import MultivariateGamPenalty, GLMGam, UnivariateGamPenalty
from statsmodels.sandbox.gam_gsoc2015.smooth_basis import (CubicSplines, PolynomialSmoother,
                                                           UnivariatePolynomialSmoother, UnivariateBSplines, CubicCyclicSplines)
from patsy import dmatrix
from statsmodels.sandbox.gam_gsoc2015.smooth_basis import _get_all_sorted_knots
n = 500
x = np.random.uniform(-1, 1, n)
y = 10*x**3 - 10*x + np.random.normal(0, 1, n)
y -= y.mean()


class NaturalCubicRegressionSplines():

    def __init__(self, x, df=10):
        self.df = df
        self.xs = dmatrix("cr(x, df=" + str(df) + ") - 1", {"x": x})

        n_inner_knots = df - 2  # +n_constraints # TODO: from patsy's CubicRegressionSplines class

        all_knots = _get_all_sorted_knots(x, n_inner_knots=n_inner_knots, inner_knots=None,
                                          lower_bound=None, upper_bound=None)
        b, d = self._get_cyclic_f(all_knots)
        self.s = d.T.dot(np.linalg.inv(b)).dot(d)
        return

    def _get_b_and_d(self, knots):
        """Returns mapping of cyclic cubic spline values to 2nd derivatives.

        .. note:: See 'Generalized Additive Models', Simon N. Wood, 2006, pp 146-147

        :param knots: The 1-d array knots used for cubic spline parametrization,
         must be sorted in ascending order.
        :return: A 2-d array mapping cyclic cubic spline values at
         knots to second derivatives.
        """
        h = knots[1:] - knots[:-1]
        n = knots.size - 1

        # b and d are defined such that the penalty matrix is equivalent to:
        # s = d.T.dot(b^-1).dot(d)
        # reference in particular to pag 146 of Wood's book
        b = np.zeros((n, n)) # the b matrix on page 146 of Wood's book
        d = np.zeros((n, n)) # the d matrix on page 146 of Wood's book

        b[0, 0] = (h[n - 1] + h[0]) / 3.
        b[0, n - 1] = h[n - 1] / 6.
        b[n - 1, 0] = h[n - 1] / 6.

        d[0, 0] = -1. / h[0] - 1. / h[n - 1]
        d[0, n - 1] = 1. / h[n - 1]
        d[n - 1, 0] = 1. / h[n - 1]

        for i in range(1, n):
            b[i, i] = (h[i - 1] + h[i]) / 3.
            b[i, i - 1] = h[i - 1] / 6.
            b[i - 1, i] = h[i - 1] / 6.

            d[i, i] = -1. / h[i - 1] - 1. / h[i]
            d[i, i - 1] = 1. / h[i - 1]
            d[i - 1, i] = 1. / h[i - 1]

        return b, d

cc = CubicCyclicSplines(x, df=12)
print(cc.basis_.shape)