# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:35:23 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np

from statsmodels.tools.decorators import cache_readonly
from statsmodels.distributions.tools import (
        _Grid, cdf2prob_grid, _eval_bernstein_dd, _eval_bernstein_2d)


class BernsteinDistribution(object):
    """Distribution based on Bernstein Polynomials on unit hypercube


    Parameters
    ----------
    cdf_grid : array_like
        cdf values on a equal spaced grid of the unit hypercube [0, 1]^d.
        The dimension of the arrays define how many random variables are
        included in the multivariate distribution.


    """

    def __init__(self, cdf_grid):
        self.cdf_grid = cdf_grid = np.asarray(cdf_grid)
        self.k_dim = cdf_grid.ndim
        self.k_grid = cdf_grid.shape
        self.k_grid_product = np.product([i-1 for i in self.k_grid])
        self._grid = _Grid(self.k_grid)

    @cache_readonly
    def prob_grid(self):
        return cdf2prob_grid(self.cdf_grid, prepend=None)

    def cdf(self, x):
        cdf_ = _eval_bernstein_dd(x, self.cdf_grid)
        return cdf_

    def pdf(self, x):
        # TODO: check usage of k_grid_product. Should this go into eval?
        pdf_ = self.k_grid_product * _eval_bernstein_dd(x, self.prob_grid)
        return pdf_

    def get_marginal(self, idx):
        """get marginal BernsteinDistribution

        currently only 1-dim margins, `idx` is int

        Status: not verified yet except for uniform margins.
        This uses the smoothed cdf values in the new marginal distribution and
        not the grid values corresponding to the margin of the multivariate
        grid. This might change.

        """

        # univariate
        if self.k_dim == 1:
            return self

        x_m = np.ones((self.k_grid[idx], self.k_dim))
        x_m[:, idx] = self._grid.x_marginal[idx]
        cdf_m = self.cdf(x_m)
        bpd_marginal = BernsteinDistribution(cdf_m)
        return bpd_marginal


class BernsteinDistributionBV(BernsteinDistribution):

    def cdf(self, x):
        cdf_ = _eval_bernstein_2d(x, self.cdf_grid)
        return cdf_

    def pdf(self, x):
        # TODO: check usage of k_grid_product. Should this go into eval?
        pdf_ = self.k_grid_product * _eval_bernstein_2d(x, self.prob_grid)
        return pdf_
