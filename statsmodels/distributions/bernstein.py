# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:35:23 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from scipy import stats

from statsmodels.tools.decorators import cache_readonly
from statsmodels.distributions.tools import (
        _Grid, cdf2prob_grid, prob2cdf_grid,
        _eval_bernstein_dd, _eval_bernstein_2d, _eval_bernstein_1d)


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

    @classmethod
    def from_data(cls, data, k_bins):
        """create distribution instance from data using histogram binning

        """
        data = np.asarray(data)
        if np.any(data < 0) or np.any(data > 1):
            raise ValueError("data needs to be in [0, 1]")

        k_dim = data.shape[1]
        if np.size(k_bins) == 1:
            k_bins = [k_bins] * k_dim
        bins = [np.linspace(-1 / ni, 1, ni + 2) for ni in k_bins]
        c, e = np.histogramdd(data, bins=bins, density=False)
        # TODO: check when we have zero observations, which bin?
        # check bins start at 0 exept leading bin
        assert all([ei[1] == 0 for ei in e])
        c /= len(data)

        cdf_grid = prob2cdf_grid(c)
        return cls(cdf_grid)

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

    def rvs(self, nobs):
        """generate random numbers from distribution

        """
        rvs_mnl = np.random.multinomial(nobs, self.prob_grid.flatten())
        k_comp = self.k_dim
        rvs_m = []
        for i in range(len(rvs_mnl)):
            if rvs_mnl[i] != 0:
                idx = np.unravel_index(i, self.prob_grid.shape)
                rvsi = []
                for j in range(k_comp):
                    n = self.k_grid[j]
                    xgi = self._grid.x_marginal[j][idx[j]]
                    # Note: x_marginal starts at 0
                    #       x_marginal ends with 1 but that is not used by idx
                    rvsi.append(stats.beta.rvs(n * xgi + 1, n * (1-xgi) + 0,
                                               size=rvs_mnl[i]))
                rvs_m.append(np.column_stack(rvsi))

        rvsm = np.concatenate(rvs_m)
        return rvsm


class BernsteinDistributionBV(BernsteinDistribution):

    def cdf(self, x):
        cdf_ = _eval_bernstein_2d(x, self.cdf_grid)
        return cdf_

    def pdf(self, x):
        # TODO: check usage of k_grid_product. Should this go into eval?
        pdf_ = self.k_grid_product * _eval_bernstein_2d(x, self.prob_grid)
        return pdf_


class BernsteinDistributionUV(BernsteinDistribution):

    def cdf(self, x, method="binom"):
        cdf_ = _eval_bernstein_1d(x, self.cdf_grid, method=method)
        return cdf_

    def pdf(self, x, method="binom"):
        # TODO: check usage of k_grid_product. Should this go into eval?
        pdf_ = self.k_grid_product * _eval_bernstein_1d(x, self.prob_grid,
                                                        method=method)
        return pdf_
