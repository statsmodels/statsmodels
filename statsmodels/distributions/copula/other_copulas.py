# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:19:45 2021

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from scipy._lib._util import check_random_state  # noqa

from statsmodels.distributions.copula.copulas import Copula


class IndependentCopula(Copula):
    """Independent copula.

    .. math::

        C_\theta(u,v) = uv

    """
    def __init__(self, d=2):
        self.d = d
        super().__init__(d=self.d)

    def random(self, n=1, random_state=None):
        rng = check_random_state(random_state)
        x = rng.random((n, self.d))
        return x

    def pdf(self, u):
        return np.ones((len(u), 1))

    def cdf(self, u):
        return np.prod(u, axis=1)

    def plot_pdf(self, *args):
        raise NotImplementedError("PDF is constant over the domain.")