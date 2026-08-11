"""
This module contains the one-parameter exponential families used
for fitting GLMs and GAMs.

These families are described in

   P. McCullagh and J. A. Nelder.  "Generalized linear models."
   Monographs on Statistics and Applied Probability.
   Chapman & Hall, London, 1983.

"""

from statsmodels.genmod.families import links
from statsmodels.tools._test_runner import PytestTester

from .family import (
    Binomial,
    Family,
    Gamma,
    Gaussian,
    InverseGaussian,
    NegativeBinomial,
    Poisson,
    Tweedie,
)

__all__ = [
           "Binomial",
           "Family",
           "Gamma",
           "Gaussian",
           "InverseGaussian",
           "NegativeBinomial",
           "Poisson",
           "Tweedie",
           "links",
           "test",
]

test = PytestTester()
