"""Test values for multinomial_proportion_confint.

Author: Sébastien Lerique
"""

import collections

import numpy as np

from statsmodels.tools.testing import Holder

res_multinomial = collections.defaultdict(Holder)

# The following examples come from the Sison & Glaz paper, and the values were
# computed using the R MultinomialCI package.

# Floating-point arithmetic errors get blown up in the Edgeworth expansion
# (starting in g1 and g2, but mostly when computing f, because of the
# polynomials), which explains why we only obtain a precision of 4 decimals
# when comparing to values computed in R.

# We test with any method name that starts with 'sison', as that is the
# criterion.
key1 = ("sison", "Sison-Glaz example 1")
res_multinomial[key1].proportions = [56, 72, 73, 59, 62, 87, 58]
res_multinomial[key1].cis = np.array(
    [
        [0.07922912, 0.1643361],
        [0.11349036, 0.1985973],
        [0.11563169, 0.2007386],
        [0.08565310, 0.1707601],
        [0.09207709, 0.1771840],
        [0.14561028, 0.2307172],
        [0.08351178, 0.1686187],
    ]
)
res_multinomial[key1].precision = 4

key2 = ("sisonandglaz", "Sison-Glaz example 2")
res_multinomial[key2].proportions = [5] * 50
res_multinomial[key2].cis = [0, 0.05304026] * np.ones((50, 2))
res_multinomial[key2].precision = 4

key3 = ("sison-whatever", "Sison-Glaz example 3")
res_multinomial[key3].proportions = (
    [1] * 10 + [12] * 10 + [5] * 10 + [3] * 10 + [4] * 10
)
res_multinomial[key3].cis = np.concatenate(
    [
        [0, 0.04120118] * np.ones((10, 2)),
        [0.012, 0.08520118] * np.ones((10, 2)),
        [0, 0.05720118] * np.ones((10, 2)),
        [0, 0.04920118] * np.ones((10, 2)),
        [0, 0.05320118] * np.ones((10, 2)),
    ]
)
res_multinomial[key3].precision = 4

# The examples from the Sison & Glaz paper only include 3 decimals.
gkey1 = ("goodman", "Sison-Glaz example 1")
res_multinomial[gkey1].proportions = [56, 72, 73, 59, 62, 87, 58]
res_multinomial[gkey1].cis = np.array(
    [
        [0.085, 0.166],
        [0.115, 0.204],
        [0.116, 0.207],
        [0.091, 0.173],
        [0.096, 0.181],
        [0.143, 0.239],
        [0.089, 0.171],
    ]
)
res_multinomial[gkey1].precision = 3

gkey2 = ("goodman", "Sison-Glaz example 2")
res_multinomial[gkey2].proportions = [5] * 50
res_multinomial[gkey2].cis = [0.005, 0.075] * np.ones((50, 2))
res_multinomial[gkey2].precision = 3

gkey3 = ("goodman", "Sison-Glaz example 3")
res_multinomial[gkey3].proportions = (
    [1] * 10 + [12] * 10 + [5] * 10 + [3] * 10 + [4] * 10
)
res_multinomial[gkey3].cis = np.concatenate(
    [
        [0, 0.049] * np.ones((10, 2)),
        [0.019, 0.114] * np.ones((10, 2)),
        [0.005, 0.075] * np.ones((10, 2)),
        [0.002, 0.062] * np.ones((10, 2)),
        [0.004, 0.069] * np.ones((10, 2)),
    ]
)
res_multinomial[gkey3].precision = 3
