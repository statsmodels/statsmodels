# -*- coding: utf-8 -*-
# Test Bland-Altman
"""

Author: Joses Ho

"""

import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Seed the random number generator.
# This ensures that the results below are reproducible.
np.random.seed(9999)
m1 = np.random.random(20)
m2 = np.random.random(20)

f, ax = plt.subplots(1, figsize = (8,5))
sm.graphics.mean_diff_plot(m1, m2, ax = ax)

plt.show()
