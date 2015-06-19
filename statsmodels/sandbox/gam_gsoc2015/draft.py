''' This file contains draft of code. Do not look at it '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from smooth_basis import make_poly_basis, make_bsplines_basis
from gam import GamPenalty, LogitGam, GLMGam, MultivariateGamPenalty
import statsmodels.api as sm

sigmoid = np.vectorize(lambda x: 1.0/(1.0 + np.exp(-x)))

x = np.linspace(1, 10, 10)
poly, d1, d2 = make_poly_basis(x, degree=3, intercept=False)

print(pd.DataFrame(poly))
