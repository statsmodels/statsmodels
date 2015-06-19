''' This file contains draft of code. Do not look at it '''
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from smooth_basis import make_poly_basis, make_bsplines_basis
from gam import GamPenalty, LogitGam, GLMGam, MultivariateGamPenalty
import statsmodels.api as sm
from statsmodels.genmod import GLMResults

sigmoid = np.vectorize(lambda x: 1.0/(1.0 + np.exp(-x)))

x = np.linspace(1, 10, 10)
poly, d1, d2 = make_poly_basis(x, degree=3, intercept=False)

print(pd.DataFrame(poly))

class GLMGAMResults(GLMResults):

    def plot_predict(self, x_values=None):
        """just to try a method in overridden Results class
        """
        import matplotlib.pyplot as plt

        if x_values is None:
            plt.plot(self.model.endog, '.')
            plt.plot(self.predict())
        else:
            plt.plot(x_values, self.model.endog, '.')
            plt.plot(x_values, self.predict())


class GLMGam(PenalizedMixin, GLM):

    _results_class = GLMGAMResults
