import numpy as N
import numpy.linalg as NL
import enthought.traits as traits
from model import Model

class RegressionModelResults:
    def __init__(self):
        return

    def t(self):
        if not hasattr(self, '_sd'):
            self.sd()
        _t = N.zeros(_beta.shape, N.Float)

        for i in range(self.beta.shape[0]):
            _t[i] = _beta[i] / (self._sd * self.sqrt(self.normalized_cov_beta[i,i]))
        return _t

    def sd(self):
        if not hasattr(self, 'resid'):
            raise ValueError, 'need residuals to estimate standard deviation'
        self._sd = sqrt(self.scale)

    def norm_resid(self):
        if not hasattr(self, 'resid'):
            raise ValueError, 'need normalized residuals to estimate standard deviation'
        if not hasattr(self, '_sd'):
            self.sd()
        test = N.greater(_sd, 0)
        sdd = (test / (self._sd + 1. - test)) / sqrt(self.df)
        norm_resid = self.resid * N.multiply.outer(N.ones(Y.shape[0]), sdd)
        return norm_resid

    def predict(self, design):
        return N.dot(design, self.beta)

    def Rsq(self):
        """
        Return the R^2 value for each row of the response Y.
        """
        Rsq = self.scale / self.Ssq
        return Rsq

    def cov_beta(self, matrix=None, column=None, scale=None, **keywords):
        """
        Returns the variance/covariance matrix of a linear contrast
        of the estimates of beta, multiplied by scale which
        will usually be an estimate of sigma^2. The linear contrast
        is either specified as a column or a matrix.
        """
        if scale is None:
            scale = self.scale

        if column is not None:
            return self.normalized_cov_beta[column,column] * scale

        elif matrix is not None:
            tmp = N.dot(matrix, N.dot(self.normalized_cov_beta, matrix))
            return tmp * scale

class OLSModel(Model, traits.HasTraits):

    def initialize(self, design, **keywords):
        self._return = RegressionModelReturn(**keywords)
        self.design = design
        Model.__init__(self, **keywords)
        self._calc_beta = NL.generalized_inverse(self.design)
        self.normalized_cov_beta = N.dot(self.calc_beta, N.transpose(self.calc_beta))
        self._df_resid = self.design.shape[0] - N.add.reduce(N.dot(self.design, self.calc_beta))

    def df_resid(self, **keywords):
        return self._df_resid

    def fit(self, Y, **keywords):

        lfit = RegressionModelResults()

        lfit.beta = N.dot(self.calc_beta, Y)
        lfit.normalized_cov_beta = self.normalized_cov_beta
        lfit.df_resid = self.df_resid
        lfit.resid = Y - N.dot(self.wdesign, lfit.beta)
        lfit.scale = N.add.reduce(lfit.resid**2) / lift.df_resid

        lfit.Ssq = N.std(Y)**2

        return lfit
