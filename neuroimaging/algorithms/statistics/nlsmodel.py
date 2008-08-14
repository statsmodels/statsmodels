"""
TODO
"""
__docformat__ = 'restructuredtext'

import numpy as np
import numpy.linalg as L
from neuroimaging.fixes.scipy.stats.models.model import Model

class NLSModel(Model):

    """
    Class representing a simple nonlinear least squares model.
    """

    def __init__(self, Y, design, f, grad, theta, niter=10):
        """
        :Parameters:
            Y : TODO
                the data in the NLS model
            design : TODO
                the deisng matrix, X
            f : TODO
                the map between the linear parameters (in the design matrix) and
                the nonlinear parameters (theta)
            grad :  TODO
                the gradient of f, this should be a function of an nxp design
                matrix X and qx1 vector theta that returns an nxq matrix
                df_i/dtheta_j where

                f_i(theta) = f(X[i], theta)

                is the nonlinear response function for the i-th instance in
                the model.
        """



        Model.__init__(self)
        self.Y = Y
        self.design = design
        self.f = f
        self.grad = grad
        self.theta = theta
        self.niter = niter
        if self.design is not None and self.Y != None:
            if self.Y.shape[0] != self.design.shape[0]:
                raise ValueError, 'Y should be same shape as design'


    def _Y_changed(self):
        if self.design is not None:
            if self.Y.shape[0] != self.design.shape[0]:
                raise ValueError, 'Y should be same shape as design'

    def _design_changed(self):
        if self.Y is not None:
            if self.Y.shape[0] != self.design.shape[0]:
                raise ValueError, 'Y should be same shape as design'

    def getZ(self):
        """
        :Returns: ``None``
        """
        self._Z = self.grad(self.design, self.theta)

    def getomega(self):
        """
        :Returns: ``None``
        """
        self._omega = self.predict() - np.dot(self._Z, self.theta)

    def predict(self, design=None):
        """
        :Parameters:
            design : TODO
                TODO

        :Returns: TODO
        """
        if design is None:
            design = self.design
        return self.f(design, self.theta)

    def SSE(self):
        """ Sum of squares error.

        :Returns; TODO
        """
        return sum((self.Y - self.predict())**2)

    def __iter__(self):
        """
        :Returns: ``self``
        """
        if self.theta is not None:
            self.initial = self.theta
        elif self.initial is not None:
            self.theta = self.initial
        else:
            raise ValueError, 'need an initial estimate for theta'

        self._iter = 0
        self.theta = self.initial
        return self

    def next(self):
        """
        :Returns: ``None``
        """
        if self._iter < self.niter:
            self.getZ()
            self.getomega()
            Zpinv = L.pinv(self._Z)
            self.theta = np.dot(Zpinv, self.Y - self._omega)
        else:
            raise StopIteration
        self._iter += 1

