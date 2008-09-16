"""
This module provides various convenience functions for extracting
statistics from regression analysis techniques to model the
relationship between the dependent and independent variables.

As well as a convenience class to output the result, RegressionOutput

"""

__docformat__ = 'restructuredtext'

import numpy as np
import numpy.linalg as L
from scipy.linalg import toeplitz
from neuroimaging.fixes.scipy.stats.models.utils import recipr

def output_T(contrast, results, effect=None, sd=None, t=None):
    """
    This convenience function outputs the results of a Tcontrast
    from a regression
    """
    r = results.Tcontrast(contrast.matrix, sd=sd,
                          t=t)

    v = []
    if effect is not None:
        v.append(r.effect)
    if sd is not None:
        v.append(r.sd)
    if t is not None:
        v.append(r.t)
    return v

def output_F(results, contrast):
    """
    This convenience function outputs the results of an Fcontrast
    from a regression
    """
    return results.Fcontrast(contrast.matrix).F

def output_resid(results):
    """
    This convenience function outputs the residuals
    from a regression
    """
    return results.resid

class RegressionOutput:
    """
    A class to output things in GLM passes through arrays of data.
    """

    def __init__(self, img, fn, output_shape=None):
        """
        :Parameters:
            `img` : the output Image
            `fn` : a function that is applied to a scipy.stats.models.model.LikelihoodModelResults instance

        """
        self.img = img
        self.fn = fn
        self.output_shape = output_shape

    def __call__(self, x):
        return self.fn(x)

    def __setitem__(self, index, value):
        self.img[index] = value


class RegressionOutputList:
    """
    A class to output more than one thing
    from a GLM pass through arrays of data.
    """

    def __call__(self, x):
        return self.fn(x)

    def __init__(self, imgs, fn):
        """
        :Parameters:
            `imgs` : the list of output images
            `fn` : a function that is applied to a scipy.stats.models.model.LikelihoodModelResults instance

        """
        self.list = imgs
        self.fn = fn

    def __setitem__(self, index, value):
        self.list[index[0]][index[1:]] = value


class TOutput(RegressionOutputList):

    """
    Output contrast related to a T contrast
    from a GLM pass through data.
    """
    def __init__(self, contrast, effect=None,
                 sd=None, t=None):
        self.fn = lambda x: output_T(contrast,
                                     x,
                                     effect=effect,
                                     sd=sd,
                                     t=t)
        self.list = []
        if effect is not None:
            self.list.append(effect)
        if sd is not None:
            self.list.append(sd)
        if t is not None:
            self.list.append(t)

class ArrayOutput(RegressionOutput):
    """
    Output an array from a GLM pass through data.

    By default, the function called is output_resid, so residuals
    are output.

    """

    def __init__(self, img, fn):
        RegressionOutput.__init__(self, img, fn)

def output_AR1(results):
    """
    Compute the usual AR(1) parameter on
    the residuals from a regression.
    """
    resid = results.resid
    rho = np.add.reduce(resid[0:-1]*resid[1:] / np.add.reduce(resid[1:-1]**2))
    return rho

class AREstimator:
    """
    A class that whose instances can estimate
    AR(p) coefficients from residuals
    """

    def __init__(self, model, p=1):
        """
        :Parameters:
            `coordmap` : TODO
                TODO
            `model` : TODO
                A scipy.stats.models.regression.OLSmodel instance
            `p` : int
                Order of AR(p) noise
        """
        self.p = p
        self._setup_bias_correct(model)

    def _setup_bias_correct(self, model):

        R = np.identity(model.design.shape[0]) - np.dot(model.design, model.calc_beta)
        M = np.zeros((self.p+1,)*2)
        I = np.identity(R.shape[0])

        for i in range(self.p+1):
            Di = np.dot(R, toeplitz(I[i]))
            for j in range(self.p+1):
                Dj = np.dot(R, toeplitz(I[j]))
                M[i,j] = np.diagonal((np.dot(Di, Dj))/(1.+(i>0))).sum()

        self.invM = L.inv(M)
        return

    def __call__(self, results):
        """
        :Parameters:
            `results` : a scipy.stats.models.model.LikelihoodModelResults instance
        :Returns: ``numpy.ndarray``
        """
        resid = results.resid.reshape((results.resid.shape[0],
                                       np.product(results.resid.shape[1:])))

        sum_sq = results.scale.reshape(resid.shape[1:]) * results.df_resid

        cov = np.zeros((self.p + 1,) + sum_sq.shape)
        cov[0] = sum_sq
        for i in range(1, self.p+1):
            cov[i] = np.add.reduce(resid[i:] * resid[0:-i], 0)
        cov = np.dot(self.invM, cov)
        output = cov[1:] * recipr(cov[0])
        return np.squeeze(output)

