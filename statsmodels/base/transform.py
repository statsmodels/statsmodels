import numpy as np
from scipy.optimize import minimize


class BoxCox(object):
    """
    Mixin class to allow for a Box-Cox transformation.
    """

    def transform_boxcox(self, x, lmbda=None, method='guerrero'):
        """

        Parameters
        ----------
        lmbda : float
            The lambda parameter for the Box-Cox transform. If None, a value
            will be estimated by means of the specified method.
        method : {'guerrero'}
            The method to estimate the lambda parameter. Will only be used if
            lmbda is None, and defaults to 'guerrero'.

        Returns
        -------
        y : array_like
            The transformed series.
        lmbda : float
            The lmbda parameter used to transform the series.
        """
        x = np.asarray(x)

        if np.any(x <= 0):
            raise ValueError("Non-positive x.")

        if lmbda is None:
            lmbda = self._est_lambda(x, method)

        if np.isclose(lmbda, 0.):
            y = np.log(x)
        else:
            y = (np.power(x, lmbda) - 1) / lmbda

        return y, lmbda

    def untransform_boxcox(self, x, lmbda, method='naive'):
        """

        Parameters
        ----------
        x : array_like
            The transformed series.
        lmbda : float
            The lmbda parameter used to transform the series.
        method : {'naive', 'normal'}
            Indicates the method to be used in the untransformation. Defaults
            to 'naive', which reverses the transformation. 'normal' is an optimal
            inversion, assuming the series is normally distributed.
        Returns
        -------
        y : array_like
            The untransformed series.
        """
        x = np.asarray(x)

        if method == 'naive':
            if np.isclose(lmbda, 0.):
                y = np.exp(x)
            else:
                y = np.power(lmbda * x + 1, 1. / lmbda)
        elif method == 'normal':
            pass  # TODO

        return y

    def _est_lambda(self, x, method='guerrero'):
        """
        Computes an estimate for the lambda parameter in the Box-Cox
        transformation using method.
        """
        if method == 'guerrero':
            return minimize(self.__guerrero_cv, 0.,
                            method='Nelder-Mead',
                            options={'maxiter': 100})

    def __guerrero_cv():
        pass  # TODO