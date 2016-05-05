import numpy as np
from scipy.optimize import minimize_scalar


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
            lmbda = self._est_lambda(x, method=method)

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
            to 'naive', which reverses the transformation. 'normal' yields an
            optimal back-transform, assuming the series is normally distributed.

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

    def _est_lambda(self, x, low=None, up=None, R=2, method='guerrero'):
        """
        Computes an estimate for the lambda parameter in the Box-Cox
        transformation using method.

        TODO: for specific methods, the number of arguments may increase
        enormously. Think about a more elegant solution.
        """
        if low is None:
            low = -1
        if up is None:
            up = 3

        if method == 'guerrero':
            res = minimize_scalar(self.__guerrero_cv,
                                   bounds=(low, up),
                                   args=(x, R),
                                   method='bounded',
                                   options={'maxiter': 100})
            return res.x

    def __guerrero_cv(self, lmbda, x, R, **kwargs):
        """
        Computes guerrero's coefficient of variation. If no seasonality
        is present in the data, R is set to 2 (p. 40, comment).

        NOTE: Seasonality-specific auxiliaries *should* provide their own
        seasonality parameter.
        """
        nobs = len(x)
        groups = int(nobs / R)

        # remove the first n < R observations from consideration.
        grouped_data = np.reshape(x[nobs - (groups * R): nobs], (R, groups))

        mean = np.mean(grouped_data, 1)
        sd = np.std(grouped_data, 1)

        rat = np.divide(sd, np.power(mean, 1 - lmbda))  # eq. 6, p. 40
        return np.std(rat) / np.mean(rat)


if __name__ == "__main__":
    bc = BoxCox()
    x = np.arange(1, 100) + np.abs(np.random.rand(99) * 100)
    print(bc.transform_boxcox(x))
