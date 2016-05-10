import numpy as np
from statsmodels.robust import mad
from scipy.optimize import minimize_scalar
from scipy.stats import boxcox_normmax

class BoxCox(object):
    """
    Mixin class to allow for a Box-Cox transformation.
    """

    def transform_boxcox(self, x, lmbda=None, method='guerrero', scale='sd'):
        """
        Performs a Box-Cox transformation on the data array x. If lmbda is None,
        the indicated method is used to estimate a suitable lambda parameter.

        Parameters
        ----------
        lmbda : float
            The lambda parameter for the Box-Cox transform. If None, a value
            will be estimated by means of the specified method.
        method : {'guerrero', 'loglik'}
            The method to estimate the lambda parameter. Will only be used if
            lmbda is None, and defaults to 'guerrero', detailed in Guerrero
            (1993). 'loglik' maximizes the profile likelihood.
        scale : {'sd', 'mad'}
            The dispersion measure to be used. 'sd' indicates the sample standard
            deviation, but the more robust 'mad' is also available.

        Returns
        -------
        y : array_like
            The transformed series.
        lmbda : float
            The lmbda parameter used to transform the series.

        References
        ----------
        Guerrero, Victor M. 1993. "Time-series analysis supported by power
        transformations". `Journal of Forecasting`. 12 (1): 37-48.

        Box, G. E. P., and D. R. Cox. 1964. "An Analysis of Transformations".
        `Journal of the Royal Statistical Society`. 26 (2): 211-252.
        """
        x = np.asarray(x)

        if np.any(x <= 0):
            raise ValueError("Non-positive x.")

        if lmbda is None:
            lmbda = self._est_lambda(x,
                                     bounds=(-1, 2),
                                     method=method,
                                     scale=scale)

        if np.isclose(lmbda, 0.):
            y = np.log(x)
        else:
            y = (np.power(x, lmbda) - 1) / lmbda

        return y, lmbda

    def untransform_boxcox(self, x, lmbda, method='naive'):
        """
        Back-transforms the Box-Cox transformed data array, by means of the
        indicated method. The provided argument lmbda should be the lambda
        parameter that was used to initially transform the data.

        Parameters
        ----------
        x : array_like
            The transformed series.
        lmbda : float
            The lambda parameter that was used to transform the series.
        method : {'naive', 'normal'}
            Indicates the method to be used in the untransformation. Defaults
            to 'naive', which reverses the transformation. 'normal' yields an
            optimal back-transform, assuming the transform series is normally
            distributed.

        Returns
        -------
        y : array_like
            The untransformed series.
        """
        method = method.lower()
        x = np.asarray(x)

        if method == 'naive':
            if np.isclose(lmbda, 0.):
                y = np.exp(x)
            else:
                y = np.power(lmbda * x + 1, 1. / lmbda)
        elif method == 'normal':
            y = x  # TODO
        else:
            raise ValueError("Method '{0}' not understood.".format(method))

        return y

    def _est_lambda(self, x, bounds=(-1, 2),
                    R=2, method='guerrero', scale='sd'):
        """
        Computes an estimate for the lambda parameter in the Box-Cox
        transformation using method.

        Parameters
        ----------
        x : array_like
            The untransformed data.
        bounds : tuple
            Numeric 2-tuple, that indicate the solution space for the lambda
            parameter. Default (-1, 2).
        R : int
            The seasonality/grouping parameter. Default 2.
        method : {'guerrero', 'loglik'}
            The method by which to estimate lambda. Defaults to 'guerrero', but
            the profile likelihood ('loglik') is also available.
        scale : {'sd', 'mad'}
            The dispersion measure to be used. 'sd' indicates the sample standard
            deviation, but the more robust 'mad' is also available.

        Returns
        -------
        lmbda : float
            The lambda parameter.
        """
        method = method.lower()

        if len(bounds) != 2:
            raise ValueError("Bounds of length {0} not understood."
                             .format(len(bounds)))
        elif bounds[0] >= bounds[1]:
            raise ValueError("Lower bound exceeds upper bound.")

        if method == 'guerrero':
            lmbda = self._guerrero_cv(x, R, bounds, scale=scale)
        elif method == 'loglik':
            lmbda = self._loglik(x, bounds)
        else:
            raise ValueError("Method '{0}' not understood.".format(method))

        return lmbda

    def _guerrero_cv(self, x, R, bounds, scale='sd'):
        """
        Computes guerrero's coefficient of variation. If no seasonality
        is present in the data, R is set to 2 (p. 40, comment).

        NOTE: Seasonality-specific auxiliaries *should* provide their own
        seasonality parameter.

        Parameters
        ----------
        x : array_like
        R : int
            Seasonality/grouping parameter. Default 2.
        bounds : tuple
            Numeric 2-tuple, that indicate the solution space for the lambda
            parameter.
        scale : {'sd', 'mad'}
            The dispersion measure to be used. 'sd' indicates the sample standard
            deviation, but the more robust 'mad' is also available.
        """
        nobs = len(x)
        groups = int(nobs / R)
        scale = scale.lower()

        # remove the first n < R observations from consideration.
        grouped_data = np.reshape(x[nobs - (groups * R): nobs], (R, groups))

        mean = np.mean(grouped_data, 1)

        if scale == 'sd':
            dispersion = np.std(grouped_data, 1)
        elif scale == 'mad':
            dispersion = mad(grouped_data, axis=1)
        else:  # TODO: IQR?
            raise ValueError("Scale '{0}' not understood.".format(scale))

        # closure; it's more elegant - and efficient - this way
        def optim(lmbda, *args, **kwargs):
            rat = np.divide(dispersion, np.power(mean, 1 - lmbda))  # eq. 6, p. 40
            return np.std(rat) / np.mean(rat)

        res = minimize_scalar(optim,
                              bounds=bounds,
                              method='bounded',
                              options={'maxiter': 50})
        return res.x

    def _loglik(self, x, bounds):
        """
        Computes the lambda parameter by means of the profile likelihood,
        assuming the series x is normally distributed.

        NOTE: Seasonality-specific auxiliaries _should_ provide their own
        seasonality parameter. This method is here only to provide a parent
        of sorts.

        Parameters
        ----------
        x : array_like
        bounds : tuple
            Numeric 2-tuple, that indicate the solution space for the lambda
            parameter.
        """
        return boxcox_normmax(x, bounds, 'mle')


if __name__ == "__main__":
    bc = BoxCox()

    for i in range(0, 25):
        x = np.arange(1, 100) + np.abs(np.random.rand(99) * i)
        y_s, l_s = bc.transform_boxcox(x, scale='sd')
        y_m, l_m = bc.transform_boxcox(x, scale='mad')
        y_m, l_l = bc.transform_boxcox(x, scale='sd', method='loglik')
        print("{}:  {}  {}  {}".format(i, l_s, l_m, l_l))
