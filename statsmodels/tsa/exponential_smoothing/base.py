from collections import OrderedDict
import contextlib

import numpy as np
import pandas as pd
from scipy.stats import norm

from statsmodels.base.data import PandasData
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, aicc, bic, hqic
import statsmodels.tsa.base.tsa_model as tsbase


class StateSpaceMLEModel(tsbase.TimeSeriesModel):
    """
    This is a temporary base model from ETS, here I just copy everything I need
    from statespace.mlemodel.MLEModel
    """

    def __init__(self, endog, exog=None, dates=None, freq=None, missing='none',
                 **kwargs):
        # TODO: this was changed from the original, requires some work when
        # using this as base class for state space and exponential smoothing
        super().__init__(endog=endog, exog=exog, dates=dates, freq=freq,
                         missing=missing)

        # Store kwargs to recreate model
        self._init_kwargs = kwargs

        # Prepared the endog array: C-ordered, shape=(nobs x k_endog)
        self.endog, self.exog = self.prepare_data(self.data)
        self.use_pandas = isinstance(self.data, PandasData)

        # Dimensions
        self.nobs = self.endog.shape[0]

        # Setup holder for fixed parameters
        self._has_fixed_params = False
        self._fixed_params = None
        self._params_index = None
        self._fixed_params_index = None
        self._free_params_index = None

    @staticmethod
    def prepare_data(data):
        raise NotImplementedError

    def clone(self, endog, exog=None, **kwargs):
        raise NotImplementedError

    def _validate_can_fix_params(self, param_names):
        for param_name in param_names:
            if param_name not in self.param_names:
                raise ValueError('Invalid parameter name passed: "%s".'
                                 % param_name)

    @property
    def k_params(self):
        return len(self.param_names)

    @contextlib.contextmanager
    def fix_params(self, params):
        """
        Fix parameters to specific values (context manager)

        Parameters
        ----------
        params : dict
            Dictionary describing the fixed parameter values, of the form
            `param_name: fixed_value`. See the `param_names` property for valid
            parameter names.

        Examples
        --------
        >>> mod = sm.tsa.SARIMAX(endog, order=(1, 0, 1))
        >>> with mod.fix_params({'ar.L1': 0.5}):
                res = mod.fit()
        """
        # Initialization (this is done here rather than in the constructor
        # because param_names may not be available at that point)
        if self._fixed_params is None:
            self._fixed_params = {}
            self._params_index = OrderedDict(
                zip(self.param_names, np.arange(self.k_params)))

        # Cache the current fixed parameters
        cache_fixed_params = self._fixed_params.copy()
        cache_has_fixed_params = self._has_fixed_params
        cache_fixed_params_index = self._fixed_params_index
        cache_free_params_index = self._free_params_index

        # Validate parameter names and values
        self._validate_can_fix_params(set(params.keys()))

        # Set the new fixed parameters, keeping the order as given by
        # param_names
        self._fixed_params.update(params)
        self._fixed_params = OrderedDict(
            [
                (name, self._fixed_params[name])
                for name in self.param_names
                if name in self._fixed_params
            ]
        )

        # Update associated values
        self._has_fixed_params = True
        self._fixed_params_index = [self._params_index[key]
                                    for key in self._fixed_params.keys()]
        self._free_params_index = list(
            set(np.arange(self.k_params)).difference(self._fixed_params_index)
        )

        try:
            yield
        finally:
            # Reset the fixed parameters
            self._has_fixed_params = cache_has_fixed_params
            self._fixed_params = cache_fixed_params
            self._fixed_params_index = cache_fixed_params_index
            self._free_params_index = cache_free_params_index

    def fit_constrained(self, constraints, start_params=None, **fit_kwds):
        """
        Fit the model with some parameters subject to equality constraints.

        Parameters
        ----------
        constraints : dict
            Dictionary of constraints, of the form `param_name: fixed_value`.
            See the `param_names` property for valid parameter names.
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            If None, the default is given by Model.start_params.
        **fit_kwds : keyword arguments
            fit_kwds are used in the optimization of the remaining parameters.

        Returns
        -------
        results : Results instance

        Examples
        --------
        >>> mod = sm.tsa.SARIMAX(endog, order=(1, 0, 1))
        >>> res = mod.fit_constrained({'ar.L1': 0.5})
        """
        with self.fix_params(constraints):
            res = self.fit(start_params, **fit_kwds)
        return res

    @property
    def start_params(self):
        """
        (array) Starting parameters for maximum likelihood estimation.
        """
        if hasattr(self, '_start_params'):
            return self._start_params
        else:
            raise NotImplementedError

    @property
    def param_names(self):
        """
        (list of str) List of human readable parameter names (for parameters
        actually included in the model).
        """
        if hasattr(self, '_param_names'):
            return self._param_names
        else:
            try:
                names = ['param.%d' % i for i in range(len(self.start_params))]
            except NotImplementedError:
                names = []
            return names

    @classmethod
    def from_formula(cls, formula, data, subset=None, drop_cols=None,
                     *args, **kwargs):
        """
        Not implemented for state space models
        """
        raise NotImplementedError

    def _wrap_data(self, data, start_idx, end_idx, names=None):
        # TODO: check if this is reasonable for statespace
        data = np.squeeze(data)
        if self.use_pandas:
            _, _, _, index = self._get_prediction_index(
                start_idx, end_idx
            )
            if data.ndim < 2:
                data = pd.Series(data, index=index, name=names)
            else:
                data = pd.DataFrame(data, index=index, columns=names)
        return data


class StateSpaceMLEResults(tsbase.TimeSeriesModelResults):
    r"""
    Class to hold results from fitting a state space model.

    Parameters
    ----------
    model : MLEModel instance
        The fitted model instance
    params : ndarray
        Fitted parameters

    Attributes
    ----------
    model : Model instance
        A reference to the model that was fit.
    nobs : float
        The number of observations used to fit the model.
    params : ndarray
        The parameters of the model.
    """
    def __init__(self, model, params, scale=1.):
        self.data = model.data
        self.endog = model.data.orig_endog

        super().__init__(model, params, None, scale=scale)

        # Save the fixed parameters
        self._has_fixed_params = self.model._has_fixed_params
        self._fixed_params_index = self.model._fixed_params_index
        self._free_params_index = self.model._free_params_index
        # TODO: seems like maybe self.fixed_params should be the dictionary
        # itself, not just the keys?
        if self._has_fixed_params:
            self._fixed_params = self.model._fixed_params.copy()
            self.fixed_params = list(self._fixed_params.keys())
        else:
            self._fixed_params = None
            self.fixed_params = []
        self.param_names = [
            '%s (fixed)' % name if name in self.fixed_params else name
            for name in (self.data.param_names or [])]

        # Dimensions
        self.nobs = self.model.nobs

    @cache_readonly
    def nobs_effective(self):
        raise NotImplementedError

    @cache_readonly
    def df_model(self):
        raise NotImplementedError

    @cache_readonly
    def df_resid(self):
        return self.nobs_effective - self.df_model

    @cache_readonly
    def aic(self):
        """
        (float) Akaike Information Criterion
        """
        return aic(self.llf, self.nobs_effective, self.df_model)

    @cache_readonly
    def aicc(self):
        """
        (float) Akaike Information Criterion with small sample correction
        """
        return aicc(self.llf, self.nobs_effective, self.df_model)

    @cache_readonly
    def bic(self):
        """
        (float) Bayes Information Criterion
        """
        return bic(self.llf, self.nobs_effective, self.df_model)

    @cache_readonly
    def fittedvalues(self):
        # TODO
        raise NotImplementedError

    @cache_readonly
    def hqic(self):
        """
        (float) Hannan-Quinn Information Criterion
        """
        # return (-2 * self.llf +
        #         2 * np.log(np.log(self.nobs_effective)) * self.df_model)
        return hqic(self.llf, self.nobs_effective, self.df_model)

    @cache_readonly
    def llf(self):
        """
        (float) The value of the log-likelihood function evaluated at `params`.
        """
        raise NotImplementedError

    @cache_readonly
    def mae(self):
        """
        (float) Mean absolute error
        """
        return np.mean(np.abs(self.resid))

    @cache_readonly
    def mse(self):
        """
        (float) Mean squared error
        """
        return self.sse / self.nobs

    @cache_readonly
    def pvalues(self):
        """
        (array) The p-values associated with the z-statistics of the
        coefficients. Note that the coefficients are assumed to have a Normal
        distribution.
        """
        pvalues = np.zeros_like(self.zvalues) * np.nan
        mask = np.ones_like(pvalues, dtype=bool)
        mask[self._free_params_index] = True
        mask &= ~np.isnan(self.zvalues)
        pvalues[mask] = norm.sf(np.abs(self.zvalues[mask])) * 2
        return pvalues

    @cache_readonly
    def resid(self):
        raise NotImplementedError

    @cache_readonly
    def sse(self):
        """
        (float) Sum of squared errors
        """
        return np.sum(self.resid**2)

    @cache_readonly
    def zvalues(self):
        """
        (array) The z-statistics for the coefficients.
        """
        return self.params / self.bse

    def _get_prediction_start_index(self, anchor):
        """Returns a valid numeric start index for predictions/simulations"""
        # TODO: once this is the base class for statespace models, use this
        # method in simulate
        if anchor is None or anchor == 'start':
            iloc = 0
        elif anchor == 'end':
            iloc = self.nobs
        else:
            iloc, _, _ = self.model._get_index_loc(anchor)
            if isinstance(iloc, slice):
                iloc = iloc.start

        if iloc < 0:
            iloc = self.nobs + iloc
        if iloc > self.nobs:
            raise ValueError('Cannot anchor simulation outside of the sample.')
        return iloc
