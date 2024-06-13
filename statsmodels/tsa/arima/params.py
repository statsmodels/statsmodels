"""
SARIMAX parameters class.

Author: Chad Fulton
License: BSD-3
"""
import warnings
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from statsmodels.tsa.statespace.tools import is_invertible
from statsmodels.tsa.arima.tools import validate_basic


class SARIMAXParams:
    """
    SARIMAX parameters.

    Parameters
    ----------
    spec : SARIMAXSpecification
        Specification of the SARIMAX model.

    Attributes
    ----------
    spec : SARIMAXSpecification
        Specification of the SARIMAX model.
    exog_names : list of str
        Names associated with exogenous parameters.
    ar_names : list of str
        Names associated with (non-seasonal) autoregressive parameters.
    ma_names : list of str
        Names associated with (non-seasonal) moving average parameters.
    seasonal_ar_names : list of str
        Names associated with seasonal autoregressive parameters.
    seasonal_ma_names : list of str
        Names associated with seasonal moving average parameters.
    param_names :list of str
        Names of all model parameters.
    k_exog_params : int
        Number of parameters associated with exogenous variables.
    k_ar_params : int
        Number of parameters associated with (non-seasonal) autoregressive
        lags.
    k_ma_params : int
        Number of parameters associated with (non-seasonal) moving average
        lags.
    k_seasonal_ar_params : int
        Number of parameters associated with seasonal autoregressive lags.
    k_seasonal_ma_params : int
        Number of parameters associated with seasonal moving average lags.
    k_params : int
        Total number of model parameters.
    """

    def __init__(self, spec):
        self.spec = spec

        # Local copies of relevant attributes
        self.exog_names = spec.exog_names
        self.ar_names = spec.ar_names
        self.ma_names = spec.ma_names
        self.seasonal_ar_names = spec.seasonal_ar_names
        self.seasonal_ma_names = spec.seasonal_ma_names
        self.param_names = spec.param_names

        self.k_exog_params = spec.k_exog_params
        self.k_ar_params = spec.k_ar_params
        self.k_ma_params = spec.k_ma_params
        self.k_seasonal_ar_params = spec.k_seasonal_ar_params
        self.k_seasonal_ma_params = spec.k_seasonal_ma_params
        self.k_params = spec.k_params

        # Cache for holding parameter values
        self._params_split = spec.split_params(
            np.zeros(self.k_params) * np.nan, allow_infnan=True)
        self._params = None

        # (GH6159) Support for free/fixed parameters: all parameters are free
        # parameters by default
        _, self._is_param_fixed_split = spec.split_fixed_params({})
        self._is_param_fixed = None  # lazy implementation

        # keeping counts for internal validation purposes
        self._k_fixed_params_split = {
            name: 0 for name in self._is_param_fixed_split.keys()
            if name != 'sigma2'
        }
        self._k_free_params_split = {
            name: len(is_param_fixed) for name, is_param_fixed
            in self._is_param_fixed_split.items()
            if name != 'sigma2'
        }

    @property
    def exog_params(self):
        """(array) Parameters associated with exogenous variables."""
        return self._params_split['exog_params'].copy()

    @exog_params.setter
    def exog_params(self, value):
        self._set_param_array(
            param_type='exog_params', value=value, k_params=self.k_exog_params,
            title='exogenous coefficients', warn_fixed_overwrite=True
        )

    @property
    def is_exog_param_fixed(self):
        """
        (array) Whether parameters associated with exogenous variables have
        previously been fixed.
        """
        return self._is_param_fixed_split['exog_params'].copy()

    @property
    def fixed_exog_params(self):
        """(array) Fixed parameters associated with exogenous variables."""
        return self.exog_params[self.is_exog_param_fixed]

    @property
    def free_exog_params(self):
        """(array) Free parameters associated with exogenous variables."""
        return self.exog_params[~self.is_exog_param_fixed]

    @free_exog_params.setter
    def free_exog_params(self, value):
        param_type = 'exog_params'
        self._set_param_array(
            param_type=param_type, value=value,
            k_params=self._k_free_params_split[param_type],
            title='exogenous coefficients',
            bool_indexer=~self.is_exog_param_fixed,
            warn_fixed_overwrite=False
        )

    @property
    def ar_params(self):
        """(array) Autoregressive (non-seasonal) parameters."""
        return self._params_split['ar_params'].copy()

    @ar_params.setter
    def ar_params(self, value):
        self._set_param_array(
            param_type='ar_params', value=value, k_params=self.k_ar_params,
            title='AR coefficients', warn_fixed_overwrite=True
        )

    @property
    def is_ar_param_fixed(self):
        """
        (array) Whether parameters associated with autoregressive
        (non-seasonal) variables have previously been fixed.
        """
        return self._is_param_fixed_split['ar_params'].copy()

    @property
    def fixed_ar_params(self):
        """
        (array) Fixed parameters associated with autoregressive (non-seasonal)
        variables.
        """
        return self.ar_params[self.is_ar_param_fixed]

    @property
    def free_ar_params(self):
        """
        (array) Free parameters associated with autoregressive (non-seasonal)
        variables.
        """
        return self.ar_params[~self.is_ar_param_fixed]

    @free_ar_params.setter
    def free_ar_params(self, value):
        param_type = 'ar_params'
        self._set_param_array(
            param_type=param_type, value=value,
            k_params=self._k_free_params_split[param_type],
            title='AR coefficients',
            bool_indexer=~self.is_ar_param_fixed,
            warn_fixed_overwrite=False
        )

    @property
    def ar_poly(self):
        """(Polynomial) Autoregressive (non-seasonal) lag polynomial."""
        coef = np.zeros(self.spec.max_ar_order + 1)
        coef[0] = 1
        ix = self.spec.ar_lags
        coef[ix] = -self._params_split['ar_params']
        return Polynomial(coef)

    @ar_poly.setter
    def ar_poly(self, value):
        # Convert from the polynomial to the parameters, and set that way
        if isinstance(value, Polynomial):
            value = value.coef
        value = validate_basic(value, self.spec.max_ar_order + 1,
                               title='AR polynomial')
        if value[0] != 1:
            raise ValueError('AR polynomial constant must be equal to 1.')
        ar_params = []
        for i in range(1, self.spec.max_ar_order + 1):
            if i in self.spec.ar_lags:
                ar_params.append(-value[i])
            elif value[i] != 0:
                raise ValueError('AR polynomial includes non-zero values'
                                 ' for lags that are excluded in the'
                                 ' specification.')
        self.ar_params = ar_params

    @property
    def ma_params(self):
        """(array) Moving average (non-seasonal) parameters."""
        return self._params_split['ma_params'].copy()

    @ma_params.setter
    def ma_params(self, value):
        self._set_param_array(
            param_type='ma_params', value=value, k_params=self.k_ma_params,
            title='MA coefficients', warn_fixed_overwrite=True
        )

    @property
    def is_ma_param_fixed(self):
        """
        (array) Whether parameters associated with moving average
        (non-seasonal) variables have previously been fixed.
        """
        return self._is_param_fixed_split['ma_params'].copy()

    @property
    def fixed_ma_params(self):
        """
        (array) Fixed parameters associated with moving average (non-seasonal)
        variables.
        """
        return self.ma_params[self.is_ma_param_fixed]

    @property
    def free_ma_params(self):
        """
        (array) Free parameters associated with moving average (non-seasonal)
        variables.
        """
        return self.ma_params[~self.is_ma_param_fixed]

    @free_ma_params.setter
    def free_ma_params(self, value):
        param_type = 'ma_params'
        self._set_param_array(
            param_type=param_type, value=value,
            k_params=self._k_free_params_split[param_type],
            title='MA coefficients',
            bool_indexer=~self.is_ma_param_fixed,
            warn_fixed_overwrite=False
        )

    @property
    def ma_poly(self):
        """(Polynomial) Moving average (non-seasonal) lag polynomial."""
        coef = np.zeros(self.spec.max_ma_order + 1)
        coef[0] = 1
        ix = self.spec.ma_lags
        coef[ix] = self._params_split['ma_params']
        return Polynomial(coef)

    @ma_poly.setter
    def ma_poly(self, value):
        # Convert from the polynomial to the parameters, and set that way
        if isinstance(value, Polynomial):
            value = value.coef
        value = validate_basic(value, self.spec.max_ma_order + 1,
                               title='MA polynomial')
        if value[0] != 1:
            raise ValueError('MA polynomial constant must be equal to 1.')
        ma_params = []
        for i in range(1, self.spec.max_ma_order + 1):
            if i in self.spec.ma_lags:
                ma_params.append(value[i])
            elif value[i] != 0:
                raise ValueError('MA polynomial includes non-zero values'
                                 ' for lags that are excluded in the'
                                 ' specification.')
        self.ma_params = ma_params

    @property
    def seasonal_ar_params(self):
        """(array) Seasonal autoregressive parameters."""
        return self._params_split['seasonal_ar_params'].copy()

    @seasonal_ar_params.setter
    def seasonal_ar_params(self, value):
        self._set_param_array(
            param_type='seasonal_ar_params', value=value,
            k_params=self.k_seasonal_ar_params,
            title='seasonal AR coefficients',
            warn_fixed_overwrite=True
        )

    @property
    def is_seasonal_ar_param_fixed(self):
        """
        (array) Whether parameters associated with seasonal autoregressive
        variables have previously been fixed.
        """
        return self._is_param_fixed_split['seasonal_ar_params'].copy()

    @property
    def fixed_seasonal_ar_params(self):
        """
        (array) Fixed parameters associated with seasonal autoregressive
        variables.
        """
        return self.seasonal_ar_params[self.is_seasonal_ar_param_fixed]

    @property
    def free_seasonal_ar_params(self):
        """
        (array) Free parameters associated with seasonal autoregressive
        variables.
        """
        return self.seasonal_ar_params[~self.is_seasonal_ar_param_fixed]

    @free_seasonal_ar_params.setter
    def free_seasonal_ar_params(self, value):
        param_type = 'seasonal_ar_params'
        self._set_param_array(
            param_type=param_type, value=value,
            k_params=self._k_free_params_split[param_type],
            title='seasonal AR coefficients',
            bool_indexer=~self.is_seasonal_ar_param_fixed,
            warn_fixed_overwrite=False
        )

    @property
    def seasonal_ar_poly(self):
        """(Polynomial) Seasonal autoregressive lag polynomial."""
        # Need to expand the polynomial according to the season
        s = self.spec.seasonal_periods
        coef = [1]
        if s > 0:
            expanded = np.zeros(self.spec.max_seasonal_ar_order)
            ix = np.array(self.spec.seasonal_ar_lags, dtype=int) - 1
            expanded[ix] = -self._params_split['seasonal_ar_params']
            coef = np.r_[1, np.pad(np.reshape(expanded, (-1, 1)),
                                   [(0, 0), (s - 1, 0)], 'constant').flatten()]
        return Polynomial(coef)

    @seasonal_ar_poly.setter
    def seasonal_ar_poly(self, value):
        s = self.spec.seasonal_periods
        # Note: assume that we are given coefficients from the full polynomial
        # Convert from the polynomial to the parameters, and set that way
        if isinstance(value, Polynomial):
            value = value.coef
        value = validate_basic(value, 1 + s * self.spec.max_seasonal_ar_order,
                               title='seasonal AR polynomial')
        if value[0] != 1:
            raise ValueError('Polynomial constant must be equal to 1.')
        seasonal_ar_params = []
        for i in range(1, self.spec.max_seasonal_ar_order + 1):
            if i in self.spec.seasonal_ar_lags:
                seasonal_ar_params.append(-value[s * i])
            elif value[s * i] != 0:
                raise ValueError('AR polynomial includes non-zero values'
                                 ' for lags that are excluded in the'
                                 ' specification.')
        self.seasonal_ar_params = seasonal_ar_params

    @property
    def seasonal_ma_params(self):
        """(array) Seasonal moving average parameters."""
        return self._params_split['seasonal_ma_params'].copy()

    @seasonal_ma_params.setter
    def seasonal_ma_params(self, value):
        self._set_param_array(
            param_type='seasonal_ma_params', value=value,
            k_params=self.k_seasonal_ma_params,
            title='seasonal MA coefficients',
            warn_fixed_overwrite=True
        )

    @property
    def is_seasonal_ma_param_fixed(self):
        """
        (array) Whether parameters associated with seasonal moving average
        variables have previously been fixed.
        """
        return self._is_param_fixed_split['seasonal_ma_params'].copy()

    @property
    def fixed_seasonal_ma_params(self):
        """
        (array) Fixed parameters associated with seasonal moving average
        variables.
        """
        return self.seasonal_ma_params[self.is_seasonal_ma_param_fixed]

    @property
    def free_seasonal_ma_params(self):
        """
        (array) Free parameters associated with seasonal moving average
        variables.
        """
        return self.seasonal_ma_params[~self.is_seasonal_ma_param_fixed]

    @free_seasonal_ma_params.setter
    def free_seasonal_ma_params(self, value):
        param_type = 'seasonal_ma_params'
        self._set_param_array(
            param_type=param_type, value=value,
            k_params=self._k_free_params_split[param_type],
            title='seasonal MA coefficients',
            bool_indexer=~self.is_seasonal_ma_param_fixed,
            warn_fixed_overwrite=False
        )

    @property
    def seasonal_ma_poly(self):
        """(Polynomial) Seasonal moving average lag polynomial."""
        # Need to expand the polynomial according to the season
        s = self.spec.seasonal_periods
        coef = np.array([1])
        if s > 0:
            expanded = np.zeros(self.spec.max_seasonal_ma_order)
            ix = np.array(self.spec.seasonal_ma_lags, dtype=int) - 1
            expanded[ix] = self._params_split['seasonal_ma_params']
            coef = np.r_[1, np.pad(np.reshape(expanded, (-1, 1)),
                                   [(0, 0), (s - 1, 0)], 'constant').flatten()]
        return Polynomial(coef)

    @seasonal_ma_poly.setter
    def seasonal_ma_poly(self, value):
        s = self.spec.seasonal_periods
        # Note: assume that we are given coefficients from the full polynomial
        # Convert from the polynomial to the parameters, and set that way
        if isinstance(value, Polynomial):
            value = value.coef
        value = validate_basic(value, 1 + s * self.spec.max_seasonal_ma_order,
                               title='seasonal MA polynomial',)
        if value[0] != 1:
            raise ValueError('Polynomial constant must be equal to 1.')
        seasonal_ma_params = []
        for i in range(1, self.spec.max_seasonal_ma_order + 1):
            if i in self.spec.seasonal_ma_lags:
                seasonal_ma_params.append(value[s * i])
            elif value[s * i] != 0:
                raise ValueError('MA polynomial includes non-zero values'
                                 ' for lags that are excluded in the'
                                 ' specification.')
        self.seasonal_ma_params = seasonal_ma_params

    @property
    def is_sigma2_fixed(self):
        """(bool) Whether innovation variance has been fixed."""
        return self._is_param_fixed_split['sigma2']

    @property
    def sigma2(self):
        """(float) Innovation variance."""
        return self._params_split['sigma2']

    @sigma2.setter
    def sigma2(self, value):
        length = int(not self.spec.concentrate_scale)
        value = validate_basic(value, length, title='sigma2').item()

        self._warn_fixed_overwrite(
            original_value=np.atleast_1d(self._params_split['sigma2']),
            new_value=np.atleast_1d(value),
            is_fixed_bool=np.atleast_1d(self.is_sigma2_fixed)
        )

        self._params_split['sigma2'] = value
        self._params = None

    @property
    def reduced_ar_poly(self):
        """(Polynomial) Reduced form autoregressive lag polynomial."""
        return self.ar_poly * self.seasonal_ar_poly

    @property
    def reduced_ma_poly(self):
        """(Polynomial) Reduced form moving average lag polynomial."""
        return self.ma_poly * self.seasonal_ma_poly

    @property
    def params(self):
        """(array) Complete parameter vector."""
        if self._params is None:
            self._params = self.spec.join_params(**self._params_split)
        return self._params.copy()

    @params.setter
    def params(self, value):
        curr_params = self.spec.join_params(**self._params_split)
        self._warn_fixed_overwrite(
            original_value=curr_params,
            new_value=value,
            is_fixed_bool=self.is_param_fixed
        )

        self._params_split = self.spec.split_params(value)
        self._params = None

    @property
    def is_param_fixed(self):
        """(array) Whether parameters have previously been fixed."""
        if self._is_param_fixed is None:
            self._is_param_fixed = self.spec.join_params(
                **self._is_param_fixed_split
            )
        return self._is_param_fixed.copy()

    @property
    def fixed_params(self):
        """(array) All fixed parameters"""
        return self.params[self.is_param_fixed]

    @property
    def free_params(self):
        """(array) All free parameters"""
        return self.params[~self.is_param_fixed]

    @free_params.setter
    def free_params(self, value):
        all_params = self.spec.join_params(**self._params_split)

        # update free parameters
        k_fixed_params = self.is_param_fixed.sum()
        k_free_params = self.k_params - k_fixed_params

        all_params[~self.is_param_fixed] = validate_basic(
            params=value, length=k_free_params,
            allow_infnan=False, title="free parameters"
        )
        self._params_split = self.spec.split_params(all_params)

        # clear cache
        self._params = None

    @property
    def is_complete(self):
        """(bool) Are current parameter values all filled in (i.e. not NaN)."""
        return not np.any(np.isnan(self.params))

    @property
    def is_valid(self):
        """(bool) Are current parameter values valid (e.g. variance > 0)."""
        valid = True
        try:
            self.spec.validate_params(self.params)
        except ValueError:
            valid = False
        return valid

    @property
    def is_stationary(self):
        """(bool) Is the reduced autoregressive lag poylnomial stationary."""
        validate_basic(self.ar_params, self.k_ar_params,
                       title='AR coefficients')
        validate_basic(self.seasonal_ar_params, self.k_seasonal_ar_params,
                       title='seasonal AR coefficients')

        ar_stationary = True
        seasonal_ar_stationary = True
        if self.k_ar_params > 0:
            ar_stationary = is_invertible(self.ar_poly.coef)
        if self.k_seasonal_ar_params > 0:
            seasonal_ar_stationary = is_invertible(self.seasonal_ar_poly.coef)

        return ar_stationary and seasonal_ar_stationary

    @property
    def is_invertible(self):
        """(bool) Is the reduced moving average lag poylnomial invertible."""
        # Short-circuit if there is no MA component
        validate_basic(self.ma_params, self.k_ma_params,
                       title='MA coefficients')
        validate_basic(self.seasonal_ma_params, self.k_seasonal_ma_params,
                       title='seasonal MA coefficients')

        ma_stationary = True
        seasonal_ma_stationary = True
        if self.k_ma_params > 0:
            ma_stationary = is_invertible(self.ma_poly.coef)
        if self.k_seasonal_ma_params > 0:
            seasonal_ma_stationary = is_invertible(self.seasonal_ma_poly.coef)

        return ma_stationary and seasonal_ma_stationary

    def set_fixed_params(self, fixed_params, validate=False,
                         **validate_kwargs):
        """
        Set fixed parameters.

        Parameters
        ----------
        fixed_params : dict
            Dictionary with names of fixed parameters as keys (e.g. 'ar.L1',
            'ma.L2'), which correspond to SARIMAXSpecification.param_names.
            Dictionary values are the values of the associated fixed
            parameters.
        validate : bool, optional
            Whether to validate fixed parameter names and values.
            See `SARIMAXSpecification.validate_fixed_params` function.
        validate_kwargs : dict, optional
            kwargs for `SARIMAXSpecification.validate_fixed_params` (e.g.
            `allow_fixed_sigma2`)
        """
        if validate:
            self.spec.validate_fixed_params(fixed_params, **validate_kwargs)

        split_fixed_params, split_is_fixed_param = (
            self.spec.split_fixed_params(fixed_params)
        )

        for param_type in split_fixed_params.keys():
            if param_type == "sigma2":
                if split_is_fixed_param[param_type]:
                    self._params_split[param_type] = validate_basic(
                        split_fixed_params[param_type],
                        int(not self.spec.concentrate_scale),
                        title="sigma2"
                    ).item()
                    self._params = None
                    self._is_param_fixed_split[param_type] = True
                    self._is_param_fixed = None
            else:
                # write to param_split
                self._set_param_array(
                    param_type=param_type,
                    value=split_fixed_params[param_type],
                    k_params=split_is_fixed_param[param_type].sum(),
                    title=None,
                    bool_indexer=split_is_fixed_param[param_type],
                    warn_fixed_overwrite=False
                )
                # update is_param_fixed_split
                self._is_param_fixed_split[param_type] = (
                    self._is_param_fixed_split[param_type] |
                    split_is_fixed_param[param_type]
                )
                self._is_param_fixed = None  # clear cache
                # update count
                self._k_fixed_params_split[param_type] = (
                    self._is_param_fixed_split[param_type].sum()
                )
                self._k_free_params_split[param_type] = (
                    (~self._is_param_fixed_split[param_type]).sum()
                )

    def reset_fixed_params(self, keep_param_value=False):
        """
        Reset statuses and (optionally) values of previously fixed parameters.

        - Set status of all fixed params to 'free';
        - Set all fixed params to np.nan if keep_param_value is False

        Parameters
        ----------
        keep_param_value : bool, optional
            Whether to keep the current fixed param value; if False, all
            fixed params are set to np.nan
        """
        for param_type in self._is_param_fixed_split.keys():
            if param_type == "sigma2":
                if not keep_param_value:
                    # set current fixed values to np.nan if requested
                    self._params_split[param_type] = np.nan
                    self._params = None
                # set the fixed param bool to False
                self._is_param_fixed_split[param_type] = False
            else:
                k_fixed_params = self._k_fixed_params_split[param_type]
                k_free_params = self._k_free_params_split[param_type]
                if k_fixed_params > 0:
                    # set current fixed values to np.nan if requested
                    if not keep_param_value:
                        bool_indexer = self._is_param_fixed_split[param_type]
                        self._set_param_array(
                            param_type=param_type,
                            value=np.nan,
                            k_params=k_fixed_params,
                            allow_infnan=True,
                            bool_indexer=bool_indexer,
                            warn_fixed_overwrite=False
                        )
                    # set all fixed param bool to False
                    self._is_param_fixed_split[param_type][:] = False
                    # set fix counts to 0
                    self._k_fixed_params_split[param_type] = 0
                    # set free counts to total count
                    self._k_free_params_split[param_type] = (
                        k_fixed_params + k_free_params
                    )

        self._is_param_fixed = None  # reset cache

    def to_dict(self):
        """
        Return the parameters split by type into a dictionary.

        Returns
        -------
        split_params : dict
            Dictionary with keys 'exog_params', 'ar_params', 'ma_params',
            'seasonal_ar_params', 'seasonal_ma_params', and (unless
            `concentrate_scale=True`) 'sigma2'. Values are the parameters
            associated with the key, based on the `params` argument.
        """
        return self._params_split.copy()

    def to_pandas(self):
        """
        Return the parameters as a Pandas series.

        Returns
        -------
        series : pd.Series
            Pandas series with index set to the parameter names.
        """
        return pd.Series(self.params, index=self.param_names)

    def __repr__(self):
        """Represent SARIMAXParams object as a string."""
        components = []
        if self.k_exog_params:
            components.append('exog=%s' % str(self.exog_params))
        if self.k_ar_params:
            components.append('ar=%s' % str(self.ar_params))
        if self.k_ma_params:
            components.append('ma=%s' % str(self.ma_params))
        if self.k_seasonal_ar_params:
            components.append('seasonal_ar=%s' %
                              str(self.seasonal_ar_params))
        if self.k_seasonal_ma_params:
            components.append('seasonal_ma=%s' %
                              str(self.seasonal_ma_params))
        if not self.spec.concentrate_scale:
            components.append('sigma2=%s' % self.sigma2)
        return 'SARIMAXParams(%s)' % ', '.join(components)

    def _set_param_array(self, param_type, value, k_params, allow_infnan=False,
                         title=None, bool_indexer=None,
                         warn_fixed_overwrite=True):
        """
        Helper function for setting entire or subset of param arrays.

        - Validates param value;
        - Check and warns if any previous fixed params are overwritten;
        - Updates `_params_split`
        - Clears `_params` cache

        Parameters
        ----------
        param_type : str
            Name of parameter type; one of 'exog_params', 'ar_params',
            'ma_params', 'seasonal_ar_params', or 'seasonal_ma_params'
        value : array_like
            Array of parameters with size `k_params`
        k_params : int
            Expected size of value array
        allow_infnan : bool, optional
            See `statsmodels.tsa.arima.tools.validate_basic`
        title : str, optional
            See `statsmodels.tsa.arima.tools.validate_basic`
        bool_indexer : array_like of bool, optional
            Array of bool indicating positions of `value` in
            `_params_split[name]`; used to set values for a subset
            of the param array (e.g. free only).
            If None, then set the entire param array by default.
        warn_fixed_overwrite : bool, optional
            Whether to check and warns if any previous fixed params
            are overwritten
        """
        if bool_indexer is None:
            # if bool_index is None, bool_index selects all by default
            bool_indexer = np.full(
                self._params_split[param_type].shape[0], True, dtype=bool
            )
        assert bool_indexer.sum() == k_params
        if np.isscalar(value):
            value = [value] * k_params

        # validates param value
        value = validate_basic(
            params=value, length=k_params,
            allow_infnan=allow_infnan, title=title
        )

        # check if overwriting previously fixed parameters
        if warn_fixed_overwrite:
            self._warn_fixed_overwrite(
                original_value=self._params_split[param_type][bool_indexer],
                new_value=value,
                is_fixed_bool=self._is_param_fixed_split[param_type][
                    bool_indexer
                ]
            )

        # updates value
        self._params_split[param_type][bool_indexer] = value

        # clears cache
        self._params = None

    @staticmethod
    def _warn_fixed_overwrite(original_value, new_value, is_fixed_bool):
        """
        Issues a warning when any previously fixed parameters are overwritten.

        Note that we only count "overwriting" when a previously fixed parameter
        is replaced with a **different** value.

        Parameters
        ----------
        original_value : array_like
        new_value : array_like
        is_fixed_bool : array_like of bool
        """
        is_value_diff = ~ np.isclose(
            original_value, new_value, rtol=0, atol=0, equal_nan=True
        )
        k_overwritten_fixed_params = sum(is_value_diff & is_fixed_bool)
        if k_overwritten_fixed_params > 0:
            warnings.warn(
                f"Overwriting {k_overwritten_fixed_params} previously "
                f"fixed parameters with different values."
            )
