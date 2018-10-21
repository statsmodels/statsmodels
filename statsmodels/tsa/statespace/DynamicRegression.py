"""
Univariate time-varying coefficient regression model

Author: Alastair Heggie
License: Simplified-BSD
"""
import numpy as np
from scipy.linalg import block_diag
import statsmodels.api as sm
import pandas as pd
import warnings
import pdb
from .tools import (constrain_stationary_univariate,
    unconstrain_stationary_univariate)

_valid_components = ['level', 'stochastic_level', 'trend', 'stochastic_trend',
                     'freq_seasonal', 'stochastic_freq_seasonal', 'irregular',
                      'AR', 'MA']
_stationary_components = ['irregular', 'AR', 'MA']
_non_stationary_components = ['level', 'stochastic_level',
                        'trend', 'stochastic_trend']#, 'freq_seasonal',
                              #'stochastic_freq_seasonal']
_stochastic_components = ['stochastic_level', 'stochastic_trend',
                          'irregular']#, 'stochastic_freq_seasonal']


# Construct the model
class DynamicRegression(sm.tsa.statespace.MLEModel):
    """
    Univariate time-varying coefficient regression models

    A model in which the coefficients of the exogonous covariates are
    described by a univariate unobserved components model. At present
    only local level and trend models are available for the coefficinets.
    Ulitmately it is intendeperiodd to add the full suite of components that
    are standard in unobserved components models:

    Parameters
    ----------
    exog : array_like or None, optional
        Exogenous variables.
    exog_models : dict or list of dicts, optional
        Either a dict, a list containing a single dict, or a list of dict
        s, one for each exogonous regressor.
        Admisble keys are the supported component types: 'irregular',
        'level', and 'trend'. If a key is included in the dict and
        it's value is not False, then it is included in the model. If the
        corresponding value is "deterministic" then the state
        corresponding to that component will not have a disturbance term.
    dynamic : bool
        Whether or not the regression coefficient updates over time. If
        False then the states corresponding to the regression
        coefficients have 0 variance.
    """
    def __init__(self, endog, exog, exog_models={"level": True}):
        # Get k_exog from size of exog data
        try:
            self.k_exog = exog.shape[1]
        except IndexError:
            exog = np.expand_dims(exog, axis=1)
            self.k_exog = exog.shape[1]
        if isinstance(exog_models, list):
            if len(exog_models) != self.k_exog and len(exog_models) != 1:
                raise ValueError('We should either recieve no exog_model'
                                 ', a single exog_model or a list of k_exog'
                                 ' exog_models')
        # If we only received a single exog_model in a list, or just one
        # dict create a k_exog length list of dicts
        if isinstance(exog_models, list):
            if len(exog_models) == 1:
                self.exog_models = [exog_models[0] for m in range(self.k_exog)]
            else:
                self.exog_models = exog_models
        elif isinstance(exog_models, dict):
            self.exog_models = [exog_models for m in range(self.k_exog)]
        # Expand each exog_model dict to specify all available component
        # types as False if not already in the dict. All exog_models
        # must include a local level

        for idx, mod in enumerate(self.exog_models):
            if 'irregular' not in mod.keys():
                mod['irregular'] = False
            if 'AR' not in mod.keys():
                mod['AR'] = False
            if 'MA' not in mod.keys():
                mod['MA'] = False
            if 'level' not in mod.keys():
                mod['level'] = False
            if 'stochastic_level' not in mod.keys():
                mod['stochastic_level'] = True
            if 'trend' not in mod.keys():
                mod['trend'] = False
            if 'stochastic_trend' not in mod.keys():
                mod['stochastic_trend'] = False
            if 'freq_seasonal' not in mod.keys():
                mod['freq_seasonal'] = False
            if 'stochastic_freq_seasonal' not in mod.keys():
                mod['stochastic_freq_seasonal'] = False
            # each model must have a local level if it has a trend
            if (mod['stochastic_trend'] or mod['trend']
                    ) and not (mod['stochastic_level'] or mod['level']):
                warnings.warn("A local level term is required with trend" +
                              "for exog" + f"{idx} . Adding stochastic local level.",
                              Warning)
                mod['stochastic_level'] = True
            # Remove supurfluous level states
            if (mod['stochastic_level'] and mod['level']):
                warnings.warn("level is supurfluous with stochastic_level" +
                              ", removing deterministic level",
                              Warning)
                mod['level'] = False
            # Remove supurfluous trend states
            if (mod['stochastic_trend'] and mod['trend']):
                warnings.warn("trend is supurfluous with stochastic_trend" +
                              ", removing deterministic trend",
                              Warning)
                mod['trend'] = False
            # Remove supurfluous freq_seasonal states
            if (mod['stochastic_freq_seasonal'] and mod['freq_seasonal']):
                warnings.warn("freq_seasonal is supurfluous with" +
                              " stochastic_freq_seasonal" +
                              ", removing deterministic freq_seasonal",
                              Warning)
                mod['freq_seasonal'] = False
            # Must have 'irregular if we have 'MA' or 'AR'
            if (mod['AR'] or mod['MA']) and not mod['irregular']:
                warnings.warn("ARMA model requires irregular error in state" +
                              ", adding irregular term",
                              Warning)
                mod['irregular'] = True
            # remove any spurious components
            for c in set(mod.keys()).difference(_valid_components):
                warnings.warn(f"{c} is not a valid component type, removing.",
                              Warning)
                del mod[c]

        # Each component of each exog_model requires one state
        # Note this is will not be true when we allow more more
        # complex exog models than irregular, level, trend
        # We need a state covariance for every stochastic component
        components = []
        stochastic = []
        r = 0
        k_AR = 0
        k_MA = 0
        for mod in self.exog_models:
            # collect states needed to represent non-stationary components
            components += [mod[key] if key in _non_stationary_components 
                           else False for key in mod.keys()]
            if mod['freq_seasonal'] or mod["stochastic_freq_seasonal"]:
                if mod['freq_seasonal']:
                    key = 'freq_seasonal'
                else:
                    key = 'stochastic_freq_seasonal'
                seas = mod[key]
                n_harmonics = seas.get('harmonics',
                                       int(np.floor(seas['period'] / 2)))
                components.append(n_harmonics*2)
            # count states needed to represent stationary components
            r += max(mod['AR'],mod['MA'] + int(mod['irregular']))
            k_AR += int(mod['AR'])
            k_MA += int(mod['MA'])
            stochastic += [mod[key] if key in _stochastic_components 
                           else False for key in mod.keys()]
            if mod["stochastic_freq_seasonal"]:
                stochastic.append(2)
        self.k_states = sum(components) + r
        self.k_state_cov = sum(stochastic)
        self.k_AR = k_AR
        self.k_MA = k_MA
        # Initialize the state space model
        super(DynamicRegression,
              self).__init__(endog, k_states=self.k_states,
                             exog=exog, k_posdef=self.k_state_cov,
                             initialization='approximate_diffuse')
        self.k_posdef = self.k_state_cov
        # Construct the design matrix
        self.ssm.shapes['design'] = (self.k_endog, self.k_states, self.nobs)
        self.ssm['design'] = self.initial_design
        # construct transition matrix
        self.ssm['transition'] = self.initial_transition
        # construct selection matrix
        self.ssm['selection'] = self.initial_selection
        # construct intercept matrices
        self.obs_intecept = np.zeros(1)
        self.ssm['obs_intercept'] = self.obs_intecept
        self.state_intercept = np.zeros((self.k_states, 1))
        self.ssm['state_intercept'] = self.state_intercept
        # construct covariance matrices
        self.obs_cov = np.zeros((1, 1))
        self.ssm['obs_cov'] = self.obs_cov
        self.state_cov = np.zeros((self.k_state_cov, self.k_state_cov))
        self.ssm['state_cov'] = self.state_cov
        # param_names
        self.data.param_names = self.param_names
        # get param indicies
        self.set_param_indices()
    
    def set_param_indices(self):
        AR_param_indices = []
        MA_param_indices = []
        state_cov_param_indices = []
        # First param is observation covariance
        param_idx = 1
        for idx, mod in enumerate(self.exog_models):
            for key in _valid_components:
                if key == "stochastic_freq_seasonal":
                    state_cov_param_indices = (state_cov_param_indices + 
                                               [param_idx,param_idx+1])
                    param_idx += 2
                if key == "AR":
                    AR_param_indices.append(slice(param_idx,
                                                  param_idx + mod['AR']))
                    param_idx += mod['AR']
                if key == "MA":
                    MA_param_indices.append(slice(param_idx,
                                                  param_idx + mod['MA']))
                    param_idx += mod['MA']
                if key in _stochastic_components and mod[key]:
                    state_cov_param_indices.append(param_idx)
                    param_idx += 1
        self.AR_param_indices = AR_param_indices
        self.MA_param_indices = MA_param_indices
        self.state_cov_param_indices = state_cov_param_indices

    @property
    def param_names(self):
        param_names=['sigma2.obs']
        for idx, mod in enumerate(self.exog_models):
            for key in _valid_components:
                if key == "AR":
                    AR_names = ['sigma2.exog' +str(idx) + '.AR' + str(i)
                                for i in range(mod[key])]
                    param_names = param_names + AR_names
                elif key == "MA":
                    MA_names = ['sigma2.exog' +str(idx) + '.MA' + str(i)
                                for i in range(mod[key])]
                    param_names = param_names + MA_names
                elif key == "stochastic_freq_seasonal":
                    seas = mod[key]
                    seas_names = ['sigma2.exog' + str(idx) + "freq_seas",
                                  'sigma2.exog' + str(idx) + "freq_seas*"]
                    param_names = param_names + seas_names
                elif mod[key]:
                    param_name = 'sigma2.exog' + str(idx) + '.' + key
                    param_names.append(param_name)
        return param_names

    @property
    def start_params(self):
        """
        Starting parameters for maximum likelihood estimation

        First param is the observation covariance
        Next k_state_cov parsm are the state covariance
        """
        params = np.zeros(self.k_state_cov + 1 +
                          self.k_AR + self.k_MA)

        params[0] = np.nanvar(self.ssm.endog)
        return params

    @property
    def initial_design(self):
        """Initial design matrix"""
        # Basic design matrix
        design = np.zeros((0, self.nobs))
        for idx, mod in enumerate(self.exog_models):
            # collect states needed to represent non-stationary components
            #components = [mod[key] if key in _non_stationary_components 
            #               else False for key in mod.keys()]
            # count states needed to represent stationary components
            r = max(mod['AR'],mod['MA'] + int(mod['irregular']))
            #k_components = sum(components) + r
            #k_components = sum([bool(component) for component in mod.values()])
            d = self.exog_design(mod)
            k_components = len(d)
            exog = self.exog[:, idx]
            d = np.tile(d, (self.nobs, 1)).transpose()*np.tile(
                exog, (k_components, 1))
            design = np.r_[design, d]
        design = np.expand_dims(design, axis=0)
        return design

    def exog_design(self, mod):
        """Part of design matric for the UC model defined by mod"""
        d = np.zeros(0)
        if mod["level"]:
            d = np.r_[d, np.ones(1)]
        if mod["stochastic_level"]:
            d = np.r_[d, np.ones(1)]
        if mod["trend"]:
            d = np.r_[d, np.zeros(1)]
        if mod["stochastic_trend"]:
            d = np.r_[d, np.zeros(1)]
        if mod["freq_seasonal"]:
            seas = mod["freq_seasonal"]
            n_harmonics = seas.get('harmonics',
                                    int(np.floor(seas['period'] / 2)))
            d = np.r_[d, np.tile([1,0],n_harmonics)]
        if mod["stochastic_freq_seasonal"]:
            seas = mod["stochastic_freq_seasonal"]
            n_harmonics = seas.get('harmonics',
                                    int(np.floor(seas['period'] / 2)))
            d = np.r_[d, np.tile([1,0],n_harmonics)]
        if mod["irregular"]:
            d = np.r_[d, np.ones(1)]
        if mod["AR"] or mod['MA']:
            d = np.r_[d, np.zeros(max(mod["AR"], mod['MA']+1)-1)]
        return d

    @property
    def initial_transition(self):
        """Initial transition matrix"""
        # Lists to hold indices of AR parameters for update method
        self.AR_transition_indices = []
        transition = np.zeros((self.k_states, self.k_states))
        start = 0

        for mod in self.exog_models:
            k_components = 0
            if (mod['level'] or mod['stochastic_level']) and not (
                 mod['trend'] or mod['stochastic_trend']):
                t = np.ones((1, 1))
                k_components += 1
            elif (mod['level'] or mod['stochastic_level']):
                t = np.array([[1, 1], [0, 1]])
                k_components += 2
            else:
                t = np.empty((0,0))
            if (mod['freq_seasonal'] or mod['stochastic_freq_seasonal']):
                if mod['freq_seasonal']:
                    key = 'freq_seasonal'
                else:
                    key = 'stochastic_freq_seasonal'
                seas = mod[key]
                n_harmonics = seas.get('harmonics',
                                       int(np.floor(seas['period'] / 2)))
                blocks = tuple(self.sin_cos_block(seas['period'],
                                             j) for j in range(n_harmonics))
                t_seas = block_diag(*blocks)
                t = block_diag(t,t_seas)
                k_components += n_harmonics*2
            if mod["AR"] or mod["MA"]:
                r = max(mod["AR"],mod["MA"]+1)
                t_arma = np.c_[np.zeros((r,1)),
                               np.r_[np.eye(r-1),
                                     np.zeros((1,r-1))]]
                t = block_diag(t,t_arma)

                row_slice = slice(start + k_components,
                                  start + k_components + mod["AR"])
                col_slice = slice(start + k_components,
                                  start + k_components + 1)
                self.AR_transition_indices.append((row_slice,col_slice))

                k_components += r
            elif mod['irregular']:
                t = block_diag(np.zeros((1, 1)), t)
                k_components += 1
            end = start + k_components
            transition[start:end, start:end] = t
            start = end
        return transition

    def sin_cos_block(self,s,j):
        lambda_s = 2 * np.pi / s
        cos_lambda_block = np.cos(lambda_s * j)
        sin_lambda_block = np.sin(lambda_s * j)
        block = np.array([[cos_lambda_block, sin_lambda_block],
                          [-sin_lambda_block, cos_lambda_block]])
        return block

    @property
    def initial_selection(self):
        """Initial selection matrix"""
        # Lists to hold indices of MA parameters for update method
        self.MA_selection_indices = []
        selection = np.zeros((self.k_states, self.k_state_cov))
        start_r = 0
        start_c = 0
        for mod in self.exog_models:
            k_components = 0
            col = 0
            s = np.empty((0,0))
            if mod['stochastic_level']:
                s = block_diag(s, np.ones((1, 1)))
                k_components += 1
                col += 1
            if mod['stochastic_trend']:
                s = block_diag(s, np.ones((1, 1)))
                k_components += 1
                col += 1
            if mod['stochastic_freq_seasonal']:
                seas = mod['stochastic_freq_seasonal']
                n_harmonics = seas.get('harmonics',
                                       int(np.floor(seas['period'] / 2)))
                s = block_diag(s, np.tile(np.eye(2),n_harmonics).transpose())
                k_components += n_harmonics*2
                col += 2
            if mod["AR"] or mod["MA"]:
                r = max(mod["AR"],mod["MA"]+1)
                s = block_diag(s, 
                               np.r_[np.ones((1, 1)),
                                     np.zeros((r-1,1))]
                               )
                row_slice = slice(start_r + k_components + 1,
                                    start_r + k_components + 1 + mod['MA'])
                col_slice= slice(start_c + col,start_c + col + 1)
                self.MA_selection_indices.append((row_slice,col_slice))
                k_components += r
                col += 1
            elif mod['irregular']:
                s = block_diag(s, np.zeros((1, 1)))
                k_components += 1
                col +=1
            end_r = start_r + k_components
            end_c = start_c + col
            selection[start_r:end_r, start_c:end_c] = s
            start_r += k_components
            start_c += col
        return selection

    def update(self, params, transformed=True, **kwargs):
        """
        Update the parameters of the model

        Updates the representation matrices to fill in the new parameter
        values.

        Parameters
        ----------
        params : array_like
            Array of new parameters.
        transformed : boolean, optional
            Whether or not `params` is already transformed. If set to False,
            `transform_params` is called. Default is True..

        Returns
        -------
        params : array_like
            Array of parameters.
        """
        obs_cov = params[0]
        self.ssm.obs_cov[0, 0, 0] = obs_cov
        state_cov_params = [params[i] for i in self.state_cov_param_indices]
        self.ssm.state_cov[:, :, 0][np.diag_indices(
            self.k_state_cov)] = state_cov_params
        for idx, indices in enumerate(self.AR_param_indices):
            AR_params = params[indices]
            if AR_params:
                transition_indices = self.AR_transition_indices[idx]
                self.ssm.transition[transition_indices] = np.expand_dims(np.expand_dims(AR_params,1),2)
        for idx, indices in enumerate(self.MA_param_indices):
            MA_params = params[indices]
            if MA_params:
                selection_indices = self.MA_selection_indices[idx]
                self.ssm.selection[selection_indices] = np.expand_dims(np.expand_dims(MA_params,1),2)

    def transform_params(self, unconstrained):
        """
        Transform unconstrained parameters used by the optimizer to constrained
        parameters used in likelihood evaluation.

        Used primarily to enforce stationarity of the autoregressive lag
        polynomial, invertibility of the moving average lag polynomial, and
        positive variance parameters.

        Parameters
        ----------
        unconstrained : array_like
            Unconstrained parameters used by the optimizer.

        Returns
        -------
        constrained : ndarray
            Constrained parameters used in likelihood evaluation.
        """
        unconstrained = np.array(unconstrained, ndmin=1)
        constrained = unconstrained
        # Transform the covariance params to be positive
        constrained[0] = unconstrained[0]**2
        for i in self.state_cov_param_indices:
            constrained[i] = unconstrained[i]**2
        # Transform ARMA params to be stationary
        for i in range(len(self.exog_models)):
            AR_params_idx = self.AR_param_indices[i]
            MA_params_idx = self.MA_param_indices[i]
            ARMA_params_idx = slice(AR_params_idx.start,
                                    MA_params_idx.stop)
            ARMA_params = unconstrained[ARMA_params_idx]
            if ARMA_params:
                ARMA_params = constrain_stationary_univariate(ARMA_params)
                constrained[ARMA_params_idx] = ARMA_params
        return constrained

    def untransform_params(self, constrained):
        """
        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer

        Used primarily to reverse enforcement of stationarity of the
        autoregressive lag polynomial and invertibility of the moving average
        lag polynomial.

        Parameters
        ----------
        constrained : array_like
            Constrained parameters used in likelihood evaluation.

        Returns
        -------
        constrained : ndarray
            Unconstrained parameters used by the optimizer.
        """
        constrained = np.array(constrained, ndmin=1)
        unconstrained = constrained
        # Transform the covariance params to be positive
        unconstrained[0] = np.sqrt(constrained[0])
        for i in self.state_cov_param_indices:
            unconstrained[i] = np.sqrt(constrained[i])
        # Transform ARMA params to be stationary
        for i in range(len(self.exog_models)):
            AR_params_idx = self.AR_param_indices[i]
            MA_params_idx = self.MA_param_indices[i]
            ARMA_params_idx = slice(AR_params_idx.start,
                                    MA_params_idx.stop)
            ARMA_params = constrained[ARMA_params_idx]
            if ARMA_params:
                ARMA_params = unconstrain_stationary_univariate(ARMA_params)
                unconstrained[ARMA_params_idx] = ARMA_params
        return unconstrained

    @property
    def _res_classes(self):
        return {'fit': (DynamicRegressionResults,
                        DynamicRegressionResultsWrapper)}

class DynamicRegressionResults(MLEResults):
    """
    Class to hold results from fitting an unobserved components model.

    Parameters
    ----------
    model : UnobservedComponents instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the unobserved components
        model instance.

    See Also
    --------
    statsmodels.tsa.statespace.kalman_filter.FilterResults
    statsmodels.tsa.statespace.mlemodel.MLEResults
    """

    def __init__(self, model, params, filter_results, cov_type='opg',
                 **kwargs):
        super(DynamicRegressionResults, self).__init__(
            model, params, filter_results, cov_type, **kwargs)


    def plot_dynamic_regression(self, which="smoothed", figsize=None,
                                fitted=True, coefficients=True):
        """
        Plot the fitted values and/or estimates of time-varying regression
        coefficients
        This should be replaced with a customer Results object that has
        properties to access the various components of the state vector
        and plotting methods.

        Parameters
        ----------
        results : MLEresults object
            results object returned from fit method
        which : string or None
            If "filtered" plot the filtered results, otherwise use smoother results
        figsize : tuple of two numbers or None
            figsize passed to pandas plot methods to determine the size of each
            plot
        fitted : bool
            if True display the fitted values
        coefficients : bool
            if True display the estimated coefficients
        """
        if which == "filtered":
            state = self.filtered_state
            fitted_values = self.filter_results.forecasts[0]
        else:
            state = self.smoothed_state
            fitted_values = self.smoother_results.smoothed_forecasts[0]
        endog = self.model.endog
        if fitted:
            pd.DataFrame({"endog": endog[:, 0], "fitted_values": fitted_values}
                         ).plot(figsize=figsize)
        if coefficients:
            design = block_diag(*tuple(self.model.exog_design(mod)
                                for mod in self.model.exog_models)
                                )@state
            pd.DataFrame(design.transpose()).plot(figsize=figsize)

class DynamicRegressionResultsWrapper(MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(MLEResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(MLEResultsWrapper._wrap_methods,
                                     _methods)
wrap.populate_wrapper(DynamicRegressionResultsWrapper,  # noqa:E305
                      DynamicRegressionResultsResults)
