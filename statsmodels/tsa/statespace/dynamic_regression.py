"""
Univariate time-varying coefficient regression model

Author: Alastair Heggie
License: Simplified-BSD
"""
import pdb
import numpy as np
from scipy.signal import lfilter
from scipy.linalg import block_diag
import statsmodels.api as sm
import pandas as pd
from .tools import (constrain_stationary_univariate,
    unconstrain_stationary_univariate)
import warnings

# Construct the model
class dynamic_regression(sm.tsa.statespace.MLEModel):
    """
    Univariate time-varying coefficient regression models
    
    A model in which the coefficients of the exogonous covariates are 
    described by a univariate unobserved components model. At present
    only local level and trend models are available for the coefficinets.
    Ulitmately it is intended to add the full suite of components that
    are standard in unobserved components models: 

    Parameters
    ----------
    exog : array_like or None, optional
        Exogenous variables.
    exog_models : dict or list of dicts, optional
        Either a dict, a list containing a single dict, or a list of dict
        s, one for each exogonous regressor.
        Admisble keys are the supported component types: 'irregular', 
        'local_level', and 'trend'. If a key is included in the dict and 
        it's value is not False, then it is included in the model. If the 
        corresponding value is "deterministic" then the state 
        corresponding to that component will not have a disturbance term.
    dynamic_regression : bool
        Whether or not the regression coefficient updates over time. If 
        False then the states corresponding to the regression 
        coefficients have 0 variance.
    """
    def __init__(self, endog, exog,exog_models={"local_level":True}):
        #Get k_exog from size of exog data
        valid_components = set(['irregular','local_level','trend'])
        try:
            self.k_exog = exog.shape[1]
        except IndexError:
            exog = np.expand_dims(exog,axis=1)
            self.k_exog = exog.shape[1]
        if isinstance(exog_models,list):
            if len(exog_models)!=self.k_exog and len(exog_models)!=1:
                raise ValueError('We should either recieve no exog_model'
                    ' ,a single exog_model or a list of k_exog exog_models ')
        # If we only received a single exog_model in a list, or just one
        # dict create a k_exog length list of dicts
        if isinstance(exog_models,list):
            if len(exog_models)==1:
                self.exog_models = [exog_models[0] for m in range(self.k_exog)]
            else:
                self.exog_models = exog_models
        elif isinstance(exog_models,dict):
            self.exog_models = [exog_models for m in range(self.k_exog)]
        # Expand each exog_model dict to specify all available component
        # types as False if not already in the dict. All exog_models
        # must include a local level 

        for idx, mod in enumerate(self.exog_models):
            if 'irregular' not in mod.keys():
                mod['irregular'] = False
            if 'local_level' not in mod.keys():
                mod['local_level'] = True
            if 'trend' not in mod.keys():
                mod['trend'] = False
            #each model must have a local level
            if not mod['local_level']:
                warnings.warn(f"A local level term is required for exog {idx}.",
                    "Adding stochastic local level.")
                mod['local_level'] = True
            # remove any spurious components
            for c in set(mod.keys()).difference(valid_components):
                warnings.warn(f"{c} is not a valid component type, removing.")
                del mod[c]


        # Each component of each exog_model requires one state
        # Note this is will not be true when we allow more more
        # complex exog models than irregular, level, trend
        # We need a state covariance for every component not specified 
        # to be "deterministic"
        components=[]
        stochastic=[]
        for mod in self.exog_models:
            components += [bool(component) for component in mod.values()]
            stochastic += [bool(component) and (component != "deterministic")
                for component in mod.values()]
        self.k_states = sum(components)
        self.k_state_cov = sum(stochastic)
        # Initialize the state space model
        super(dynamic_regression, self).__init__(endog, k_states=self.k_states, exog=exog,
         k_posdef=self.k_state_cov, initialization='approximate_diffuse')
        self.k_posdef=self.k_state_cov
        #Construct the design matrix
        self.ssm.shapes['design'] = (self.k_endog,self.k_states,self.nobs)
        self.ssm['design'] = self.initial_design
        #construct transition matrix
        self.ssm['transition'] = self.initial_transition
        #construct selection matrix
        self.ssm['selection'] = self.initial_selection
        #construct intercept matrices
        self.obs_intecept = np.zeros(1)
        self.ssm['obs_intercept'] = self.obs_intecept
        self.state_intercept = np.zeros((self.k_states,1))
        self.ssm['state_intercept'] = self.state_intercept
        #construct covariance matrices
        self.obs_cov =  np.zeros((1,1))
        self.ssm['obs_cov'] = self.obs_cov
        self.state_cov = np.zeros((self.k_state_cov,self.k_state_cov))
        self.ssm['state_cov'] = self.state_cov

    @property
    def start_params(self):
        """
        Starting parameters for maximum likelihood estimation
        
        First param is the observation covariance
        Next k_state_cov parsm are the state covariance
        """
        params = np.zeros(self.k_state_cov + 1)
        params[0] = np.nanvar(self.ssm.endog)
        return params

    @property
    def initial_design(self):
        """Initial design matrix"""
        # Basic design matrix
        design = np.zeros((0,self.nobs))
        for idx ,mod in enumerate(self.exog_models):
            k_components = sum([bool(component) for component in mod.values()])
            d = self.exog_design(mod)
            exog = self.exog[:,idx]
            d = np.tile(d,(self.nobs,1)).transpose()*np.tile(
                exog,(k_components,1))
            design = np.r_[design,d]
        design = np.expand_dims(design,axis=0)
        return design

    def exog_design(self,mod):
        """Part of design matric for the UC model defined by mod"""
        d = np.zeros(0)
        if mod["irregular"]:
            d = np.r_[d,np.ones(1)]
        if mod["local_level"]:
            d = np.r_[d,np.ones(1)]
        if mod["trend"]:
            d = np.r_[d,np.zeros(1)]
        return d

    @property
    def initial_transition(self):
        """Initial transition matrix"""
        transition = np.zeros((self.k_states,self.k_states))
        start = 0
        for mod in self.exog_models:
            k_components = sum([bool(component) for component in mod.values()])
            end = start + k_components
            if mod['local_level'] and not mod['trend']:
                t = np.ones((1,1))
            else:
                t = np.array([[1,1],[0,1]])
            if mod['irregular']:
                t = block_diag(np.zeros((1,1)),t)
            transition[start:end,start:end] = t
            start = end
        return transition

    
    @property
    def initial_selection(self):
        """Initial selection matrix"""
        selection = np.zeros((self.k_states,self.k_state_cov))
        start = 0
        for mod in self.exog_models:
            k_components = 0
            if mod['local_level'] != "deterministic":
                s = np.ones((1,1))
                k_components += 1
            if mod['trend'] & (mod['trend']!="deterministic"):
                s = block_diag(s,np.ones((1,1)))
                k_components += 1
            if mod['irregular']:
                s = block_diag(np.zeros((1,1)),s)
                k_components += 1
            end = start + k_components
            selection[start:end,start:end] = s
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
        state_cov_params = params[1:]
        self.ssm.obs_cov[0,0,0]= obs_cov
        self.ssm.state_cov[:,:,0][np.diag_indices(
            self.k_state_cov)] = state_cov_params
        

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
        constrained : array_like
            Constrained parameters used in likelihood evaluation.
        """
        unconstrained = np.array(unconstrained, ndmin=1)
        constrained = np.zeros(unconstrained.shape, unconstrained.dtype)
        start = 0
        # Transform the covariance params to be positive
        constrained[start:] = unconstrained[start:]**2
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
        constrained : array_like
            Unconstrained parameters used by the optimizer.
        """
        constrained = np.array(constrained, ndmin=1)
        unconstrained = np.zeros(constrained.shape, constrained.dtype)
        # Transform the AR parameters (phi) to be stationary
        start = 0
        # Untransform the covariance params which have been transformed
        # to be positive
        unconstrained[start:] = np.sqrt(constrained[start:])
        return unconstrained

def plot_dynamic_regression(results,which="smoothed",figsize=None,
    fitted=True,coefficients=True):
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
        state = results.filtered_state
        fitted_values = results.filter_results.forecasts[0]
    else:
        state = results.smoothed_state
        fitted_values = results.smoother_results.smoothed_forecasts[0]
    endog = results.model.endog
    if fitted:
        pd.DataFrame({"endog":endog[:,0],"fitted_values":fitted_values}
            ).plot(figsize=figsize)
    if coefficients:
        if which == "filtered":
            design = block_diag(tuple(results.model.exog_design(mod) 
                for mod in results.model.exog_models))@results.filtered_state
        else:
            design = block_diag(tuple(results.model.exog_design(mod) 
                for mod in results.model.exog_models))@results.smoothed_state
        pd.DataFrame(design[0,:]).plot(figsize=figsize)