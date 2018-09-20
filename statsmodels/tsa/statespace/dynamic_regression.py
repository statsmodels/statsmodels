"""
Univariate time-varying coefficient regression model

Author: Alastair Heggie
License: Simplified-BSD
"""
import pdb
import numpy as np
from scipy.signal import lfilter
import statsmodels.api as sm
import pandas as pd
from .tools import (constrain_stationary_univariate,
    unconstrain_stationary_univariate)

# Construct the model
class SDR(sm.tsa.statespace.MLEModel):
    """
    Univariate time-varying coefficient regression models
    
    A model in which the coefficients of the exogonous covariates are 
    described by a univariate structural model

    Parameters
    ----------
    exog : array_like or None, optional
        Exogenous variables.
    freq_seasonal: list of dicts or None, optional.
        Whether (and how) to model seasonal component(s) with trig. functions.
        If specified, there is one dictionary for each frequency-domain
        seasonal component.  Each dictionary must have the key, value pair for
        'period' -- integer and may have a key, value pair for
        'harmonics' -- integer. If 'harmonics' is not specified in any of the
        dictionaries, it defaults to the floor of period/2.
    arma_order : tuple
        A tuple giving the order of the autoregressive and moving average
        components of the error term respectively. (0,0) by default.
    dynamic_regression : bool
        Whether or not the regression coefficient updates over time. If False
        then the states corresponding to the regression coefficients have 0
        variance.
    """
    def __init__(self, endog, exog, freq_seasonal,arma_order=(0,0),
        dynamic_regression=True):
        self.arma_order = arma_order
        self.freq_seasonal_periods = [d['period'] for d in freq_seasonal]
        self.freq_seasonal_harmonics = [d.get(
            'harmonics', int(np.floor(d['period'] / 2))) for
            d in freq_seasonal]
        self.ar_order = self.arma_order[0]
        self.ma_order = self.arma_order[1]
        self.r = max(self.ar_order,self.ma_order+1)
        self.enforce_stationarity = True
        self.enforce_invertibility = True
        self.dynamic_regression = dynamic_regression

        #Get k_exog from size of exog data
        try:
            self.k_exog = exog.shape[1]
        except IndexError:
            exog = np.expand_dims(exog,axis=1)
            self.k_exog = exog.shape[1]
        k_states = (self.r + self.k_exog +
            sum(self.freq_seasonal_harmonics)*2)
        k_state_cov = (self.dynamic_regression*self.k_exog * 3 + 1)
        # Initialize the state space model
        super(SDR, self).__init__(endog, k_states=k_states, exog=exog,
         k_posdef=k_state_cov, initialization='approximate_diffuse')
        self.k_posdef=k_state_cov
        self.k_state_cov=k_state_cov
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
        self.obs_cov =  np.zeros(1)
        self.ssm['obs_cov'] = self.obs_cov
        self.state_cov = np.zeros((self.k_state_cov,self.k_state_cov))
        self.ssm['state_cov'] = self.state_cov

    @property
    def start_params(self):
        """
        Starting parameters for maximum likelihood estimation
        
        first ar_order terms are ar_params, next ma_order terms are ma params
        next k_state_cov params are the state covariance, the first of which
        is the observation covariance which is in the state due to ARMA model
        """
        params = np.zeros(self.ar_order + self.ma_order + self.k_state_cov)
        params[self.ar_order + self.ma_order] = np.nanvar(self.ssm.endog)
        return params

    @property
    def initial_design(self):
        """Initial design matrix"""
        # Basic design matrix
        design = np.vstack((np.ones(self.nobs),np.zeros((self.r-1,self.nobs)),self.exog.transpose()))
        for ix, h in enumerate(self.freq_seasonal_harmonics):
            series = self.exog[:,ix]
            lines=np.array([series,np.repeat(0,self.nobs)])
            array=np.vstack(tuple(lines for i in range(0,h)))
            design = np.vstack((design,array))
        design = np.expand_dims(design,axis=0)
        return design

    @property
    def initial_transition(self):
        """Initial transition matrix"""
        transition = np.zeros((self.k_states,self.k_states))
        #ARMA
        transition[1:self.r,0:self.r-1] = np.eye(self.r-1)
        #Exog level
        i = self.r
        transition[i:i+self.k_exog,i:i+self.k_exog] = np.eye(self.k_exog)
        #Exog seasonal
        i=self.r + self.k_exog
        for ix, h in enumerate(self.freq_seasonal_harmonics):
            n = 2 * h
            p = self.freq_seasonal_periods[ix]
            lambda_p = 2 * np.pi / float(p)
            t = 0 # frequency transition matrix offset
            for block in range(1, h + 1):
                cos_lambda_block = np.cos(lambda_p * block)
                sin_lambda_block = np.sin(lambda_p * block)
                trans = np.array([[cos_lambda_block, sin_lambda_block],
                                  [-sin_lambda_block, cos_lambda_block]])
                trans_s = np.s_[i + t:i + t + 2]
                transition[trans_s, trans_s] = trans
                t += 2
            i += n
        return transition
    
    @property
    def initial_selection(self):
        """Initial selection matrix"""
        selection = np.zeros((self.k_states,self.k_state_cov))
        #ARMA
        i,j = 0,0
        selection[i,j] = 1
        i += self.r
        j += 1
        if self.dynamic_regression:
            #Exog level
            selection[i:i+self.k_exog,j:j+self.k_exog] = np.eye(self.k_exog)
            i += self.k_exog
            j += self.k_exog
            #Exog seasonal
            for ix, h in enumerate(self.freq_seasonal_harmonics):
                select = np.vstack([np.eye(2) for i in range(h)])
                selection[i:i+2*h,j:j+2] = select
                i += 2*h
                j += 2
        selection = np.expand_dims(selection,axis=2)
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
        ar_params = params[0:self.ar_order]
        ma_params = params[self.ar_order:self.ma_order+self.ar_order]
        state_cov_params = params[self.ma_order+self.ar_order:]
        if self.ma_order>0:
            self.ssm.design[0,1:1+self.ma_order,:] = ma_params
        if self.ar_order>0:
            self.ssm.transition[0,:self.ar_order,0] = ar_params
        di = np.diag_indices(self.k_state_cov) + tuple(
            [np.zeros(self.k_state_cov,dtype=np.int)])
        self.ssm.state_cov[di] = state_cov_params


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
        # Transform the AR parameters (phi) to be stationary
        start = 0
        end = 0
        if self.ar_order > 0:
            end += self.ar_order
            if self.enforce_stationarity:
                constrained[start:end] = (
                    constrain_stationary_univariate(unconstrained[start:end])
                )
            else:
                constrained[start:end] = unconstrained[start:end]
            start += self.ar_order

        # Transform the MA parameters (theta) to be invertible
        if self.ma_order > 0:
            end += self.ma_order
            if self.enforce_invertibility:
                constrained[start:end] = (
                    -constrain_stationary_univariate(unconstrained[start:end])
                )
            else:
                constrained[start:end] = unconstrained[start:end]
            start += self.ma_order
        # Transform the state covariance to be positive
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
        end=0
        if self.ar_order > 0:
            end += self.ar_order
            if self.enforce_stationarity:
                unconstrained[start:end] = (
                    unconstrain_stationary_univariate(constrained[start:end])
                )
            else:
                unconstrained[start:end] = constrained[start:end]
            start += self.ar_order

        # Transform the MA parameters (theta) to be invertible
        if self.ma_order > 0:
            end += self.ma_order
            if self.enforce_invertibility:
                unconstrained[start:end] = (
                    unconstrain_stationary_univariate(-constrained[start:end])
                )
            else:
                unconstrained[start:end] = constrained[start:end]
            start += self.ma_order
        # Transform the state covariance to be positive
        unconstrained[start:] = np.sqrt(constrained[start:])
        return unconstrained

def plot_SDR(results,which="smoothed",alpha=None,figsize=None,combine_coefs=False,
    combine_components=False,fitted=True,components=True):
    if which == "filtered":
        state = results.filtered_state
        fitted_values = results.filter_results.forecasts[0]
    else:
        state = results.smoothed_state
        fitted_values = results.smoother_results.smoothed_forecasts[0]
    endog = results.model.endog
    if fitted == True:
        pd.DataFrame({"endog":endog[:,0],"fitted_values":fitted_values}
            ).plot(figsize=figsize)
    if components == True:
        k_arma_states = results.model.r
        k_exog = results.model.k_exog
        nobs = results.model.nobs
        freq_seasonal_harmonics = results.model.freq_seasonal_harmonics
        nobs = results.model.nobs

        regression_coefs = state[range(k_arma_states,k_arma_states+k_exog),:]

        if combine_coefs == True:
            regression_coefs = regression_coefs.sum(axis=0)
            df1 = pd.DataFrame(regression_coefs,columns=
            ["regression_coefs"]
            )
        else:
            df1 = pd.DataFrame(regression_coefs.transpose(),columns=
            ["regression_coef{!r}".format(ix) for ix in range(k_exog)]
            )


        seasonal_regression_coefs = {}
        offset=k_exog + k_arma_states
        for ix, harmonics in enumerate(freq_seasonal_harmonics):
            h_idx = np.array([i*2 for i in range(0,harmonics)])+offset
            offset += harmonics*2
            seasonal_regression_coefs['regression_coef{!r}'.format(ix)] = state[h_idx,:].sum(axis=0)
        if combine_coefs == True:
            df2= pd.DataFrame(seasonal_regression_coefs)
            df2 = pd.DataFrame(df2.sum(axis=1),columns=["regression_coefs"])
        else:
            df2= pd.DataFrame(seasonal_regression_coefs
            )
        if combine_components:
            (df1+df2).plot(figsize=figsize,subplots=True)
        else:
            df1.plot(figsize=figsize,subplots=True)
            df2.plot(figsize=figsize,subplots=True)   