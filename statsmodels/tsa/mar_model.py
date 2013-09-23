"""
Markov Autoregressive Models

Author: Chad Fulton
License: BSD

References
----------

Hamilton, James D. 1989.
"A New Approach to the Economic Analysis of
Nonstationary Time Series and the Business Cycle."
Econometrica 57 (2) (March 1): 357-384.

Hamilton, James D. 1994.
Time Series Analysis.
Princeton, N.J.: Princeton University Press.

Kim, Chang-Jin, and Charles R. Nelson. 1999.
"State-Space Models with Regime Switching:
Classical and Gibbs-Sampling Approaches with Applications".
MIT Press Books. The MIT Press.

Notes
-----

Roadmap:
- Correctly estimate covariance matrix
- Add expected regime duration
- Add results class
- Add plotting capabilities (e.g. for showing probabilities of each regime at
  time t - can do a line plot for M=2, otherwise an area plot)
- Add support for model specification testing (the usual nuisence approach)
- Add support for time varying parameters
- Add support for Regime-specific error variances
- Add support for MS-VAR
- Add support for state-space models
- Add support for the EM algorithm



The MAR model has four types of parameters:
- transition probabilities
- AR parameters
- standard deviation parameters
- mean parameters

The standard case is the assumption of fixed transition probabilities. In this
case, there are nstates * (nstates - 1) parameters to be estimated, and
nstates^2 parameters used in the model. See below for more details.
If the transition probabilities are allowed to change over time, it is called a
Time-Varying Transition Probabilites (TVTP) Markov-Switching Model. In this
case, an additional exogenous matrix must be supplied that are assumed to
determine the transition probabilities at each point in time. The number of
parameters to be estimated is (k+1)(nstates) * (k+1)(nstates-1), and the number
of parameters used in the model is (k+1)^2 * nstates^2.

The AR, standard deviation, and mean parameters may be allowed to differ
across states, or may be restricted to be the same.
If the AR parameters are allowed to differ, there are `order`*`nstates`
parameters to be estimated and used in the model. If they are not, then there
are `order` parameters.
If the standard deviation (or the mean) parameter is allowed to differ,
there are `nstates` standard deviation (or mean) parameters to estimate and use
in the model, otherwise there is only 1.

Parameters are used in two ways:

(1) Optimization: the optimization routine requires a flat array of parameters
    where each parameter can range over (-Inf, Inf), and it adjusts each
    parameter while finding the values that optimize the objective function,
    here the log likelihood. Thus if there are M states with regime
    homoskedasticity, there must be only a single standard deviation parameter
    in the array passed to the optimizer. If there are M states with regime
    heteroskedasticity, there must be M standard deviation parameters in the
    array.
    These are the parameters passed to the MAR.loglike() method.
(2) Initializing the filter: the parameters selected by the optimizer at each
    iteration are then used to calculate the inputs to the filtering routines
    (i.e. joint_probabilities and marginal_conditional_densities). For this,
    they need to be (a) transformed to their actual ranges (e.g. probabilities
    to lie within [0,1]) and (b) expanded to the full state range. In the
    regime homoskedasticity example above, the single standard deviation
    parameter must be expanded so that there is one parameter per regime. In
    this case, each regime's standard deviation parameter would be identical.
    These are the parameters passed to the MAR.filter() and
    MAR.initialize_filter() methods.

To achieve this, several helper methods are employed:
- MAR.expand_params()
  - Takes an array of parameters from the optimizer, and returns an expanded
    array of parameters suitable for use in the model.
  - (If not TVTP) Expands the probability vector into a transition vector
  - Expands restrictions (e.g. if parameters are restricted to not change, it
    expands the single set of `order` parameters to `nstates`*`order`
    parameters).
  - Always returns `nparams` parameters.
- MAR.contract_params()
  - Takes an array of parameters suitable for use in the model, and returns a
    contracted array of parameters to be passed to the optimizer.
- MAR.fuse_params()
  - Takes each set of parameters separately and fuses them into a single array
    (used to maintain consistent parameter ordering in e.g. the optimization
    setting).
- MAR.separate_params()
   - Takes an array of parameters and separates it into the component parts.
- MAR.transform_params()
   - Takes an array of parameters (either the contracted or expanded set of
     parameters) that are unconstrained (as would be used in the optimizer) and
     transforms them to a constrained form suitable for use in the model (e.g.
     transforms probabilities from (-Inf, Inf) to [0,1])
- MAR.untransform_params()
   - Takes an array of parameters (either the contracted or expanded set of
     parameters) that are constrained (as would be used in the model) and
     reverses the transformation to make them be unconstrained (e.g. transforms
     probabilities from [0,1] to (-Inf, Inf))

The flow of parameters through the model looks like:

(1) MAR.fit() is called, optionally with the start_params argument.
(2) MAR.loglike() is called by the optimizer, and is passed the contracted,
    untransformed (i.e. unconstrained) params.
(3) The parameters are
    a. Transformed (i.e. constrained to lie within the actual parameter spaces)
    b. Expanded
(4) MAR.initialize_filter() is called with the expanded, transformed
    parameters.

The default functionality

To allow arbitrary specification of regime-switching for the parameters, 


Internally, the transition matrix is constructed to be left stochastic, and 
the transition vector is created from it by stacking its columns.

Notes about the (internally used, left stochastic) transition matrix:
    The nstates x nstates Markov chain transition matrix.

    The [i,j]th element of the matrix corresponds to the
    probability of moving to the i-th state given that you are
    in the j-th state. This means that it is the columns that
    sum to one, aka that the matrix is left stochastic.

    It looks like:

    | p11 p12 ... p1M |
    | p21  .       .  |
    |  .       .   .  |
    | pM1   ...   pMM |

    Here the element pM1 is the probability of moving from the first state
    to the M-th state. This representation of the matrix:
    - is consistent with usual row / column indexing, but
    - inconveniently represents the "from" state as the second index and the
      "to" state as the first index.

    Kim and Nelson (1999) represent this same matrix (on p.70) as:

    | p11 p21 ... pM1 |
    | p12  .       .  |
    |  .       .   .  |
    | p1M   ...   pMM |

    This is the same, identical, matrix, just with a different indexing
    convention. Here, p1M is the probability of moving from the first state to
    the M-th state. This representation of the matrix is:
    - inconsitent with usual row / column indexing, but
    - conveniently represents the "from" and "to" states as the, respectively,
      first and second indices

    Constructing the internally used transition vector (from column stacking)
    is easily achieved via:

        P.reshape((1, P.size), order='F')
    or
        P.reshape(-1, order='F')
    or
        P.ravel(order='F')
    or
        P.T.ravel()
    etc.

    Two convenience functions to assist with transition matrices are:

    - MAR.transition_vector() accepts a transition matrix and converts it to a
      transition vector. The default options create the vector via column
      stacking. (Note: this function also may accept a probabilities vector,
      which is then converted to a transition vector - see below for details)

    - MAR.transition_matrix() accepts a transition vector and converts it to a
      transition matrix. The default options create the matrix by setting the
      first M elements of the vector to the first column of the matrix, etc.

Notes about the (internally used, constructed via column stacking) transition
vector:

    The full (column stacked, M^2 length) transition vector is of the form:

    [ p11, p21, ..., pM1, p12, p22, ..., pM2, ..., p1M, p2M, ..., pMM ]

    This is the version that used internally, which means that it is:
        - Returned by mar_c.hamilton_filter() and MAR.hamilton_filter()
        - Accepted as an argument by MAR.smooth()

    However, it is not the version that is accepted to externally facing
    methods, because it contains redundant information. Each column of the
    (left stochastic) transition matrix has M entries, but e.g. the M-th entry
    could be left out, and instead calculated as the sum of the first M-1
    entries. This is significant because the M-th (or whichever arbitrary
    entry is left out) is constrained, and so is not estimated separately from
    the other M-1 entries. Thus the optimizer will only optimize over M * (M-1)
    probability values.

    Some normalization must be specified, and the convention here is that the
    last row of the transition matrix will be left off. This means that from
    the full transition vector above, each M-th element must be left off (this
    corresponds to eliminating the last row of the transition matrix before
    creating the vector by stacking the columns). It is of the form:

    [ p11, p21, ..., p(M-1)1, p12, p22, ..., p(M-1)2, ..., p1M, p2M, ..., p(M-1)M ]

    and the last elements are calculated as:

    PM* = 1 - p1* - p2* - ... - p(M-1)*

    To distinguish between these two, the following terminology will be used:
    - `transition_vector` refers to the full transition vector
    - `probabilities` refers to the version without each M-th value

    There are convenience functions to assist in moving between these two
    representations:

    - probabilities() accepts a transition vector and returns the
      corresponding probabilities vector by removing each M-th value
    - transition_vector() accepts a probabilities vector and returns the
      corresponding transition vector by calculating and adding the M-th values
      (this is its behavior if its first argument has ndim=1. If the first
      argument has ndim=2, then it is assumed to be converting a transition
      matrix to a transition vector by column stacking)

"""

from __future__ import division
import numpy as np
import pandas as pd
import statsmodels.tsa.base.tsa_model as tsbase
import statsmodels.base.model as base
from statsmodels.base import data
from statsmodels.tsa.tsatools import add_constant, lagmat
from statsmodels.regression.linear_model import OLS, OLSResults
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.tools.decorators import (cache_readonly, cache_writable,
                                          resettable_cache)
import statsmodels.base.wrapper as wrap
from scipy import stats
from mar_c import (hamilton_filter, tvtp_transition_vectors_left,
                   tvtp_transition_vectors_right,
                   marginal_conditional_densities)
import resource

class MAR(tsbase.TimeSeriesModel):
    """
    "An autoregressive model of order k with first-order , M-state
    Markov-switching mean and variance"

    Parameters
    ----------
    endog : array-like
        The endogenous variable. Assumed not to be in deviation-from-mean form.
    order : integer
        The order of the autoregressive parameters.
    nstates : integer
        The number of states in the Markov chain.
    switch_ar : boolean, optiona
        Whether or not AR parameters are allowed to switch with regimes.
    switch_var : boolean, optional
        Whether or not the variances are allowed to vary across regimes.
        (Regime-specific Heteroskedasticity)
    switch_means : boolean, optional
        Whether or not the means are allowed to vary across regimes.
    tvtp_data : array-like, optional
        A vector or matrix of exogenous or lagged variables to use in
        calculating time varying transition probabilities (TVTP). TVTP is only
        used if this variable is provided.

    References
    ----------
    Kim, Chang-Jin, and Charles R. Nelson. 1999.
    "State-Space Models with Regime Switching:
    Classical and Gibbs-Sampling Approaches with Applications".
    MIT Press Books. The MIT Press.

    Notes
    -----
    States are zero-indexed.
    """

    def __init__(self, endog, order, nstates,
                 switch_ar=False, switch_var=False, switch_mean=True,
                 tvtp_exog=None,
                 dates=None, freq=None, missing='none'):

        # "Immutable" properties
        self.nobs_initial = order
        self.nobs = endog.shape[0] - order
        self.order = order
        self.nstates = nstates

        # Determine switching parameters

        # Transition probabilities
        if tvtp_exog is None:
            self.tvtp_exog = np.ones((self.nobs + self.nobs_initial + 1, 1))
        else:
            self.tvtp_exog = add_constant(tvtp_exog)
        self.tvtp_order = self.tvtp_exog.shape[1]
        if not self.tvtp_exog.shape[0] == self.nobs + self.nobs_initial + 1:
            raise ValueError('Length of exogenous data determining the time'
                             ' varying transition probabilities must have'
                             ' length equal to %d: the number of observations'
                             ' plus one. Got length %d.' %
                             (self.nobs + self.nobs_initial + 1,
                              self.tvtp_exog.shape[0]))
        self.nparams_prob = (
            self.nstates * (self.nstates - 1) * self.tvtp_order
        )

        # AR parameters
        if switch_ar == True:
            self.nparams_ar = self.nstates*self.order
            self.switch_ar = True
            self.switch_method_ar = 'all'
        elif switch_ar == False:
            self.nparams_ar = self.order
            self.switch_ar = False
            self.switch_method_ar = 'none'
        elif isinstance(switch_ar, (list, np.ndarray)):
            self.nparams_ar = 0
            self.switch_ar = np.asarray(switch_ar)
            if not self.switch_ar.shape[0] == nstates:
                raise ValueError('Fixed switching definitions for AR'
                                 ' parameters must be an array specifying a'
                                 ' fixed value for each state. Expected length'
                                 ' %d, got length %d.' %
                                 (nstates, self.switch_ar.shape[0]))
            self.switch_method_ar = 'fixed'
        elif isinstance(switch_ar, tuple) and callable(switch_ar[1]):
            self.nparams_ar, self.switch_ar = switch_ar
            self.switch_method_ar = 'custom'
        else:
            raise ValueError('Custom switching definitions for AR'
                             ' parameters must be an array of fixed values or'
                             ' must be a tuple with the number of parameters'
                             ' to estimate as the first value and a callback'
                             ' as the second value.')

        # Variance parameters
        if switch_var == True:
            self.nparams_var = self.nstates
            self.switch_var = True
            self.switch_method_var = 'all'
        elif switch_var == False:
            self.nparams_var = 1
            self.switch_var = False
            self.switch_method_var = 'none'
        elif isinstance(switch_var, (list, np.ndarray)):
            self.nparams_var = 0
            self.switch_var = np.asarray(switch_var)
            if not self.switch_var.shape[0] == nstates:
                raise ValueError('Fixed switching definitions for variance'
                                 ' parameters must be an array specifying a'
                                 ' fixed value for each state. Expected length'
                                 ' %d, got length %d.' %
                                 (nstates, self.switch_var.shape[0]))
            self.switch_method_var = 'fixed'
        elif isinstance(switch_var, tuple) and callable(switch_var[1]):
            self.nparams_var, self.switch_var = switch_var
            self.switch_method_var = 'custom'
        else:
            raise ValueError('Custom switching definitions for variance'
                             ' parameters must be an array of fixed values or'
                             ' must be a tuple with the number of parameters'
                             ' to estimate as the first value and a callback'
                             ' as the second value.')

        # Mean parameters
        if switch_mean == True:
            self.nparams_mean = self.nstates
            self.switch_mean = True
            self.switch_method_mean = 'all'
        elif switch_mean == False:
            self.nparams_mean = 1
            self.switch_mean = False
            self.switch_method_mean = 'none'
        elif isinstance(switch_mean, (list, np.ndarray)):
            self.nparams_mean = 0
            self.switch_mean = np.asarray(switch_mean)
            if not self.switch_mean.shape[0] == nstates:
                raise ValueError('Fixed switching definitions for mean'
                                 ' parameters must be an array specifying a'
                                 ' fixed value for each state. Expected length'
                                 ' %d, got length %d.' %
                                 (nstates, self.switch_mean.shape[0]))
            self.switch_method_mean = 'fixed'
        elif isinstance(switch_mean, tuple) and callable(switch_mean[1]):
            self.nparams_mean, self.switch_mean = switch_mean
            self.switch_method_mean = 'custom'
        else:
            raise ValueError('Custom switching definitions for mean'
                             ' parameters must be an array of fixed values or'
                             ' must be a tuple with the number of parameters'
                             ' to estimate as the first value and a callback'
                             ' as the second value.')

        # The number of parameters used by the optimizer
        self.nparams = (
            self.nparams_prob +
            self.nparams_ar +
            self.nparams_var +
            self.nparams_mean
        )
        # The number of parameters used by the model
        # (not quite right for nparams_prob, in case of TVTP)
        self.nparams_prob_full = self.nparams_prob
        self.nparams_ar_full = self.order * self.nstates
        self.nparams_var_full = self.nstates
        self.nparams_mean_full = self.nstates
        self.nparams_full = (
            self.nparams_prob_full +
            self.nparams_ar_full +
            self.nparams_var_full +
            self.nparams_mean_full
        )

        # If we got custom (callable) switch functions, test them
        test_args = self.separate_params(np.ones((self.nparams,)))
        if self.switch_method_ar == 'custom':
            test_ar = len(self.switch_ar(*test_args))
            if not test_ar == self.nparams_ar_full:
                raise ValueError('Invalid custom switching function for AR'
                                 ' parameters. Must return a vector of length'
                                 ' %d. Got a parameter of length %d.' %
                                 (self.nparams_ar_full, test_ar))
        if self.switch_method_var == 'custom':
            test_var = len(self.switch_var(*test_args))
            if not test_var == self.nparams_var_full:
                raise ValueError('Invalid custom switching function for'
                                 ' variance parameters. Must return a vector'
                                 ' of length %d. Got a parameter of length'
                                 ' %d.' % (self.nparams_ar_full, test_var))
        if self.switch_method_mean == 'custom':
            test_mean = len(self.switch_mean(*test_args))
            if not test_mean == self.nparams_mean_full:
                raise ValueError('Invalid custom switching function for mean'
                                 ' parameters. Must return a vector of length'
                                 ' %d. Got a parameter of length %d.' %
                                 (self.nparams_mean_full, test_mean))


        # Make a copy of original datasets
        orig_endog = endog
        orig_exog = lagmat(orig_endog, order)

        # Create datasets / complete initialization
        endog = orig_endog[self.nobs_initial:]
        exog = orig_exog[self.nobs_initial:]
        super(MAR, self).__init__(endog, exog, missing=missing)

        # Overwrite originals
        self.data.orig_endog = orig_endog
        self.data.orig_exog = orig_exog

        # Cache
        self.augmented = np.c_[endog, exog]

    def expand_params(self, params):
        params = np.asarray(params)
        # Make sure they're not already expanded
        if params.shape == (self.nparams_full,):
            return params
        elif params.shape != (self.nparams,):
            raise ValueError('Unexpected parameter vector shape. Expected %s,'
                             ' got %s.' % ((self.nparams,), params.shape))

        transitions, ar_params, stddevs, means = self.separate_params(params)

        # Transition probabilities
        # (these are expanded later, due to possibility of TVTP)

        # AR parameters
        if self.switch_method_ar == 'all':
            pass
        elif self.switch_method_ar == 'none':
            ar_params = np.tile(ar_params, self.nstates)
        elif self.switch_method_ar == 'fixed':
            ar_params = self.switch_ar
        else:
            ar_params = self.switch_ar(transitions, ar_params, stddevs, means)

        # Variance parameters
        if self.switch_method_var == 'all':
            pass
        elif self.switch_method_var == 'none':
            stddevs = np.tile(stddevs, self.nstates)
        elif self.switch_method_var == 'fixed':
            stddevs = self.switch_var
        else:
            stddevs = self.switch_var(transitions, ar_params, stddevs, means)

        # Mean parameters
        if self.switch_method_mean == 'all':
            pass
        elif self.switch_method_mean == 'none':
            means = np.tile(means, self.nstates)
        elif self.switch_method_mean == 'fixed':
            means = self.switch_mean
        else:
            means = self.switch_mean(transitions, ar_params, stddevs, means)

        return self.fuse_params(transitions, ar_params, stddevs, means)

    def contract_params(self, params):
        raise NotImplementedError

    def fuse_params(self, transitions, ar_params, stddevs, means):
        """
        Combines the component parameters into a single array.

        Parameters
        ----------
        transitions : array-like
            A vector of transition probabilities
        ar_params : array-like
            The AR parameters
        stddevs : array-like
            The standard deviations for each state
        means : array-like
            The means for each state

        Returns
        -------
        params : array-like
            An array of parameters
        """
        return np.r_[transitions, ar_params, stddevs, means]

    def separate_params(self, params):
        """
        Separates a single array of parameters into the component pieces.

        Parameters
        ----------
        params : array-like
            An array of parameters

        Returns
        -------
        transitions : array-like
            A vector of transition probabilities
        ar_params : array-like
            The AR parameters
        stddevs : array-like
            The standard deviations for each state
        means : array-like
            The means for each state
        """
        params = np.asarray(params)

        # Separate the parameters
        if params.shape == (self.nparams,):
            nparams = np.cumsum((self.nparams_prob, self.nparams_ar,
                       self.nparams_var, self.nparams_mean))
        elif params.shape == (self.nparams_full,):
            nparams = np.cumsum((self.nparams_prob_full, self.nparams_ar_full,
                       self.nparams_var_full, self.nparams_mean_full))
        else:
            raise ValueError('Invalid number of parameters. Expected %s or %s,'
                             ' got %s.' % ((self.nparams,),
                             (self.nparams_full,), params.shape))
        transitions = params[:nparams[0]]
        ar_params = params[nparams[0]:nparams[1]]
        stddevs = params[nparams[1]:nparams[2]]
        means = params[nparams[2]:]

        return transitions, ar_params, stddevs, means

    def transform_params(self, params, method='logit'):
        """
        Transforms a set of unconstrained parameters to a set of contrained
        parameters.

        Optimization methods (e.g scipy.optimize) work on sets of unconstrained
        parameters, but the model requires e.g. that probability values lie in
        the range [0, 1]. This function takes the unconstrained parameters from
        the optimizer and transforms them into parameters usable in the model
        (e.g to evaluate the likelihood).

        Parameters
        ----------
        params : array-like
            An array of unconstrained parameters
        method : {'logit', 'abs'}, optional
            The method used to transform parameters on the entire real line to
            parameters in the range (0,1).

        Returns
        -------
        params : an array of constrained parameters
        """
        transitions, ar_params, stddevs, means = self.separate_params(params)

        # Standard deviations: transform to always be positive
        stddevs = np.exp(-stddevs)

        return self.fuse_params(transitions, ar_params, stddevs, means)

    def untransform_params(self, params, method='logit'):
        """
        Transforms a set of constrained parameters to a set of uncontrained
        parameters.

        Optimization methods (e.g scipy.optimize) work on sets of unconstrained
        parameters, but the model requires e.g. that probability values lie in
        the range [0, 1]. This function takes the constrained parameters used
        in the model and transforms them into parameters usable by the
        optimizer (e.g to take step sizes, etc.).

        Parameters
        ----------
        params : array-like
            An array of constrained parameters
        method : {'logit', 'abs'}, optional
            The method used to transform parameters on the entire real line to
            parameters in the range (0,1).

        Returns
        -------
        params : an array of unconstrained parameters
        """
        transitions, ar_params, stddevs, means = self.separate_params(params)

        stddevs = -np.log(stddevs)

        return self.fuse_params(transitions, ar_params, stddevs, means)

    def transform_jacobian(self, params):
        """
        Evaluates the jacobian of the transformation function.

        Used to calculate standard errors via the delta method (the method of
        propagation of errors).

        Parameters
        ----------
        params : array-like
            An array of parameters

        Returns
        -------
        jacobian : array-like
            The jacobian matrix of the transformation function, evaluated at
            the given parameters.
        """
        transitions, ar_params, stddevs, means = self.separate_params(params)

        # The only transformation to take the gradient of is on probabilities
        if not tvtp:
            transitions = (
                np.exp(transitions) / (1 + np.exp(transitions))**2
            )
            vector = np.r_[transitions, [1]*(self.nparams-len(transitions))]
        else:
            vector = [1] * nparams

        return np.diag(vector)

    def loglike(self, params):
        """
        Calculate the log likelihood.

        Parameters
        ----------
        params : array-like
            An array of unconstrained, contracted parameters

        Returns
        -------
        loglike : float
            The log likelihood of the model evaluated at the given parameters.

        Notes
        -----
        Uses unconstrained parameters because it is meant to be called via
        the optimization routine, which uses unconstrained parameters.
        """
        params = self.transform_params(params)
        params = self.expand_params(params)

        (joint_probabilities,
         marginal_conditional_densities) = self.initialize_filter(params)

        transitions, _, _, _ = self.separate_params(params)
        transition_vectors = self.tvtp_transition_vectors(transitions, 'right')
        transition_vectors = transition_vectors[self.nobs_initial:]

        marginal_densities, _, _ = hamilton_filter(
            self.nobs, self.nstates, self.order,
            transition_vectors, joint_probabilities,
            marginal_conditional_densities
        )

        return np.sum(np.log(marginal_densities))

    def tvtp_transition_vectors(self, transitions, matrix_type='left'):
        """
        Create a vector of time varying transition probability vectors

        Each transition vector is the vectorized version of the transition
        matrix.

        Parameters
        ----------
        transitions : array-like
            A vector of transition parameters, with length
            self.nstates * (self.nstates - 1) * self.tvtp_order
        matrix_type : {'left', 'right'}, optional
            The method by which the corresponding transition matrix would be
            constructed from the returned transition vector.
            - If 'left', the transition matrix would be constructed to be left
              stochastic.
            - If 'right', the transition matrix would be constructed to be
              right stochastic.
            See MAR.transition_matrix() or the module docstring for details.

        Returns
        -------
        transition_vector : array
            An (nobs+1) x (nstates*nstates) matrix (i.e. an nobs+1 vector of
            nstates*nstates transition vectors).
        """
        transitions = transitions.reshape(
            self.nstates*(self.nstates-1), self.tvtp_order
        )

        if matrix_type == 'left':
            fn = tvtp_transition_vectors_left
        elif matrix_type == 'right':
            fn = tvtp_transition_vectors_right
        else:
            raise ValueError("Invalid matrix type method. Must be one of"
                             " {'left', 'right'}, corresponding to a left"
                             " stochastic or right stochastic transition"
                             " matrix. Got %s." % matrix_type)

        transition_vectors = fn(
            self.nobs + self.nobs_initial, self.nstates, self.tvtp_order,
            transitions, self.tvtp_exog
        )
        return transition_vectors

    def probability_vector(self, transitions, matrix_type='left'):
        """
        Create a probability vector

        The probability vector is the vectorized version of the transition
        matrix, excluding its last row.

        Parameters
        ----------
        transitions : array-like
            A vector of transition values for the probability vector. It can be
            either:
            - a transition vector, if it has 1-dimension
            - a transition matrix, if it has 2-dimensions
            See the module docstring for more information about the difference.
        matrix_type : {'left', 'right'}, optional
            The method by which the corresponding transition matrix would be
            constructed from the returned probability vector.
            - If 'left', the transition matrix would be constructed to be left
              stochastic.
            - If 'right', the transition matrix would be constructed to be
              right stochastic.
            See MAR.transition_matrix() or the module docstring for details.

        Returns
        -------
        probability_vector : array
            A 1-dimensional probability vector

        Notes
        -----
        See module docstring for details on the distinction between the terms
        `transitions`, `probability_vector`, `transition_vector`, and
        `transition_matrix`.
        """

        # Figure out which type of stochastic matrix we have
        if matrix_type == 'left':
            order = 'F'
        elif matrix_type == 'right':
            order = 'C'
        else:
            raise ValueError("Invalid matrix type method. Must be one of"
                             " {'left', 'right'}, corresponding to a left"
                             " stochastic or right stochastic transition"
                             " matrix. Got %s." % matrix_type)

        # Handle transition vector (convert to a transition matrix first)
        if transitions.ndim == 1:
            transitions = self.transition_matrix(array, order)
        if not transitions.ndim == 2:
            raise ValueError('Invalid input array. Must be 1-dimensional (a'
                             ' transition vector) or 2-dimensional (a'
                             ' transition matrix. Got %d dimensions.' %
                             transitions.ndim)

        # Transition matrix to probabilities vector
        return transitions[:-1,:].ravel(order=order)

    def transition_matrix(self, transitions, matrix_type='left'):
        """
        Create a transition matrix from a vector of probability values.

        Parameters
        ----------
        transitions : array-like
            A vector of probability values for the transition matrix. It can be
            either:
            - a transition vector, if it has length self.nstates^2)
            - a probabilities vector, if it has length
              self.nstates*(self.nstates - 1)
            See the module docstring for more information about the difference.
        matrix_type : {'left', 'right'}, optional
            The method by which the transition matrix is constructed.
            - If 'left', the transition matrix is constructed to be left
              stochastic by converting each set of `self.nstates` values in the
              transition vector into columns of the transition matrix. This
              corresponds to creating the matrix by unstacking the vector into
              columns, and the operation is equivalent to reshaping the vector
              into the matrix using Fortran ordering.
            - If 'right', the transition matrix is constructed to be right
              stochastic by converting each set of `self.nstates` values in the
              transition vector into rows of the transition matrix. This
              corresponds to creating the matrix by unstacking the vector into
              rows, and the operation is equivalent to reshaping the vector
              into the matrix using C ordering.

        Returns
        -------
        transition_matrix : array
            A 2-dimensional transition matrix

        Notes
        -----
        See module docstring for details on the distinction between the terms
        `transitions`, `probability_vector`, `transition_vector`, and
        `transition_matrix`.
        """
        transitions = np.asarray(transitions)

        # Figure out which type of stochastic matrix we have
        if matrix_type == 'left':
            order = 'F'
        elif matrix_type == 'right':
            order = 'C'
        else:
            raise ValueError("Invalid matrix type method. Must be one of"
                             " {'left', 'right'}, corresponding to a left"
                             " stochastic or right stochastic transition"
                             " matrix. Got %s." % matrix_type)

        # If we already have a transition matrix
        if transitions.ndim == 2:
            transition_matrix = transitions
        elif transitions.ndim == 1:
            # Handle a probabilities vector by converting it to a transition
            # vector first
            if transitions.shape[0] == self.nstates*(self.nstates-1):
                transitions = self.transition_vector(transitions, matrix_type)

            if not transitions.shape[0] == self.nstates**2:
                raise ValueError('Invalid vector of probability values. Must'
                                 ' have length %d if it is a transition vector'
                                 ' or length %d if it is a probabilities vector'
                                 ' (see module docstring for details). Got '
                                 ' length %d.' %
                                 (self.nstates**2,
                                  self.nstates*(self.nstates-1),
                                  transitions.shape[0]))
            transition_matrix = transitions.reshape(
                (self.nstates, self.nstates),
                order=order
            )
        else:
            raise ValueError('Invalid input array. Must be 1-dimensional (a'
                             ' probability or transition vector) or '
                             ' 2-dimensional (a transition matrix. Got %d'
                             ' dimensions.' % transitions.ndim)

        # Transition vector to transition matrix
        return transition_matrix

    def transition_vector(self, transitions, matrix_type='left'):
        """
        Create a transition vector

        The transition vector is the vectorized version of the transition
        matrix.

        Parameters
        ----------
        transitions : array-like
            A vector of transition values for the transition vector. It can be
            either:
            - a probability vector, if it has 1-dimension
            - a transition matrix, if it has 2-dimensions
            See the module docstring for more information about the difference.
        matrix_type : {'left', 'right'}, optional
            The method by which the corresponding transition matrix would be
            constructed from the returned transition vector.
            - If 'left', the transition matrix would be constructed to be left
              stochastic.
            - If 'right', the transition matrix would be constructed to be
              right stochastic.
            See MAR.transition_matrix() or the module docstring for details.

        Returns
        -------
        transition_vector : array
            A 1-dimensional transition vector

        Notes
        -----
        See module docstring for details on the distinction between the terms
        `transitions`, `probability_vector`, `transition_vector`, and
        `transition_matrix`.
        """
        transitions = np.asarray(transitions)

        if matrix_type == 'left':
            order = 'F'
        elif matrix_type == 'right':
            order = 'C'
        else:
            raise ValueError("Invalid matrix type method. Must be one of"
                             " {'left', 'right'}, corresponding to a left"
                             " stochastic or right stochastic transition"
                             " matrix. Got %s." % matrix_type)
        
        # If we already have a transition vector
        if transitions.ndim == 1 and transitions.size == self.nstates**2:
            transition_vector = transitions
        # Probabilities vector -> transition vector
        elif transitions.ndim == 1:
            # Get a transition matrix, but missing the last row
            transition_matrix = transitions.reshape(
                (self.nstates-1, self.nstates),
                order=order
            )
            # Calculate and append the last row
            transitions = np.c_[
                transition_matrix.T, 1-transition_matrix.sum(0)
            ].T
            # Vectorize
            transition_vector = transitions.ravel(order=order)
        # Transition matrix -> transition vector
        elif transitions.ndim == 2:
            transition_vector = transitions.ravel(order=order)
        else:
            raise ValueError('Invalid input array. Must be 1-dimensional (a'
                             ' probability vector) or 2-dimensional (a'
                             ' transition matrix. Got %d dimensions.' %
                             transitions.ndim)

        return transition_vector

    def unconditional_probabilities(self, transitions):
        """
        Calculate the unconditional probabilities ("ergodic probabilities")
        from a (left stochastic) transition matrix.

        Parameters
        ----------
        transitions : array-like
            A probability vector, transition vector, or transition matrix.

        Returns
        -------
        unconditional_probabilities : array
            A 1-dimensional, self.nstates length vector of the unconditional
            probabilities of each state.
        """
        transition_matrix = self.transition_matrix(transitions, 'right')
        A = np.r_[
            np.eye(self.nstates) - transition_matrix,
            np.ones((self.nstates, 1)).T
        ]
        return np.linalg.pinv(A)[:,-1]

    def marginalize_probabilities(self, joint_probabilities, nremaining=1):
        """
        Calculate marginal(ized) probabilities from joint probabilities.

        This is used in two ways:
        1. With the output from the filter, to calculate the marginal
           probabilities that the time period t is in each of the possible
           states given time t information
        2. With the output from the smoother, to calculate the marginal
           probabilities that the time period t is in each of the possible
           states given time T information.

        By default it integrates out all but one state.

        Parameters
        ----------
        joint_probabilities : array-like
            A vector of joint probabilities of state sequences ordered in
            increasing lexicographic fashion.
        nremaining : integer, optional
            The dimension the state sequences remaining after the
            marginalization is performed.

        Returns
        -------
        marginalized_probabilities : array-like
            An M^(nremaining) length vector of probabilities; marginal
            probabilities if nremaining is 1, otherwise joint probabilities.
        
        Notes
        -----
        Given joint_probabilities[t] - which is an M^k length vector of the
        joint probabilities of state sequences ordered in increasing
        lexicographic fashion where the 0-th element corresponds to the s_t,
        the state at time t - the marginal probability of (S_t = s_t) is
        achieved by integrating out the other k-1 states,
        (S_{t-1}, ..., S_{t-k}).

        This can be computed for the i-th
        (zero-indexed) state simply by summing the M^(k-1) elements of the
        joint_probabilities[t] vector corresponding to vector locations
        [i*M^(k-1), (i+1)*M^(k-1)-1]. For example, for i=0, this corresponds to
        array locations [0, (i+1)*M^(k-1)-1], inclusive (actually retrieving
        this using slice notation is joint_probabilities[t][0:(i+1)*M^(k-1)]).
        """

        marginalized_probabilities = joint_probabilities.reshape((
            joint_probabilities.shape[0],
            self.nstates**nremaining,
            joint_probabilities.shape[1] / self.nstates**nremaining
        )).sum(-1)
        return marginalized_probabilities

    def smooth(self, joint_probabilities, joint_probabilities_t1, transitions):
        """
        Calculate smoothed probabilities (using all information in the sample),
        using Kim's smoothing algorithm.

        Calculates the marginal probability that the time period t is in each
        of the possible states, given time T information

        Parameters
        ----------
        joint_probabilities : array-like
            The nobs+1 x M^k output from the hamilton filter; the t-th row is
            conditional on time t information.
        joint_probabilities_t1 : array-like, optional
            The nobs+1 x M^(k+1) output from the hamilton filter; the t-th row
            is conditional on time t-1 information.
        transitions : array-like
            A probability vector, transition vector, or transition matrix.

        Returns
        -------
        smoothed_marginal_probabilities : array-like
            An nobs x M length vector of marginal probabilities that the time
            period t is in each of the possible states given time T information.
        """
        transition_vector = self.tvtp_transition_vectors(transitions, 'right')[0]
        transition_matrix = self.transition_matrix(transition_vector, 'right')

        marginal_probabilities = self.marginalize_probabilities(
            joint_probabilities[1:]
        )
        marginal_probabilities_t1 = self.marginalize_probabilities(
            joint_probabilities_t1[1:]
        )

        smoothed_marginal_probabilities = np.zeros((self.nobs, self.nstates))
        smoothed_marginal_probabilities[self.nobs-1] = marginal_probabilities[self.nobs-1]

        for t in range(self.nobs-1, 0, -1): 
            smoothed_marginal_probabilities[t-1] = (
                marginal_probabilities[t-1] * np.dot(
                    transition_matrix.T,
                    (smoothed_marginal_probabilities[t] / 
                     marginal_probabilities_t1[t-1])
                )
            )

        return smoothed_marginal_probabilities

    def filter(self, params):
        """
        Filter the data via the Hamilton Filter

        Parameters
        ----------
        params : array-like
            An array of constrained parameters
        method : {'c', 'python'}, optional
            The method used to run the Hamilton Filter.
            - 'c' runs the filter using an optimized version written in Cython
            - 'python' runs the filter using a slower Python implementation

        Returns
        -------
        marginal_densities : array-like
            The marginal densities of endog at each time t; byproduct of the
            hamilton filter.
        joint_probabilities : array-like
            The nobs+1 x M^k output from the hamilton filter; the t-th row is
            conditional on time t information.
        joint_probabilities_t1 : array-like, optional
            The nobs+1 x M^(k+1) output from the hamilton filter; the t-th row
            is conditional on time t-1 information.

        """
        params = self.transform_params(params)
        params = self.expand_params(params)
        transitions, _, _, _ = self.separate_params(params)
        transition_vectors = self.tvtp_transition_vectors(transitions, 'right')

        (joint_probabilities,
         marginal_conditional_densities) = self.initialize_filter(params)

        args = (self.nobs, self.nstates, self.order, transition_vectors,
                joint_probabilities, marginal_conditional_densities)

        marginal_densities, joint_probabilities, joint_probabilities_t1 = hamilton_filter(*args)

        return (
            marginal_densities, joint_probabilities, joint_probabilities_t1
        )

    def initial_joint_probabilities(self, transitions):
        # The initialized values for the joint probabilities of states are
        # calculated from the unconditional probabilities
        # Note: considering k states
        # The order of the states is lexicographic, increasing
        """
        The initialized values for the joint probabilities of each length k
        sequence of states are calculated from the unconditional probabilities

        At this stage, we are interested in calculating this for each of the
        possible k-permutations (with replacement) of the states (a set with M
        elements), so there are M^k values. Any particular sequence looks like:
        (s_0, s_{-1}, ..., s_{-k+1})
        where the lowecase s denotes a particular realization of one of the
        random state variables S.

        The sequences of states are ordered in increasing lexicographic order:

            (0, 0, ..., 0),
            (0, 0, ..., 1),
            ...
            (M, M, ..., M-1)
            (M, M, ..., M)

        (this is also equivalent to sequences of bits in left-zero-padded
        base-M counting to k^M)

        For each sequence of states, the order is descending in time, so that
        (0, 0, 2) corresponds to (s_{0} = 0, s_{-1} = 0, s_{-2} = 1)
        
        The joint probability of each of the M^k possible state combinations
        (s_0, s_{-1}, ..., s_{-k+1}), is computed using two elements:
        - The unconditional probability of state s_{-k+1}: P[S_{-k+1}=s_{-k+1}]
          This is just the s_{-k+1}-th element of the \pi vector
        - The Markov transition probabilities (there are k of these)
          P[S_{-k+2}=s_{-k+2} | S_{-k+1}=s_{-k+1}]
          P[S_{-k+3}=s_{-k+3} | S_{-k+2}=s_{-k+2}]
          ...
          P[S_{-k+k}s_{-k+k} | S_{-k+(k-1)}=s_{-k+(k-1)}]

        Example (k=2, M=2):

            In general, each joint probability will be of the form:
            P[S_{-2+2}=s_{-2+2} | S_{-2+1}=s_{-2+1}] * P[S_{-2+1}=s_{-2+1}]
            or
            P[S_{0}=s_{0} | S_{-1}=s_{-1}] * P[S_{-1}=s_{-1}]

            The 2^2=4 sequences of states, with their joint probabilities are:
            (0, 0)    => P[S_{0} = 0 | S_{-1} = 0] * P[S_{-1} = 0]
            (0, 1)    => P[S_{0} = 0 | S_{-1} = 1] * P[S_{-1} = 1]
            (1, 0)    => P[S_{0} = 1 | S_{-1} = 0] * P[S_{-1} = 0]
            (1, 1)    => P[S_{0} = 1 | S_{-1} = 1] * P[S_{-1} = 1]

        The result is a M^k length vector giving the resultant joint
        probabilities. It could be reshaped into an (M x M x ... x M) (k times)
        dimensional matrix. In the example above, if you construe each sequence
        of states as the coordinates to a matrix, the corresponding matrix
        would be:

            | (0,0), (0,1) |
            | (1,0), (1,1) |

        Given the vector, it is trivial to reshape into a matrix:

            joint_probabilities.reshape([self.nstates] * self.order)

        Note that the conditional probabilities are elements of the transition
        matrix, with indices corresponding to the *reverse* of the state
        sequence (it is the reverse because in a (i,j) segment of a state
        sequence, we are moving from the jth to the ith state, but because the
        transition matrix is in left stochastic form, the row corresponds to
        the state being moved to. Thus the matrix indices are (j,i) - the
        reverse of the segment (i,j)). It will be convenient to vectorize this
        matrix (i.e. convert it to a vector by "stacking" the rows). This is a
        simple reshaping of the matrix.

            transition_vector = transition_matrix.reshape(
                (1, transition_matrix.size)
            )

        In the k=2, M=2 case, the transition vector is:

            | P[S_{t} = 0 | S_{t-1} = 0] |
            | P[S_{t} = 0 | S_{t-1} = 1] |
            | P[S_{t} = 1 | S_{t-1} = 0] |
            | P[S_{t} = 1 | S_{t-1} = 1] |

        Or represented more compactly:

            | P[0|0] |
            | P[0|1] |
            | P[1|0] |
            | P[1|1] |

        The vector is constructed using vectorized operations rather than
        iteration. As may be seen even in the example above, this requires
        multiplying the following vectors:

            | P[S_{0} = 0 | S_{-1} = 0] |     | P[S_{-1} = 0] |
            | P[S_{0} = 0 | S_{-1} = 1] |     | P[S_{-1} = 1] |
            | P[S_{0} = 1 | S_{-1} = 0] |  *  | P[S_{-1} = 0] |
            | P[S_{0} = 1 | S_{-1} = 1] |     | P[S_{-1} = 1] |

        We can represent the above vector operation more compactly:

            | P[0|0] |   | P[0] |
            | P[0|1] |   | P[1] |
            | P[1|0] | * | P[0] |
            | P[1|1] |   | P[1] |

        Notice that:
        - The last vector is just the \pi vector tiled 2^1 = 2 times
        - The first vector is just the transition vector repeated 2^0 = 1 times
          and tiled 2^0 = 1 times.

        Consider increasing the order to k=3.
        Now the 2^3 = 8 sequences of states are:

            (0,0,0)
            (0,0,1)
            (0,1,0)
            (0,1,1)
            (1,0,0)
            (1,0,1)
            (1,1,0)
            (1,1,1)

        And the vector operation to create the joint probabilities is:

            | P[0|0] |   | P[0|0] |   | P[0] |
            | P[0|0] |   | P[0|1] |   | P[1] |
            | P[0|1] |   | P[1|0] |   | P[0] |
            | P[0|1] |   | P[1|1] |   | P[1] |
            | P[1|0] | * | P[0|0] | * | P[0] |
            | P[1|0] |   | P[0|1] |   | P[1] |
            | P[1|1] |   | P[1|0] |   | P[0] |
            | P[1|1] |   | P[1|1] |   | P[1] |

        Notice that:
        - The total length of the vectors is M^k = 2^3 = 8
        - The last vector is the \pi vector (length M=2) tiled M^2 = 4 times
        - The middle vector is the transition vector (length M^2=4)
          repeated M^0 = 1 time and tiled M^1 = 2 times
        - The first vector is the transition vector (length M^2=4)
          repeated M^1 = 2 times and tiled M^0 = 1 time

        In general, given an order k and number of states M:

        1. The joint probabilities will have M^k elements
        2. The joint probabilities will be calculated as the result of k-1
           multiplication operations on k-1 conditional probability vectors
           constructed from the transition vector and 1 unconditional
           probability vector constructed from the \pi vector.
        3. Tiling and repeating each create a new vector the length of which is
           the length of the vector multiplied by the number of times it is
           tiled or repeated.
        4. The transition vector has length M^2. To achieve a length of M^k, it
           must be tiled or repeated a total of M^(k-2) times. Note that the
           tiling and repeating is multiplicative, in the sense that tiling
           n times and repeating m times multiplies the length by n*m. Thus we
           must have n * m = M^{k-2}. In general, n and m will be exponents of
           M, so M^n * M^m = M^(n+m) = M^{k-2}, or n+m = k-2.
        5. The rightmost conditional probability is constructed by repeating
           the transition vector M^0 = 1 time and tiling it M^(k-2-0) = M^(k-2)
           times.
        6. The next left conditional probability is constructed by repeating
           the transition vector M^1 = times and tiling it M^(k-2-1) = M^(k-3)
           times.
        7. The leftmost conditional probability is constructed by repeating
           the transition vector M^(k-2) times and tiling it
           M^(k-2-(k-2)) = M^0 = 1 time.
        7. There are k-1 conditional probability vectors: The i-th vector is
           constructed by (assume i is zero-indexed):
           - Repeating the transition vector M^(k-2-i) times
           - Tiling the transition vector M^i times
           Note that k-2-i+i = k-2, which is the total number of times required
        8. Note that k >= 1. If k == 1 (so that k-2-i < 0) then the joint
           probabilities are only the marginal unconditional probabilities
           (i.e. there are no conditional probability vectors at all).
        8. The unconditional probabilities vector is constructed from only
           tiling the \pi vector, with no repeating. Since the \pi vector
           has length M and the resultant vector needs to have length M^k, it
           must be tiled M^(k-1) times.
        """
        transition_vectors = self.tvtp_transition_vectors(transitions, 'right')

        # Get the unconditional probabilities of the states, given a set of
        # transition probabilities
        unconditional_probabilities = self.unconditional_probabilities(
            transition_vectors[0]
        )

        if self.order > 1:
            conditional_probabilities = [
                np.tile(
                    transition_vectors[self.order-i-1].repeat(self.nstates**(self.order-2-i)),
                    self.nstates**i
                )[:,None] # need to add the second dimension to concatenate
                for i in range(self.order-1) # k-1 values; 0=first, k-2=last
            ]

            unconditional_probabilities = np.tile(
                unconditional_probabilities, self.nstates**(self.order-1)
            )[:,None]

            joint_probabilities = reduce(np.multiply,
                conditional_probabilities + [unconditional_probabilities]
            ).squeeze()
        else:
            joint_probabilities = unconditional_probabilities

        return joint_probabilities

    def marginal_conditional_densities(self, params, stddevs, means):
        return marginal_conditional_densities(
            self.nobs, self.nstates, self.order,
            params, stddevs, means, self.augmented
        )
        
    def initialize_filter(self, params):
        """
        Calculate the joint probability of S_{t-1} = j and S_{t} = i
        for j=0,1; i=0,1; and t in [1,T], given time t-1 information

        Parameters
        ----------
        params : array-like
            The parameters of the model. In order, they are (with the expected
            lengths of the components in paretheses):
            - transition probabilities (nstates^2 or nstates*(nstates-1))
            - AR parameters (order)
            - stddevs (nstates)
            - means (nstates)
            TODO specify the lengths of stddevs and means in the constructor, so
            that they can be different (e.g. constrain some or all regimes to
            have the same values, constrain certain regimes to have certain
            values, etc.)

        Returns
        -------
        joints : array-like
            An nobs x (nstates^(self.order+1)) array, where
            joints[t][i_0, i_1, ..., i_k] corresponds to the joint probability
            of S_{t} = i_0, S_{t-1} = i_1, ... S_{t-k} = i_k, given time t-1
            information. So, importantly, lower-numbered axes corresponds to
            the more recent time periods (i.e. the zero-index is time t)
            :math:`Pr(S_t, \dots, S_{t-k}|\psi_{t-1})`
        marginals : array
            A (T+1) x nstates array, where marginals[t][i] corresponds to the
            marginal probability that S_{t} = i given time t information.
            :math:`Pr(S_t|\psi_t)`
        params : iterable
            The AR parameters (1 x self.order)
        stddevs : iterable
            A vector of standard deviations, corresponding to each state.
            (1 x self.order+1)
        means : iterable
            A vector of means, corresponding to each state.
            (1 x self.orrder+1)
        """
        transitions, ar_params, stddevs, means = self.separate_params(params)

        # Joint probabilities (of states): (nobs+1) x (M x ... x M), ndim = k+1
        # It's time dimension is nobs+1 because the 0th joint probability is
        # the input (calculated from the unconditional probabilities) for the
        # first iteration of the algorithm, which starts at time t=1
        order = max(self.order, 1)
        joint_probabilities = np.zeros((self.nobs+1, self.nstates**order))
        joint_probabilities[0] = self.initial_joint_probabilities(transitions)

        # Marginal conditional densities
        params = np.c_[
            [1]*self.nstates,
            -ar_params.reshape((self.nstates, self.order))
        ]
        mcds = self.marginal_conditional_densities(
            params, stddevs, means
        )

        return joint_probabilities, mcds
        
    def score(self, params):
        '''
        Gradient of log-likelihood evaluated at params
        '''
        kwds = {}
        kwds.setdefault('centered', True)
        return approx_fprime(params, self.loglike, **kwds).ravel()

    def jac(self, params, **kwds):
        '''
        Jacobian/Gradient of log-likelihood evaluated at params for each
        observation.
        '''
        #kwds.setdefault('epsilon', 1e-4)
        kwds.setdefault('centered', True)
        return approx_fprime(params, self.loglikeobs, **kwds)

    def hessian(self, params):
        '''
        Hessian of log-likelihood evaluated at params
        '''
        from statsmodels.tools.numdiff import approx_hess
        # need options for hess (epsilon)
        return approx_hess(params, self.loglike)

class MARResults(tsbase.TimeSeriesModelResults):
    """
    Class to hold results from fitting a MAR model.

    Parameters
    ----------
    model : ARMA instance
        The fitted model instance
    params : array
        Fitted parameters
    normalized_cov_params : array, optional
        The normalized variance covariance matrix
    scale : float, optional
        Optional argument to scale the variance covariance matrix.
    """

    _cache = {}

    def __init__(self, model, params, normalized_cov_params, scale=1.):
        super(MARResults, self).__init__(model, params,
                                           normalized_cov_params, scale)

    def summary(self, yname=None, title=None, alpha=.05):
        """
        Summarize the MAR Results

        Parameters
        ----------
        yname : string, optional
            Default is `y`
        title : string, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results

        """

        xname = self._make_exog_names()

        model = (
            self.model.__class__.__name__ + '('
            + repr(self.model.order) + ';'
            + ','.join([repr(self.model.ar_order), repr(self.model.delay)])
            + ')'
        )

        try:
            dates = self.data.dates
            sample = [('Sample:', [dates[0].strftime('%m-%d-%Y')])]
            sample += [('', [' - ' + dates[-1].strftime('%m-%d-%Y')])]
        except:
            start = self.model.nobs_initial + 1
            end = repr(self.model.data.orig_endog.shape[0])
            sample = [('Sample:', [repr(start) + ' - ' + end])]

        top_left = [('Dep. Variable:', None),
                    ('Model:', [model]),
                    ('Method:', ['Least Squares']),
                    ('Date:', None),
                    ('Time:', None)
                    ] + sample

        top_right = [('No. Observations:', None),
                     ('Df Residuals:', None),
                     ('Df Model:', None),
                     ('Log-Likelihood:', None),
                     ('AIC:', ["%#8.4g" % self.aic]),
                     ('BIC:', ["%#8.4g" % self.bic])
                     ]

        if title is None:
            title = self.model.__class__.__name__ + ' ' + "Regression Results"

        # Create summary table instance
        from statsmodels.iolib.summary import Summary, summary_params, forg
        from statsmodels.iolib.table import SimpleTable
        from statsmodels.iolib.tableformatting import fmt_params
        smry = Summary()
        warnings = []

        # Add model information
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                             yname=yname, xname=xname, title=title)

        # Add hyperparameters summary table
        if (1 - alpha) not in self.model.threshold_crits:
            warnings.append("Critical value for threshold estimates is"
                            " unavailable at the %d%% level. Using 95%%"
                            " instead." % ((1-alpha)*100))
            alpha = 0.05
        alp = str((1-alpha)*100)+'%'
        conf_int = self.conf_int_thresholds(alpha)

        # (see summary_params())
        confint = [
            "%s %s" % tuple(map(forg, conf_int[i]))
            for i in range(len(conf_int))
        ]
        confint.insert(0, '')
        len_ci = map(len, confint)
        max_ci = max(len_ci)
        min_ci = min(len_ci)

        if min_ci < max_ci:
            confint = [ci.center(max_ci) for ci in confint]

        thresholds = list(self.model.thresholds)
        param_header = ['coef', '[' + alp + ' Conf. Int.]']
        param_stubs = ['Delay'] + ['\gamma_%d' % (threshold_idx + 1)
                                   for threshold_idx in range(len(thresholds))]
        param_data = zip([self.model.delay] + map(forg, thresholds), confint)

        parameter_table = SimpleTable(param_data,
                                      param_header,
                                      param_stubs,
                                      title=None,
                                      txt_fmt=fmt_params)
        smry.tables.append(parameter_table)

        # Add parameter tables for each regime
        results = np.c_[
            self.params, self.bse, self.tvalues, self.pvalues,
        ].T
        conf = self.conf_int(alpha)
        k = self.model.ar_order + self.model.k_trend
        regime_desc = self._make_regime_descriptions()
        max_len = max(map(len, regime_desc))
        for regime in range(1, self.model.order + 1):
            res = (self,)
            res += tuple(results[:, k*(regime - 1):k*regime])
            res += (conf[k*(regime - 1):k*regime],)
            table = summary_params(res, yname=yname,
                                   xname=xname[k*regime:k*(regime+1)],
                                   alpha=alpha, use_t=True)

            # Add regime descriptives, if multiple regimes
            if self.model.order > 1:
                # Replace the header row
                header = ["\n" + str(cell) for cell in table.pop(0)]
                title = ("Regime %d" % regime).center(max_len)
                desc = regime_desc[regime - 1].center(max_len)
                header[0] = "%s \n %s" % (title, desc)
                table.insert_header_row(0, header)
                # Add diagnostic information
                nobs = [
                    'nobs_%d' % regime, self.model.nobs_regimes[regime - 1],
                    '', '', '', ''
                ]
                table.insert(len(table), nobs, 'header')

            smry.tables.append(table)

        # Add notes / warnings, added to text format only
        warnings.append("Reported parameter standard errors are White's (1980)"
                        " heteroskedasticity robust standard errors.")
        warnings.append("Threshold confidence intervals calculated as"
                        " Hansen's (1997) conservative (non-disjoint)"
                        " intervals")

        if self.model.exog.shape[0] < self.model.exog.shape[1]:
            wstr = "The input rank is higher than the number of observations."
            warnings.append(wstr)

        if warnings:
            etext = [
                "[{0}] {1}".format(i + 1, text)
                for i, text in enumerate(warnings)
            ]
            etext.insert(0, "Notes / Warnings:")
            smry.add_extra_txt(etext)

        return smry


class MARResultsWrapper(tsbase.TimeSeriesResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(
        tsbase.TimeSeriesResultsWrapper._wrap_methods,
        _methods
    )
wrap.populate_wrapper(MARResultsWrapper, MARResults)