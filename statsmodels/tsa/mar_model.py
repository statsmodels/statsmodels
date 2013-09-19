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

The AR, standard deviation, and mean parameters may be allowed to differ
across states, or may be restricted to be the same.

If the transition probabilities are allowed to change over time, it is called a
Time-Varying Probabilites (TVP) Markov-Switching Model.



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
from mar_c import hamilton_filter
import resource

class MAR(base.LikelihoodModel):
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
                 dates=None, freq=None, missing='none'):

        # "Immutable" properties
        self.nobs_initial = order
        self.nobs = endog.shape[0] - order
        self.order = order
        self.nstates = nstates
        self.nparams = (
            self.nstates*(self.nstates - 1) + # Probabilities
            self.order +                      # AR parameters
            self.nstates +                    # Standard deviatons
            self.nstates                      # Means
        )

        # Make a copy of original datasets
        orig_endog = endog
        orig_exog = lagmat(orig_endog, order)

        # Create datasets / complete initialization
        endog = orig_endog[self.nobs_initial:]
        exog = orig_exog[self.nobs_initial:]
        super(MAR, self).__init__(endog, exog,
                                  hasconst=0, missing=missing)

        # Overwrite originals
        self.data.orig_endog = orig_endog
        self.data.orig_exog = orig_exog

        # Cache
        self.augmented = np.c_[endog, exog]
        self.cache_transition = {}
        self.cache_marginal_conditional_densities = {}

    def combine_params(self, probabilities, ar_params, stddevs, means):
        """
        Combines the component parameters into a single array.

        Parameters
        ----------
        probabilities : array-like
            A probability vector
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
        return np.r_[probabilities, ar_params, stddevs, means]

    def separate_params(self, params):
        """
        Separates a single array of parameters into the component pieces.

        Parameters
        ----------
        params : array-like
            An array of parameters

        Returns
        -------
        probabilities : array-like
            A probability vector
        ar_params : array-like
            The AR parameters
        stddevs : array-like
            The standard deviations for each state
        means : array-like
            The means for each state
        """
        params = np.asarray(params)

        # Anything called "params" has a probabilities vector - meaning not
        # the full transition vector of transition probabilities.
        nprobs = self.nstates*(self.nstates - 1)

        # Make sure we have the right number of parameters
        if not params.shape[0] == self.nparams:
            raise ValueError('Invalid number of parameters. Expected %d,'
                             ' got %d.' % (self.nparams, params.shape[0]))

        # Separate the parameters
        transitions = params[:nprobs]
        ar_params = params[nprobs:self.order+nprobs]
        stddevs = params[self.order+nprobs:self.order+nprobs+self.nstates]
        means = params[-self.nstates:]

        return transitions, ar_params, stddevs, means

    def transform(self, params, method='logit'):
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

        #  transitions: transform to always be in (0, 1)
        if method == 'logit':
             transitions = np.exp( transitions) / (1 + np.exp( transitions))
        elif method == 'abs':
             transitions = np.abs( transitions) / (1 + np.abs( transitions))
        else:
            raise VaueError('Invalid transformation method')

        # Standard deviations: transform to always be positive
        stddevs = np.abs(stddevs)

        return np.r_[transitions, ar_params, stddevs, means]

    def untransform(self, params, method='logit'):
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

        # Probabilities: untransform to always be in (-Inf, Inf)
        if method == 'logit':
            transitions = np.log(transitions / (1 - transitions))
        elif method == 'abs':
            transitions = transitions / (1 - transitions)
        else:
            raise VaueError('Invalid transformation method')
        # No other untransformations

        return np.r_[transitions, ar_params, stddevs, means]

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
        transitions = (
            np.exp(transitions) / (1 + np.exp(transitions))**2
        )

        return np.diag(np.r_[transitions, [1]*(self.nparams-len(transitions))])

    def loglike(self, params):
        """
        Calculate the log likelihood.

        Parameters
        ----------
        params : array-like
            An array of unonstrained parameters

        Returns
        -------
        loglike : float
            The log likelihood of the model evaluated at the given parameters.

        Notes
        -----
        Uses unconstrained parameters because it is meant to be called via
        the optimization routine, which uses unconstrained parameters.
        """
        params = self.transform(params)

        transitions, _, _, _ = self.separate_params(params)
        transition_vector = self.transition_vector(transitions, 'right')

        (joint_probabilities,
         marginal_conditional_densities) = self.initialize_filter(params)

        marginal_densities, _, _ = hamilton_filter(
            self.nobs, self.nstates, self.order,
            transition_vector, joint_probabilities,
            marginal_conditional_densities
        )

        return np.sum(np.log(marginal_densities))

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
        transition_matrix = self.transition_matrix(transitions)
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
        transition_matrix = self.transition_matrix(transitions)

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

    def filter(self, params, method='c'):
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
        #params = self.transform(params)

        transitions, _, _, _ = self.separate_params(params)
        transition_vector = self.transition_vector(transitions, 'right')

        (joint_probabilities,
         marginal_conditional_densities) = self.initialize_filter(params)

        if method == 'c':
            fn = hamilton_filter
            args = (self.nobs, self.nstates, self.order, transition_vector,
                    joint_probabilities, marginal_conditional_densities)
        elif method == 'python':
            fn = self.hamilton_filter
            args = (transition_vector, joint_probabilities,
                    marginal_conditional_densities)
        else:
            raise ValueError('Invalid filter method')

        marginal_densities, joint_probabilities, joint_probabilities_t1 = fn(*args)

        return (
            marginal_densities, joint_probabilities, joint_probabilities_t1
        )

    def initial_joint_probabilities(self, transitions,
                                    unconditional_probabilities):
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
        # Make sure we have our transitions in vector form
        transition_vector = self.transition_vector(transitions, 'right')

        conditional_probabilities = [
            np.tile(
                transition_vector.repeat(self.nstates**(self.order-2-i)),
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

        return joint_probabilities

    def marginal_conditional_densities(self, ar_params, sds, means):
        # Calculate the marginal conditional densities
        ar_params = np.r_[1, -ar_params]
        # The order of the states is lexicographic, increasing
        x = self.augmented.dot(ar_params)
        """
        The location for each state sequence is the vector of mean values
        corresponding to each state, dotted by the [1, -ar_params].

        The [1, -ar_params] vector is a set of weights to apply to the means of
        the k+1 length state sequence (S_{0}, ..., S_{-k}).

        The resultant vector is of length M^(k+1) (i.e. one row for each possible
        sequence of k+1 states). Proceed by first creating a matrix of means
        where each row has a set of mean values, corresponds to a state
        sequence lexicographically increasing ordered (as above in construction
        of the joint probabilities). This matrix is M^(k+1) x (k+1).

        The sequence of states looks like:

            (0, 0, ..., 0),
            (0, 0, ..., 1),
            ...
            (M, M, ..., M-1)
            (M, M, ..., M)

        So that the means matrix looks like:

            | \mu_{0}  \mu_{0}  ...   \mu_{0}  |
            | \mu_{0}  \mu_{0}  ...   \mu_{1}  |
            ...
            | \mu_{M}  \mu_{M}  ...  \mu_{M-1} |
            | \mu_{M}  \mu_{M}  ...   \mu_{M}  |

        It is constructed from the vector of means in each state, which has
        length M. To achieve a length of M^(k+1), it must be tiled or repeated
        a total of M^k times. The i-th column (from the left, assume i is
        zero-indexed) is constructed by:
        - Repeating the means vector M^(k-i) times
        - Tiling the means vector M^i times

        For example:
        The leftmost column (i=0) is tiled M^0 = 1 time and repeated M^k times
        The rightmost column (i=k) is tiled M^k times and repeated M^0 = 1 time
        """
        loc = np.dot(
            np.concatenate([
                np.tile(
                    means.repeat(self.nstates**(self.order-i)),
                    self.nstates**i
                )[:, None]
                for i in range(self.order+1) # k+1 values; 0=first, k=last
            ], 1),
            ar_params
        )
        """
        The scale for each k+1 length state sequence (S_t, ..., S_{t-k}),
        corresponding to the standard deviation under state S_t.

        Recall that the state sequences look like:

            (0, 0, ..., 0),
            (0, 0, ..., 1),
            ...
            (M, M, ..., M-1)
            (M, M, ..., M)

        sds is a M length vector which needs to be expanded to length M^(k+1)
        in such a way that it corresponds with the first state of each state
        sequence. Thus we repeat it M^k times (M * M^k = M^(k+1) as desired).
        """
        scale = np.asarray(sds).repeat(self.nstates**(self.order))
        """
        Compute the marginal conditional densities.

        x is a self.nobs length vector
        loc and scale are M^(k+1) length vectors

        The resultant vector should be a self.nobs x M^(k+1) matrix, such that
        marginal_conditional_densities[t] is the set of marginal conditional
        densities corresponding to the data at time t, for each state sequence
        S_t, ..., S_{t-k}.

        Thus to allow broadcasting to work, we need to perform the operation on
        x repeated M^(k+1) times, and on loc and scale tiled self.nobs times.

        Since we need to convert scale to var anyway, the tiling is done then.
        For x and loc the repeating or tiling is done inline, since we don't
        need to save those objects.
        """
        var = np.tile(scale**2, self.nobs)
        marginal_conditional_densities = (
            (1 / np.sqrt(2*np.pi*var)) * np.exp(
                -( (np.repeat(x, self.nstates**(self.order+1)) - np.tile(loc, self.nobs))**2 ) / (2*var)
            )
        ).reshape((self.nobs, self.nstates**(self.order+1)))

        return marginal_conditional_densities
        
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
            - sds (nstates)
            - means (nstates)
            TODO specify the lengths of sds and means in the constructor, so
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
        sds : iterable
            A vector of standard deviations, corresponding to each state.
            (1 x self.order+1)
        means : iterable
            A vector of means, corresponding to each state.
            (1 x self.orrder+1)
        """
        transitions, ar_params, stddevs, means = self.separate_params(params)

        # Get the unconditional probabilities of the states, given a set of 
        # transition probabilities
        unconditional_probabilities = self.unconditional_probabilities(transitions)

        # Joint probabilities (of states): (nobs+1) x (M x ... x M), ndim = k+1
        # It's time dimension is nobs+1 because the 0th joint probability is
        # the input (calculated from the unconditional probabilities) for the
        # first iteration of the algorithm, which starts at time t=1
        joint_probabilities = np.zeros((self.nobs+1, self.nstates**self.order))
        joint_probabilities[0] = self.initial_joint_probabilities(
            transitions, unconditional_probabilities
        )

        # Marginal conditional densities
        marginal_conditional_densities = self.marginal_conditional_densities(
            ar_params, stddevs, means
        )

        return joint_probabilities, marginal_conditional_densities
        
    def hamilton_filter(self, transitions, joint_probabilities,
                        marginal_conditional_densities):
        transition_vector = self.transition_vector(transitions)

        # Marginal density (not conditional on states) of y_t: t x 1
        marginal_densities = np.zeros((self.nobs, 1))
        joint_probabilities_t1 = np.zeros((self.nobs, self.nstates**(self.order+1)))

        # Note: Need to extend the states grid; now considering k+1 states
        for t in range(1, self.nobs+1):
            # Step 1:
            # Iterate over all combinations of states for periods
            # t, t-1, ..., t-k to calculate joint probabilities and marginal
            # densities (conditional on states), given time t-1 information
            """
            Compute the joint probabilities over the k+1 length sequence of
            states S_{t}, ..., S_{t-k}, given time t-1 information.

            The given joint probabilities (from the joint_probabilities vector)
            are over the k length sequence of states
            S_{t-1}, ..., S_{t-k}, given time t-1 information.

            To calculate the joint probability over the new state S_{t}, use
            the previous joint probabilities along with the conditional
            probabilities in the transition vector (due to the Markov
            property).

            Consider M=2, k=2. Then the transition vector is length 2^2 = 4,
            and the joint_probabilities[t-1] vector is length 2^2 = 4.

            This operation moves from length 2^2=4:

                (0, 0)
                (0, 1)
                (1, 0)
                (1, 1)

            to to length 2^3=8:

                (0,0,0)
                (0,0,1)

                (0,1,0)
                (0,1,1)


                (1,0,0)
                (1,0,1)

                (1,1,0)
                (1,1,1)

            This requires repeating the transition vector M^(2-1) = M = 2 times
            and tiling the joint probabilities vector M = 2 times.

            Or, consider M=3, k=2. Then the transition vector is
            length 3^2 = 9, and the joint_probabilities[t-1] vector is
            length 3^2 = 9.

            This operation moves from length 3^2=9:

                (0, 0)
                (0, 1)
                (0, 2)
                (1, 0)
                (1, 1)
                (1, 2)
                (2, 0)
                (2, 1)
                (2, 2)

            to to length 3^3=27:

                idx = i*9 + j*3 + k

                (0, 0, 0) i=0, j=0, k=0  => 0,0   => transition[0]
                (0, 0, 1)           k=1  => 1,1
                (0, 0, 2)           k=2  => 2,2

                (0, 1, 0)      j=1, k=0  => 3,3   => transition[1]
                (0, 1, 1)           k=1  => 4,4
                (0, 1, 2)           k=2  => 5,5

                (0, 2, 0)      j=2, k=0  => 6,6   => transition[2]
                (0, 2, 1)           k=1  => 7,7
                (0, 2, 2)           k=2  => 8,8


                (1, 0, 0) i=1, j=0, k=0  => 9,0   => transition[3]
                (1, 0, 1)
                (1, 0, 2)

                (1, 1, 0)
                (1, 1, 1)
                (1, 1, 2)

                (1, 2, 0)
                (1, 2, 1)
                (1, 2, 2)


                (2, 0, 0)
                (2, 0, 1)
                (2, 0, 2)

                (2, 1, 0)
                (2, 1, 1)
                (2, 1, 2)

                (2, 2, 0)
                (2, 2, 1)
                (2, 2, 2)

            This requires repeating the transition vector M^(2-1) = M = 3 times
            and tiling the joint probabilities vector M = 3 times.

            Or, consider M=2, k=4. Then the transition vector is
            length 2^2 = 4, and the joint_probabilities[t-1] vector is
            length 2^4 = 16.

            In the M=2, k=4 case, the transition vector is:

                | P[S_{t} = 0 | S_{t-1} = 0] |
                | P[S_{t} = 0 | S_{t-1} = 1] |
                | P[S_{t} = 1 | S_{t-1} = 0] |
                | P[S_{t} = 1 | S_{t-1} = 1] |

            This operation moves from length 2^4=16:

                (0,0,0,0)
                (0,0,0,1)
                (0,0,1,0)
                (0,0,1,1)
                (0,1,0,0)
                (0,1,0,1)
                (0,1,1,0)
                (0,1,1,1)
                (1,0,0,0)
                (1,0,0,1)
                (1,0,1,0)
                (1,0,1,1)
                (1,1,0,0)
                (1,1,0,1)
                (1,1,1,0)
                (1,1,1,1)

            to to length 2^5=32:

                (0,0,0,0,0)         => transition_vector[0]
                (0,0,0,0,1)         => transition_vector[0]
                (0,0,0,1,0)         => transition_vector[0]
                (0,0,0,1,1)         => transition_vector[0]
                (0,0,1,0,0)         => transition_vector[0]
                (0,0,1,0,1)         => transition_vector[0]
                (0,0,1,1,0)         => transition_vector[0]
                (0,0,1,1,1)         => transition_vector[0]

                (0,1,0,0,0)         => transition_vector[1]
                (0,1,0,0,1)         => transition_vector[1]
                (0,1,0,1,0)         => transition_vector[1]
                (0,1,0,1,1)         => transition_vector[1]
                (0,1,1,0,0)         => transition_vector[1]
                (0,1,1,0,1)         => transition_vector[1]
                (0,1,1,1,0)         => transition_vector[1]
                (0,1,1,1,1)         => transition_vector[1]


                (1,0,0,0,0)         => transition_vector[2]
                (1,0,0,0,1)         => transition_vector[2]
                (1,0,0,1,0)         => transition_vector[2]
                (1,0,0,1,1)         => transition_vector[2]
                (1,0,1,0,0)         => transition_vector[2]
                (1,0,1,0,1)         => transition_vector[2]
                (1,0,1,1,0)         => transition_vector[2]
                (1,0,1,1,1)         => transition_vector[2]

                (1,1,0,0,0)         => transition_vector[3]
                (1,1,0,0,1)         => transition_vector[3]
                (1,1,0,1,0)         => transition_vector[3]
                (1,1,0,1,1)         => transition_vector[3]
                (1,1,1,0,0)         => transition_vector[3]
                (1,1,1,0,1)         => transition_vector[3]
                (1,1,1,1,0)         => transition_vector[3]
                (1,1,1,1,1)         => transition_vector[3]

            This requires repeating the transition vector M^(2-1) = M = 2 times
            and tiling the joint probabilities vector M = 2 times.

            This just expands the state space, and is constructed by tiling the
            entire M^k length joint probabilities vector M times (once for each
            possible) value of M) - for a new length of M^k * M^1 = M^(k+1) -
            and adding a new leftmost vector which repeats the M^2 length
            transition matrix M^(k-1) times - for a new length of
            M^2 * M^(k-1) = M^(k+1).
            """
            _joint_probabilities_t1 = (
                np.repeat(transition_vector, self.nstates**(self.order-1)) * 
                np.tile(joint_probabilities[t-1], self.nstates)
            )
            joint_probabilities_t1[t-1] = _joint_probabilities_t1

            # Step 2 (ctnd):
            # Compute the joint densities (conditional on time t-1 information)
            # PR_VL
            joint_densities = np.multiply(
                marginal_conditional_densities[t-1], _joint_probabilities_t1
            )

            # Step 3:
            # Calculate the marginal density (conditional on time t-1 information)
            # PR_VAL
            marginal_densities[t-1] = np.sum(joint_densities)

            # Step 4:
            # Compute the joint probabilities based on time t information
            # :math:`Pr(S_t = s_t, S_{t-1} = s_{t-1}, \dots, S_{t-k} = s_{t-k} | \phi_{t-1})`
            """
            Compute the joint probabilities over the k+1 length sequence of
            states S_{t}, ..., S_{t-k}, given time t information.
            """
            joint_probabilities_t = joint_densities / marginal_densities[t-1]
            
            # Step 5:
            # Compute the output joint probability based on time t information
            # by integrating out the last state
            """
            Compute the joint probabilities over the k length sequence of
            states S_{t}, ..., S_{t-k+1}, given time t information.

            The given joint probabilities (from the joint_probabilities_t
            vector) are over the k+1 length sequence of states
            S_{t}, ..., S_{t-k}, given time t information. This integrates out
            the last state S_{t-k}.

            This moves from a M^(k+1) length vector to an M^k length vector.

            Consider the case of M=2, k=2. This operation moves from a vector
            of state sequences that looks like this:

                (0,0,0)
                (0,0,1)
                (0,1,0)
                (0,1,1)
                (1,0,0)
                (1,0,1)
                (1,1,0)
                (1,1,1)

            to one that looks like this:

                (0, 0)
                (0, 1)
                (1, 0)
                (1, 1)

            Thus integrating out the last state amounts to creating a new
            vector where the i-th (zero-indexed) element is the sum of the
            [i*M, ((i+1)*M)-1] elements of the existing vector.

            For example, the first (i=0) element of the resultant vector in the
            M=2, k=2 case is the sum of elements [0,1] = [0*2, ((0+1)*2)-1)] in
            the original vector. The second (i=1) element of the resultant
            vector is the the sum of elements [2,3] = [1*2, ((1+1)*2)-1]
            """
            joint_probabilities[t] = joint_probabilities_t.reshape(
                (self.nstates**self.order, self.nstates)
            ).sum(1)

        return marginal_densities, joint_probabilities, joint_probabilities_t1

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