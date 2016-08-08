"""
Markov Switching State Space Model

Author: Valery Likhosherstov
License: Simplified-BSD
"""
import numpy as np
from statsmodels.tsa.statespace.api import MLEModel, MLEResults
from .tools import MarkovSwitchingParams
from .kim_smoother import KimSmoother


class SwitchingMLEModel(MLEModel):
    r"""
    Markov switching state space model for maximum likelihood estimation

    Parameters
    ----------
    k_regimes : int
        The number of switching regimes.
    endog : array_like
        The observed time-series process :math:`y`
    k_states : int
        The dimension of the unobserved state process.
    param_k_regimes : int, optional
        Order of regime transition matrix in parameters vector. If not
        specified, which is usual, `k_regimes` is used instead.
        Regime transition matrix in parameters vector can be different from
        transition matrix, used internally by switching state space
        representation. For example, Markov switching :math:`AR(p)` model of
        :math:`k` regimes is internally evaluated using `k^{p + 1}` regimes.
        See `MarkovAutoregression` class for details.
    **kwargs
        This additional arguments are used in superclass intializer. See
        `MLEModel` documentation for details.

    Attributes
    ----------
    ssm : KimSmoother
        Underlying Markov switching state space representation.

    Notes
    -----
    This class wraps the Markov switching state space model with Kim filtering
    to add in functionality for maximum likelihood estimation. In particular,
    it adds the concept of updating the state space representation based on a
    defined set of parameters, through the `update` method, and it adds a `fit`
    method which uses a numerical optimizer to select the parameters that
    maximize the likelihood of the model.
    The `start_params` `update` method must be overridden in the
    child class (and the `transform_*` and `untransform_*` methods, if needed).

    This class also has a feature of using non-switching model to evaluate
    starting parameters of optimization. This feature is based on hypothesis, that
    parameters of non-switching model provide a good starting parameters for
    switching model fit. To non-switching initalization feature,
    `get_nonswitching_model` and `update_params` methods must be overridden.

    See Also
    --------
    SwitchingMLEResults
    statsmodels.tsa.statespace.regime_switching.switching_representation. \
    SwitchingRepresentation
    statsmodels.tsa.statespace.regime_switching.kim_filter.KimFilter
    statsmodels.tsa.statespace.regime_switching.kim_smoother.KimSmoother
    statsmodels.tsa.statespace.mlemodel.MLEModel
    statsmodels.tsa.statespace.regime_switching.ms_ar.MarkovAutoregression
    """

    def __init__(self, k_regimes, endog, k_states, param_k_regimes=None,
            **kwargs):

        self.k_regimes = k_regimes

        # If `param_k_regimes` is not specified, use `k_regimes`
        if param_k_regimes is None:
            self.param_k_regimes = k_regimes
        else:
            self.param_k_regimes = param_k_regimes

        # A convenient organizing of parameters
        self.parameters = MarkovSwitchingParams(self.param_k_regimes)

        # Parameters vector saves the transition matrix without last row, which
        # can be easily recovered due to left stochastic feature of the matrix
        self.parameters['regime_transition'] = [False] * \
                self.param_k_regimes * (self.param_k_regimes - 1)

        # Create param names with regime transition prob names
        self._param_names = ['Pr[{0}->{1}]'.format(j, i) for i in range(
                self.param_k_regimes - 1) for j in range(self.param_k_regimes)]

        # Superclass initialization
        super(SwitchingMLEModel, self).__init__(endog, k_states, **kwargs)

    def initialize_statespace(self, **kwargs):
        """
        Initialize the state space representation

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the state space class
            constructor.

        Notes
        -----
        This method is overridden to change base class `ssm` attribute type
        from `KalmanSmoother` to `KimSmoother`.
        """

        # Match the shape, required by `KimSmoother`
        endog = self.endog.T

        # Instantiate the state space objest
        self.ssm = KimSmoother(endog.shape[0], self.k_states, self.k_regimes,
                **kwargs)
        # Bind the data to the model
        self.ssm.bind(endog)

        # Save endog vector length
        self.k_endog = self.ssm.k_endog

    def get_nonswitching_model(self):
        """
        Get a non-switching model, corresponding to this switching model.
        To override, if non-switching initialization is used.

        Returns
        -------
        subclass of MLEModel instance

        Notes
        -----
        See existing models code, notebooks and tests for example of
        `get_nonswitching_model` implementation.
        """

        raise NotImplementedError

    def update_params(self, params, nonswitching_params):
        """
        Update constrained parameters of the model, using parameters of
        non-switching model.
        To override, if non-switching initialization is used.

        Parameters
        ----------
        params : array_like
            Parameters vector to update.
        nonswitching_params : array_like
            Parameters vector of the non-switching analog, used to update
            `params`.

        Returns
        -------
        result_params : array_like
            Updated parameters.

        Notes
        -----
        Parameters vector for Markov switching model can be logically splitted
        into two parts:
        - regime transition parameters, which define regime transtition matrix.
        - model parameters, which describe common and different for every
            regime set of parameters, which are used to recover state space
            representations of every regime.
        This method is supposed to update starting parameters using
        non-switching fit, that is, regimes common parameters values are set
        to their non-switching analog, and regimes switching parameters are all
        set to one non-switching value. But don't forget to add some random or
        determinate noise to switching parameters to break the symmetry.
        See existing models code, notebooks and tests for example of
        `update_params` implementation.
        """

        return np.array(params, ndmin=1)

    def transform_regime_transition(self, unconstrained):
        """
        Transform regime transition parameters of `unconstrained` vector.

        Parameters
        ----------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer, to be
            transformed.

        Returns
        -------
        parameters : array_like
            Input vector with constrained regime transition parameters.

        Notes
        -----
        Parameters vector for Markov switching model can be logically splitted
        into two parts:
        - regime transition parameters, which define regime transtition matrix.
        - model parameters, which describe common and different for every
            regime set of parameters, which are used to recover state space
            representations of every regime.
        This method transforms regime transition parameters using
        logistic transformation, leaving model parameters unchanged. Usually no
        need to override this method.
        """

        param_k_regimes = self.param_k_regimes

        constrained = np.array(unconstrained)

        # Unconstrained regime transition matrix without last row
        unconstrained_transition = \
                unconstrained[self.parameters['regime_transition']].reshape(
                (param_k_regimes - 1, param_k_regimes))

        # Logistic transformation
        constrained_transition = np.exp(unconstrained_transition)
        constrained_transition /= \
                (1 + constrained_transition.sum(axis=0)).reshape((1, -1))

        # Copying result to parameters vector
        constrained[self.parameters['regime_transition']] = \
                constrained_transition.ravel()

        return constrained

    def untransform_regime_transition(self, constrained):
        """
        Untransform regime transition parameters of `constrained` vector.

        Parameters
        ----------
        constrained : array_like
            Array of constrained parameters used in likelihood evalution, to be
            transformed.

        Returns
        -------
        parameters : array_like
            Input vector with unconstrained regime transition parameters.

        Notes
        -----
        Parameters vector for Markov switching model can be logically splitted
        into two parts:
        - regime transition parameters, which define regime transtition matrix.
        - model parameters, which describe common and different for every
            regime set of parameters, which are used to recover state space
            representations of every regime.
        This method untransforms regime transition parameters using
        logistic transformation, leaving model parameters unchanged. Usually no
        need to override this method.
        """

        param_k_regimes = self.param_k_regimes

        unconstrained = np.array(constrained)

        eps = 1e-8

        # Constrained regime transition matrix without last row
        constrained_transition = \
                constrained[self.parameters['regime_transition']].reshape(
                (param_k_regimes - 1, param_k_regimes))

        unconstrained_transition = np.array(constrained_transition)

        # Setting zero probabilities to a small value to avoid dealing with
        # -np.inf after switching to logarithms
        unconstrained_transition[unconstrained_transition == 0] = eps

        # Logistic transformation
        unconstrained_transition /= \
                (1 - unconstrained_transition.sum(axis=0)).reshape(1, -1)
        unconstrained_transition = np.log(unconstrained_transition)

        # Copying result to parameters vector
        unconstrained[self.parameters['regime_transition']] = \
                unconstrained_transition.ravel()

        return unconstrained

    def transform_model_params(self, unconstrained):
        """
        Transform model parameters of `unconstrained` vector.
        To override in subclasses.

        Parameters
        ----------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer, to be
            transformed.

        Returns
        -------
        parameters : array_like
            Input vector with constrained model parameters.

        Notes
        -----
        Parameters vector for Markov switching model can be logically splitted
        into two parts:
        - regime transition parameters, which define regime transtition matrix.
        - model parameters, which describe common and different for every
            regime set of parameters, which are used to recover state space
            representations of every regime.
        This method transforms model parameters leaving regime transition
        parameters unchanged.
        """

        return np.array(unconstrained)

    def untransform_model_params(self, constrained):
        """
        Untransform model parameters of `constrained` vector.
        To override in subclasses.

        Parameters
        ----------
        constrained : array_like
            Array of constrained parameters used in likelihood evalution, to
            be transformed.

        Returns
        -------
        parameters : array_like
            Input vector with unconstrained model parameters.

        Notes
        -----
        Parameters vector for Markov switching model can be logically splitted
        into two parts:
        - regime transition parameters, which define regime transtition matrix.
        - model parameters, which describe common and different for every
            regime set of parameters, which are used to recover state space
            representations of every regime.
        This method untransforms model parameters leaving regime transition
        parameters unchanged.
        """

        return np.array(constrained)

    def transform_params(self, unconstrained):
        """
        Transform unconstrained parameters used by the optimizer to constrained
        parameters used in likelihood evaluation

        Parameters
        ----------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer, to be
            transformed.

        Returns
        -------
        constrained : array_like
            Array of constrained parameters which may be used in likelihood
            evalation.

        Notes
        -----
        Unlike in `MLEModel` class, no need to override this method. Instead
        consider overriding `transform_model_params`.
        """

        return self.transform_model_params(
                self.transform_regime_transition(unconstrained))

    def untransform_params(self, constrained):
        """
        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer

        Parameters
        ----------
        constrained : array_like
            Array of constrained parameters used in likelihood evalution, to be
            transformed.

        Returns
        -------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer.

        Notes
        -----
        Unlike in `MLEModel` class, no need to override this method. Instead
        consider overriding `untransform_model_params`.
        """

        return self.untransform_model_params(
                self.untransform_regime_transition(constrained))

    def _permute_regimes(self, params, permutation):

        # This is a useful method, permuting regime switching parameters in
        # `params` vector.
        # It is used in `normalize_regimes`.

        param_k_regimes = self.param_k_regimes
        dtype = self.ssm.dtype

        # Get regime transition matrix, encoded in `params` vector
        regime_transition = self._get_param_regime_transition(params)

        # Permute regimes in matrix
        new_regime_transition = np.zeros((param_k_regimes, param_k_regimes),
                dtype=dtype)
        for i in range(param_k_regimes):
            for j in range(param_k_regimes):
                new_regime_transition[i, j] = \
                        regime_transition[permutation[i],
                        permutation[j]]

        # Instantiating new parameters vector
        new_params = np.zeros((self.parameters.k_params,), dtype=dtype)

        # Copying permuted regime transition matrix into new vector
        self._set_param_regime_transition(new_params, new_regime_transition)

        # Permuting model parameters
        for i in range(param_k_regimes):
            new_params[self.parameters[i]] = \
                    params[self.parameters[permutation[i]]]

        return new_params

    def get_normal_regimes_permutation(self, params):
        """
        Return normal permutation of regimes.
        To override in subclass, when required.

        Parameters
        ----------
        params : array_like
            Constrained parameters of the model.

        Returns
        -------
        permutation : array_like
            Array of size `param_k_regimes`, containing permutation of
            indices `[0, 1, ..., param_k_regimes - 1]`.

        Notes
        -----
        Several unique parameter vectors can represent the only model
        configuration, because of different order of regime enumeration. To
        compare two configurations (e.g. for testing), we need to
        normalize both parameter vectors first, that is to use permutation of
        regimes, which is determined by model configuration only.
        See `MarkovAutoregression` for `get_normal_regimes_permutation`
        example, where sorting of switching parameters is used.
        """

        param_k_regimes = self.param_k_regimes

        # Identity permutation by default
        return list(range(param_k_regimes))

    def normalize_params(self, params, transformed=True):
        """
        Normalization of parameters vector.

        Parameters
        ----------
        params : array_like
            Parameters of the model.
        transformed : bool
            Whether or not parameters are transformed.

        Returns
        -------
        result : array_like
            Normalized parameters vector.

        Notes
        -----
        Several unique parameter vectors can represent the only model
        configuration, because of different order of regime enumeration. To
        compare two configurations (e.g. for testing), we need to
        normalize both parameter vectors first, that is to use permutation of
        regimes, which is determined by model configuration only.

        This method relies on user-defined `get_normal_regimes_permutation`
        method.
        """
        if not transformed:
            params = self.transform_params(params)

        permutation = self.get_normal_regimes_permutation(params)
        params = self._permute_regimes(params, permutation)

        if not transformed:
            params = self.untransform_params(params)

        return params

    def initialize_known_regime_probs(self, *args):

        self.ssm.initialize_known_regime_probs(*args)

    def initialize_uniform_regime_probs(self):

        self.ssm.initialize_uniform_regime_probs()

    def initialize_stationary_regime_probs(self):

        self.ssm.initialize_stationary_regime_probs()

    def _get_param_regime_transition(self, constrained_params):

        # Useful method, extracting regime transition matrix from contrained
        # parameters.

        dtype = self.ssm.dtype
        param_k_regimes = self.param_k_regimes

        # Matrix initialization.
        regime_transition = np.zeros((param_k_regimes, param_k_regimes),
                dtype=dtype)

        # Fill all elements except of the last row
        regime_transition[:-1, :] = constrained_params[
                self.parameters['regime_transition']].reshape((-1,
                param_k_regimes))

        # Fill the last row
        regime_transition[-1, :] = 1 - regime_transition[:-1, :].sum(axis=0)

        return regime_transition

    def _set_param_regime_transition(self, constrained_params, regime_transition):

        # Useful method, encoding regime transition matrix into constrained
        # parameters vector.

        constrained_params[self.parameters['regime_transition']] = \
                regime_transition[:-1, :].ravel()

    @property
    def start_params(self):
        """
        (array) Starting parameters for maximum likelihood estimation. Note,
        that this is a bad initialization with identical regimes. Consider using
        user-defined starting parameters, non-switching fit or model-specific
        initialization (e.g. EM-algorithm in case of MS AR).
        """
        return self.transform_params(np.ones((self.parameters.k_params,),
            dtype=self.ssm.dtype))

    def fit(self, start_params=None, fit_nonswitching_first=False,
            default_transition_probs=None, **kwargs):
        """
        Fits the model by maximum likelihood via Kim filter.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            If `None`, the default is given by Model.start_params. If
            `fit_nonswitching_first=True`, then this is passed to
            non-switching model `fit` arguments.
        fit_nonswitching_first : bool, optional
            Use non-switching initialization feature. Default is `False`.
        default_transition_probs : array_like, optional
            If `fit_nonswitching_first=True`, this argument is used to specify
            regime transition matrix initial guess, because non-switching model
            only guesses parameters related to regimes representation. Equal
            probabilities by default.
        **kwargs
            Additional keyword arguments, which are passed to superclass `fit`
            method and also to non-switching model `fit` method, if it is used.

        Returns
        -------
        params : array_like
            Estimated parameters

        Notes
        -----
        Kalman-filter-specific options `optim_score='harvey'` and
        `optim_hessian='oim'` are unavailable.

        See also
        --------
        statsmodels.base.model.LikelihoodModel.fit
        statsmodels.tsa.statespace.mlemodel.fit

        """

        dtype = self.ssm.dtype

        if fit_nonswitching_first:
            # Initializing of non-switching model
            nonswitching_model = self.get_nonswitching_model()
            # Copying kwargs
            nonswitching_kwargs = dict(kwargs)
            # Need to return parameters
            nonswitching_kwargs['return_params'] = True

            start_nonswitching_params = start_params
            # Fit non-switching model
            nonswitching_params = nonswitching_model.fit(
                    start_params=start_nonswitching_params,
                    **nonswitching_kwargs)

            # Constructing starting parameters for switching model
            start_params = np.zeros((self.parameters.k_params,), dtype=dtype)

            # If default distributions are not provided, use uniform
            if default_transition_probs is None:
                default_transition_probs = \
                        np.ones((self.param_k_regimes, self.param_k_regimes),
                        dtype=self.ssm.dtype) / self.param_k_regimes

            # Encoding regime transition matrix
            self._set_param_regime_transition(start_params,
                    default_transition_probs)

            # Encoding model parameters
            start_params = self.update_params(start_params, nonswitching_params)

        kwargs['start_params'] = start_params

        #kwargs['return_params'] = True

        # Kalman-filter-specific Harvey method is not available
        if 'optim_score' in kwargs and kwargs['optim_score'] == 'harvey':
            raise NotImplementedError
        if 'optim_hessian' in kwargs and kwargs['optim_hessian'] == 'oim':
            raise NotImplementedError

        return super(SwitchingMLEModel, self).fit(**kwargs)

    def filter(self, params, transformed=True, complex_step=False,
            cov_type=None, cov_kwds=None, return_ssm=False,
            results_class=None, results_wrapper_class=None, **kwargs):
        """
        Kim filtering

        Notes
        -----
        This method is inherited from base `MLEModel` class, see arguments
        explanation, etc. in the corresponding docs.
        Also, Kalman-filter-specific Harvey method (`cov_type == 'oim'`) is
        unavailable here.

        See Also
        --------
        MLEModel.filter
        """

        if not return_ssm and results_class is None:
            # In this case base class returns `MLEResults` instance, so we need
            # to specify results class
            results_class = SwitchingMLEResults

        return super(SwitchingMLEModel, self).filter(params,
                transformed=transformed, complex_step=complex_step,
                cov_type=cov_type, cov_kwds=cov_kwds, return_ssm=return_ssm,
                results_class=results_class,
                results_wrapper_class=results_wrapper_class, **kwargs)

    def smooth(self, params, transformed=True, complex_step=False,
            cov_type=None, cov_kwds=None, return_ssm=False,
            results_class=None, results_wrapper_class=None, **kwargs):
        """
        Kim smoothing

        Notes
        -----
        This method is inherited from base `MLEModel` class, see arguments
        explanation, etc. in the corresponding docs.
        Also, Kalman-filter-specific Harvey method (`cov_type == 'oim'`) is
        unavailable here.

        See Also
        --------
        MLEModel.smooth
        """

        if not return_ssm and results_class is None:
            # In this case base class returns `MLEResults` instance, so we need
            # to specify results class
            results_class = SwitchingMLEResults

        return super(SwitchingMLEModel, self).smooth(params,
                transformed=transformed, complex_step=complex_step,
                cov_type=cov_type, cov_kwds=cov_kwds, return_ssm=return_ssm,
                results_class=results_class,
                results_wrapper_class=results_wrapper_class, **kwargs)

    #def loglike(self, *args, **kwargs):
    #    raise NotImplementedError

    #def loglikeobs(self, *args, **kwargs):
    #    raise NotImplementedError

    def update(self, params, transformed=True, complex_step=False):
        """
        Update the parameters of the model

        Parameters
        ----------
        params : array_like
            Array of new parameters.
        transformed : boolean, optional
            Whether or not `params` is already transformed. If set to False,
            `transform_params` is called. Default is True.

        Returns
        -------
        params : array_like
            Array of parameters.

        Notes
        -----
        This method should be overridden by subclasses to perform actual
        updating steps.

        See Also
        --------
        MLEModel.update
        """

        # When `complex_step` is used, params can have a small imaginary part
        if complex_step:
            params = np.real(params)

        return super(SwitchingMLEModel, self).update(params,
                transformed=transformed, complex_step=complex_step)

    #TODO: add this functionality
    def set_smoother_output(self, **kwargs):
        raise NotImplementedError

    def initialize_approximate_diffuse(self, **kwargs): 
        raise NotImplementedError(
                'Diffuse initialization is not defined for Kim filtering.')

    @property
    def initial_variance(self):
        raise NotImplementedError(
                'Diffuse initialization is not defined for Kim filtering.')

    @initial_variance.setter
    def initial_variance(self, value):
        raise NotImplementedError(
                'Diffuse initialization is not defined for Kim filtering.')

    def simulation_smoother(self, *args, **kwargs):
        # Simulation is not implemented yet
        raise NotImplementedError

    def _forecast_error_partial_derivatives(self, *args, **kwargs): 
        raise NotImplementedError('Kalman filter specific functionality.')

    def observed_information_matrix(self, *args, **kwargs):
        raise NotImplementedError('Kalman filter specific functionality.')

    #def opg_information_matrix(self, *args, **kwargs):
    #    raise NotImplementedError

    #def _score_complex_step(self, *args, **kwargs):
    #    raise NotImplementedError

    #def _score_finite_difference(self, *args, **kwargs):
    #    raise NotImplementedError

    def _score_harvey(self, *args, **kwargs):
        raise NotImplementedError('Kalman filter specific functionality.')

    def _score_obs_harvey(self, *args, **kwargs):
        raise NotImplementedError('Kalman filter specific functionality.')

    def score(self, *args, **kwargs):
        """
        Compute the score function at params

        Notes
        -----
        This method is inherited from base `MLEModel` class, see arguments
        explanation, etc. in the corresponding docs.
        Also, Kalman-filter-specific Harvey method (`method == 'harvey'`) is
        unavailable here.

        See Also
        --------
        MLEModel.score
        """

        return super(SwitchingMLEModel, self).score(*args, **kwargs)

    def score_obs(self, *args, **kwargs):
        """
        Compute the score per observation, evaluated at params

        Notes
        -----
        This method is inherited from base `MLEModel` class, see arguments
        explanation, etc. in the corresponding docs.
        Also, Kalman-filter-specific Harvey method (`method == 'harvey'`) is
        unavailable here.

        See Also
        --------
        MLEModel.score_obs
        """

        return super(SwitchingMLEModel, self).score_obs(*args, **kwargs)

    def hessian(self, *args, **kwargs):
        """
        Hessian matrix of the likelihood function, evaluated at the given
        parameters

        Notes
        -----
        This method is inherited from base `MLEModel` class, see arguments
        explanation, etc. in the corresponding docs.
        Also, Kalman-filter-specific Harvey method (`method == 'oim'`) is
        unavailable here.

        See Also
        --------
        MLEModel.hessian
        """

        return super(SwitchingMLEModel, self).hessian(*args, **kwargs)

    def _hessian_oim(self, *args, **kwargs):
        raise NotImplementedError('Kalman filter specific functionality.')

    #def _hessian_opg(self, *args, **kwargs):
    #    raise NotImplementedError

    #def _hessian_finite_difference(self, *args, **kwargs):
    #    raise NotImplementedError

    #def _hessian_complex_step(self, *args, **kwargs):
    #    raise NotImplementedError

    def transform_jacobian(self, *args, **kwargs):
        """
        Jacobian matrix matrix for the parameter transformation function

        Notes
        -----
        This method is inherited from base `MLEModel` class, see arguments
        explanation, etc. in the corresponding docs.

        See Also
        --------
        MLEModel.transform_jacobian
        """

        return super(SwitchingMLEModel, self).transform_jacobian(*args,
                **kwargs)

    def simulate(self, *args, **kwargs):
        # Simulation is not implemented yet
        raise NotImplementedError

    def impulse_responses(self, *args, **kwargs):
        # Not implemented yet
        raise NotImplementedError

class SwitchingMLEResults(MLEResults):

    _filter_and_smoother_attributes = ['filtered_state', 'filtered_state_cov',
            'filtered_regime_probs', 'predicted_regime_probs',
            'initial_regime_probs', 'smoothed_regime_probs',
            'smoothed_curr_and_next_regime_probs']

    def __init__(self, model, params, results, cov_type='opg',
            cov_kwds=None, **kwargs):

        if cov_type == 'oim' or cov_type == 'robust_oim':
            raise NotImplementedError('Kalman filter specific functionality.')

        #TODO: check for correctness
        #TODO: take away attributes array
        super(SwitchingMLEResults, self).__init__(model, params, results,
                cov_type=cov_type, cov_kwds=cov_kwds, **kwargs)

    def _get_robustcov_results(self, cov_type='opg', **kwargs):

        if cov_type == 'oim' or cov_type == 'robust_oim':
            raise NotImplementedError('Kalman filter specific functionality.')

        return super(SwitchingMLEResults, self)._get_robustcov_results(
                cov_type=cov_type, **kwargs)

    #def aic(self, *args, **kwargs):
    #    raise NotImplementedError

    #def bic(self, *args, **kwargs):
    #    raise NotImplementedError

    #def _cov_params_approx(self, *args, **kwargs):
    #    raise NotImplementedError

    #def cov_params_approx(self, *args, **kwargs):
    #    raise NotImplementedError

    def _cov_params_oim(self, *args, **kwargs):
        raise NotImplementedError('Kalman filter specific functionality.')

    def cov_params_oim(self, *args, **kwargs):
        raise NotImplementedError('Kalman filter specific functionality.')

    #def _cov_params_opg(self, *args, **kwargs):
    #    raise NotImplementedError

    #def cov_params_opg(self, *args, **kwargs):
    #    raise NotImplementedError

    def cov_params_robust(self, *args, **kwargs):
        raise NotImplementedError('Kalman filter specific functionality.')

    def _cov_params_robust_oim(self, *args, **kwargs):
        raise NotImplementedError('Kalman filter specific functionality.')

    def cov_params_robust_oim(self, *args, **kwargs):
        raise NotImplementedError('Kalman filter specific functionality.')

    #def _cov_params_robust_approx(self, *args, **kwargs):
    #    raise NotImplementedError

    #def cov_params_robust_approx(self, *args, **kwargs):
    #    raise NotImplementedError

    #def fitted_values(self, *args, **kwargs):
    #    raise NotImplementedError

    #def hqic(self, *args, **kwargs):
    #    raise NotImplementedError

    #def llf_obs(self, *args, **kwargs):
    #    raise NotImplementedError

    #def llf(self, *args, **kwargs):
    #    raise NotImplementedError

    #def loglikelihood_burn(self, *args, **kwargs):
    #    raise NotImplementedError

    #def pvalues(self, *args, **kwargs):
    #    raise NotImplementedError

    #def resid(self, *args, **kwargs):
    #    raise NotImplementedError

    #def zvalues(self, *args, **kwargs):
    #    raise NotImplementedError

    #def test_normality(self, *args, **kwargs):
    #    raise NotImplementedError

    #def test_heteroscedasticity(self, *args, **kwargs):
    #    raise NotImplementedError

    #def test_serial_correlation(self, *args, **kwargs):
    #    raise NotImplementedError

    #def get_prediction(self, *args, **kwargs):
    #    raise NotImplementedError

    #def get_forecast(self, *args, **kwargs):
    #    raise NotImplementedError

    #def predict(self, *args, **kwargs):
    #    raise NotImplementedError

    #def forecast(self, *args, **kwargs):
    #    raise NotImplementedError

    def simulate(self, *args, **kwargs):
        # Simulation is not implemented yet
        raise NotImplementedError

    def impulse_responses(self, *args, **kwargs):
        # Not implemented yet
        raise NotImplementedError

    #def plot_diagnostics(self, *args, **kwargs):
    #    raise NotImplementedError

    def summary(self, title=None, **kwargs):
        """
        Summarize the Model

        Notes
        -----
        This method is inherited from base `MLEModel` class, see arguments
        explanation, etc. in the corresponding docs.

        See Also
        --------
        MLEModel.summary
        """

        # change statespace model title
        if title is None:
            title = 'Markov Switching Statespace Model Results'

        return super(SwitchingMLEResults, self).summary(title=title, **kwargs)
