import numpy as np
from .kim_filter import KimFilter, KimFilterResults
from scipy.misc import logsumexp


class _KimSmoother(object):

    def __init__(self, model, filtered_regime_logprobs,
            predicted_regime_logprobs, regime_partition):

        # This class does smoothing work

        # Save reference to state space representation
        self.model = model

        # Save probabilities which are used during smoothing
        self.filtered_regime_logprobs = filtered_regime_logprobs
        self.predicted_regime_logprobs = predicted_regime_logprobs

        # Save partition
        self.regime_partition = regime_partition

    def __call__(self):

        # This method is based on chapter 5.3 of
        # Kim, Chang-Jin, and Charles R. Nelson. 1999.
        # "State-Space Models with Regime Switching:
        # Classical and Gibbs-Sampling Approaches with Applications".
        # MIT Press Books. The MIT Press.

        # Notation in comments follows this chapter

        # Method calculates Pr[ S_t | \psi_T ] and Pr[ S_t, S_{t+1} | \psi_T ]

        model = self.model
        regime_partition = self.regime_partition

        dtype = model._dtype
        nobs = model._nobs
        k_regimes = model._k_regimes

        # If partition is provided, its elements are treated as regimes.
        if regime_partition is None:
            # Create references, used during smoothing
            log_transition = model._log_regime_transition
            filtered_regime_logprobs = self.filtered_regime_logprobs
            predicted_regime_logprobs = self.predicted_regime_logprobs
            partition_size = k_regimes
        else:
            # Get transition probability matrix for partition (or raise error,
            # if partition doesn't form Markov chain)
            transition_probs = regime_partition.get_transition_probabilities(
                    np.exp(model._log_regime_transition))
            log_transition = np.log(transition_probs)

            # Allocating partition probabilities array
            filtered_regime_logprobs = np.zeros((nobs, regime_partition.size),
                    dtype=dtype)
            predicted_regime_logprobs = np.zeros((nobs, regime_partition.size),
                    dtype=dtype)

            # Collapsing values to get probabilities of subsets, forming
            # partition
            for i in range(regime_partition.size):
                mask = regime_partition.get_mask(i)
                filtered_regime_logprobs[:, i] = logsumexp(
                        self.filtered_regime_logprobs[:, mask], axis=1)
                predicted_regime_logprobs[:, i] = logsumexp(
                        self.predicted_regime_logprobs[:, mask], axis=1)

            partition_size = regime_partition.size

        # Allocation of result

        # Pr[ S_t | \psi_T ]
        self.smoothed_regime_logprobs = np.zeros((nobs, partition_size),
                dtype=dtype)
        # Pr[ S_t, S_{t+1} | \psi_T ]
        self.smoothed_curr_and_next_regime_logprobs = np.zeros((nobs - 1,
                partition_size, partition_size), dtype=dtype)

        # Allocation of buffers, reused during iterations of smoothing

        # Pr[ S_t, S_{t+1} | \psi_t ]
        predicted_curr_and_next_regime_logprobs = np.zeros((partition_size,
                partition_size), dtype=dtype)
        # Pr[ S_t | S_{t+1}, \psi_t ]
        filtered_curr_regime_cond_on_next_logprobs = np.zeros((partition_size,
                partition_size), dtype=dtype)

        # Initialization of smoothing
        self.smoothed_regime_logprobs[-1, :] = filtered_regime_logprobs[-1, :]

        # Backward pass iterations
        for t in range(nobs - 2, -1, -1):

            # Pr[ S_t, S_{t+1} | \psi_t ] = Pr[ S_t | \psi_t ] *
            # Pr[ S_{t+1} | S_t ]
            np.add(log_transition.transpose(),
                    filtered_regime_logprobs[t, :].reshape(-1, 1),
                    out=predicted_curr_and_next_regime_logprobs)

            # Pr[ S_t | S_{t+1}, \psi_t ] = Pr[ S_t, S_{t+1} | \psi_t ] /
            # Pr[ S_{t+1} | \psi_t ]
            for i in range(partition_size):
                # Condition to avoid -np.inf - (-np.inf) operation
                if predicted_regime_logprobs[t + 1, i] != -np.inf:
                    np.subtract(predicted_curr_and_next_regime_logprobs[:, i],
                            predicted_regime_logprobs[t + 1, i],
                            out=filtered_curr_regime_cond_on_next_logprobs[:,
                            i])
                else:
                    filtered_curr_regime_cond_on_next_logprobs[:, i] = -np.inf

            # Pr[ S_t, S_{t+1} | \psi_T ] \approx Pr[ S_{t+1} | \psi_T ] * \
            # Pr[ S_t | S_{t+1}, \psi_t ]
            np.add(self.smoothed_regime_logprobs[t + 1, :].reshape(1, -1),
                    filtered_curr_regime_cond_on_next_logprobs,
                    out=self.smoothed_curr_and_next_regime_logprobs[t, :, :])

            # Pr[ S_t | \psi_T ] = \sum_{S_{t+1}} Pr[ S_t, S_{t+1} | \psi_T ]
            self.smoothed_regime_logprobs[t, :] = logsumexp(
                    self.smoothed_curr_and_next_regime_logprobs[t, :, :], axis=1)


class KimSmoother(KimFilter):
    """
    Markov switching state space representation of a time series process, with
    Kim filter and smoother

    Parameters
    ----------
    k_endog : int
        The number of variables in the process.
    k_states : int
        The dimension of the unobserved state process.
    k_regimes : int
        The number of switching regimes.
    results_class : class, optional
        Default results class to use to save filtering output. Default is
        `KimSmootherResults`. If specified, class must extend from
        `KimSmootherResults`.
    **kwargs
        Additional keyword arguments, passed to superclass initializer.

    Notes
    -----
    This class extends `KimFilter` and performs Kim smoothing.

    See Also
    --------
    _KimSmoother
    KimSmootherResults
    statsmodels.tsa.statespace.regime_switching.kim_filter.KimFilter
    """

    def __init__(self, k_endog, k_states, k_regimes, results_class=None, **kwargs):

        # If no `results_class` provided, set it to default value
        if results_class is None:
            results_class = KimSmootherResults

        super(KimSmoother, self).__init__(k_endog, k_states, k_regimes,
                results_class=results_class, **kwargs)

    def smooth(self, results=None, run_filter=True, regime_partition=None,
            **kwargs):
        """
        Apply the Kim smoother to the Markov switching statespace model.

        Parameters
        ----------
        results : class or object, optional
            If a class, then that class is instantiated and returned with the
            result of filtering. It must be a subclass of `KimFilterResults`.
            If an object, then that object is updated with the filtering data.
            Its class should extend `KimFilterResults`.
            If `None`, then a `KimFilterResults` object is returned.
        run_filter : bool, optional
            Whether or not to run the Kim filter prior to smoothing. Default is
            `True`.
        regime_partition : RegimePartition, optional
            Partition of regimes set, forming a Markov chain. If this is
            specified, smoothing is performed towards elements of partition as
            regimes. See `tools.RegimePartition` for details about partitions.
        **kwargs
            Additional keyword arguments, passed to `filter` method, if
            filtering happens.

        Notes
        -----
        Smoothing is impossible without preliminary filtering, so make sure
        that you've chosen `run_filter=True` option or provided `results`
        object with filtering results.

        Since smoothing is approximate, applying smoothing to regimes and than
        collapsing values to form partition smoothed probabilities gives
        different result from that achieved by treating partition probabilities
        inside smoothing iterations. This is why `regime_partition` can be
        specified.
        See example of `regime_partition` option usage in
        `tests.test_ms_ar_hamilton1989.TestHamilton1989_Smoothing`.

        Returns
        -------
        KimSmootherResults

        See Also
        --------
        statsmodels.tsa.statespace.regime_switching.tools.RegimePartition
        statsmodels.tsa.statespace.regime_switching.tests.\
        test_ms_ar_hamilton1989.TestHamilton1989_Smoothing
        """

        if run_filter:
            # Run filtering first
            results = self.filter(results=results, **kwargs)
        elif results is None or isinstance(results, type) or \
                results.filtered_regime_logprobs is None:
            # Raise exception, if no filtering happened before.
            raise ValueError(
                    'Can\'t perform smoothing without filtering first')

        # Check if given `results` instance is valid
        if not isinstance(results, KimSmootherResults):
            raise ValueError('Invalid results type.')

        # Actual calculations are done by `_KimSmoother` class. See this class
        # for details.
        smoother = _KimSmoother(self, results.filtered_regime_logprobs,
                results.predicted_regime_logprobs, regime_partition)
        smoother()

        # Save smoothing data in results
        results.update_smoother(smoother)

        return results


class KimSmootherResults(KimFilterResults):
    """
    Results from applying the Kim smoother and filter to a Markov switching
    state space model.

    Parameters
    ----------
    model : SwitchingRepresentation
        A Markov switching state space representation

    Attributes
    ----------
    smoothed_regime_logprobs : array_like
        Smoothed log-probabilities of given regime being active at given moment.
        `(nobs, k_regimes)`-shaped array.
    smoothed_curr_and_next_regime_logprobs : array_like
        Smoothed log-probabilities of two given regimes being active at given
        and previous moment.
        `(nobs, k_regimes, k_regimes)`-shaped array.
    """

    _smoother_attributes = ['smoothed_regime_logprobs',
            'smoothed_curr_and_next_regime_logprobs']

    _attributes = KimFilterResults._attributes + _smoother_attributes

    def update_smoother(self, smoother):
        """
        Update the smoothing results

        Parameters
        ----------
        smoother : _KimSmoother
            Object, handling smoothing, which to take updated values from.

        Notes
        -----
        This method is rarely required except for internal usage.
        """

        self.smoothed_regime_logprobs = smoother.smoothed_regime_logprobs
        self.smoothed_curr_and_next_regime_logprobs = \
                smoother.smoothed_curr_and_next_regime_logprobs

    @property
    def smoothed_regime_probs(self):
        """
        (array) Smoothed probabilities of given regime being active at given
        moment.
        """
        return np.exp(self.smoothed_regime_logprobs)

    @property
    def smoothed_curr_and_next_regime_probs(self):
        """
        (array) Smoothed log-probabilities of two given regimes being active at
        given and previous moment.
        """
        return np.exp(self.smoothed_curr_and_next_regime_logprobs)
