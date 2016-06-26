import numpy as np
from .kim_filter import KimFilter, KimFilterResults
from scipy.misc import logsumexp
from .tools import RegimePartition


class _KimSmoother(object):

    def __init__(self, model, filtered_regime_logprobs,
            predicted_regime_logprobs, regime_partition):

        self.model = model
        self.filtered_regime_logprobs = filtered_regime_logprobs
        self.predicted_regime_logprobs = predicted_regime_logprobs
        self.regime_partition = regime_partition

    def _get_subsets_log_transition(self):

        model = self.model
        regime_partition = self.regime_partition

        dtype = model._dtype
        k_regimes = model._k_regimes
        partition_size = regime_partition.size

        log_transition = np.zeros((partition_size, partition_size),
                dtype=dtype)
        log_transition[:, :] = -np.inf

        logprob_initialized = np.zeros((partition_size, partition_size),
                dtype=bool)

        for prev_regime in range(k_regimes):
            prev_subset_index = \
                    regime_partition.get_subset_index(prev_regime)
            for curr_regime in range(k_regimes):
                curr_subset_index = \
                        regime_partition.get_subset_index(curr_regime)
                log_regime_transition = model._log_regime_transition[ \
                        curr_regime, prev_regime]
                if log_regime_transition == -np.inf:
                    continue
                if logprob_initialized[curr_subset_index, prev_subset_index]:
                    if log_transition[curr_subset_index, prev_subset_index] != \
                            log_regime_transition:
                        raise ValueError(
                                'Provided partition doesn\'t form Markov chain')
                else:
                    log_transition[curr_subset_index, prev_subset_index] = \
                            log_regime_transition
                    logprob_initialized[curr_subset_index, \
                            prev_subset_index] = True

        if not model._is_left_stochastic(np.exp(log_transition)):
            raise ValueError('Provided partition doesn\'t form Markov chain')

        return log_transition

    def __call__(self):
        '''
        Get smoothed marginal (smoothed_regime_logprobs) and joint
        (smoothed_curr_and_next_regime_logprobs) regime probabilities.
        regime_partition (RegimePartition instance) is used when we need
        smoothed probs of subsets of regimes, forming a partition of the
        regimes set. This subsets can be considered as superior regimes.
        Method produces sensible result in a case when superior regimes
        form a Markov chain.
        The main usage of partition feature right now is smoothing in
        MarkovAutoregression, which is slightly different from smoothing
        in its state space representation.

        this method is tested in test_ms_ar.py

        p. 107 Kim-Nelson
        '''

        model = self.model
        regime_partition = self.regime_partition

        dtype = model._dtype
        nobs = model._nobs
        k_regimes = model._k_regimes

        if regime_partition is None:
            log_transition = model._log_regime_transition
            filtered_regime_logprobs = self.filtered_regime_logprobs
            predicted_regime_logprobs = self.predicted_regime_logprobs
            regime_partition = RegimePartition(list(range(k_regimes)))
        else:
            log_transition = self._get_subsets_log_transition()
            filtered_regime_logprobs = np.zeros((nobs, regime_partition.size),
                    dtype=dtype)
            predicted_regime_logprobs = np.zeros((nobs, regime_partition.size),
                    dtype=dtype)
            for i in range(regime_partition.size):
                mask = regime_partition.get_mask(i)
                filtered_regime_logprobs[:, i] = logsumexp(
                        self.filtered_regime_logprobs[:, mask], axis=1)
                predicted_regime_logprobs[:, i] = logsumexp(
                        self.predicted_regime_logprobs[:, mask], axis=1)

        partition_size = regime_partition.size

        self.smoothed_regime_logprobs = np.zeros((nobs, partition_size),
                dtype=dtype)
        self.smoothed_curr_and_next_regime_logprobs = np.zeros((nobs - 1,
                partition_size, partition_size), dtype=dtype)

        predicted_curr_and_next_regime_logprobs = np.zeros((partition_size,
                partition_size), dtype=dtype)

        filtered_curr_regime_cond_on_next_logprobs = np.zeros((partition_size,
                partition_size), dtype=dtype)

        self.smoothed_regime_logprobs[-1, :] = filtered_regime_logprobs[-1, :]

        for t in range(nobs - 2, -1, -1):

            np.add(log_transition.transpose(),
                    filtered_regime_logprobs[t, :].reshape(-1, 1),
                    out=predicted_curr_and_next_regime_logprobs)

            for i in range(partition_size):
                # Condition to avoid -np.inf - (-np.inf) operation
                if predicted_regime_logprobs[t + 1, i] != -np.inf:
                    np.subtract(predicted_curr_and_next_regime_logprobs[:, i],
                            predicted_regime_logprobs[t + 1, i],
                            out=filtered_curr_regime_cond_on_next_logprobs[:,
                            i])
                else:
                    filtered_curr_regime_cond_on_next_logprobs[:, i] = -np.inf

            np.add(self.smoothed_regime_logprobs[t + 1, :].reshape(1, -1),
                    filtered_curr_regime_cond_on_next_logprobs,
                    out=self.smoothed_curr_and_next_regime_logprobs[t, :, :])

            self.smoothed_regime_logprobs[t, :] = logsumexp(
                    self.smoothed_curr_and_next_regime_logprobs[t, :, :], axis=1)


class KimSmoother(KimFilter):
    '''
    Kim smoother
    '''

    def __init__(self, k_endog, k_states, k_regimes, results_class=None, **kwargs):

        if results_class is None:
            results_class = KimSmootherResults

        super(KimSmoother, self).__init__(k_endog, k_states, k_regimes,
                results_class=results_class, **kwargs)

    def smooth(self, results=None, run_filter=True,
            regime_partition=None, **kwargs):

        if run_filter:
            results = self.filter(results=results, **kwargs)
        elif results is None or isinstance(results, type) or \
                results.filtered_regime_logprobs is None:
            raise ValueError(
                    'Can\'t perform smoothing without filtering first')

        if not isinstance(results, KimSmootherResults):
            raise ValueError('Invalid results type.')

        smoother = _KimSmoother(self, results.filtered_regime_logprobs,
                results.predicted_regime_logprobs, regime_partition)
        smoother()

        results.update_smoother(smoother)

        return results


class KimSmootherResults(KimFilterResults):

    _smoother_attributes = ['smoothed_regime_logprobs',
            'smoothed_curr_and_next_regime_logprobs']

    _attributes = KimFilterResults._attributes + _smoother_attributes

    def update_smoother(self, smoother):

        self.smoothed_regime_logprobs = smoother.smoothed_regime_logprobs
        self.smoothed_curr_and_next_regime_logprobs = \
                smoother.smoothed_curr_and_next_regime_logprobs

    @property
    def smoothed_regime_probs(self):

        return np.exp(self.smoothed_regime_logprobs)

    @property
    def smoothed_curr_and_next_regime_probs(self):

        return np.exp(self.smoothed_curr_and_next_regime_logprobs)
