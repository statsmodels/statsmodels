import numpy as np
from collections import OrderedDict


def _is_left_stochastic(matrix):
    # This method checks if `matrix` is left stochastic

    # Matrix should have square shape
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False

    # Comparing by eps is highly important due to floating point imperfect
    # accuracy
    eps = 1e-8

    # Check if all elements are non-negative by eps
    if np.any(matrix < -eps):
        return False

    # If some elements are negative, but insignificantly small, set them
    # to zero
    matrix[matrix < 0] = 0

    # Check if every column represents a discrete probability distribution
    if not np.all(np.fabs(matrix.sum(axis=0) - 1) < eps):
        return False

    # If all checks are passed, return True
    return True


class RegimePartition(object):
    r"""
    Markov switching regimes partition, forming a Markov chain itself

    Parameters
    ----------
    partition : array_like
        Array of size, equal to the number of regimes, encoding the partition
        of regimes set. That is, i-th position of `partition` array contains
        index of partition the i-th regime belongs. If there are :math:`k`
        partitions, than partition indices are from segment :math:`[0, k-1]`.
        Example of 6 regimes partition into 3 sets: `[0, 2, 1, 0, 1, 2]`.

    Notes
    -----
    Suppose that we have :math:`n` Markov switching regimes and partition
    :math:`\lbrace A_1, A_2, ... , A_n \rbrace` of this regime set. Sometimes
    it happens that these subsets form a Markov chain themselves. That is,
    for every :math:`1 \leqslant k, l \leqslant n`:

    .. math::

        Pr[x_t \in A_l | x_{t-1} = i_1 ] = Pr[x_t \in A_l | x_{t-1} = i_2] = ...
        = Pr[x_t \in A_l | x_{t-1} = i_{m_k}]

    where :math:`A_k = \lbrace i_1, i_2, ... , i_{m_k} \rbrace` and :math:`x_t`
    is a sequence of regimes.
    Then we can speak of probabilities
    .. math::

        p_{lk} = Pr[x_t \in A_l | x_{t-1} in A_k ] =
        Pr[x_t \in A_l | x_{t-1} = i_{j}]

    which do not depend on moment :math:`t` and initial regimes distribution.

    This class handles this partition: validates that it forms a Markov chain
    and finds its transition probability matrix.

    See Also
    --------
    statsmodels.tsa.statespace.regime_switching.kim_smoother.KimSmoother.smooth
    """

    def __init__(self, partition):

        self.size = max(partition) + 1
        self._partition = np.array(partition)

    def get_subset_index(self, regime):
        """
        Get an element of partition to which regime belongs

        Parameters
        ----------
        regime : int
            Regime index

        Returns
        -------
        subset_index : int
            Index of partition, containing `regime`
        """
        return self._partition[regime]

    def get_mask(self, subset_index):
        """
        Get a boolean mask of regimes, belonging to subset

        Parameters
        ----------
        subset_index : int
            Index of partition element

        Returns
        -------
        mask : array_like
            Array of size, equal to the number of regimes, where each boolean
            element indicates whether corresponding regime is in the subset.
        """
        return self._partition == subset_index

    def get_transition_probabilities(self, regime_transition):
        """
        Get a transition probability matrix of partition, if it forms Markov
        chain. Otherwise, raise a `ValueError`.

        Parameters
        ----------
        regime_transition : array_like
            Left stochastic matrix of order, equal to the number of regimes,
            representing their transition probabilities.

        Returns
        -------
        transition_probs : array_like
            Left-stochastic matrix, representing transition probabilities
            between elements of partition.

        See Also
        --------
        statsmodels.tsa.statespace.regime_switching.kim_smoother._KimSmoother.\
        __call__
        """

        # Transform provided matrix into numpy array
        regime_transition = np.asarray(regime_transition)

        # Check whether provided matrix is left stochastic
        if not _is_left_stochastic(regime_transition):
            raise ValueError('Provided matrix is not left stochastic')

        # Check whether shapes of matrix and partition conform
        if regime_transition.shape[0] != self._partition.shape[0]:
            raise ValueError('Order of the matrix and number of regimes are \
                    unequal')

        dtype = regime_transition.dtype
        k_regimes = self._partition.shape[0]
        partition_size = self.size

        transition_probs = np.zeros((partition_size, partition_size),
                dtype=dtype)

        regimes = np.arange(k_regimes)

        # All comparisons should use epsilon error
        eps = 1e-8

        # For every (k, l)
        for prev_subset_index in range(partition_size):
            for curr_subset_index in range(partition_size):

                probability_not_set = True

                # For every i_{j} \in A_k
                for prev_regime in regimes[self.get_mask(prev_subset_index)]:

                    # Calculate Pr[ x_t \in A_l | x_{t-1} = i_{j} ]
                    transition_prob = 0.0
                    for curr_regime in regimes[self.get_mask(
                            curr_subset_index)]:
                        transition_prob += regime_transition[curr_regime,
                                prev_regime]

                    # if all such probabilities are equal, fill them in
                    # `transition_probs`, else raise exception
                    if probability_not_set:
                        transition_probs[curr_subset_index,
                                prev_subset_index] = transition_prob
                        probability_not_set = False
                    else:
                        if np.fabs(transition_probs[curr_subset_index,
                                prev_subset_index] - transition_prob) > eps:
                            raise ValueError('Provided partition doesn\'t form \
                                    Markov chain')

        return transition_probs

class MarkovSwitchingParams(object):
    def __init__(self, k_regimes):
        self.k_regimes = k_regimes

        self.k_params = 0
        self.k_parameters = OrderedDict()
        self.switching = OrderedDict()
        self.slices_purpose = OrderedDict()
        self.relative_index_regime_purpose = [
            OrderedDict() for i in range(self.k_regimes)]
        self.index_regime_purpose = [
            OrderedDict() for i in range(self.k_regimes)]
        self.index_regime = [[] for i in range(self.k_regimes)]

    def __getitem__(self, key):
        _type = type(key)

        # Get a slice for a block of parameters by purpose
        if _type is str:
            return self.slices_purpose[key]
        # Get a slice for a block of parameters by regime
        elif _type is int:
            return self.index_regime[key]
        elif _type is tuple:
            if not len(key) == 2:
                raise IndexError('Invalid index')
            if type(key[1]) == str and type(key[0]) == int:
                return self.index_regime_purpose[key[0]][key[1]]
            elif type(key[0]) == str and type(key[1]) == int:
                return self.index_regime_purpose[key[1]][key[0]]
            else:
                raise IndexError('Invalid index')
        else:
            raise IndexError('Invalid index')

    def __setitem__(self, key, value):
        _type = type(key)

        if _type is str:
            value = np.array(value, dtype=bool, ndmin=1)
            k_params = self.k_params
            self.k_parameters[key] = (
                value.size + np.sum(value) * (self.k_regimes - 1))
            self.k_params += self.k_parameters[key]
            self.switching[key] = value
            self.slices_purpose[key] = np.s_[k_params:self.k_params]

            for j in range(self.k_regimes):
                self.relative_index_regime_purpose[j][key] = []
                self.index_regime_purpose[j][key] = []

            offset = 0
            for i in range(value.size):
                switching = value[i]
                for j in range(self.k_regimes):
                    # Non-switching parameters
                    if not switching:
                        self.relative_index_regime_purpose[j][key].append(
                            offset)
                    # Switching parameters
                    else:
                        self.relative_index_regime_purpose[j][key].append(
                            offset + j)
                offset += 1 if not switching else self.k_regimes

            for j in range(self.k_regimes):
                offset = 0
                indices = []
                for k, v in self.relative_index_regime_purpose[j].items():
                    v = (np.r_[v] + offset).tolist()
                    self.index_regime_purpose[j][k] = v
                    indices.append(v)
                    offset += self.k_parameters[k]
                self.index_regime[j] = np.concatenate(indices)
        else:
            raise IndexError('Invalid index')
