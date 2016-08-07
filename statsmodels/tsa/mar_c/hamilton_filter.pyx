import numpy as np
cimport numpy as np
cimport cython
DTYPE = np.float64
ctypedef np.float64_t dtype_t

cdef extern from "math.h":
    double log(double x)
cdef extern from "math.h":
    double exp(double x)
cdef extern from "math.h":
    double sqrt(double x)
cdef extern from "math.h":
    double PI

@cython.boundscheck(False)
@cython.wraparound(False)
def hamilton_filter(int nobs,
                    int nstates,
                    int order,
                    np.ndarray[dtype_t, ndim = 2] transition_vectors not None,
                    np.ndarray[dtype_t, ndim = 2, mode='c'] joint_probabilities not None,
                    np.ndarray[dtype_t, ndim = 2, mode='c'] marginal_conditional_densities not None):

    if order == 0:
        return hamilton_filter_uncorrelated(nobs, nstates, transition_vectors, joint_probabilities, marginal_conditional_densities)

    cdef np.ndarray[dtype_t, ndim = 2] joint_probabilities_t1
    cdef np.ndarray[dtype_t, ndim = 1] joint_densities, marginal_densities
    #cdef np.ndarray[dtype_t, ndim = 1] _joint_probabilities_t1 #, joint_probabilities_t
    #cdef np.ndarray[dtype_t, ndim = 1] _joint_probabilities, _marginal_conditional_densities
    cdef int nstatesk_1, nstatesk, nstatesk1, t, i, j, k, idx
    cdef dtype_t transition

    nstatesk_1 = nstates**(order-1)
    nstatesk = nstates**order
    nstatesk1 = nstates**(order+1)

    joint_probabilities_t1 = np.zeros((nobs, nstatesk1))
    joint_densities = np.zeros((nstatesk1,))
    marginal_densities = np.zeros((nobs,))
    #_joint_probabilities_t1 = np.zeros((nstatesk1,))
    #joint_probabilities_t = np.zeros((nstatesk1,))
    #_joint_probabilities = np.zeros((nstatesk,))
    #_marginal_conditional_densities = np.zeros((nstatesk1,))

    for t in range(1, nobs+1):
        #_joint_probabilities = joint_probabilities[t-1]
        #_marginal_conditional_densities = marginal_conditional_densities[t-1]
        idx = 0
        for i in range(nstates):
            for j in range(nstates):
                transition = transition_vectors[t-1, i*nstates + j]
                for k in range(nstatesk_1):
                    joint_probabilities_t1[t-1, idx] = transition * joint_probabilities[t-1, j*nstatesk_1 + k]
                    joint_densities[idx] = joint_probabilities_t1[t-1, idx] * marginal_conditional_densities[t-1, idx]
                    marginal_densities[t-1] += joint_densities[idx]
                    idx += 1
        #joint_probabilities_t1[t-1] = _joint_probabilities_t1

        #joint_probabilities_t1 = (
        #    np.repeat(transition_vectors[t], nstates**(order-1)) * 
        #    np.tile(joint_probabilities[t-1], nstates)
        #)

        #joint_densities = np.multiply(
        #    marginal_conditional_densities[t-1], joint_probabilities_t1
        #)

        #for i in range(nstatesk1):
        #    marginal_densities[t-1] += joint_densities[i]
        #marginal_densities[t-1] = np.sum(joint_densities)

        #for i in range(nstatesk1):
        #    joint_probabilities_t[i] = joint_densities[i] / marginal_densities[t-1]

        #joint_probabilities_t = joint_densities / marginal_densities[t-1]

        for i in range(nstatesk):
            #_joint_probabilities[i] = 0
            idx = i*nstates
            for j in range(nstates):
                #joint_probabilities_t[idx+j] = joint_densities[idx+j] / marginal_densities[t-1]
                joint_probabilities[t, i] += joint_densities[idx+j] / marginal_densities[t-1]
        #joint_probabilities[t] = _joint_probabilities
        #joint_probabilities[t] = joint_probabilities_t.reshape(
        #    (nstates**order, nstates)
        #).sum(1)
    return marginal_densities, joint_probabilities, joint_probabilities_t1

@cython.boundscheck(False)
@cython.wraparound(False)
def hamilton_filter_uncorrelated(int nobs,
                                 int nstates,
                                 np.ndarray[dtype_t, ndim = 2] transition_vectors not None,
                                 np.ndarray[dtype_t, ndim = 2, mode='c'] joint_probabilities not None,
                                 np.ndarray[dtype_t, ndim = 2, mode='c'] marginal_conditional_densities not None):

    cdef np.ndarray[dtype_t, ndim = 2] joint_probabilities_t1, marginal_probabilities_t1
    cdef np.ndarray[dtype_t, ndim = 1] joint_densities, marginal_densities
    cdef int t, i, j, k, idx
    cdef dtype_t transition

    joint_probabilities_t1 = np.zeros((nobs, nstates**2))
    marginal_probabilities_t1 = np.zeros((nobs, nstates))
    joint_densities = np.zeros((nstates,))
    marginal_densities = np.zeros((nobs,))

    for t in range(1, nobs+1):
        for i in range(nstates):        # Range over S_t
            for j in range(nstates):    # Range over S_{t-1}
                # This step is what dictates whether transition is in left or right stochastic form
                # Here, i represents the row (S_t = the state to which we're moving),
                # and j represents the column (S_{t-1} = the state from which we're moving)
                # Thus the vector needs to be of the form:
                # [P11 P12 ... P1M P21 ... P2M ... PMM ]
                transition = transition_vectors[t-1, i*nstates + j]
                joint_probabilities_t1[t-1, i*nstates + j] = transition * joint_probabilities[t-1, j]
                marginal_probabilities_t1[t-1, i] += joint_probabilities_t1[t-1, i*nstates + j]
            joint_densities[i] = marginal_probabilities_t1[t-1, i] * marginal_conditional_densities[t-1, i]
            marginal_densities[t-1] += joint_densities[i]

        joint_probabilities[t] = joint_densities / marginal_densities[t-1]
    return marginal_densities, joint_probabilities, joint_probabilities_t1

def tvtp_transition_vectors_right(int nobs,
                                 int nstates,
                                 int tvtp_order,
                                 np.ndarray[dtype_t, ndim = 2] transitions,     # nstates * (nstates-1) x tvtp_order
                                 np.ndarray[dtype_t, ndim = 2, mode='c'] exog): # t+1 x tvtp_order
    cdef int n, t, i, j, k, idx
    cpdef dtype_t transition, colsum
    cdef np.ndarray[dtype_t, ndim = 2] transition_vectors

    transition_vectors = np.zeros((nobs+1, nstates**2))

    for t in range(nobs+1):
        for i in range(nstates): # iterate over "columns" in the transition matrix
            colsum = 0
            for j in range(nstates-1): # iterate all but last "row" in the transition matrix
                transition = 0
                for k in range(tvtp_order):
                    transition += exog[t,k] * transitions[i*(nstates-1)+j, k]
                transition = exp(transition)
                transition_vectors[t, i + j*nstates] = transition
                colsum += transition
            # iterate over all but the last "row" again, now that we have all
            # of the values
            for j in range(nstates-1):
                transition_vectors[t, i + j*nstates] /= (1 + colsum)
            # Add in last row
            transition_vectors[t,i + (nstates-1)*nstates] = 1 - (colsum / (1 + colsum))

    return transition_vectors

def tvtp_transition_vectors_left(int nobs,
                                 int nstates,
                                 int tvtp_order,
                                 np.ndarray[dtype_t, ndim = 2] transitions,     # nstates * (nstates-1) x tvtp_order
                                 np.ndarray[dtype_t, ndim = 2, mode='c'] exog): # t+1 x tvtp_order
    cdef int n, t, i, j, k, idx
    cpdef dtype_t transition, colsum
    cdef np.ndarray[dtype_t, ndim = 2] transition_vectors

    transition_vectors = np.zeros((nobs+1, nstates**2))

    for t in range(nobs+1):
        idx = 0
        for i in range(nstates): # iterate over "columns" in the transition matrix
            colsum = 0
            for j in range(nstates-1): # iterate all but last "row" in the transition matrix
                transition = 0
                for k in range(tvtp_order):
                    transition += exog[t,k] * transitions[i*(nstates-1)+j, k]
                transition = exp(transition)
                transition_vectors[t, idx] = transition
                colsum += transition
            # iterate over all but the last "row" again, now that we have all
            # of the values
            for j in range(nstates-1):
                transition_vectors[t, idx] /= (1 + colsum)
                idx += 1
            # Add in last row
            transition_vectors[t,idx] = 1 - (colsum / (1 + colsum))
            idx += 1

    return transition_vectors

def marginal_conditional_densities(int nobs,
                                   int nstates,
                                   int order,
                                   np.ndarray[dtype_t, ndim=2] params,
                                   np.ndarray[dtype_t, ndim=1] stddevs,
                                   np.ndarray[dtype_t, ndim=1] means,
                                   np.ndarray[dtype_t, ndim=2] augmented):
    cdef int nstatesk, t, i, j, k, idx, idx2, num, state
    cdef dtype_t var, top
    cdef np.ndarray[dtype_t, ndim = 1] state_means, variances
    cdef np.ndarray[dtype_t, ndim = 2] marginal_conditional_densities

    nstatesk = nstates**order
    marginal_conditional_densities = np.zeros((nobs, nstates**(order+1)))
    variances = stddevs**2

    state_means = np.zeros((order+1,))
    for t in range(nobs):
        idx = 0
        for i in range(nstates):
            var = variances[i]
            for j in range(nstatesk):
                num = idx
                top = 0
                for k in range(order+1):
                    state = num % nstates
                    top += (augmented[t, -(k+1)] - means[state]) * params[i, -(k+1)]
                    num = num // nstates
                marginal_conditional_densities[t, idx] = (
                    (1 / sqrt(2*np.pi*var)) * exp(
                        -( top**2 / (2*var))
                    )
                )
                idx += 1

    return marginal_conditional_densities
