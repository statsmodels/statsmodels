import numpy as np
cimport numpy as np
cimport cython
DTYPE = np.float64
ctypedef np.float64_t dtype_t

@cython.boundscheck(False)
@cython.wraparound(False)
def hamilton_filter(int nobs,
                    int nstates,
                    int order,
                    np.ndarray[dtype_t, ndim = 1] transition_vector not None,
                    np.ndarray[dtype_t, ndim = 2, mode='c'] joint_probabilities not None,
                    np.ndarray[dtype_t, ndim = 2, mode='c'] marginal_conditional_densities not None):

    cdef np.ndarray[dtype_t, ndim = 2] joint_probabilities_t1
    cdef np.ndarray[dtype_t, ndim = 1] joint_densities, marginal_densities
    #cdef np.ndarray[dtype_t, ndim = 1] _joint_probabilities_t1 #, joint_probabilities_t
    #cdef np.ndarray[dtype_t, ndim = 1] _joint_probabilities, _marginal_conditional_densities
    cdef int nstatesk_1, nstatesk, nstatesk1, t, i, j, k, idx
    cdef dtype_t transition

    nstatesk_1 = nstates**(order-1)
    nstatesk = nstatesk_1*nstates
    nstatesk1 = nstatesk*nstates

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
        for i in range(nstates):
            for j in range(nstates):
                transition = transition_vector[i*nstates + j]
                for k in range(nstatesk_1):
                    idx = j*nstatesk_1 + k
                    joint_probabilities_t1[t-1, i*nstatesk + idx] = transition * joint_probabilities[t-1, idx]
                    joint_densities[i*nstatesk + idx] = transition * joint_probabilities[t-1, idx] * marginal_conditional_densities[t-1, i*nstatesk + idx]
                    marginal_densities[t-1] += joint_densities[i*nstatesk + idx]
        #joint_probabilities_t1[t-1] = _joint_probabilities_t1

        #joint_probabilities_t1 = (
        #    np.repeat(transition_vector, nstates**(order-1)) * 
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
