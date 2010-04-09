"""
Kalman Filter following Hamilton 13.1 - 13.3

Notes
-----
Script should be general (for known F,Q,A,H, and R)
No smoothing.  No likelihood.
"""

from scipy import optimize
from var import chain_dot #TODO: move this to tools

def kalmanfilter(f, h, y, xi10, q, ntrain, history=False):
    """

    Parameters
    -----------
    f : array-like
        f is the transition matrix for the hidden state
    h : array-like
        Relates the observable state to the hidden state.
    y : array-like
        Observed data
    x10 : array-like
        Is the initial prior on the initial state vector
    q : array-like
        Variance/Covariance matrix on the error term in the hidden state
    ntrain : int
        The number of training periods for the filter.


    Returns
    -------
    likelihood
        The negatiev of the log likelihood
    history or priors, history of posterior

    TODO: change API, update names

    No input checking is done.
    """
    f = np.asarray(f)
    h = np.asarray(h)
    y = np.asarray(y)
    if y.ndim == 1: # note that Y is in rows for now
        y = y[:,None]
        y = y.T

    xi10 = np.asarray(xi10) # could this be 1d?
    q = np.asarray(q)
    n = h.shape[1]
    nobs = y.shape[1]
    if history == False:
        xi10History = xi10
        xi11History = xi10History

    p10 = q # eq 13.2.21
    loglikelihood = 0

    for i in range(nobs):
        hP_p10_h = np.linalg.inv(chain_dot(h.T,p10,h))
        part1 = y[:,i] - np.dot(h.T,xi10History)

        # after training, don't throw this away
        if i > ntrain:
            part2 = -0.5 * chain_dot(part1.T,hP_p10_h,part1)
            det_hpp10h = np.linalg.det(chain_dot(h.T,p10,h))
            if det_hpp10h > 10e-300: # not singular
                loglike_int = (-n/2.)*np.log(2*np.pi)-.5*np.log(det_hpp10h)+part2
                if loglike_int > 10e300:
                    raise ValueError("There was an error in forming likelihood")
                loglikelihood += loglike_int

        # 13.2.15
        xi11History = xi10History + chain_dot(p10,h,hP_p10_h,
                part1)
        # 13.2.16

        p11 = p10 - chain_dot(p10,h,hP_p10_h,h.T,p10)
        # 13.2.17
        xi10History = np.dot(f,xi11History)
        # 13.2.21
        p10 = chain_dot(f,p11,f.T) + q
    return -loglikelihood


def kalmanupdate(params, y, xi10, ntrain, penalty, upperbound, lowerbound):
    """
    TODO: change API, update names

    This isn't general
    """
    paramsorig = params
    # are the bounds binding?
    params = np.min((np.max((lowerbound,params),axis=0),upperbound), axis=0)
    rho = params[0]
    sigma1 = params[1]
    sigma2 = params[2]

    F = np.array([[rho, 0],[0,0]])
    cholQ = np.array([[sigma1,0],[0,sigma2]])
    H = np.ones((2,1))
    q = np.dot(cholQ,cholQ.T)
    loglike = kalmanfilter(F,H,y,xi10,q,ntrain)
    loglike = loglike + penalty*np.sum((params-params)**2)
    return loglike


if __name__ == "__main__":
    import numpy as np
    # Make our observations as in 13.1.13
    np.random.seed(54321)
    nobs = 600
    y = np.zeros(nobs)
    rho = [.5, -.25, .35, .25]
    sigma = 2.0 # std dev. or noise
    for i in range(4,nobs):
        y[i] = np.dot(rho,y[i-4:i][::-1]) + np.random.normal(scale=sigma)
    y = y[100:]

    # make an MA(2) observation equation as in example 13.3
    # y = mu + [1 theta][e_t e_t-1]'
    mu = 2.
    theta = .8
    rho = np.array([1, theta])
    np.random.randn(54321)
    e = np.random.randn(101)
    y = mu + rho[0]*e[1:]+rho[1]*e[:-1]
    # might need to add an axis
    r = len(rho)
    x = np.ones_like(y)

    # For now, assume that F,Q,A,H, and R are known
    F = np.array([[0,0],[1,0]])
    Q = np.array([[1,0],[0,0]])
    A = np.array([mu])
    H = rho[:,None]
    R = 0

    # remember that the goal is to solve recursively for the
    # state vector, xi, given the data, y (in this case)
    # we can also get a MSE matrix, P, associated with *each* observation

    # given that our errors are ~ NID(0,variance)
    # the starting E[e(1),e(0)] = [0,0]
    xi0 = np.array([[0],[0]])
    # with variance = 1 we know that
#    P0 = np.eye(2)  # really P_{1|0}

# Using the note below
    P0 = np.dot(np.linalg.inv(np.eye(r**2)-np.kron(F,F)),Q.ravel('F'))
    P0 = np.reshape(P0, (r,r), order='F')

    # more generally, if the eigenvalues for F are in the unit circle
    # (watch out for rounding error in LAPACK!) then
    # the DGP of the state vector is var/cov stationary, we know that
    # xi0 = 0
    # Furthermore, we could start with
    # vec(P0) = np.dot(np.linalg.inv(np.eye(r**2) - np.kron(F,F)),vec(Q))
    # where vec(X) = np.ravel(X, order='F') with a possible [:,np.newaxis]
    # if you really want a "2-d" array
    # a fortran (row-) ordered raveled array
    # If instead, some eigenvalues are on or outside the unit circle
    # xi0 can be replaced with a best guess and then
    # P0 is a positive definite matrix repr the confidence in the guess
    # larger diagonal elements signify less confidence


    # we also know that y1 = mu
    # and MSE(y1) = variance*(1+theta**2) = np.dot(np.dot(H.T,P0),H)

    state_vector = [xi0]
    forecast_vector = [mu]
    MSE_state = [P0]    # will be a list of matrices
    MSE_forecast = []
    # must be numerical shortcuts for some of this...
    # this should be general enough to be reused
    for i in range(len(y)-1):
        # update the state vector
        sv = state_vector[i]
        P = MSE_state[i]
        HTPHR = np.dot(np.dot(H.T,P),H)+R
        if np.ndim(HTPHR) < 2: # we have a scalar
            HTPHRinv = 1./HTPHR
        else:
            HTPHRinv = np.linalg.inv(HTPHR)
        FPH = np.dot(np.dot(F,P),H)
        gain_matrix = np.dot(FPH,HTPHRinv)  # correct
        new_sv = np.dot(F,sv)
        new_sv += np.dot(gain_matrix,y[i] - np.dot(A.T,x[i]) -
                np.dot(H.T,sv))
        state_vector.append(new_sv)
        # update the MSE of the state vector forecast using 13.2.28
        new_MSEf = np.dot(np.dot(F - np.dot(gain_matrix,H.T),P),F.T - np.dot(H,
            gain_matrix.T)) + np.dot(np.dot(gain_matrix,R),gain_matrix.T) + Q
        MSE_state.append(new_MSEf)
        # update the in sample forecast of y
        forecast_vector.append(np.dot(A.T,x[i+1]) + np.dot(H.T,new_sv))
        # update the MSE of the forecast
        MSE_forecast.append(np.dot(np.dot(H.T,new_MSEf),H) + R)
    MSE_forecast = np.array(MSE_forecast).squeeze()
    MSE_state = np.array(MSE_state)
    forecast_vector = np.array(forecast_vector)
    state_vector = np.array(state_vector).squeeze()

##########
#    Luca's example
    # choose parameters governing the signal extraction problem
    rho = .9
    sigma1 = 1
    sigma2 = 1
    nobs = 100

# get the state space representation (Hamilton's notation)\
    F = np.array([[rho, 0],[0, 0]])
    cholQ = np.array([[sigma1, 0],[0,sigma2]])
    H = np.ones((2,1))

# generate random data
    np.random.seed(12345)
    xihistory = np.zeros((2,nobs))
    for i in range(1,nobs):
        xihistory[:,i] = np.dot(F,xihistory[:,i-1]) + \
                np.dot(cholQ,np.random.randn(2,1)).squeeze()
                # this makes an ARMA process?
                # check notes, do the math
    y = np.dot(H.T, xihistory)

    params = np.array([rho, sigma1, sigma2])
    penalty = 1e5
    upperbounds = np.array([.999, 100, 100])
    lowerbounds = np.array([-.999, .001, .001])
    xi10 = xihistory[:,0]
    ntrain = 1
    bounds = zip(lowerbounds,upperbounds) # if you use fmin_l_bfgs_b
    results = optimize.fmin_bfgs(kalmanupdate, params,
        args=(y,xi10,ntrain,penalty,upperbounds,lowerbounds),
        gtol = 1e-8, epsilon=1e-10)
