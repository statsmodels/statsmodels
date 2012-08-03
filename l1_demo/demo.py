import statsmodels.api as sm
import scipy as sp
from scipy import linalg
import statsmodels.discrete.l1 as l1
import pdb
from scipy.optimize import fmin_slsqp
# pdb.set_trace()



def main():
    """
    Demonstrates l1 regularization for MNLogit model.

    How to use this demo
    --------------------
    Adjust the "Commonly adjusted params" at the top, then run with 
        $ python demo.py

    The Story
    ---------
    The maximum likelihood (ML) solution works well when the number of data 
    points is large and the noise is small.  When the ML solution starts 
    "breaking", the regularized solution will do better.

    The l1 Solvers
    --------------
    The solvers often converge with errors.  Nonetheless, the final solution
        makes sense and is often (see above) better than the ML solution.
    The l1_slsqp solver is included with scipy.  It sometimes has trouble when
        the data size is large.
    The l1_cvxopt_cp solver is part of CVXOPT.  It works well even for larger
        data sizes.
    """
    ##########################################################################
    #### Commonly adjusted params
    N = 500 # Number of data points
    num_targets = 3  # Targets are the dependent variables
    num_nonconst_covariates = 10 # For every target
    num_zero_params = 8 # For every target
    print_long_results = False
    get_l1_slsqp_results = True
    get_l1_cvxopt_results = True
    save_results = False
    load_old_results = False
    # The regularization parameter
    # Here we scale it with N for simplicity.  In practice, you should
    # use cross validation to pick alpha
    alpha = 0.01 * N * sp.ones((num_nonconst_covariates+1, num_targets))
    alpha[0,:] = 0  # Don't regularize the intercept
    # Correlation length for the independent variables
    # Higher makes the problem more ill-posed, and easier to screw
    # up with noise.
    # BEWARE:  With long correlation lengths, you often get a singular KKT
    # matrix (during the l1_cvxopt_cp fit)
    cor_length = 2 
    noise_level = 0.2  # As a fraction of the "signal"
    ##########################################################################

    #### Make the arrays
    exog = get_exog(N, num_nonconst_covariates, cor_length) 
    exog = sm.add_constant(exog, prepend=True)
    true_params = sp.rand(num_nonconst_covariates+1, num_targets)
    if num_zero_params:
        true_params[-num_zero_params:, :] = 0
    endog = get_multinomial_endog(num_targets, true_params, exog, noise_level)
    if save_results:
        sp.save('endog.npy', endog)
        sp.save('exog.npy', exog)
        sp.save('true_params.npy', true_params)
    if load_old_results:
        endog = sp.load('endog.npy')
        exog = sp.load('exog.npy')
        true_params = sp.load('true_params.npy')

    #### Train the models
    print_str = '=========== Brief Result Printout ============'
    model = sm.MNLogit(endog, exog)
    # Get ML results
    results_ML = model.fit(method='newton')
    MSE_ML = get_MSE(results_ML, true_params)
    print_str += '\n ML mean square error = %.4f'%MSE_ML
    # Get l1 results
    start_params = results_ML.params.ravel(order='F')
    if get_l1_slsqp_results:
        results_l1_slsqp = model.fit(method='l1_slsqp', alpha=alpha, 
                maxiter=70, start_params=start_params, trim_params=True, 
                retall=True)
        MSE_l1_slsqp = get_MSE(results_l1_slsqp, true_params)
        print_str += '\n l1_slsqp mean square error = %.4f'%MSE_l1_slsqp
    if get_l1_cvxopt_results:
        results_l1_cvxopt_cp = model.fit(method='l1_cvxopt_cp', alpha=alpha, 
                maxiter=50, start_params=start_params, trim_params=True, 
                retall=True, feastol=1e-5)
        MSE_l1_cvxopt_cp = get_MSE(results_l1_cvxopt_cp, true_params)
        print_str += '\n l1_cvxopt_cp mean square error = %.4f'%MSE_l1_cvxopt_cp
    #### Prints results
    print_str += '\n========== More detail ============='
    print_str += "\nTrue parameters: \n%s"%true_params
    if print_long_results:
        if get_l1_slsqp_results:
            print_str += '\n' + results_l1_slsqp.summary().as_text()
        if get_l1_cvxopt_results:
            print_str += '\n' + results_l1_cvxopt_cp.summary().as_text()
    else:
        if get_l1_slsqp_results:
            print_str += '\nThe l1_slsqp params are %s'%results_l1_slsqp.params
        if get_l1_cvxopt_results:
            print_str += '\nThe l1_cvxopt_cp params are %s'%results_l1_cvxopt_cp.params
    print print_str


def get_MSE(results, true_params):
    """
    Gets the (normalized) mean square error.
    """
    raw_MSE = sp.sqrt(((results.params - true_params)**2).sum()) 
    param_norm = sp.sqrt((true_params**2).sum())
    return raw_MSE / param_norm

def get_multinomial_endog(num_targets, true_params, exog, noise_level):
    """
    Gets an endogenous response that is consistent with the true_params,
        perturbed by noise at noise_level.
    """
    N = exog.shape[0]
    ### Create the probability of entering the different classes, 
    ### given exog and true_params
    # Create a model just to access its cdf method
    temp_endog = sp.random.randint(0, num_targets, size=N)
    model = sm.MNLogit(temp_endog, exog)
    Xdotparams = sp.dot(exog, true_params)
    noise = noise_level * sp.randn(*Xdotparams.shape)
    class_probabilities = model.cdf(Xdotparams + noise)
    
    ### Create the endog 
    cdf = class_probabilities.cumsum(axis=1) 
    endog = sp.zeros(N)
    for n in xrange(N):
        endog[n] = sp.searchsorted(cdf[n, :], sp.rand())

    return endog

def get_exog(N, num_nonconst_covariates, cor_length):
    """
    Returns an exog array with correlations determined by cor_length.
    The covariance matrix of exog will have (asymptotically, as 
    :math:'N\\to\\inf')
    .. math:: Cov[i,j] = \\exp(-|i-j| / cor_length)
    """
    ## Create the noiseless exog
    uncorrelated_exog = sp.randn(N, num_nonconst_covariates) 
    if cor_length == 0:
        exog = uncorrelated_exog
    else:
        cov_matrix = sp.zeros((num_nonconst_covariates, num_nonconst_covariates))
        j = sp.arange(num_nonconst_covariates)
        for i in xrange(num_nonconst_covariates):
            cov_matrix[i,:] = sp.exp(-sp.fabs(i-j) / cor_length)
        chol = linalg.cholesky(cov_matrix)  # cov_matrix = sp.dot(chol.T, chol)
        exog = sp.dot(uncorrelated_exog, chol)
    ## Return
    return exog
    


if __name__ == '__main__':
    main()
