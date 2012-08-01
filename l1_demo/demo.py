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
    Both l1 solvers run into issues when the number of data points is  small.  
    Ironically...the solvers run into issues precisely when the number of 
    data points is small enough that regularization helps...
    """
    #### Commonly adjusted params
    N = 5000 # Number of data points
    num_targets = 3  # Targets are the dependent variables
    num_nonconst_covariates = 20 # For every target
    num_zero_params = 5 # For every target
    # The regularization parameter
    # Here we scale it with N for simplicity.  In practice, you should
    # use cross validation to pick alpha
    alpha = 0.0005 * N * sp.ones((num_nonconst_covariates+1, num_targets))
    alpha[0,:] = 0  # Don't regularize the intercept
    # Correlation length for the independent variables
    # Higher makes the problem more ill-posed, and easier to screw
    # up with noise.
    cor_length = 10 
    noise_level = 0.2  # As a fraction of the "signal"

    #### Make the arrays
    exog = get_exog(N, num_nonconst_covariates, cor_length) 
    exog = sm.add_constant(exog, prepend=True)
    true_params = sp.rand(num_nonconst_covariates+1, num_targets)
    if num_zero_params:
        true_params[-num_zero_params:, :] = 0
    # TODO Add noise to endog
    endog = get_multinomial_endog(num_targets, true_params, exog, noise_level)
    #### Use these lines to save results and try again with new alpha
    #sp.save('endog.npy', endog)
    #sp.save('exog.npy', exog)
    #sp.save('true_params.npy', true_params)
    #endog = sp.load('endog.npy')
    #exog = sp.load('exog.npy')
    #true_params = sp.load('true_params.npy')
    #### Train the models
    model = sm.MNLogit(endog, exog)
    results_ML = model.fit(method='newton')
    start_params = results_ML.params.ravel(order='F')
    results_l1_slsqp = model.fit(method='l1_slsqp', alpha=alpha, maxiter=70, 
            start_params=start_params, trim_params=True, retall=True)
    results_l1_cvxopt_cp = model.fit(method='l1_cvxopt_cp', alpha=alpha, 
            maxiter=70, start_params=start_params, trim_params=True, 
            retall=True)
    #### Compute MSE
    MSE_ML = get_MSE(results_ML, true_params)
    MSE_l1_slsqp = get_MSE(results_l1_slsqp, true_params)
    MSE_l1_cvxopt_cp = get_MSE(results_l1_cvxopt_cp, true_params)
    #### Prints results
    print "MSEs:  ML = %.4f,  l1_slsqp = %.4f,  l1_cvxopt_cp = %.4f"%(
            MSE_ML, MSE_l1_slsqp, MSE_l1_cvxopt_cp)
    #print "The true parameters are \n%s"%true_params
    #print "\nML had a MSE of %f and the parameters are \n%s"%(
    #        MSE_ML, results_ML.params)
    #print "\nl1_slsqp had a MSE of %f and the parameters are \n%s"%(
    #        MSE_l1_slsqp, results_l1_slsqp.params)
    #print "\nl1_cvxopt_cp had a MSE of %f and the parameters are \n%s"%(
    #        MSE_l1_cvxopt_cp, results_l1_cvxopt_cp.params)
    #print "\n"
    #print "\nThe ML fit results are"
    #print results_ML.summary()
    #print "\nThe l1_slsqp fit results are"
    #print results_l1_slsqp.summary()
    #print "\nThe l1_cvxopt_cp fit results are"
    #print results_l1_cvxopt_cp.summary()

def get_MSE(results, true_params):
    raw_MSE = sp.sqrt(((results.params - true_params)**2).sum()) 
    param_norm = sp.sqrt((true_params**2).sum())
    return raw_MSE / param_norm

def get_multinomial_endog(num_targets, true_params, exog, noise_level):
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
