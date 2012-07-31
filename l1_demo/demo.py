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
    """
    ## Commonly adjusted params
    N = 10000 # Number of data points
    num_targets = 2  # Targets are the dependent variables
    num_nonconst_covariates = 3 # For every target
    num_zero_params = 1 # For every target
    cor_length = 1 # Correlation length for the independent variables
    noise_level = 0.05  # As a fraction of the "signal"
    ## Make the arrays
    exog = get_exog(N, num_nonconst_covariates, cor_length) 
    exog = sm.add_constant(exog, prepend=True)
    true_params = sp.rand(exog.shape[1], num_targets)
    if num_zero_params:
        true_params[-num_zero_params:, :] = 0
    alpha = 0.005 * N * sp.ones(true_params.shape) # Regularization parameter
    alpha[0,:] = 0  # Don't regularize the intercept
    # TODO Add noise to endog
    endog = get_multinomial_endog(num_targets, true_params, exog)
    ## Use these lines to save results and try again with new alpha
    #sp.save('endog.npy', endog)
    #sp.save('exog.npy', exog)
    #endog = sp.load('endog.npy')
    #exog = sp.load('exog.npy')
    ## Train the models
    model = sm.MNLogit(endog, exog)
    results_ML = model.fit(method='newton')
    start_params = results_ML.params.ravel(order='F')
    results_l1_slsqp = model.fit(method='l1_slsqp', alpha=alpha, maxiter=70, 
            start_params=start_params, trim_params=True)
    results_l1_cvxopt_cp = model.fit(method='l1_cvxopt_cp', alpha=alpha, 
            maxiter=70, start_params=start_params, trim_params=True, 
            retall=True)
    ## Compute MSE
    MSE_ML = sp.sqrt(((results_ML.params - true_params)**2).sum())
    MSE_l1_slsqp = sp.sqrt(((results_l1_slsqp.params - true_params)**2).sum())
    MSE_l1_cvxopt_cp = sp.sqrt(((results_l1_cvxopt_cp.params - true_params)**2).sum())
    ## Prints results
    print "The true parameters are \n%s"%true_params
    print "\nML had a MSE of %f and the parameters are \n%s"%(
            MSE_ML, results_ML.params)
    print "\nl1_slsqp had a MSE of %f and the parameters are \n%s"%(
            MSE_l1_slsqp, results_l1_slsqp.params)
    print "\nl1_cvxopt_cp had a MSE of %f and the parameters are \n%s"%(
            MSE_l1_cvxopt_cp, results_l1_cvxopt_cp.params)
    #print "\n"
    #print "\nThe ML fit results are"
    #print results_ML.summary()
    #print "\nThe l1 fit results are"
    #print results_l1.summary()

def get_multinomial_endog(num_targets, true_params, exog):
    N = exog.shape[0]
    ### Create the probability of entering the different classes, 
    ### given exog and true_params
    # Create a model just to access its cdf method
    temp_endog = sp.random.randint(0, num_targets, size=N)
    model = sm.MNLogit(temp_endog, exog)
    class_probabilities = model.cdf(sp.dot(exog, true_params))
    
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
