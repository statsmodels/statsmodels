import statsmodels.api as sm
import scipy as sp
import statsmodels.discrete.l1 as l1
import pdb
from scipy.optimize import fmin_slsqp
# pdb.set_trace()



def main():
    """
    Demonstrates l1 regularization for MNLogit model.
    """
    ## Commonly adjusted params
    N = 100000  # Number of data points
    num_targets = 3  # Targets are the dependent variables
    num_nonconst_covariates = 4 # For every target
    num_zero_params = 2 # For every target
    ## Make the arrays
    exog = sp.rand(N, num_nonconst_covariates)
    exog = sm.add_constant(exog, prepend=True)
    true_params = sp.rand(exog.shape[1], num_targets)
    true_params[-num_zero_params:, :] = 0
    alpha = 0.0005 * N * sp.ones(true_params.shape) # Regularization parameter
    alpha[0,:] = 0  # Don't regularize the intercept
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
    results_l1 = model.fit(method='l1', alpha=alpha, maxiter=70, 
            start_params=start_params, trim_params=True)
    ## Prints results
    print "The true parameters are \n%s"%true_params
    print "The ML fit parameters are \n%s"%results_ML.params
    print "The l1 fit parameters are \n%s"%results_l1.params
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



if __name__ == '__main__':
    main()
