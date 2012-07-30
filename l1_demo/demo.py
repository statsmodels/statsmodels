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
    N = 10000  # Number of data points
    alpha = 0.005 * N  # Regularization parameter
    num_nonconst_params = 2 
    num_targets = 2
    prepend_constant = True
    ## Make the arrays
    exog = sp.rand(N, num_nonconst_params)
    if prepend_constant:
        exog = sm.add_constant(exog, prepend=True)
    true_params = sp.rand(exog.shape[1], num_targets)
    true_params[-1:] = 0
    endog = get_multinomial_endog(num_targets, true_params, exog)
    ## Use these lines to save results and try again with new alpha
    #sp.save('endog.npy', endog)
    #sp.save('exog.npy', exog)
    #endog = sp.load('endog.npy')
    #exog = sp.load('exog.npy')
    ## Train the models
    model = sm.MNLogit(endog, exog)
    results_ML = model.fit(method='newton')
    results_l1 = model.fit(method='l1', alpha=alpha, maxiter=70, 
            constant=prepend_constant, trim_params=True)
    ## Prints results
    print "The true parameters are \n%s"%true_params
    print "The ML fit parameters are \n%s"%results_ML.params
    print "The l1 fit parameters are \n%s"%results_l1.params
    print "\n"
    print "The ML fit results are"
    print results_ML.summary()
    print "The l1 fit results are"
    print results_l1.summary()


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
