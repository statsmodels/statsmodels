import statsmodels.api as sm
import scipy as sp
import statsmodels.discrete.l1 as l1
import pdb
from scipy.optimize import fmin_slsqp

# pdb.set_trace()

def main():
    N = 100
    num_nonconst_params = 10 # Not including the leading constant
    num_targets = 3
    prepend_constant = True
    exog = sp.rand(N, num_nonconst_params)
    if prepend_constant:
        exog = sm.add_constant(exog, prepend=True)
    true_params = sp.rand(exog.shape[1], num_targets)
    endog = get_endog(num_targets, true_params, exog)

    #endog = sp.random.randint(0, num_targets, size=N)
    #endog = 2*sp.ones(N)
    #endog[-1] = 0
    #endog[-2] = 1
    sp.save('exog.npy', exog)
    sp.save('endog.npy', endog)
    exog = sp.load('exog.npy')
    endog = sp.load('endog.npy')

    model = sm.MNLogit(endog, exog)
    results = model.fit(method='newton')
    print "Newton results"
    print results.summary()
    x0 = results.params.reshape((exog.shape[1])*(num_targets-1), order='F')
    results = model.fit(method='l1', alpha=0.5, epsilon=1.5e-8, maxiter=70, constant=True, trim_params=True, start_params=x0)
    bic = l1.modified_bic(results, model, prepend_constant)
    pdb.set_trace()
    print "l1 results"
    print results.summary()


def get_endog(num_targets, true_params, exog):
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
