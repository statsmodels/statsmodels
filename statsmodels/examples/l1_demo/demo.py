from optparse import OptionParser

import scipy as sp
from scipy import linalg, stats

import statsmodels.api as sm

docstr = """
Demonstrates l1 regularization for likelihood models.
Use different models by setting mode = mnlogit, logit, or probit.

Examples
-------
$ python demo.py --get_l1_slsqp_results  logit

>>> import demo
>>> demo.run_demo('logit')

The Story
---------
The maximum likelihood (ML) solution works well when the number of data
points is large and the noise is small.  When the ML solution starts
"breaking", the regularized solution should do better.

The l1 Solvers
--------------
The solvers are slower than standard Newton, and sometimes have
    convergence issues Nonetheless, the final solution makes sense and
    is often better than the ML solution.
The standard l1 solver is fmin_slsqp and is included with scipy.  It
    sometimes has trouble verifying convergence when the data size is
    large.
The l1_cvxopt_cp solver is part of CVXOPT and this package needs to be
    installed separately.  It works well even for larger data sizes.
"""


def main():
    """
    Provides a CLI for the demo.
    """
    usage = "usage: %prog [options] mode"
    usage += '\n'+docstr
    parser = OptionParser(usage=usage)
    # base_alpha
    parser.add_option("-a", "--base_alpha",
            help="Size of regularization param (the param actully used will "\
                    "automatically scale with data size in this demo) "\
                    "[default: %default]",
            dest='base_alpha', action='store', type='float', default=0.01)
    # num_samples
    parser.add_option("-N", "--num_samples",
            help="Number of data points to generate for fit "\
                    "[default: %default]",
            dest='N', action='store', type='int', default=500)
    # get_l1_slsqp_results
    parser.add_option("--get_l1_slsqp_results",
            help="Do an l1 fit using slsqp. [default: %default]", \
            action="store_true",dest='get_l1_slsqp_results', default=False)
    # get_l1_cvxopt_results
    parser.add_option("--get_l1_cvxopt_results",
            help="Do an l1 fit using cvxopt. [default: %default]", \
            action="store_true",dest='get_l1_cvxopt_results', default=False)
    # num_nonconst_covariates
    parser.add_option("--num_nonconst_covariates",
            help="Number of covariates that are not constant "\
                    "(a constant will be prepended) [default: %default]",
                    dest='num_nonconst_covariates', action='store',
                    type='int', default=10)
    # noise_level
    parser.add_option("--noise_level",
            help="Level of the noise relative to signal [default: %default]",
                    dest='noise_level', action='store', type='float',
                    default=0.2)
    # cor_length
    parser.add_option("--cor_length",
            help="Correlation length of the (Gaussian) independent variables"\
                    "[default: %default]",
                    dest='cor_length', action='store', type='float',
                    default=2)
    # num_zero_params
    parser.add_option("--num_zero_params",
            help="Number of parameters equal to zero for every target in "\
                    "logistic regression examples.  [default: %default]",
                    dest='num_zero_params', action='store', type='int',
                    default=8)
    # num_targets
    parser.add_option("-J", "--num_targets",
            help="Number of choices for the endogenous response in "\
                    "multinomial logit example [default: %default]",
                    dest='num_targets', action='store', type='int', default=3)
    # print_summaries
    parser.add_option("-s", "--print_summaries",
            help="Print the full fit summary. [default: %default]", \
            action="store_true",dest='print_summaries', default=False)
    # save_arrays
    parser.add_option("--save_arrays",
            help="Save exog/endog/true_params to disk for future use. "\
                    "[default: %default]",
                    action="store_true",dest='save_arrays', default=False)
    # load_old_arrays
    parser.add_option("--load_old_arrays",
            help="Load exog/endog/true_params arrays from disk.  "\
                    "[default: %default]",
                    action="store_true",dest='load_old_arrays', default=False)

    (options, args) = parser.parse_args()

    assert len(args) == 1
    mode = args[0].lower()

    run_demo(mode, **options.__dict__)


def run_demo(mode, base_alpha=0.01, N=500, get_l1_slsqp_results=False,
        get_l1_cvxopt_results=False, num_nonconst_covariates=10,
        noise_level=0.2, cor_length=2, num_zero_params=8, num_targets=3,
        print_summaries=False, save_arrays=False, load_old_arrays=False):
    """
    Run the demo and print results.

    Parameters
    ----------
    mode : str
        either 'logit', 'mnlogit', or 'probit'
    base_alpha :  Float
        Size of regularization param (the param actually used will
        automatically scale with data size in this demo)
    N : int
        Number of data points to generate for fit
    get_l1_slsqp_results : bool,
        Do an l1 fit using slsqp.
    get_l1_cvxopt_results : bool
        Do an l1 fit using cvxopt
    num_nonconst_covariates : int
        Number of covariates that are not constant
        (a constant will be prepended)
    noise_level : float (non-negative)
        Level of the noise relative to signal
    cor_length : float (non-negative)
        Correlation length of the (Gaussian) independent variables
    num_zero_params : int
        Number of parameters equal to zero for every target in logistic
        regression examples.
    num_targets : int
        Number of choices for the endogenous response in multinomial logit
        example
    print_summaries : bool
        print the full fit summary.
    save_arrays : bool
        Save exog/endog/true_params to disk for future use.
    load_old_arrays
        Load exog/endog/true_params arrays from disk.
    """
    if mode != 'mnlogit':
        print("Setting num_targets to 2 since mode != 'mnlogit'")
        num_targets = 2
    models = {
            'logit': sm.Logit, 'mnlogit': sm.MNLogit, 'probit': sm.Probit}
    endog_funcs = {
            'logit': get_logit_endog, 'mnlogit': get_logit_endog,
            'probit': get_probit_endog}
    # The regularization parameter
    # Here we scale it with N for simplicity.  In practice, you should
    # use cross validation to pick alpha
    alpha = base_alpha * N * sp.ones((num_nonconst_covariates+1, num_targets-1))
    alpha[0,:] = 0  # Do not regularize the intercept

    #### Make the data and model
    exog = get_exog(N, num_nonconst_covariates, cor_length)
    exog = sm.add_constant(exog)
    true_params = sp.rand(num_nonconst_covariates+1, num_targets-1)
    if num_zero_params:
        true_params[-num_zero_params:, :] = 0
    endog = endog_funcs[mode](true_params, exog, noise_level)

    endog, exog, true_params = save_andor_load_arrays(
            endog, exog, true_params, save_arrays, load_old_arrays)
    model = models[mode](endog, exog)

    #### Get the results and print
    results = run_solvers(model, true_params, alpha,
            get_l1_slsqp_results, get_l1_cvxopt_results, print_summaries)

    summary_str = get_summary_str(results, true_params, get_l1_slsqp_results,
            get_l1_cvxopt_results, print_summaries)

    print(summary_str)


def run_solvers(model, true_params, alpha, get_l1_slsqp_results,
        get_l1_cvxopt_results, print_summaries):
    """
    Runs the solvers using the specified settings and returns a result string.
    Works the same for any l1 penalized likelihood model.
    """
    results = {}
    #### Train the models
    # Get ML results
    results['results_ML'] = model.fit(method='newton')
    # Get l1 results
    start_params = results['results_ML'].params.ravel(order='F')
    if get_l1_slsqp_results:
        results['results_l1_slsqp'] = model.fit_regularized(
                method='l1', alpha=alpha, maxiter=1000,
                start_params=start_params, retall=True)
    if get_l1_cvxopt_results:
        results['results_l1_cvxopt_cp'] = model.fit_regularized(
                method='l1_cvxopt_cp', alpha=alpha, maxiter=50,
                start_params=start_params, retall=True, feastol=1e-5)

    return results


def get_summary_str(results, true_params, get_l1_slsqp_results,
        get_l1_cvxopt_results, print_summaries):
    """
    Gets a string summarizing the results.
    """
    #### Extract specific results
    results_ML = results['results_ML']
    RMSE_ML = get_RMSE(results_ML, true_params)
    if get_l1_slsqp_results:
        results_l1_slsqp = results['results_l1_slsqp']
    if get_l1_cvxopt_results:
        results_l1_cvxopt_cp = results['results_l1_cvxopt_cp']

    #### Format summaries
    # Short summary
    print_str = '\n\n=========== Short Error Summary ============'
    print_str += '\n\n The maximum likelihood fit RMS error = %.4f' % RMSE_ML
    if get_l1_slsqp_results:
        RMSE_l1_slsqp = get_RMSE(results_l1_slsqp, true_params)
        print_str += '\n The l1_slsqp fit RMS error = %.4f' % RMSE_l1_slsqp
    if get_l1_cvxopt_results:
        RMSE_l1_cvxopt_cp = get_RMSE(results_l1_cvxopt_cp, true_params)
        print_str += '\n The l1_cvxopt_cp fit RMS error = %.4f' % RMSE_l1_cvxopt_cp
    # Parameters
    print_str += '\n\n\n============== Parameters ================='
    print_str += "\n\nTrue parameters: \n%s" % true_params
    # Full summary
    if print_summaries:
        print_str += '\n' + results_ML.summary().as_text()
        if get_l1_slsqp_results:
            print_str += '\n' + results_l1_slsqp.summary().as_text()
        if get_l1_cvxopt_results:
            print_str += '\n' + results_l1_cvxopt_cp.summary().as_text()
    else:
        print_str += '\n\nThe maximum likelihood params are \n%s' % results_ML.params
        if get_l1_slsqp_results:
            print_str += '\n\nThe l1_slsqp params are \n%s' % results_l1_slsqp.params
        if get_l1_cvxopt_results:
            print_str += '\n\nThe l1_cvxopt_cp params are \n%s' % \
                    results_l1_cvxopt_cp.params
    # Return
    return print_str


def save_andor_load_arrays(
        endog, exog, true_params, save_arrays, load_old_arrays):
    if save_arrays:
        sp.save('endog.npy', endog)
        sp.save('exog.npy', exog)
        sp.save('true_params.npy', true_params)
    if load_old_arrays:
        endog = sp.load('endog.npy')
        exog = sp.load('exog.npy')
        true_params = sp.load('true_params.npy')
    return endog, exog, true_params


def get_RMSE(results, true_params):
    """
    Gets the (normalized) root mean square error.
    """
    diff = results.params.reshape(true_params.shape) - true_params
    raw_RMSE = sp.sqrt(((diff)**2).sum())
    param_norm = sp.sqrt((true_params**2).sum())
    return raw_RMSE / param_norm


def get_logit_endog(true_params, exog, noise_level):
    """
    Gets an endogenous response that is consistent with the true_params,
        perturbed by noise at noise_level.
    """
    N = exog.shape[0]
    ### Create the probability of entering the different classes,
    ### given exog and true_params
    Xdotparams = sp.dot(exog, true_params)
    eXB = sp.column_stack((sp.ones(len(Xdotparams)), sp.exp(Xdotparams)))
    class_probabilities = eXB / eXB.sum(1)[:, None]

    ### Create the endog
    cdf = class_probabilities.cumsum(axis=1)
    endog = sp.zeros(N)
    for i in range(N):
        endog[i] = sp.searchsorted(cdf[i, :], sp.rand())

    return endog


def get_probit_endog(true_params, exog, noise_level):
    """
    Gets an endogenous response that is consistent with the true_params,
        perturbed by noise at noise_level.
    """
    N = exog.shape[0]
    ### Create the probability of entering the different classes,
    ### given exog and true_params
    Xdotparams = sp.dot(exog, true_params)

    ### Create the endog
    cdf = stats.norm._cdf(-Xdotparams)
    endog = sp.zeros(N)
    for i in range(N):
        endog[i] = sp.searchsorted(cdf[i, :], sp.rand())

    return endog


def get_exog(N, num_nonconst_covariates, cor_length):
    """
    Returns an exog array with correlations determined by cor_length.
    The covariance matrix of exog will have (asymptotically, as
    :math:'N\\to\\inf')
    .. math:: Cov[i,j] = \\exp(-|i-j| / cor_length)

    Higher cor_length makes the problem more ill-posed, and easier to screw
        up with noise.
    BEWARE:  With very long correlation lengths, you often get a singular KKT
        matrix (during the l1_cvxopt_cp fit)
    """
    ## Create the noiseless exog
    uncorrelated_exog = sp.randn(N, num_nonconst_covariates)
    if cor_length == 0:
        exog = uncorrelated_exog
    else:
        cov_matrix = sp.zeros((num_nonconst_covariates, num_nonconst_covariates))
        j = sp.arange(num_nonconst_covariates)
        for i in range(num_nonconst_covariates):
            cov_matrix[i,:] = sp.exp(-sp.fabs(i-j) / cor_length)
        chol = linalg.cholesky(cov_matrix)  # cov_matrix = sp.dot(chol.T, chol)
        exog = sp.dot(uncorrelated_exog, chol)
    ## Return
    return exog



if __name__ == '__main__':
    main()
