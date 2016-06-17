from statsmodels.base.elastic_net import fit_elasticnet
from statsmodels.base.model import Results
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
import statsmodels.base.wrapper as wrap
import numpy as np

"""
distributed regularized estimation.

Routines for fitting regression models using a distributed
approach outlined in 

Jason D. Lee, Qiang Liu, Yuekai Sun and Jonathan E. Taylor.
"Communication-Efficient Sparse Regression: A One-Shot Approach." 
arXiv:1503.04337. 2015.
"""


# TODO this is just a temporary function for testing it is
# also currently out of date
def _generator(model, partitions):
    """
    yields the partitioned model
    """

    n_exog = model.exog.shape[0]
    n_part = np.floor(n_exog / partitions)

    # TODO this is memory inefficient and should be
    # fixed
    exog = model.exog.copy()
    endog = model.endog.copy()

    ii = 0
    while ii < n_exog:
        jj = int(min(ii + n_part, n_exog))
        tmodel = model
        tmodel.exog = exog[ii:jj, :].copy()
        tmodel.endog = endog[ii:jj].copy()
        yield tmodel
        ii += int(n_part)


def _gen_grad(mod, beta_hat, n, alpha, L1_wt, score_kwds):
    """
    generates the log-likelihood gradient for the debiasing

    Parameters
    ----------
    mod : statsmodels model class
        The model for the current machine.
    beta_hat : array-like
        The estimated coefficients for the current machine.
    n : scalar
        machine specific sample size.
    alpha : scalar or array-like
        The penalty weight.  If a scalar, the same penalty weight
        applies to all variables in the model.  If a vector, it
        must have the same length as `params`, and contains a
        penalty weight for each coefficient.
    L1_wt : scalar
        The fraction of the penalty given to the L1 penalty term.
        Must be between 0 and 1 (inclusive).  If 0, the fit is
        a ridge fit, if 1 it is a lasso fit.
    score_kwds : dict-like or None
        Keyword arguments for the score function.
    
    Returns
    -------
    An array-like object of the same dimension as beta_hat
   
    Notes
    -----
    In general:
    
    nabla l_k(beta)

    For the simple linear case:

    X^T(y - X^T beta) / n
    """

    grad = -mod.score(np.r_[beta_hat], **score_kwds) / n
    # this second part comes from the elastic net penalty
    grad += alpha * (1 - L1_wt)
    return grad


def _gen_wdesign_mat(mod, beta_hat, hess_kwds):
    """
    generates the weighted design matrix necessary to generate
    the approximate inverse covariance matrix

    Parameters
    ----------
    mod : statsmodels model class
        The model for the current machine.
    beta_hat : array-like
        The estimated coefficients for the current machine.
    hess_kwds : dict-like or None
        Keyword arguments for the hessian function.

    Returns
    -------
    An array-like object, updated design matrix, same dimension
    as mod.exog
    """
   
    return mod.hessian_obs(np.r_[beta_hat], **hess_kwds)
    # TODO need to handle duration and other linear model classes


def _gen_gamma_hat(X_beta, pi, p, n, alpha):
    """
    generates the gamma hat values for the pith variable, used to
    estimate theta hat.

    Parameters
    ----------
    X_beta : array-like
        The weighted design matrix for the current machine
    pi : scalar
        Index of the current variable
    p : scalar
        Number of variables
    n : scalar
        sample size of current machine
    alpha : scalar or array-like
        The penalty weight.  If a scalar, the same penalty weight
        applies to all variables in the model.  If a vector, it
        must have the same length as `params`, and contains a
        penalty weight for each coefficient.

    Returns
    -------
    An array-like object of length p-1

    Notes
    -----

    gamma_hat_i = arg min 1/(2n) ||X_beta,i - X_beta,-i gamma||_2^2
                          + alpha ||gamma||_1
    """
   
    ind = range(pi) + range(pi + 1, p)

    # TODO use elastic net optimization routine directly
    tmod = OLS(X_beta[:, pi], X_beta[:, ind])

    #def func(gamma):
    #    d = np.linalg.norm(X_beta[:, pi] - X_beta[:, ind].dot(gamma))**2
    #    d = d / (2 * n) + alpha * np.linalg.norm(gamma, 1)
    #    return d

    gamma_hat = tmod.fit_regularized(alpha=alpha).params
    #gamma_hat = opt.minimize(func, np.ones(p - 1)).x
    return gamma_hat


def _gen_tau_hat(X_beta, gamma_hat, pi, p, n, alpha):
    """
    generates the tau hat value for the pith variable, used to
    estimate theta hat.

    Parameters
    ----------
    X_beta : array-like
        The weighted design matrix for the current machine
    gamma_hat : array-like
        The gamma_hat values for the current variable
    pi : scalar
        Index of the current variable
    p : scalar
        Number of variables
    n : scalar
        sample size of current machine
    alpha : scalar or array-like
        The penalty weight.  If a scalar, the same penalty weight
        applies to all variables in the model.  If a vector, it
        must have the same length as `params`, and contains a
        penalty weight for each coefficient.

    Returns
    -------
    A scalar

    Notes
    -----

    tau_hat_i = sqrt(1/n ||X_beta,i - X_beta,-i gamma_hat||_2^2
                     + alpha ||gamma||_1)
    """

    ind = range(pi) + range(pi + 1, p)
    d = np.linalg.norm(X_beta[:, pi] - X_beta[:, ind].dot(gamma_hat))**2
    d = np.sqrt(d / n + alpha * np.linalg.norm(gamma_hat, 1))
    return d


def _gen_theta_hat(gamma_hat_l, tau_hat_l, p):
    """
    generates the theta hat matrix

    Parameters
    ----------
    gamma_hat_l : list
        A list of array-like object where each object corresponds to
        the gamma hat values for the corresponding variable, should 
        be length p.
    tau_hat_l : list
        A list of scalars where each scalar corresponds to the tau hat
        value for the corresponding variable, should be length p.
    p : scalar
        Number of variables

    Returns
    ------
    An array-like object, p x p matrix
   
    Notes
    -----

    theta_hat_j = - 1 / tau_hat_j [gamma_hat_j,1,...,1,...gamma_hat_j,p]
    """
    
    theta_hat = np.eye(p)
    for pi in range(p):
        ind = range(pi) + range(pi + 1, p)
        theta_hat[pi,ind] = gamma_hat_l[pi]
        theta_hat[pi,:] = (- 1. / tau_hat_l[pi]**2) * theta_hat[pi,:]

    return theta_hat


def _est_regularized_distributed(mod, mnum, partitions, score_kwds=None,
                                 hess_kwds=None, fit_kwds=None):
    """
    generates the regularized fitted parameters, is the default
    estimation_method for distributed_estimation

    Parameters
    ----------
    mod : statsmodels model class
        The model for the current machine.
    mnum : scalar
        Index of current machine
    partitions : scalar
        Total number of machines
    score_kwds : dict-like or None
        Keyword arguments for the score function.
    hess_kwds : dict-like or None
        Keyword arguments for the Hessian function.
    fit_kwds : dict-like or None
        Keyword arguments to be given to fit_regularized

    Returns
    -------
    A tuple of paramters for regularized fit
        An array-like object of the fitted parameters, beta hat
        An array-like object for the gradient
        A list of array like objects for gamma hat
        A list of array like objects for tau hat
    """

    # TODO this is currently going to fail with some alpha
    # values
    alpha = fit_kwds["alpha"]
    L1_wt = fit_kwds["L1_wt"]

    n, p = mod.exog.shape
    p_part = int(np.floor(p / partitions)

    beta_hat = mod.fit_regularized(**fit_kwargs).params
    grad = _gen_grad(mod, beta_hat, n, alpha, L1_wt, score_kwds)

    X_beta = _gen_wdesign_mat(mod, beta_hat, hess_kwds)

    gamma_hat_l = []
    tau_hat_l = []
    for pi in range(mnum * p_part, min((mnum + 1) * p_part, p)):
        
        gamma_hat = _gen_gamma_hat(X_beta, pi, p, n, alpha)
        gamma_hat_l.append(gamma_hat)

        tau_hat = _gen_tau_hat(X_beta, gamma_hat, pi, p, n, alpha)
        tau_hat_l.append(tau_hat)

    return beta_hat, grad, gamma_hat_l, tau_hat_l


def _join_debiased(model_results_l, partitions, threshold=0):
    """
    """

    # TODO currently the way we extract p is roundabout, should be
    # handled better but ideally not as an argument to _join_debiased

    beta_hat_l = []
    grad_l = []
    gamma_hat_l = []
    tau_hat_l = []

    for r in model_results_l:
        beta_hat_l.append(r[0])
        grad_l.append(r[1])
        gamma_hat_l.extend(r[2])
        tau_hat_l.extend(r[3])

    p = len(gamma_hat_l)

    beta_mn = np.zeros(p)
    for beta_hat in beta_hat_l:
        beta_mn += beta_hat
    beta_mn = beta_mn / partitions

    grad_mn = np.zeros(p)
    for grad in grad_l:
        grad_mn += grad
    # TODO fix this
    grad_mn = grad_mn / (partitions ** 2)

    theta_hat = _gen_theta_hat(gamma_hat_l, tau_hat_l, p)

    beta_tilde = beta_mn + theta_hat.dot(grad_mn)

    beta_tilde[np.abs(beta_tilde) < threshold] = 0

    return beta_tilde


def distributed_estimation(endog_generator, exog_generator, partitions,
                           model_class=None, init_kwds=None, fit_kwds=None,
                           estimation_method=None, estimation_kwds=None,
                           join_method=None, join_kwargs=None):
    """
    This functions handles a general approach to distributed estimation,
    the user is expected to provide generators for the data as well as
    a model and methods for performing the estimation and recombining
    the results

    Paramters
    ---------

    Returns
    -------

    Notes
    -----

    """

    # set defaults
    if model_class is None:
        model_class = OLS

    if estimation_method is None:
        estimation_method = _est_regularized_distributed

    if join_method is None:
        join_method = _join_debiased

    model_results_l = []

    # index for machine
    mnum = 0

    # TODO given that we already have an example where generators should
    # produce more than just exog and endog (partition for variables)
    # this should probably be handled differently
    for endog, exog in zip(endog_generator, exog_generator):
        
        model = model_class(endog, exog, **init_kwds)

        # TODO possibly fit_kwds should be handled within
        # estimation_kwds to make more general?
        results = estimation_method(model, mnum, partitions, **estimation_kwds,
                                    fit_kwds)
        model_results_l.append(results)

        mnum += 1

    return join_method(model_results_l, **join_kwds)


# TODO This doesn't work with the current implementation, need
# to determine the best way to handle results
class DistributedResults(Results):

    def __init__(self, model, params):
        super(DistributedResults, self).__init__(model, params)

    @cache_readonly
    def fittedvalues(self):
        return self.model.predict(self.params)


class DistributedResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
        'params': 'columns',
        'resid': 'rows',
        'fittedvalues': 'rows',
    }

    _wrap_attrs = _attrs

wrap.populate_wrapper(DistributedResultsWrapper,
                      DistributedResults)
