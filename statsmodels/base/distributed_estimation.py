from statsmodels.base.elastic_net import fit_elasticnet
from statsmodels.base.model import Results
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regress.linear_model import OLS
import statsmodels.base.wrapper as wrap
import scipy.optimize as opt
import numpy as np

"""
distributed regularized estimation.

Routines for fitting regression models using a distributed
approach outlined in 

Jason D. Lee, Qiang Liu, Yuekai Sun and Jonathan E. Taylor.
"Communication-Efficient Sparse Regression: A One-Shot Approach." 
arXiv:1503.04337. 2015.
"""


# NOTE this is just a temporary function for testing
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


def _gen_grad(beta_hat, n, alpha, L1_wt, score_kwds):
    """
    generates the log-likelihood gradient for the debiasing

    Parameters
    ----------
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


def _gen_wdesign_mat(mod, beta_hat, n, hess_kwds):
    """
    generates the weighted design matrix necessary to generate
    the approximate inverse covariance matrix

    Parameters
    ----------
    mod : statsmodels model class
        The model for the current machine.
    beta_hat : array-like
        The estimated coefficients for the current machine.
    n : scalar
        sample size for current machine
    hess_kwds : dict-like or None
        Keyword arguments for the hessian function.

    Returns
    -------
    An array-like object, updated design matrix, same dimension
    as mod.exog
    """
    
    if isinstance(mod, OLS):
        return mod.exog
    if isinstance(mod, GLM):
        factor = mod.hessian_factor(np.r_[beta_hat], **hess_kwds)
        W = np.diag(factor)
        return W.dot(mod.exog)
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
    tmod = sm.OLS(X_beta[:, pi], X_beta[:, ind])

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


def _gen_dist_params(mod_gen, partitions, p, elastic_net_kwds,
                     score_kwds, hess_kwds):
    """
    generates lists of each of the parameters required to estimate
    the final coefficients

    Parameters
    ----------
    mod_gen : generator
        A generator object that yields a chunk of data to be used
        for the estimation.
    partitions : scalar
        The number of partitions that the data is split into.
    p : scalar
        Number of variables.
    elastic_net_kwds : dict_like or None
        Keyword arguments for the fit_regularized function.
    score_kwds : dict-like or None
        Keyword arguments for the score function.
    hess_kwds : dict-like or None
        Keyword arguments for the Hessian function.

    Returns
    -------
    A tuple of lists, the lists are as follows:
        beta_hat_l
        grad_l
        gamma_hat_l
        tau_hat_l
    """

    pi = 0
    part_ind = 0
    p_part = int(np.floor(p / partitions))
    
    beta_hat_l = []
    grad_l = []
    gamma_hat_l = []
    tau_hat_l = []

    # TODO, the handling of this is not great and will break in
    # some cases, we don't necessarily want the same alpha for
    # estimating theta_hat as we use for beta_hat
    alpha = elastic_net_kwds["alpha"]
    L1_wt = elastic_net_kwds["L1_wt"]

    for mod in mod_gen:

        n = mod.exog.shape[0]

        # estimate beta_hat
        # TODO probably should refer to elastic_net_kwds as something
        # else in case other methods are added, regularized_kwds maybe
        beta_hat = mod.fit_regularized(**elastic_net_kwds).params
        beta_hat_l.append(beta_hat)
        
        # generate gradient
        grad = _gen_grad(beta_hat, n, alpha, L1_wt, score_kwds)
        grad_l.append(grad)

        # generate weighted design matrix
        X_beta = _gen_wdesign_mat(mod, beta_hat, n, hess_kwds)

        # now we loop over the subset of variables assigned to the
        # current machine and estimate gamma_hat and tau_hat for
        # each variable
        for pj in range(pi, min(pi + p_part, p)):
            
            # generate gamma_hat
            gamma_hat = _gen_gamma_hat(X_beta, pj, p, n, alpha)
            gamma_hat_l.append(gamma_hat)

            # generate tau_hat
            tau_hat = _gen_tau_hat(X_beta, gamma_hat, pj, p, n, alpha)
            tau_hat_l.append(tau_hat)

        pi += p_part
        part_ind += 1

    return beta_hat_l, grad_l, gamma_hat_l, tau_hat_l


def _gen_debiased_params(beta_hat_l, grad_l, gamma_hat_l, tau_hat_l,
                         partitions, p, threshold):
    """
    """

    beta_mn = np.zeros(p)
    for beta_hat in beta_hat_l:
        beta_mn += beta_hat
    beta_mn = beta_mn / partitions

    grad_mn = np.zeros(p)
    for grad in grad_l:
        grad_mn += grad
    grad_mn = grad_mn / (partitions ** 2)

    theta_hat = _gen_theta_hat(gamma_hat_l, tau_hat_l, p)

    beta_tilde = beta_mn + theta_hat.dot(grad_mn)

    beta_tilde[np.abs(beta_tilde) < threshold] = 0

    return beta_tilde


def fit_distributed(model, generator=None, partitions=1, threshold=0.,
                    elastic_net_kwds=None, score_kwds=None, hess_kwds=None):
    """
    Returns an elastic net regularized fit to a regression model
    estimated sequentially over a series of partiions
    """

    # TODO the handling of p may change when we better integrate the
    # generators
    p = model.exog.shape[1]
    if generator is None:
        mod_gen = _generator(model, partitions)
    else:
        mod_gen = generator(model, partitions)

    dres = _gen_dist_params(mod_gen, partitions, p, elastic_net_kwds,
                            score_kwds, hess_kwds)
    beta_hat_l, grad_l, gamma_hat_l, tau_hat_l = dres

    beta_tilde = _gen_debiased_params(beta_hat_l, grad_l, gamma_hat_l,
                                      partitions, p, threshold) 

    return DistributedResults(model, beta_tilde)


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
