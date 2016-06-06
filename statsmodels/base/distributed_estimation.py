from statsmodels.base.elastic_net import fit_elasticnet
from scipy.optimize import brent
import numpy as np


# NOTE this is just a temporary function for testing
def _generator(model, partitions):
    """
    yields the partitioned model
    """

    n_exog = model.exog.shape[0]
    n_part = np.floor(n_exog / partitions)

    ii = 0
    while ii < n_exog:
        jj = min(ii + n_part, n_exog)
        tmodel = model
        tmodel.exog = tmodel.exog[ii:jj]
        tmodel.endog = tmodel.endog[ii:jj, :]
        yield tmodel
        ii += n_part


def _gen_beta_hat_n_grad(mod_gen, alpha, L1_wt, elastic_net_kwds, score_kwds):
    """
    generates the beta hat list as well as a list of the gradients for the debiasing
    """

    beta_hat_l = []
    grad_l = []
    for mod in mod_gen:
        n = mod.endog.shape[0]
        beta_hat = fit_elasticnet(mod, alpha, L1_wt, **elastic_net_kwds).params
        grad = mod.grad(np.r_[beta_hat], **score_kwds) / n + alpha * (1 - L1_wt)
        beta_hat_l.append(beta_hat)
        grad_l.append(grad)

    return beta_hat_l, grad_l


def _gen_gamma_n_tau_hat(mod_gen, beta_hat_l, alpha, L1_wt, partitions, p, hess_kwds):
    """
    generates the gamma hat values for each parameter and the 
    corresponding tau values
    """

    pi = 0
    part_ind = 0
    gamma_l = []
    tau_l = []
    p_part = np.floor(p / partitions)
    for mod in mod_gen:
        X = mod.endog
        n = X.shape[0]
        beta_hat = beta_hat_l[part_ind]
        hess = -mod.hessian(np.r_[beta_hat], **hess_kwds) / n + alpha * (1 - L1_wt)
        X_beta = np.sqrt(hess).dot(X)
        for pj in range(pi, min(pi + p_part, p)):
            ind = range(pj) + range(pj + 1, p)
            func = lambda gamma: np.linalg.norm(X_beta[:, pj] - X_beta[:, ind].dot(gamma)) + alpha * np.linalg.norm(gamma, 1)
            gamma_hat_j = brent(func)
            tau_hat_j = np.sqrt(func(gamma_hat_j) / n)
            gamma_hat_l.append(gamma_hat_j)
            tau_hat_l.append(tau_hat_j)

    return gamma_hat_l, tau_hat_l


def _gen_theta_hat(gamma_hat_l, tau_hat_l, p):
    """
    generates theta hat
    """

    theta_hat = np.eye(p)
    for pi in range(p):
        ind = range(pi) + range(pi + 1, p)
        theta_hat[pi,ind] = - gamma_hat_l[pi] / tau_hat_l[pi]**2

    return theta_hat


def _gen_debiased_params(beta_hat_l, grad_l, gamma_hat_l, tau_hat_l, partitions, threshold):
    """
    performs the debiasing
    """

    beta_mn = np.sum(beta_hat_l) / partitions

    grad_mn = np.sum(grad_l) / partitions

    theta_hat = _gen_theta_hat(gamma_hat_l, tau_hat_l)

    beta_tilde = beta_mn + theta_hat.dot(grad_mn)    
        
    beta_tilde[np.abs(beta_tilde) < threshold] = 0

    return beta_tilde


def fit_distributed(model, partitions=1, threshold=0, alpha=0., L1_wt=1.,
                    elastic_net_kwds=None, score_kwds=None, hess_kwds=None):
    """
    Returns an elastic net regularized fit to a regression model
    estimated sequentially over a series of partiions
    """

    N, p = model.endog.shape
    gen_mod = _generator(model, partitions)

    beta_hat_l, grad_l = _gen_beta_hat_n_grad(mod_gen, elastic_net_kwds, score_kwds)
    gamma_hat_l, tau_hat_l = _gen_gamma_n_tau_hat(mod_gen, beta_hat_l, alpha, L1_wt, partitions, p, hess_kwds)

    beta_tilde = _gen_debiased_params(beta_hat_l, grad_l, gamma_hat_l, tau_hat_l, partitions, threshold)

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
