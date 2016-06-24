from statsmodels.base.elastic_net import fit_elasticnet
from statsmodels.base.model import Results
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
#from statsmodels.genmod.generalized_linear_model import GLM
#from statsmodels.genmod.families import Binomial
import statsmodels.base.wrapper as wrap
import numpy as np

"""
distributed regularized estimation.

Routines for fitting regression models using a distributed
approach outlined in

Jason D. Lee, Qiang Liu, Yuekai Sun and Jonathan E. Taylor.
"Communication-Efficient Sparse Regression: A One-Shot Approach."
arXiv:1503.04337. 2015. http://arxiv.org/abs/1503.04337.

Currently the primary usecase is for out of memory computation,
this approach currently does not support mutli-core processing
and processing must be handled sequentially.
"""


def _gen_grad(mod, beta_hat, n, alpha, L1_wt, score_kwds):
    """generates the log-likelihood gradient for the debiasing

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
    """generates the weighted design matrix necessary to generate
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

    # TODO need to handle duration and other linear model classes
    return np.sqrt(mod.hessian_obs(np.r_[beta_hat], **hess_kwds)).dot(mod.exog)


def _gen_gamma_hat(X_beta, pi, p, n, alpha):
    """generates the gamma hat values for the pith variable, used to
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

    # TODO this is not a very elegant solution, we can't concat
    # range in Py 3 because it is now a generator.
    ind = list(range(pi)) + list(range(pi + 1, p))

    # TODO use elastic net optimization routine directly
    tmod = OLS(X_beta[:, pi], X_beta[:, ind])

    # TODO this is currently going to fail with some alpha
    # values, the issue is that may have an array for of
    # alphas for beta hat and we want each individual element
    # here
    gamma_hat = tmod.fit_regularized(alpha=alpha).params

    return gamma_hat


def _gen_tau_hat(X_beta, gamma_hat, pi, p, n, alpha):
    """generates the tau hat value for the pith variable, used to
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

    ind = list(range(pi)) + list(range(pi + 1, p))
    d = np.linalg.norm(X_beta[:, pi] - X_beta[:, ind].dot(gamma_hat))**2
    d = np.sqrt(d / n + alpha * np.linalg.norm(gamma_hat, 1))
    return d


def _gen_theta_hat(gamma_hat_l, tau_hat_l, p):
    """generates the theta hat matrix

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
        ind = list(range(pi)) + list(range(pi + 1, p))
        theta_hat[pi,ind] = gamma_hat_l[pi]
        theta_hat[pi,:] = (- 1. / tau_hat_l[pi]**2) * theta_hat[pi,:]

    return theta_hat


def _est_regularized_distributed(mod, mnum, partitions, fit_kwds=None,
                                 score_kwds=None, hess_kwds=None):
    """generates the regularized fitted parameters, is the default
    estimation_method for distributed_estimation

    Parameters
    ----------
    mod : statsmodels model class
        The model for the current machine.
    mnum : scalar
        Index of current machine
    partitions : scalar
        Total number of machines
    fit_kwds : dict-like or None
        Keyword arguments to be given to fit_regularized
    score_kwds : dict-like or None
        Keyword arguments for the score function.
    hess_kwds : dict-like or None
        Keyword arguments for the Hessian function.

    Returns
    -------
    A tuple of paramters for regularized fit
        An array-like object of the fitted parameters, beta hat
        An array-like object for the gradient
        A list of array like objects for gamma hat
        A list of array like objects for tau hat
    """

    score_kwds = {} if score_kwds is None else score_kwds
    hess_kwds = {} if hess_kwds is None else hess_kwds

    if fit_kwds is None:
        raise ValueError("_est_regularized_distributed currently " +
                         "requires that fit_kwds not be None.")
    else:
        alpha = fit_kwds["alpha"]

    if "L1_wt" in fit_kwds:
        L1_wt = fit_kwds["L1_wt"]
    else:
        L1_wt = 1

    n, p = mod.exog.shape
    p_part = int(np.ceil((1. * p) / partitions))

    beta_hat = mod.fit_regularized(**fit_kwds).params
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
    """joins the results from each run of _est_regularized_distributed
    and returns the debiased estimate of the coefficients

    Parameters
    ----------
    model_results_l : list
        A list of tuples each one containing the beta_hat, grad,
        gamma_hat and tau_hat values for each partition.
    partitions : scalar
        The number of partitions that the data will be split into.
    threshold : scalar
        The threshold at which the coefficients will be cut.
    """

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
    beta_mn /= partitions

    grad_mn = np.zeros(p)
    for grad in grad_l:
        grad_mn += grad
    grad_mn *= -1. / partitions

    theta_hat = _gen_theta_hat(gamma_hat_l, tau_hat_l, p)

    beta_tilde = beta_mn + theta_hat.dot(grad_mn)

    beta_tilde[np.abs(beta_tilde) < threshold] = 0

    return beta_tilde
#    return beta_tilde, beta_mn, theta_hat, grad_mn, gamma_hat_l, theta_hat.dot(grad_mn)


def distributed_estimation(data_generator, partitions,
                           model_class=None, init_kwds=None, fit_kwds=None,
                           estimation_method=None, estimation_kwds=None,
                           join_method=None, join_kwds=None):
    """This functions handles a general approach to distributed estimation,
    the user is expected to provide generators for the data as well as
    a model and methods for performing the estimation and recombining
    the results

    Parameters
    ---------
    data_generator : generator
        A generator that produces a sequence of tuples where the first
        element in the tuple corresponds to an endog array and the
        element corresponds to an exog array.
    partitions : scalar
        The number of partitions that the data will be split into.
    model_class : statsmodels model class
        The model class which will be used for estimation. If None
        this defaults to OLS.
    init_kwds : dict-like or None
        Keywords needed for initializing the model, in addition to
        endog and exog.
    fit_kwds : dict-like or None
        Keywords needed for the model fitting.
    estimation_method : function or None
        The function that performs the estimation for each partition.
        If None this defaults to _est_regularized_distributed.
    estimation_kwds : dict-like or None
        Keywords to be passed to estimation_method.
    join_method : function or None
        The function used to recombine the results from each partition.
        If None this defaults to _join_debiased.

    Returns
    -------

    join_method result.  For the default _join_debiased, it returns a
    p length array.
    """

    init_kwds = {} if init_kwds is None else init_kwds
    estimation_kwds = {} if estimation_kwds is None else estimation_kwds
    join_kwds = {} if join_kwds is None else join_kwds

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
    for endog, exog in data_generator:

        model = model_class(endog, exog, **init_kwds)

        # TODO possibly fit_kwds should be handled within
        # estimation_kwds to make more general?
        results = estimation_method(model, mnum, partitions, fit_kwds,
                                    **estimation_kwds)
        model_results_l.append(results)

        mnum += 1

    return join_method(model_results_l, partitions, **join_kwds)
