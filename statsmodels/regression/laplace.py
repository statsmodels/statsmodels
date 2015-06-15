import numpy as np
import statsmodels.api as sm
import scipy
from statsmodels.tools.numdiff import (approx_fprime,
                                       approx_hess_cs)
from numpy.testing import assert_allclose


n = 100
k_fe = 3
k_re = 2

# Simulate from a logistic mixed GLM
exog = np.random.normal(size=(n, k_fe))
exog_re = np.random.normal(size=(n, k_re))
groups = np.kron(np.arange(n / 2), np.r_[1, 1])
re = np.random.normal(size=(n / 2, k_re))
re = np.kron(re, np.ones((2, 1)))
lin_pred = exog.sum(1) + (exog_re * re).sum(1)
pr = 1 / (1 + np.exp(-lin_pred))
endog = 1 * (np.random.uniform(size=n) < pr)


family = sm.families.Binomial()

ugroups = np.unique(groups)
n_groups = len(ugroups)

def split(ar):
    rs = []
    for g in ugroups:
        ii = np.flatnonzero(groups == g)
        if ar.ndim == 1:
            rs.append(ar[ii])
        else:
            rs.append(ar[ii, :])
    return rs


endog_split = split(endog)
exog_split = split(exog)
exog_re_split = split(exog_re)


def gen_fungrad(params, cov_re, scale):
    """
    Function and gradient of the joint log likelihood.

    The log-likelihood is for the data and random effects, viewed as a
    function of the random effects.

    Parameters
    ----------
    params : array-like, 1d
        The fixed effects parameters
    cov_re : array-like, 2d
        The covariance matrix of the random effects
    scale : float
        The scale parameter

    Returns a function that takes a state for the random effects
    (vectorized) and returns the value of the negative joint log
    likelihood and its gradient evaluated at the given state.
    """

    lin_pred = np.dot(exog, params)
    lin_pred = split(lin_pred)

    def fun(ref):

        ref = np.reshape(ref, (n_groups, k_re))

        s = np.linalg.solve(cov_re, ref.T)
        f = -(ref.T * s).sum() / 2 # r.e. log likelihood

        d = np.zeros((n_groups, k_re))

        for k, g in enumerate(ugroups):
            lin_predr = lin_pred[k] + np.dot(exog_re_split[k], ref[k, :]) # eta_i
            mean = family.fitted(lin_predr) # mu_i = h(eta_i)
            f += family.loglike(endog_split[k], mean,scale=scale)

            d[k, :] = ((endog_split[k] - mean)[:, None] * exog_re_split[k] / scale).sum(0)
            d[k, :] -= s[:, k]

        return -f, -d.ravel()

    return fun


def test_gen_fungrad():

    np.random.seed(4234)

    for k in range(10):

        params = np.random.normal(size=k_fe)
        cov_re = np.random.normal(size=(k_re, k_re))
        cov_re = np.dot(cov_re, cov_re.T)
        fungrad = gen_fungrad(params, cov_re, 1)
        fun = lambda x : fungrad(x)[0]
        grad = lambda x : fungrad(x)[1]

        ref = np.random.normal(size=(n_groups, k_re)).ravel()

        # Check that the numerical gradient matches the analytic
        # gradient
        fp1 = approx_fprime(ref, fun)
        fp2 = grad(ref)

        assert_allclose(fp1, fp2, rtol=1e-1, atol=1e-5)




def funhess(ref, params, cov_re, scale):
    """
    Function and Hessian of the joint log likelihood evaluated at MAP.

    The log-likelihood is for the data and random effects, viewed as a
    function of the random effects.

    Parameters
    ----------
    ref : array-like
        The random effects
    params : array-like, 1d
        The fixed effects parameters
    cov_re : array-like, 2d
        The covariance matrix of the random effects
    scale : float
        The scale parameter

    Returns the negative joint log likelihood and the determinant of
    its Hessian evaluated at the given state.
    """

    lin_pred = np.dot(exog, params)
    lin_pred = split(lin_pred)

    ref = np.reshape(ref, (n_groups, k_re))

    s = np.linalg.solve(cov_re, ref.T)
    f = -(ref.T * s).sum() / 2

    d2 = np.zeros(n_groups)

    for k, g in enumerate(ugroups):
        lin_predr = lin_pred[k] + np.dot(exog_re_split[k], ref[k, :])
        mean = family.fitted(lin_predr)
        f += family.loglike(endog_split[k], mean,scale=scale)
        va = family.variance(mean)
        hmat = va[:, None] * exog_re_split[k]
        hmat = np.dot(exog_re_split[k].T, hmat)
        hmat += np.linalg.inv(cov_re)
        _, d2[k] = np.linalg.slogdet(hmat)

    return -f, d2.sum()


def test_funhess():

    np.random.seed(4234)

    for k in range(10):

        params = np.random.normal(size=k_fe)
        cov_re = np.random.normal(size=(k_re, k_re))
        cov_re = np.dot(cov_re, cov_re.T)
        fun = lambda x : funhess(x, params, cov_re, 1)[0]
        hess = lambda x : funhess(x, params, cov_re, 1)[1]

        fungrad = gen_fungrad(params, cov_re, 1)
        fun1 = lambda x : fungrad(x)[0]
        grad = lambda x : fungrad(x)[1]

        ref = np.random.normal(size=(n_groups, k_re)).ravel()

        # Check that the numerical gradient matches the analytic
        # gradient
        he1 = approx_fprime(ref, grad)
        _, dhe1 = np.linalg.slogdet(he1)
        he2 = hess(ref)

        assert_allclose(dhe1, he2, rtol=1e-3, atol=1e-5)



def get_map(params, cov_re, scale):
    """
    Obtain the MAP predictor of the random effects.

    The MAP (maximum a posteriori) predictor is the mode of the joint
    likelihood of the data and the random effects, viewed as a
    function of the random effects.

    Parameters
    ----------
    params : array-like, 1d
        The fixed effects parameters
    cov_re : array-like, 2d
        The covariance matrix of the random effects
    scale : float
        The scale parameter

    Returns the MAP predictor of the random effects.
    """

    fun = gen_fungrad(params, cov_re, scale)

    x0 = np.zeros(n)

    result = scipy.optimize.minimize(fun, x0, jac=True, method='Newton-CG')
    
    print(result)

    if not result.success:
        print("OPTIMIZATION FAILED")
        1/0

    mp = np.reshape(result.x, (n_groups, k_re))
    return mp


def laplace(params, cov_re, scale):
    """
    Evaluate the marginal log-likelihood.

    The Laplace method is used to approximate the integral over the
    random effects distribution.

    Parameters
    ----------
    params : array-like, 1d
        The fixed effects parameters
    cov_re : array-like, 2d
        The covariance matrix of the random effects
    scale : float
        The scale parameter


    """

    mp = get_map(params, cov_re, scale)

    f, h = funhess(mp, params, cov_re, scale)

    d = len(mp)
    ival = np.exp(-f) * (2 * np.pi)**(d / 2) / np.sqrt(np.exp(h))
    return(ival)


params = np.ones(k_fe)
cov_re = np.random.normal(size=(2, 2))
cov_re = np.dot(cov_re.T, cov_re)
scale = 1.1
mp = laplace(params, cov_re, scale)



