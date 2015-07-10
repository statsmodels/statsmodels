import statsmodels.api as sm
from statsmodels.tools.numdiff import approx_fprime, approx_hess
import numpy as np
import pandas as pd
from statsmodels.regression.mixed_glm import *

def gen_mixed(family, n_group):

    # Number of observations per group
    gsize = 2

    exog = np.random.normal(size=(n_group * gsize, 2))
    lin_pred = exog.sum(1)

    re = 0.5 * np.random.normal(size=n_group)
    re = np.kron(re, np.ones(gsize))
    lin_pred += re

    if family == "Binomial":
        pr = 1 / (1 + np.exp(-lin_pred))
        endog = 1 * (np.random.uniform(size=n_group * gsize) < pr)
    elif family == "Poisson":
        mn = np.exp(lin_pred)
        endog = np.random.poisson(mn)
    elif family == "Gamma":
        # TODO: What link?
        mn = np.exp(lin_pred)
        endog = np.random.gamma(shape=1,scale=mn)
    else:
        1/0

    groups = np.kron(np.arange(n_group), np.ones(gsize))

    return endog, exog, groups



class CheckFamily(object):

    def test_joint_loglike(self):
        """
        Test the function, gradient and Hessian of the joint
        log-likelihood of the random effects and the data.
        """

        fa = self.family()

        np.random.seed(423)
        endog, exog, groups = gen_mixed(fa.__class__.__name__, 10)

        model = MixedGLM(endog, exog, groups=groups, family=fa)

        # Set some parameters
        fe_params = np.random.normal(size=2)
        cov_re = np.abs(np.random.normal(size=(1,1)))
        scale = 1.0

        # Get the log-likelihood function, and its gradient and Hessian.
        fungrad = model._gen_joint_like_score(fe_params, cov_re, scale)
        fun1 = lambda x : fungrad(x)[0]
        grad = lambda x : fungrad(x)[1]
        fun2 = lambda x : model._joint_like_hess(x, fe_params, cov_re, scale)[0]
        hess = lambda x : model._joint_like_hess(x, fe_params, cov_re, scale)[1]

        # Check that the gradients vanish at the mode.
        mp = model._get_map(fe_params, cov_re, scale, "BFGS")
        fp1 = approx_fprime(mp.ravel(), fun1)
        fp2 = grad(mp.ravel())

        np.testing.assert_allclose(fp1, 0, atol=1e-3)
        np.testing.assert_allclose(fp2, 0, atol=1e-3)

        # Check the Hessian at the mode.
        he1 = approx_hess(mp.ravel(), fun1)
        he2 = approx_fprime(mp.ravel(), grad)
        he3 = hess(mp.ravel())

        np.testing.assert_allclose(he1, he2, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(np.linalg.slogdet(he2)[1], he3)

        for k in range(10):

            # An arbitrary state for the random effects
            ref = np.random.normal(size=(model.n_groups, model.k_re)).ravel()

            # Check that the two functions values are equal.
            np.testing.assert_allclose(fun1(ref), fun2(ref))

            # Check the gradient
            fp1 = approx_fprime(ref.ravel(), fun1)
            fp2 = grad(ref.ravel())
            np.testing.assert_allclose(fp1, fp2, atol=1e-3)

            # Check the Hessian
            he1 = approx_hess(ref.ravel(), fun1)
            he2 = approx_fprime(ref.ravel(), grad)
            he3 = hess(ref.ravel())
            np.testing.assert_allclose(he1, he2, atol=1e-4, rtol=1e-4)
            np.testing.assert_allclose(np.linalg.slogdet(he2)[1], he3)


def test_laplace_loglike_binomial():
    """
    Python:
    np.random.seed(313)
    endog, exog, groups = gen_mixed('Binomial', 30)
    mat = np.hstack((endog[:, None], groups[:, None], exog))
    np.savetxt('mat.txt', mat)

    R:
    df = read.table('mat.txt')
    names(df) = c('y', 'g', 'x1', 'x2')
    library(lme4)
    glmer(y ~ 0 + x1 + x2 + (1 | g), family=binomial(), data=df)

    logLik -31.7866107
    """

    np.random.seed(313)
    endog, exog, groups = gen_mixed('Binomial', 30)

    model = MixedGLM(endog, exog, groups=groups, family=sm.families.Binomial())
    par = MixedGLMParams.from_components(fe_params=np.asarray([0.7400841, 1.279978]),
                                         cov_re=np.asarray([[0.5818]]))
    logLik = model.loglike(par)
    np.testing.assert_allclose(logLik, -31.7866107)


def test_laplace_loglike_poisson():
    """
    Python:
    np.random.seed(313)
    endog, exog, groups = gen_mixed('Poisson', 30)
    mat = np.hstack((endog[:, None], groups[:, None], exog))
    np.savetxt('mat.txt', mat)

    R:
    df = read.table('mat.txt')
    names(df) = c('y', 'g', 'x1', 'x2')
    library(lme4)
    glmer(y ~ 0 + x1 + x2 + (1 | g), family=poisson(), data=df)

    logLik -85.9951
    """

    np.random.seed(313)
    endog, exog, groups = gen_mixed('Poisson', 30)

    model = MixedGLM(endog, exog, groups=groups, family=sm.families.Poisson())
    par = MixedGLMParams.from_components(fe_params=np.asarray([1.01505, 0.9464392]),
                                         cov_re=np.asarray([[0.1195]]))
    logLik = model.loglike(par)
    np.testing.assert_allclose(logLik, -85.9951, rtol=1e-5)


# Not working for non-canonical GLM links yet. 

# def test_laplace_loglike_gamma():
#     """
#     Python:
#     np.random.seed(313)
#     endog, exog, groups = gen_mixed('Gamma', 30)
#     mat = np.hstack((endog[:, None], groups[:, None], exog))
#     np.savetxt('mat.txt', mat)
#
#     R:
#     df = read.table('mat.txt')
#     names(df) = c('y', 'g', 'x1', 'x2')
#     library(lme4)
#     glmer(y ~ 0 + x1 + x2 + (1 | g), family=Gamma(link=log), data=df)
#
#     logLik -61.3776
#     """
#
#     np.random.seed(313)
#     endog, exog, groups = gen_mixed('Gamma', 30)
#
#     loglink=sm.families.links.log
#     model = MixedGLM(endog, exog, groups=groups, family=sm.families.Gamma(loglink))
#     par = MixedGLMParams.from_components(fe_params=np.asarray([0.9921, 0.9012]),
#                                          cov_re=np.asarray([[0.3590]]))
#     logLik = model.loglike(par)
#     np.testing.assert_allclose(logLik, -61.3776, rtol=1e-1)


# Can't check Gaussian since it uses the concentrated log-likelihood

class TestBinomial(CheckFamily):
    family = sm.families.Binomial


class TestPoisson(CheckFamily):
    family = sm.families.Poisson

#class TestGamma(CheckFamily):
#    loglink=sm.families.links.log
#    family = sm.families.Gamma