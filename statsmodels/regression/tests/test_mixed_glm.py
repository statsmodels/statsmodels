import statsmodels.api as sm
from statsmodels.tools.numdiff import approx_fprime, approx_hess
import numpy as np
import pandas as pd
from statsmodels.regression.mixed_glm import *

def endog_gen(family,lin_pred, n_group, gsize):
    if family == "Binomial":
        pr = 1 / (1 + np.exp(-lin_pred))
        endog = 1 * (np.random.uniform(size=n_group * gsize) < pr)
    elif family == "Poisson" or family == "NegativeBinomial":
        mn = np.exp(lin_pred)
        endog = np.random.poisson(mn)
    elif family == "Gamma" or family == "InverseGaussian":
        mn = np.exp(lin_pred)
        endog = np.random.gamma(shape=1,scale=mn)
    elif family == "Gaussian":
        endog = .5 * lin_pred + np.random.randn(len(lin_pred)) #.sum()
    else:
        1/0

    return endog

def gen_mixed(family, n_group):

    # Number of observations per group
    gsize = 2

    exog = np.random.normal(size=(n_group * gsize, 2))
    lin_pred = exog.sum(1)

    re = 0.5 * np.random.normal(size=n_group)
    re = np.kron(re, np.ones(gsize))
    lin_pred += re

    endog = endog_gen(family,lin_pred,n_group,gsize)
    groups = np.kron(np.arange(n_group), np.ones(gsize))

    return endog, exog, groups

def gen_mixed2(family, n_group):

    # Number of observations per group
    gsize = 3

    exog = np.random.normal(size=(n_group * gsize, 2))
    lin_pred = exog.sum(1)

    re = 0.5 * np.random.normal(size=n_group)
    re = np.kron(re, np.ones(gsize))
    lin_pred += re

    rs = np.random.normal(size=n_group * gsize)
    re2 = np.random.normal(size=n_group)
    re2 = np.kron(re2, np.ones(gsize))
    re2 = rs * re2
    lin_pred += re2

    endog = endog_gen(family,lin_pred,n_group,gsize)
    groups = np.kron(np.arange(n_group), np.ones(gsize))

    return endog, exog, groups, rs


## Test Loglike
def test_laplace_loglike_binomial():
    code = """
    Python:
    np.random.seed(313)
    endog, exog, groups = gen_mixed('Binomial', 10)
    mat = np.hstack((endog[:, None], groups[:, None], exog))
    np.savetxt('mat.txt', mat)

    R:
    df = read.table('mat.txt')
    names(df) = c('y', 'g', 'x1', 'x2')
    library(lme4)
    m = glmer(y ~ 0 + x1 + x2 + (1 | g), family=binomial(), data=df)
    logLik(m)
    summary(m)$coef
    VarCorr(m)

    logLik -31.78661
    """

    np.random.seed(313)
    endog, exog, groups = gen_mixed('Binomial', 30)

    model = MixedGLM(endog, exog, groups=groups, family=sm.families.Binomial())
    par = MixedGLMParams.from_components(fe_params=np.asarray([0.7400841,1.2799780]),
                                         cov_re=np.asarray([[0.76278**2]]))
    logLik = model.loglike(par)
    np.testing.assert_allclose(logLik, -31.78661, rtol=1e-6)

# TODO:Need to get working for more than one RE. Not yet working.
def t_est_laplace_loglike_binomial2():
    code = """
    Python:
    np.random.seed(313)
    endog, exog, groups, rs = gen_mixed2('Binomial', 20)
    mat = np.hstack((endog[:, None], groups[:, None], rs[:,None], exog))
    np.savetxt('mat.txt', mat)

    R:
    df = read.table('mat.txt')
    names(df) = c('y', 'g', 'rs', 'x1', 'x2')
    library(lme4)
    m = glmer(y ~ 0 + x1 + x2 + (1 + rs| g), family=binomial(), data=df)

    logLik -31.7866107
    """

    np.random.seed(313)
    endog, exog, groups, rs = gen_mixed2('Binomial', 20)
    exog_re = np.vstack((np.ones(len(groups)),rs)).T

    model = MixedGLM(endog, exog, groups=groups, exog_re=np.ones(len(groups)), family=sm.families.Binomial())
    cmat = np.asarray([[0.54829**2]])
    fps = np.asarray([0.4996,0.4391])
    par = MixedGLMParams.from_components(fe_params=fps, cov_re= cmat)
    logLik = model.loglike(par)
    np.testing.assert_allclose(logLik, -38.22342, rtol=1E-5)

    model = MixedGLM(endog, exog, groups=groups, exog_re= exog_re, family=sm.families.Binomial())
    cmat = np.asarray([[0.65090**2,0.65090*0.74996*0.311],[0.65090*0.74996*0.311,0.74996**2]])
    fps = np.asarray([0.5950,0.4952])
    par = MixedGLMParams.from_components(fe_params=fps, cov_re= cmat)
    logLik = model.loglike(par)
    np.testing.assert_allclose(logLik, -37.81756)

def test_laplace_loglike_poisson():
    code = """
    Python:
    np.random.seed(313)
    endog, exog, groups = gen_mixed('Poisson', 30)
    mat = np.hstack((endog[:, None], groups[:, None], exog))
    np.savetxt('mat.txt', mat)

    R:
    df = read.table('mat.txt')
    names(df) = c('y', 'g', 'x1', 'x2')
    library(lme4)
    m = glmer(y ~ 0 + x1 + x2 + (1 | g), family=poisson(), data=df)
    logLik(m)
    summary(m)$coef
    VarCorr(m)

    logLik -85.99506
    """

    np.random.seed(313)
    endog, exog, groups = gen_mixed('Poisson', 30)

    model = MixedGLM(endog, exog, groups=groups, family=sm.families.Poisson())
    par = MixedGLMParams.from_components(fe_params=np.asarray([1.01505, 0.9464392]),
                                         cov_re=np.asarray([[0.34564**2]]))
    logLik = model.loglike(par)
    np.testing.assert_allclose(logLik, -85.99506, rtol=1e-5)

# TODO: Non canonical and hence doesn't match R
def t_est_laplace_loglike_poisson_sqrt():
    code = """
    Python:
    np.random.seed(313)
    endog, exog, groups = gen_mixed('Poisson', 30)
    mat = np.hstack((endog[:, None], groups[:, None], exog))
    np.savetxt('mat.txt', mat)

    R:
    df = read.table('mat.txt')
    names(df) = c('y', 'g', 'x1', 'x2')
    library(lme4)
    m = glmer(y ~ 0 + x1 + x2 + (1 | g), family=poisson(link=sqrt), data=df)
    logLik(m)
    summary(m)$coef
    VarCorr(m)

    logLik -118.5216
    """

    np.random.seed(313)
    endog, exog, groups = gen_mixed('Poisson', 30)

    sqrt_link = sm.families.links.sqrt
    model = MixedGLM(endog, exog, groups=groups, family=sm.families.Poisson(link=sqrt_link))
    par = MixedGLMParams.from_components(fe_params=np.asarray([0.5840612,0.5596115]),
                                         cov_re=np.asarray([[1.2337**2]]))
    logLik = model.loglike(par)
    np.testing.assert_allclose(logLik, -118.5216, rtol=1e-4)

# TODO: Non canonical and hence doesn't match R
def t_est_laplace_loglike_gamma_log():
    code = """
    Python:
    np.random.seed(313)
    endog, exog, groups = gen_mixed('Gamma', 30)
    mat = np.hstack((endog[:, None], groups[:, None], exog))
    np.savetxt('mat.txt', mat)

    R:
    df = read.table('mat.txt')
    names(df) = c('y', 'g', 'x1', 'x2')
    library(lme4)
    m = glmer(y ~ 0 + x1 + x2 + (1 | g), family=Gamma(link=log), data=df)
    logLik(m)
    summary(m)$coef
    VarCorr(m)

    logLik -61.37757
    """

    np.random.seed(313)
    endog, exog, groups = gen_mixed('Gamma', 30)

    loglink=sm.families.links.log
    model = MixedGLM(endog, exog, groups=groups, family=sm.families.Gamma(loglink))
    par = MixedGLMParams.from_components(fe_params=np.asarray([0.9920529, 0.9011868]),
                                         cov_re=np.asarray([[0.59920**2]]))
    logLik = model.loglike(par)
    np.testing.assert_allclose(logLik, -61.3776, rtol=1e-6)

# Can't check Gaussian since it uses the concentrated log-likelihood

## Test Fitted Params
def test_fitted_poisson_log():
    code = """
    Python:
    np.random.seed(313)
    endog, exog, groups = gen_mixed('Poisson', 30)
    mat = np.hstack((endog[:, None], groups[:, None], exog))
    np.savetxt('mat.txt', mat)

    R:
    df = read.table('mat.txt')
    names(df) = c('y', 'g', 'x1', 'x2')
    library(lme4)
    m = glmer(y ~ 0 + x1 + x2 + (1 | g), family=poisson(), data=df)
    logLik(m)
    summary(m)$coef
    VarCorr(m)

    coefs 1.0150502 0.9464392 0.34564
    """

    np.random.seed(313)
    endog, exog, groups = gen_mixed('Poisson', 30)

    model = MixedGLM(endog, exog, groups=groups, family=sm.families.Poisson())
    fit = model.fit()

    np.testing.assert_allclose(fit.params, [1.0150502, 0.9464392, 0.34564], rtol=1e-2)

def test_fitted_binomial_logit():
    code = """
    Python:
    np.random.seed(313)
    endog, exog, groups = gen_mixed('Binomial', 30)
    mat = np.hstack((endog[:, None], groups[:, None], exog))
    np.savetxt('mat.txt', mat)

    R:
    df = read.table('mat.txt')
    names(df) = c('y', 'g', 'x1', 'x2')
    library(lme4)
    m = glmer(y ~ 0 + x1 + x2 + (1 | g), family=binomial(), data=df)
    logLik(m)
    summary(m)$coef
    VarCorr(m)

    coefs 0.7400841 1.2799780 0.76278

    """

    np.random.seed(313)
    endog, exog, groups = gen_mixed('Binomial', 30)

    model = MixedGLM(endog, exog, groups=groups, family=sm.families.Binomial())
    fit = model.fit()

    np.testing.assert_allclose(fit.params, [0.7400841, 1.2799780, 0.76278], rtol=1e-2)

# TODO: Takes too long to fit. Unclear why.
def t_est_fitted_negative_binomial():
    code = """
    Python:
    np.random.seed(313)
    endog, exog, groups = gen_mixed('NegativeBinomial', 30)
    mat = np.hstack((endog[:, None], groups[:, None], exog))
    np.savetxt('mat.txt', mat)

    R:
    df = read.table('mat.txt')
    names(df) = c('y', 'g', 'x1', 'x2')
    library(lme4)
    m = glmer.nb(y ~ 0 + x1 + x2 + (1 | g), data=df)
    logLik(m)
    summary(m)$coef
    VarCorr(m)

    coefs 1.0150092 0.9463241 0.32494
    """

    np.random.seed(313)
    endog, exog, groups = gen_mixed('NegativeBinomial', 30)

    model = MixedGLM(endog, exog, groups=groups, family=sm.families.NegativeBinomial())
    fit = model.fit()

    np.testing.assert_allclose(fit.params, [1.0150092, 0.9463241, 0.32494], rtol=1e-2)

## Test Derivatives
import statsmodels.genmod.families.links as links

class CheckFamily(object):

    family = sm.families.Family
    test_links = []

    def test_links_derivs(self):
        for lnk in self.test_links:
            try:
                self.like_hess_grad(self.family, lnk)
            except Exception as e:
                e.args += ("Link type: " + str(lnk),)
                raise

    def like_hess_grad(self, fam, lnk):
        # Test the function, gradient and Hessian of the joint
        # log-likelihood of the random effects and the data.

        fa = fam(link=lnk)

        np.random.seed(423)
        endog, exog, groups = gen_mixed(fa.__class__.__name__, 10)

        model = MixedGLM(endog, exog, groups=groups, family=fa)

        # Set some parameters
        fe_params = np.random.normal(size=2)
        cov_re = np.abs(np.random.normal(size=(1,1)))
        scale = 1.0

        self.like_hess_grad_helper(model, fe_params, cov_re, scale)

    def test_links_derivs2(self):
        for lnk in self.test_links:
            try:
                self.like_hess_grad2(self.family, lnk)
            except Exception as e:
                e.args += ("Link type: " + str(lnk),)
                raise

    def like_hess_grad2(self, fam, lnk):
        # Test the function, gradient and Hessian of the joint
        # log-likelihood of the random effects and the data.

        fa = fam(link=lnk)

        np.random.seed(423)
        endog, exog, groups, rs = gen_mixed2(fa.__class__.__name__, 10)
        exog_re = np.vstack((np.ones(len(groups)),rs)).T

        model = MixedGLM(endog, exog, groups=groups, exog_re=exog_re, family=fa)

        # Set some parameters
        fe_params = np.random.normal(size=2)
        cov_re = np.abs(np.random.normal(size=(2,2)))
        cov_re = np.dot(cov_re.T,cov_re)
        scale = 1.0

        self.like_hess_grad_helper(model, fe_params, cov_re, scale)


    def like_hess_grad_helper(self, model, fe_params, cov_re, scale):

        # Get the log-likelihood function, and its gradient and Hessian.
        fungradhess = model._gen_joint_like_grad_hess(fe_params, cov_re, scale)
        fun1 = lambda x : fungradhess(x)[0]
        grad = lambda x : fungradhess(x)[1]
        hess = lambda x : fungradhess(x)[2]

        for k in range(10):

            RTOL = 1e-2
            ATOL = 1e-2

            # An arbitrary state for the random effects
            ref = np.random.normal(size=(model.n_groups, model.k_re)).ravel()

            # Check the gradient
            fp1 = approx_fprime(ref, fun1, centered=True)
            fp2 = grad(ref)
            np.testing.assert_allclose(fp1, fp2, atol=ATOL, rtol=RTOL)

            # Check the Hessian
            he1 = approx_hess(ref, fun1)
            he1ld = np.linalg.slogdet(he1)[1]
            he2 = approx_fprime(ref, grad, centered=True)
            he2ld = np.linalg.slogdet(he2)[1]
            he3 = hess(ref)

            np.testing.assert_allclose(he1ld, he3, atol=ATOL, rtol=RTOL)
            np.testing.assert_allclose(he2ld, he3, atol=ATOL, rtol=RTOL)


class TestPoisson(CheckFamily):
    family = sm.families.Poisson
    test_links = [links.log, links.sqrt]


# TODO: Gaussian Throws weird errors about 'unrecognized data structure'
class T_estGaussian(CheckFamily):
    family = sm.families.Gaussian
    test_links = family.safe_links


class TestGamma(CheckFamily):
    family = sm.families.Gamma
    test_links = [links.log]

class TestBinomial(CheckFamily):
    family = sm.families.Binomial
    test_links = [links.logit, links.cauchy]
    # TODO: cloglog and probit throw errors when testing derivs
    #test_links = [links.logit, links.probit, links.cauchy, links.cloglog]


# TODO: Throws weird errors with NaNs in derivs
class T_estInverseGaussian(CheckFamily):
    family = sm.families.InverseGaussian
    test_links = family.safe_links


class TestNegativeBinomial(CheckFamily):
    family = sm.families.NegativeBinomial
    test_links = [links.log]
