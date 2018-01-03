import numpy as np
from .mixed_glm import MixedGLM, BinomialMixedGLM
import statsmodels.api as sm
from scipy import sparse
from numpy.testing import assert_allclose
from scipy.optimize import approx_fprime
import warnings


def gen_simple_logit(s):

    np.random.seed(3799)

    nc = 100
    cs = 500

    exog_vc = np.kron(np.eye(nc), np.ones((cs, 1)))
    exog_fe = np.random.normal(size=(nc*cs, 2))
    vc = s*np.random.normal(size=nc)
    lp = np.dot(exog_fe, np.r_[1, -1]) + np.dot(exog_vc, vc)
    pr = 1 / (1 + np.exp(-lp))
    y = 1*(np.random.uniform(size=nc*cs) < pr)
    ident = np.zeros(nc, dtype=np.int)

    return y, exog_fe, exog_vc, ident


def gen_logit_crossed(s1, s2):

    np.random.seed(3799)

    nc = 100
    cs = 500

    a = np.kron(np.eye(nc), np.ones((cs, 1)))
    b = np.kron(np.ones((cs, 1)), np.eye(nc))
    exog_vc = np.concatenate((a, b), axis=1)

    exog_fe = np.random.normal(size=(nc*cs, 1))
    vc = s1 * np.random.normal(size=2*nc)
    vc[nc:] *= s2 / s1
    lp = np.dot(exog_fe, np.r_[-0.5]) + np.dot(exog_vc, vc)
    pr = 1 / (1 + np.exp(-lp))
    y = 1*(np.random.uniform(size=nc*cs) < pr)
    ident = np.zeros(2*nc, dtype=np.int)
    ident[nc:] = 1

    return y, exog_fe, exog_vc, ident


def test_logit_map():

    y, exog_fe, exog_vc, ident = gen_simple_logit(2)
    exog_vc = sparse.csr_matrix(exog_vc)

    glmm = MixedGLM(y, exog_fe, exog_vc, ident, family=sm.families.Binomial())
    rslt = glmm.fit_map(minim_opts={"gtol": 1e-4})

    assert_allclose(glmm.logposterior_grad(rslt.params),
                    np.zeros_like(rslt.params), atol=1e-4)


def test_logit_map_crossed():

    y, exog_fe, exog_vc, ident = gen_logit_crossed(1, 2)
    exog_vc = sparse.csr_matrix(exog_vc)

    glmm = MixedGLM(y, exog_fe, exog_vc, ident, family=sm.families.Binomial())
    rslt = glmm.fit_map(minim_opts={"gtol": 1e-4})

    assert_allclose(glmm.logposterior_grad(rslt.params),
                    np.zeros_like(rslt.params), atol=1e-4)


def test_logit_elbo_grad():

    for j in range(2):

        if j == 0:
            y, exog_fe, exog_vc, ident = gen_simple_logit(2)
        else:
            y, exog_fe, exog_vc, ident = gen_logit_crossed(1, 2)

        exog_vc = sparse.csr_matrix(exog_vc)

        glmm1 = BinomialMixedGLM(y, exog_fe, exog_vc, ident)
        rslt1 = glmm1.fit_map(minim_opts={"gtol": 1e-4})

        n = glmm1.k_fep + glmm1.k_vcp + glmm1.k_vc

        for k in range(3):

            if k == 0:
                vb_mean = rslt1.params
                vb_sd = np.ones_like(vb_mean)
            elif k == 1:
                vb_mean = np.zeros(len(vb_mean))
                vb_sd = np.ones_like(vb_mean)
            else:
                vb_mean = np.random.normal(size=len(vb_mean))
                vb_sd = np.random.uniform(1, 2, size=len(vb_mean))

            mean_grad, sd_grad = glmm1.vb_elbo_grad(vb_mean, vb_sd)

            def elbo(vec):
                n = len(vec) // 2
                return glmm1.vb_elbo(vec[:n], vec[n:])

            x = np.concatenate((vb_mean, vb_sd))
            g1 = approx_fprime(x, elbo, 1e-5)
            n = len(x) // 2

            mean_grad_n = g1[:n]
            sd_grad_n = g1[n:]

            assert_allclose(mean_grad, mean_grad_n, atol=1e-2, rtol=1e-2)
            assert_allclose(sd_grad, sd_grad_n, atol=1e-2, rtol=1e-2)


def test_logit_vb():

    y, exog_fe, exog_vc, ident = gen_simple_logit(0)
    exog_vc = sparse.csr_matrix(exog_vc)

    glmm1 = BinomialMixedGLM(y, exog_fe, exog_vc, ident)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rslt1 = glmm1.fit_map(minim_opts={"gtol": 1e-4, "maxiter": 5})

    glmm2 = BinomialMixedGLM(y, exog_fe, exog_vc, ident)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rslt2 = glmm2.fit_vb(mean=rslt1.params, minim_opts={"maxiter": 2})

    rslt1.summary()
    rslt2.summary()

    assert_allclose(rslt1.params[0:5], np.r_[
        0.64644962, -0.61266869, -1., -0.00961027, 0.02411796],
                    rtol=1e-4, atol=1e-4)

    assert_allclose(rslt2.params[0:5], np.r_[
        0.9017295, -0.95958884, -0.70822657, -0.00711374, 0.02673195],
                    rtol=1e-4, atol=1e-4)


def test_logit_vb_crossed():

    y, exog_fe, exog_vc, ident = gen_logit_crossed(1, 2)
    exog_vc = sparse.csr_matrix(exog_vc)

    glmm1 = BinomialMixedGLM(y, exog_fe, exog_vc, ident)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rslt1 = glmm1.fit_map(minim_opts={"gtol": 1e-4, "maxiter": 2})

    glmm2 = BinomialMixedGLM(y, exog_fe, exog_vc, ident)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rslt2 = glmm2.fit_vb(mean=rslt1.params, minim_opts={"maxiter": 2})

    rslt1.summary()
    rslt2.summary()

    assert_allclose(rslt1.params[0:5], np.r_[
        -0.84192649, 0.81152304, 0.81056098, -0.76727982, -0.94713751],
                    rtol=1e-4, atol=1e-4)

    assert_allclose(rslt2.params[0:5], np.r_[
        -0.68311938,  0.75472554,  0.75218755, -0.71387273, -0.76462306],
                    rtol=1e-4, atol=1e-4)
