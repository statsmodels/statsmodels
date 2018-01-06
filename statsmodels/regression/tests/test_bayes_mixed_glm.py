import numpy as np
from statsmodels.regression.bayes_mixed_glm import (
    BayesMixedGLM, BinomialBayesMixedGLM)
import statsmodels.api as sm
import pandas as pd
from scipy import sparse
from numpy.testing import assert_allclose
from scipy.optimize import approx_fprime


def gen_simple_logit(nc, cs, s):

    np.random.seed(3799)

    exog_vc = np.kron(np.eye(nc), np.ones((cs, 1)))
    exog_fe = np.random.normal(size=(nc*cs, 2))
    vc = s*np.random.normal(size=nc)
    lp = np.dot(exog_fe, np.r_[1, -1]) + np.dot(exog_vc, vc)
    pr = 1 / (1 + np.exp(-lp))
    y = 1*(np.random.uniform(size=nc*cs) < pr)
    ident = np.zeros(nc, dtype=np.int)

    return y, exog_fe, exog_vc, ident


def gen_logit_crossed(nc, cs, s1, s2):

    np.random.seed(3799)

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


def gen_logit_crossed_pandas(nc, cs, s1, s2):

    np.random.seed(3799)

    a = np.kron(np.arange(nc), np.ones(cs))
    b = np.kron(np.ones(cs), np.arange(nc))
    fe = np.ones(nc * cs)

    vc = np.zeros(nc * cs)
    for i in np.unique(a):
        ii = np.flatnonzero(a == i)
        vc[ii] += s1*np.random.normal()
    for i in np.unique(b):
        ii = np.flatnonzero(b == i)
        vc[ii] += s2*np.random.normal()

    lp = -0.5 * fe + vc
    pr = 1 / (1 + np.exp(-lp))
    y = 1*(np.random.uniform(size=nc*cs) < pr)

    ident = np.zeros(2*nc, dtype=np.int)
    ident[nc:] = 1

    df = pd.DataFrame({"fe": fe, "a": a, "b": b, "y": y})

    return df


def test_logit_map():

    y, exog_fe, exog_vc, ident = gen_simple_logit(10, 10, 2)
    exog_vc = sparse.csr_matrix(exog_vc)

    glmm = BayesMixedGLM(y, exog_fe, exog_vc, ident,
                         family=sm.families.Binomial(),
                         vcp_p=0.5)
    rslt = glmm.fit_map()

    assert_allclose(glmm.logposterior_grad(rslt.params),
                    np.zeros_like(rslt.params), atol=1e-3)


def test_logit_map_crossed():

    y, exog_fe, exog_vc, ident = gen_logit_crossed(10, 10, 1, 2)
    exog_vc = sparse.csr_matrix(exog_vc)

    glmm = BayesMixedGLM(y, exog_fe, exog_vc, ident,
                         family=sm.families.Binomial(),
                         vcp_p=0.5)
    rslt = glmm.fit_map()

    assert_allclose(glmm.logposterior_grad(rslt.params),
                    np.zeros_like(rslt.params), atol=1e-4)


def test_logit_map_crosed_formula():

    data = gen_logit_crossed_pandas(10, 10, 1, 2)

    fml = "y ~ fe"
    fml_vc = ["0 + C(a)", "0 + C(b)"]
    glmm = BayesMixedGLM.from_formula(
        fml, fml_vc, data, family=sm.families.Binomial(), vcp_p=0.5)
    rslt = glmm.fit_map()

    assert_allclose(glmm.logposterior_grad(rslt.params),
                    np.zeros_like(rslt.params), atol=1e-4)

    rslt.summary()


def test_logit_elbo_grad():

    for j in range(2):

        if j == 0:
            y, exog_fe, exog_vc, ident = gen_simple_logit(10, 10, 2)
        else:
            y, exog_fe, exog_vc, ident = gen_logit_crossed(10, 10, 1, 2)

        exog_vc = sparse.csr_matrix(exog_vc)

        glmm1 = BinomialBayesMixedGLM(y, exog_fe, exog_vc, ident, vcp_p=0.5)
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

    y, exog_fe, exog_vc, ident = gen_simple_logit(10, 10, 0)
    exog_vc = sparse.csr_matrix(exog_vc)

    glmm1 = BayesMixedGLM(y, exog_fe, exog_vc, ident, vcp_p=0.5,
                          fe_p=0.5, family=sm.families.Binomial())
    rslt1 = glmm1.fit_map()

    glmm2 = BinomialBayesMixedGLM(y, exog_fe, exog_vc, ident, vcp_p=0.5,
                                  fe_p=0.5)
    rslt2 = glmm2.fit_vb(rslt1.params)

    rslt1.summary()
    rslt2.summary()

    assert_allclose(rslt1.params[0:5], np.r_[
        0.75330405, -0.71643228, -1., -0.00959806,  0.00450254],
                    rtol=1e-4, atol=1e-4)

    assert_allclose(rslt2.params[0:5], np.r_[
        0.79338836, -0.7599833, -0.64149356, -0.24772884,  0.10775366],
                    rtol=1e-4, atol=1e-4)


def test_logit_vb_crossed():

    y, exog_fe, exog_vc, ident = gen_logit_crossed(10, 10, 1, 2)

    glmm1 = BayesMixedGLM(y, exog_fe, exog_vc, ident, vcp_p=0.5,
                          fe_p=0.5, family=sm.families.Binomial())
    rslt1 = glmm1.fit_map()

    glmm2 = BinomialBayesMixedGLM(y, exog_fe, exog_vc, ident, vcp_p=0.5,
                                  fe_p=0.5)
    rslt2 = glmm2.fit_vb(mean=rslt1.params)

    rslt1.summary()
    rslt2.summary()

    assert_allclose(rslt1.params[0:5], np.r_[
        -0.54307398, -1., -1., -0.0096403, 0.00232701],
                    rtol=1e-4, atol=1e-4)

    assert_allclose(rslt2.params[0:5], np.r_[
        -0.70834417, -0.3571011, 0.19126823, -0.36074489, 0.058976],
                    rtol=1e-4, atol=1e-4)


def test_logit_vb_crossed_formula():

    data = gen_logit_crossed_pandas(10, 10, 1, 2)

    fml = "y ~ fe"
    fml_vc = ["0 + C(a)", "0 + C(b)"]
    glmm1 = BinomialBayesMixedGLM.from_formula(
        fml, fml_vc, data, vcp_p=0.5)
    rslt1 = glmm1.fit_vb()

    glmm2 = BinomialBayesMixedGLM(glmm1.endog, glmm1.exog_fe, glmm1.exog_vc,
                                  glmm1.ident, vcp_p=0.5)
    rslt2 = glmm2.fit_vb()

    assert_allclose(rslt1.params, rslt2.params, atol=1e-4)

    rslt1.summary()
    rslt2.summary()
