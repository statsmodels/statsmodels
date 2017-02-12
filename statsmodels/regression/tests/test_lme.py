import warnings
import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM, MixedLMParams
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
                           dec, assert_)
from . import lme_r_results
from statsmodels.base import _penalties as penalties
from numpy.testing import dec
import statsmodels.tools.numdiff as nd
import os
import csv
import scipy

# TODO: add tests with unequal group sizes

v = scipy.__version__.split(".")[1]
old_scipy = int(v) < 16

class R_Results(object):
    """
    A class for holding various results obtained from fitting one data
    set using lmer in R.

    Parameters
    ----------
    meth : string
        Either "ml" or "reml".
    irfs : string
        Either "irf", for independent random effects, or "drf" for
        dependent random effects.
    ds_ix : integer
        The number of the data set
    """

    def __init__(self, meth, irfs, ds_ix):

        bname = "_%s_%s_%d" % (meth, irfs, ds_ix)

        self.coef = getattr(lme_r_results, "coef" + bname)
        self.vcov_r = getattr(lme_r_results, "vcov" + bname)
        self.cov_re_r = getattr(lme_r_results, "cov_re" + bname)
        self.scale_r = getattr(lme_r_results, "scale" + bname)
        self.loglike = getattr(lme_r_results, "loglike" + bname)

        if hasattr(lme_r_results, "ranef_mean" + bname):
            self.ranef_postmean = getattr(lme_r_results, "ranef_mean"
                                          + bname)
            self.ranef_condvar = getattr(lme_r_results,
                                         "ranef_condvar" + bname)
            self.ranef_condvar = np.atleast_2d(self.ranef_condvar)

        # Load the data file
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        rdir = os.path.join(cur_dir, 'results')
        fname = os.path.join(rdir, "lme%02d.csv" % ds_ix)
        fid = open(fname)
        rdr = csv.reader(fid)
        header = next(rdr)
        data = [[float(x) for x in line] for line in rdr]
        data = np.asarray(data)

        # Split into exog, endog, etc.
        self.endog = data[:, header.index("endog")]
        self.groups = data[:, header.index("groups")]
        ii = [i for i, x in enumerate(header) if
              x.startswith("exog_fe")]
        self.exog_fe = data[:, ii]
        ii = [i for i, x in enumerate(header) if
              x.startswith("exog_re")]
        self.exog_re = data[:, ii]


def loglike_function(model, profile_fe, has_fe):
    # Returns a function that evaluates the negative log-likelihood for
    # the given model.

    def f(x):
        params = MixedLMParams.from_packed(
            x, model.k_fe, model.k_re, model.use_sqrt, has_fe=has_fe)
        return -model.loglike(params, profile_fe=profile_fe)

    return f


class TestMixedLM(object):

    # Test analytic scores and Hessian using numeric differentiation
    @dec.slow
    def test_compare_numdiff(self):

        n_grp = 200
        grpsize = 5
        k_fe = 3
        k_re = 2

        for use_sqrt in False, True:
            for reml in False, True:
                for profile_fe in False, True:

                    np.random.seed(3558)
                    exog_fe = np.random.normal(size=(n_grp * grpsize, k_fe))
                    exog_re = np.random.normal(size=(n_grp * grpsize, k_re))
                    exog_re[:, 0] = 1
                    exog_vc = np.random.normal(size=(n_grp * grpsize, 3))
                    slopes = np.random.normal(size=(n_grp, k_re))
                    slopes[:, -1] *= 2
                    slopes = np.kron(slopes, np.ones((grpsize, 1)))
                    slopes_vc = np.random.normal(size=(n_grp, 3))
                    slopes_vc = np.kron(slopes_vc, np.ones((grpsize, 1)))
                    slopes_vc[:, -1] *= 2
                    re_values = (slopes * exog_re).sum(1)
                    vc_values = (slopes_vc * exog_vc).sum(1)
                    err = np.random.normal(size=n_grp * grpsize)
                    endog = exog_fe.sum(1) + re_values + vc_values + err
                    groups = np.kron(range(n_grp), np.ones(grpsize))

                    vc = {"a": {}, "b": {}}
                    for i in range(n_grp):
                        ix = np.flatnonzero(groups == i)
                        vc["a"][i] = exog_vc[ix, 0:2]
                        vc["b"][i] = exog_vc[ix, 2:3]

                    model = MixedLM(endog, exog_fe, groups,
                                    exog_re, exog_vc=vc, use_sqrt=use_sqrt)
                    rslt = model.fit(reml=reml)

                    loglike = loglike_function(
                        model, profile_fe=profile_fe, has_fe=not profile_fe)

                    # Test the score at several points.
                    for kr in range(5):
                        fe_params = np.random.normal(size=k_fe)
                        cov_re = np.random.normal(size=(k_re, k_re))
                        cov_re = np.dot(cov_re.T, cov_re)
                        vcomp = np.random.normal(size=2) ** 2
                        params = MixedLMParams.from_components(
                            fe_params, cov_re=cov_re, vcomp=vcomp)
                        params_vec = params.get_packed(
                            has_fe=not profile_fe, use_sqrt=use_sqrt)

                        # Check scores
                        gr = -model.score(params, profile_fe=profile_fe)
                        ngr = nd.approx_fprime(params_vec, loglike)
                        assert_allclose(gr, ngr, rtol=1e-3)

                    # Check Hessian matrices at the MLE (we don't have
                    # the profile Hessian matrix and we don't care
                    # about the Hessian for the square root
                    # transformed parameter).
                    if (profile_fe is False) and (use_sqrt is False):
                        hess = -model.hessian(rslt.params_object)
                        params_vec = rslt.params_object.get_packed(
                            use_sqrt=False, has_fe=True)
                        loglike_h = loglike_function(
                            model, profile_fe=False, has_fe=True)
                        nhess = nd.approx_hess(params_vec, loglike_h)
                        assert_allclose(hess, nhess, rtol=1e-3)

    def test_default_re(self):

        np.random.seed(3235)
        exog = np.random.normal(size=(300, 4))
        groups = np.kron(np.arange(100), [1, 1, 1])
        g_errors = np.kron(np.random.normal(size=100), [1, 1, 1])
        endog = exog.sum(1) + g_errors + np.random.normal(size=300)
        mdf1 = MixedLM(endog, exog, groups).fit()
        mdf2 = MixedLM(endog, exog, groups, np.ones(300)).fit()
        assert_almost_equal(mdf1.params, mdf2.params, decimal=8)

    def test_history(self):

        np.random.seed(3235)
        exog = np.random.normal(size=(300, 4))
        groups = np.kron(np.arange(100), [1, 1, 1])
        g_errors = np.kron(np.random.normal(size=100), [1, 1, 1])
        endog = exog.sum(1) + g_errors + np.random.normal(size=300)
        mod = MixedLM(endog, exog, groups)
        rslt = mod.fit(full_output=True)
        assert_equal(hasattr(rslt, "hist"), True)

    def test_profile_inference(self):
        # Smoke test
        np.random.seed(9814)
        k_fe = 2
        gsize = 3
        n_grp = 100
        exog = np.random.normal(size=(n_grp * gsize, k_fe))
        exog_re = np.ones((n_grp * gsize, 1))
        groups = np.kron(np.arange(n_grp), np.ones(gsize))
        vca = np.random.normal(size=n_grp * gsize)
        vcb = np.random.normal(size=n_grp * gsize)
        errors = 0
        g_errors = np.kron(np.random.normal(size=100), np.ones(gsize))
        errors += g_errors + exog_re[:, 0]
        rc = np.random.normal(size=n_grp)
        errors += np.kron(rc, np.ones(gsize)) * vca
        rc = np.random.normal(size=n_grp)
        errors += np.kron(rc, np.ones(gsize)) * vcb
        errors += np.random.normal(size=n_grp * gsize)

        endog = exog.sum(1) + errors
        vc = {"a": {}, "b": {}}
        for k in range(n_grp):
            ii = np.flatnonzero(groups == k)
            vc["a"][k] = vca[ii][:, None]
            vc["b"][k] = vcb[ii][:, None]
        rslt = MixedLM(endog, exog, groups=groups,
                       exog_re=exog_re, exog_vc=vc).fit()
        rslt.profile_re(0, vtype='re', dist_low=1, num_low=3, dist_high=1,
                        num_high=3)
        rslt.profile_re('b', vtype='vc', dist_low=0.5, num_low=3,
                        dist_high=0.5, num_high=3)

    # Fails on old versions of scipy/numpy
    @dec.skipif(old_scipy)
    def test_vcomp_1(self):
        # Fit the same model using constrained random effects and
        # variance components.

        np.random.seed(4279)
        exog = np.random.normal(size=(400, 1))
        exog_re = np.random.normal(size=(400, 2))
        groups = np.kron(np.arange(100), np.ones(4))
        slopes = np.random.normal(size=(100, 2))
        slopes[:, 1] *= 2
        slopes = np.kron(slopes, np.ones((4, 1))) * exog_re
        errors = slopes.sum(1) + np.random.normal(size=400)
        endog = exog.sum(1) + errors

        free = MixedLMParams(1, 2, 0)
        free.fe_params = np.ones(1)
        free.cov_re = np.eye(2)
        free.vcomp = np.zeros(0)

        model1 = MixedLM(endog, exog, groups, exog_re=exog_re)
        result1 = model1.fit(free=free)

        exog_vc = {"a": {}, "b": {}}
        for k, group in enumerate(model1.group_labels):
            ix = model1.row_indices[group]
            exog_vc["a"][group] = exog_re[ix, 0:1]
            exog_vc["b"][group] = exog_re[ix, 1:2]
        model2 = MixedLM(endog, exog, groups, exog_vc=exog_vc)
        result2 = model2.fit()
        result2.summary()

        assert_allclose(result1.fe_params, result2.fe_params, atol=1e-4)
        assert_allclose(np.diag(result1.cov_re),
                        result2.vcomp, atol=1e-2, rtol=1e-4)
        assert_allclose(result1.bse[[0, 1, 3]],
                        result2.bse, atol=1e-2, rtol=1e-2)

    def test_vcomp_2(self):
        # Simulated data comparison to R

        np.random.seed(6241)
        n = 1600
        exog = np.random.normal(size=(n, 2))
        groups = np.kron(np.arange(n / 16), np.ones(16))

        # Build up the random error vector
        errors = 0

        # The random effects
        exog_re = np.random.normal(size=(n, 2))
        slopes = np.random.normal(size=(n // 16, 2))
        slopes = np.kron(slopes, np.ones((16, 1))) * exog_re
        errors += slopes.sum(1)

        # First variance component
        subgroups1 = np.kron(np.arange(n / 4), np.ones(4))
        errors += np.kron(2 * np.random.normal(size=n // 4), np.ones(4))

        # Second variance component
        subgroups2 = np.kron(np.arange(n / 2), np.ones(2))
        errors += np.kron(2 * np.random.normal(size=n // 2), np.ones(2))

        # iid errors
        errors += np.random.normal(size=n)

        endog = exog.sum(1) + errors

        df = pd.DataFrame(index=range(n))
        df["y"] = endog
        df["groups"] = groups
        df["x1"] = exog[:, 0]
        df["x2"] = exog[:, 1]
        df["z1"] = exog_re[:, 0]
        df["z2"] = exog_re[:, 1]
        df["v1"] = subgroups1
        df["v2"] = subgroups2

        # Equivalent model in R:
        # df.to_csv("tst.csv")
        # model = lmer(y ~ x1 + x2 + (0 + z1 + z2 | groups) + (1 | v1) + (1 |
        # v2), df)

        vcf = {"a": "0 + C(v1)", "b": "0 + C(v2)"}
        model1 = MixedLM.from_formula("y ~ x1 + x2", groups=groups,
                                      re_formula="0+z1+z2",
                                      vc_formula=vcf, data=df)
        result1 = model1.fit()

        # Compare to R
        assert_allclose(result1.fe_params, [
                        0.16527, 0.99911, 0.96217], rtol=1e-4)
        assert_allclose(result1.cov_re, [
                        [1.244,  0.146], [0.146, 1.371]], rtol=1e-3)
        assert_allclose(result1.vcomp, [4.024, 3.997], rtol=1e-3)
        assert_allclose(result1.bse.iloc[0:3], [
                        0.12610, 0.03938, 0.03848], rtol=1e-3)

    @dec.skipif(old_scipy)
    def test_vcomp_3(self):
        # Test a model with vcomp but no other random effects, using formulas.

        np.random.seed(4279)
        x1 = np.random.normal(size=400)
        groups = np.kron(np.arange(100), np.ones(4))
        slopes = np.random.normal(size=100)
        slopes = np.kron(slopes, np.ones(4)) * x1
        y = slopes + np.random.normal(size=400)
        vc_fml = {"a": "0 + x1"}
        df = pd.DataFrame({"y": y, "x1": x1, "groups": groups})

        model = MixedLM.from_formula("y ~ 1", groups="groups",
                                     vc_formula=vc_fml,
                                     data=df)
        result = model.fit()
        result.summary()

        assert_allclose(result.resid.iloc[0:4],
                        np.r_[-1.180753, 0.279966, 0.578576, -0.667916],
                        rtol=1e-3)
        assert_allclose(result.fittedvalues.iloc[0:4],
                        np.r_[-0.101549, 0.028613, -0.224621, -0.126295],
                        rtol=1e-3)

    @dec.skipif(old_scipy)
    def test_sparse(self):

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        rdir = os.path.join(cur_dir, 'results')
        fname = os.path.join(rdir, 'pastes.csv')

        # Dense
        data = pd.read_csv(fname)
        vcf = {"cask": "0 + cask"}
        model = MixedLM.from_formula("strength ~ 1", groups="batch",
                                     re_formula="1", vc_formula=vcf,
                                     data=data)
        result = model.fit()

        # Sparse
        model2 = MixedLM.from_formula("strength ~ 1", groups="batch",
                                      re_formula="1", vc_formula=vcf,
                                      use_sparse=True, data=data)
        result2 = model2.fit()

        assert_allclose(result.params, result2.params)
        assert_allclose(result.bse, result2.bse)

    def test_pastes_vcomp(self):
        # pastes data from lme4
        #
        # Fit in R using formula:
        #
        # strength ~ (1|batch) + (1|batch:cask)

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        rdir = os.path.join(cur_dir, 'results')
        fname = os.path.join(rdir, 'pastes.csv')

        # REML
        data = pd.read_csv(fname)
        vcf = {"cask": "0 + cask"}
        model = MixedLM.from_formula("strength ~ 1", groups="batch",
                                     re_formula="1", vc_formula=vcf,
                                     data=data)
        result = model.fit()

        assert_allclose(result.fe_params.iloc[0], 60.0533, rtol=1e-3)
        assert_allclose(result.bse.iloc[0], 0.6769, rtol=1e-3)
        assert_allclose(result.cov_re.iloc[0, 0], 1.657, rtol=1e-3)
        assert_allclose(result.scale, 0.678, rtol=1e-3)
        assert_allclose(result.llf, -123.49, rtol=1e-1)
        assert_equal(result.aic, np.nan)  # don't provide aic/bic with REML
        assert_equal(result.bic, np.nan)

        resid = np.r_[0.17133538, -0.02866462, -
                      1.08662875, 1.11337125, -0.12093607]
        assert_allclose(result.resid[0:5], resid, rtol=1e-3)

        fit = np.r_[62.62866, 62.62866, 61.18663, 61.18663, 62.82094]
        assert_allclose(result.fittedvalues[0:5], fit, rtol=1e-4)

        # ML
        data = pd.read_csv(fname)
        vcf = {"cask": "0 + cask"}
        model = MixedLM.from_formula("strength ~ 1", groups="batch",
                                     re_formula="1", vc_formula=vcf,
                                     data=data)
        result = model.fit(reml=False)
        assert_allclose(result.fe_params.iloc[0], 60.0533, rtol=1e-3)
        assert_allclose(result.bse.iloc[0], 0.642, rtol=1e-3)
        assert_allclose(result.cov_re.iloc[0, 0], 1.199, rtol=1e-3)
        assert_allclose(result.scale, 0.67799, rtol=1e-3)
        assert_allclose(result.llf, -123.997, rtol=1e-1)
        assert_allclose(result.aic, 255.9944, rtol=1e-3)
        assert_allclose(result.bic, 264.3718, rtol=1e-3)

    def test_vcomp_formula(self):

        np.random.seed(6241)
        n = 800
        exog = np.random.normal(size=(n, 2))
        exog[:, 0] = 1
        ex_vc = []
        groups = np.kron(np.arange(n / 4), np.ones(4))
        errors = 0
        exog_re = np.random.normal(size=(n, 2))
        slopes = np.random.normal(size=(n // 4, 2))
        slopes = np.kron(slopes, np.ones((4, 1))) * exog_re
        errors += slopes.sum(1)
        ex_vc = np.random.normal(size=(n, 4))
        slopes = np.random.normal(size=(n // 4, 4))
        slopes[:, 2:] *= 2
        slopes = np.kron(slopes, np.ones((4, 1))) * ex_vc
        errors += slopes.sum(1)
        errors += np.random.normal(size=n)
        endog = exog.sum(1) + errors

        exog_vc = {"a": {}, "b": {}}
        for k, group in enumerate(range(int(n / 4))):
            ix = np.flatnonzero(groups == group)
            exog_vc["a"][group] = ex_vc[ix, 0:2]
            exog_vc["b"][group] = ex_vc[ix, 2:]
        model1 = MixedLM(endog, exog, groups, exog_re=exog_re, exog_vc=exog_vc)
        result1 = model1.fit()

        df = pd.DataFrame(exog[:, 1:], columns=["x1", ])
        df["y"] = endog
        df["re1"] = exog_re[:, 0]
        df["re2"] = exog_re[:, 1]
        df["vc1"] = ex_vc[:, 0]
        df["vc2"] = ex_vc[:, 1]
        df["vc3"] = ex_vc[:, 2]
        df["vc4"] = ex_vc[:, 3]
        vc_formula = {"a": "0 + vc1 + vc2", "b": "0 + vc3 + vc4"}
        model2 = MixedLM.from_formula("y ~ x1", groups=groups,
                                      re_formula="0 + re1 + re2",
                                      vc_formula=vc_formula, data=df)
        result2 = model2.fit()

        assert_allclose(result1.fe_params, result2.fe_params, rtol=1e-8)
        assert_allclose(result1.cov_re, result2.cov_re, rtol=1e-8)
        assert_allclose(result1.vcomp, result2.vcomp, rtol=1e-8)
        assert_allclose(result1.params, result2.params, rtol=1e-8)
        assert_allclose(result1.bse, result2.bse, rtol=1e-8)

    def test_formulas(self):
        np.random.seed(2410)
        exog = np.random.normal(size=(300, 4))
        exog_re = np.random.normal(size=300)
        groups = np.kron(np.arange(100), [1, 1, 1])
        g_errors = exog_re * np.kron(np.random.normal(size=100),
                                     [1, 1, 1])
        endog = exog.sum(1) + g_errors + np.random.normal(size=300)

        mod1 = MixedLM(endog, exog, groups, exog_re)
        # test the names
        assert_(mod1.data.xnames == ["x1", "x2", "x3", "x4"])
        assert_(mod1.data.exog_re_names == ["x_re1"])
        assert_(mod1.data.exog_re_names_full == ["x_re1 RE"])
        rslt1 = mod1.fit()

        # Fit with a formula, passing groups as the actual values.
        df = pd.DataFrame({"endog": endog})
        for k in range(exog.shape[1]):
            df["exog%d" % k] = exog[:, k]
        df["exog_re"] = exog_re
        fml = "endog ~ 0 + exog0 + exog1 + exog2 + exog3"
        re_fml = "0 + exog_re"
        mod2 = MixedLM.from_formula(fml, df, re_formula=re_fml,
                                    groups=groups)

        assert_(mod2.data.xnames == ["exog0", "exog1", "exog2", "exog3"])
        assert_(mod2.data.exog_re_names == ["exog_re"])
        assert_(mod2.data.exog_re_names_full == ["exog_re RE"])

        rslt2 = mod2.fit()
        assert_almost_equal(rslt1.params, rslt2.params)

        # Fit with a formula, passing groups as the variable name.
        df["groups"] = groups
        mod3 = MixedLM.from_formula(fml, df, re_formula=re_fml,
                                    groups="groups")
        assert_(mod3.data.xnames == ["exog0", "exog1", "exog2", "exog3"])
        assert_(mod3.data.exog_re_names == ["exog_re"])
        assert_(mod3.data.exog_re_names_full == ["exog_re RE"])

        rslt3 = mod3.fit(start_params=rslt2.params)
        assert_allclose(rslt1.params, rslt3.params, rtol=1e-4)

        # Check default variance structure with non-formula model
        # creation, also use different exog_re that produces a zero
        # estimated variance parameter.
        exog_re = np.ones(len(endog), dtype=np.float64)
        mod4 = MixedLM(endog, exog, groups, exog_re)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rslt4 = mod4.fit()
        from statsmodels.formula.api import mixedlm
        mod5 = mixedlm(fml, df, groups="groups")
        assert_(mod5.data.exog_re_names == ["groups"])
        assert_(mod5.data.exog_re_names_full == ["groups RE"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rslt5 = mod5.fit()
        assert_almost_equal(rslt4.params, rslt5.params)

    @dec.skipif(old_scipy)
    def test_regularized(self):

        np.random.seed(3453)
        exog = np.random.normal(size=(400, 5))
        groups = np.kron(np.arange(100), np.ones(4))
        expected_endog = exog[:, 0] - exog[:, 2]
        endog = expected_endog +\
            np.kron(np.random.normal(size=100), np.ones(4)) +\
            np.random.normal(size=400)

        # L1 regularization
        md = MixedLM(endog, exog, groups)
        mdf1 = md.fit_regularized(alpha=1.)
        mdf1.summary()

        # L1 regularization
        md = MixedLM(endog, exog, groups)
        mdf2 = md.fit_regularized(alpha=10 * np.ones(5))
        mdf2.summary()

        # L2 regularization
        pen = penalties.L2()
        mdf3 = md.fit_regularized(method=pen, alpha=0.)
        mdf3.summary()

        # L2 regularization
        pen = penalties.L2()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdf4 = md.fit_regularized(method=pen, alpha=100.)
        mdf4.summary()

        # Pseudo-Huber regularization
        pen = penalties.PseudoHuber(0.3)
        mdf5 = md.fit_regularized(method=pen, alpha=1.)
        mdf5.summary()

    def do1(self, reml, irf, ds_ix):

        # No need to check independent random effects when there is
        # only one of them.
        if irf and ds_ix < 6:
            return

        irfs = "irf" if irf else "drf"
        meth = "reml" if reml else "ml"

        rslt = R_Results(meth, irfs, ds_ix)

        # Fit the model
        md = MixedLM(rslt.endog, rslt.exog_fe, rslt.groups,
                     rslt.exog_re)
        if not irf:  # Free random effects covariance
            if np.any(np.diag(rslt.cov_re_r) < 1e-5):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mdf = md.fit(gtol=1e-7, reml=reml)
            else:
                mdf = md.fit(gtol=1e-7, reml=reml)
        else:  # Independent random effects
            k_fe = rslt.exog_fe.shape[1]
            k_re = rslt.exog_re.shape[1]
            free = MixedLMParams(k_fe, k_re, 0)
            free.fe_params = np.ones(k_fe)
            free.cov_re = np.eye(k_re)
            free.vcomp = np.array([])
            if np.any(np.diag(rslt.cov_re_r) < 1e-5):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mdf = md.fit(reml=reml, gtol=1e-7, free=free)
            else:
                mdf = md.fit(reml=reml, gtol=1e-7, free=free)

        assert_almost_equal(mdf.fe_params, rslt.coef, decimal=4)
        assert_almost_equal(mdf.cov_re, rslt.cov_re_r, decimal=4)
        assert_almost_equal(mdf.scale, rslt.scale_r, decimal=4)

        k_fe = md.k_fe
        assert_almost_equal(rslt.vcov_r, mdf.cov_params()[0:k_fe, 0:k_fe],
                            decimal=3)

        assert_almost_equal(mdf.llf, rslt.loglike[0], decimal=2)

        # Not supported in R except for independent random effects
        if not irf:
            assert_almost_equal(mdf.random_effects[0], rslt.ranef_postmean,
                                decimal=3)
            assert_almost_equal(mdf.random_effects_cov[0],
                                rslt.ranef_condvar,
                                decimal=3)

    # Run all the tests against R
    def test_r(self):

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        rdir = os.path.join(cur_dir, 'results')
        fnames = os.listdir(rdir)
        fnames = [x for x in fnames if x.startswith("lme")
                  and x.endswith(".csv")]

        for fname in fnames:
            for reml in False, True:
                for irf in False, True:
                    ds_ix = int(fname[3:5])

                    yield self.do1, reml, irf, ds_ix


def test_mixed_lm_wrapper():
    # a bit more complicated model to test
    np.random.seed(2410)
    exog = np.random.normal(size=(300, 4))
    exog_re = np.random.normal(size=300)
    groups = np.kron(np.arange(100), [1, 1, 1])
    g_errors = exog_re * np.kron(np.random.normal(size=100),
                                 [1, 1, 1])
    endog = exog.sum(1) + g_errors + np.random.normal(size=300)

    # Fit with a formula, passing groups as the actual values.
    df = pd.DataFrame({"endog": endog})
    for k in range(exog.shape[1]):
        df["exog%d" % k] = exog[:, k]
    df["exog_re"] = exog_re
    fml = "endog ~ 0 + exog0 + exog1 + exog2 + exog3"
    re_fml = "~ exog_re"
    mod2 = MixedLM.from_formula(fml, df, re_formula=re_fml,
                                groups=groups)
    result = mod2.fit()
    result.summary()

    xnames = ["exog0", "exog1", "exog2", "exog3"]
    re_names = ["Intercept", "exog_re"]
    re_names_full = ["Intercept RE", "Intercept RE x exog_re RE",
                     "exog_re RE"]

    assert_(mod2.data.xnames == xnames)
    assert_(mod2.data.exog_re_names == re_names)
    assert_(mod2.data.exog_re_names_full == re_names_full)

    params = result.params
    assert_(params.index.tolist() == xnames + re_names_full)
    bse = result.bse
    assert_(bse.index.tolist() == xnames + re_names_full)
    tvalues = result.tvalues
    assert_(tvalues.index.tolist() == xnames + re_names_full)
    cov_params = result.cov_params()
    assert_(cov_params.index.tolist() == xnames + re_names_full)
    assert_(cov_params.columns.tolist() == xnames + re_names_full)
    fe = result.fe_params
    assert_(fe.index.tolist() == xnames)
    bse_fe = result.bse_fe
    assert_(bse_fe.index.tolist() == xnames)
    cov_re = result.cov_re
    assert_(cov_re.index.tolist() == re_names)
    assert_(cov_re.columns.tolist() == re_names)
    cov_re_u = result.cov_re_unscaled
    assert_(cov_re_u.index.tolist() == re_names)
    assert_(cov_re_u.columns.tolist() == re_names)
    bse_re = result.bse_re
    assert_(bse_re.index.tolist() == re_names_full)

def test_random_effects():

    np.random.seed(23429)

    # Default model (random effects only)
    ngrp = 100
    gsize = 10
    rsd = 2
    gsd = 3
    mn = gsd*np.random.normal(size=ngrp)
    gmn = np.kron(mn, np.ones(gsize))
    y = gmn + rsd*np.random.normal(size=ngrp*gsize)
    gr = np.kron(np.arange(ngrp), np.ones(gsize))
    x = np.ones(ngrp * gsize)
    model = MixedLM(y, x, groups=gr)
    result = model.fit()
    re = result.random_effects
    assert_(isinstance(re, dict))
    assert_(len(re) == ngrp)
    assert_(isinstance(re[0], pd.Series))
    assert_(len(re[0]) == 1)

    # Random intercept only, set explicitly
    model = MixedLM(y, x, exog_re=x, groups=gr)
    result = model.fit()
    re = result.random_effects
    assert_(isinstance(re, dict))
    assert_(len(re) == ngrp)
    assert_(isinstance(re[0], pd.Series))
    assert_(len(re[0]) == 1)

    # Random intercept and slope
    xr = np.random.normal(size=(ngrp*gsize, 2))
    xr[:, 0] = 1
    qp = np.linspace(-1, 1, gsize)
    xr[:, 1] = np.kron(np.ones(ngrp), qp)
    model = MixedLM(y, x, exog_re=xr, groups=gr)
    result = model.fit()
    re = result.random_effects
    assert_(isinstance(re, dict))
    assert_(len(re) == ngrp)
    assert_(isinstance(re[0], pd.Series))
    assert_(len(re[0]) == 2)

if __name__ == "__main__":

    import nose

    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
