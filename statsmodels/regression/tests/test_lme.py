import warnings
import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM, MixedLMParams
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
                           dec, assert_)
from . import lme_r_results
from statsmodels.base import _penalties as penalties
import os
import csv

# TODO: add tests with unequal group sizes

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
        self.endog = data[:,header.index("endog")]
        self.groups = data[:,header.index("groups")]
        ii = [i for i,x in enumerate(header) if
              x.startswith("exog_fe")]
        self.exog_fe = data[:,ii]
        ii = [i for i,x in enumerate(header) if
              x.startswith("exog_re")]
        self.exog_re = data[:,ii]


class TestMixedLM(object):

    # Test analytic scores using numeric differentiation
    # TODO: better checks on Hessian
    @dec.slow
    def test_compare_numdiff(self):

        import statsmodels.tools.numdiff as nd

        n_grp = 200
        grpsize = 5
        k_fe = 3
        k_re = 2

        for jl in 0,1:
            for reml in False,True:
                for cov_pen_wt in 0,10:

                    cov_pen = penalties.PSD(cov_pen_wt)

                    np.random.seed(3558)
                    exog_fe = np.random.normal(size=(n_grp*grpsize, k_fe))
                    exog_re = np.random.normal(size=(n_grp*grpsize, k_re))
                    exog_re[:, 0] = 1
                    slopes = np.random.normal(size=(n_grp, k_re))
                    slopes = np.kron(slopes, np.ones((grpsize,1)))
                    re_values = (slopes * exog_re).sum(1)
                    err = np.random.normal(size=n_grp*grpsize)
                    endog = exog_fe.sum(1) + re_values + err
                    groups = np.kron(range(n_grp), np.ones(grpsize))

                    if jl == 0:
                        md = MixedLM(endog, exog_fe, groups, exog_re)
                        score = lambda x: -md.score_sqrt(x)
                        hessian = lambda x : -md.hessian_sqrt(x)
                    else:
                        md = MixedLM(endog, exog_fe, groups, exog_re,
                                     use_sqrt=False)
                        score = lambda x: -md.score_full(x)
                        hessian = lambda x: -md.hessian_full(x)
                    md.reml = reml
                    md.cov_pen = cov_pen
                    loglike = lambda x: -md.loglike(x)
                    rslt = md.fit()

                    # Test the score at several points.
                    for kr in range(5):
                        fe_params = np.random.normal(size=k_fe)
                        cov_re = np.random.normal(size=(k_re, k_re))
                        cov_re = np.dot(cov_re.T, cov_re)
                        params = MixedLMParams.from_components(fe_params,
                                                               cov_re)
                        if jl == 0:
                            params_vec = params.get_packed()
                        else:
                            params_vec = params.get_packed(use_sqrt=False)

                        # Check scores
                        gr = score(params)
                        ngr = nd.approx_fprime(params_vec, loglike)
                        assert_allclose(gr, ngr, rtol=1e-2)

                        # Hessian matrices don't agree well away from
                        # the MLE.
                        #if cov_pen_wt == 0:
                        #    hess = hessian(params)
                        #    nhess = nd.approx_hess(params_vec, loglike)
                        #    assert_allclose(hess, nhess, rtol=1e-2)

                    # Check Hessian matrices at the MLE.
                    if cov_pen_wt == 0:
                        hess = hessian(rslt.params_object)
                        params_vec = rslt.params_object.get_packed()
                        nhess = nd.approx_hess(params_vec, loglike)
                        assert_allclose(hess, nhess, rtol=1e-2)

    def test_default_re(self):

        np.random.seed(3235)
        exog = np.random.normal(size=(300,4))
        groups = np.kron(np.arange(100), [1,1,1])
        g_errors = np.kron(np.random.normal(size=100), [1,1,1])
        endog = exog.sum(1) + g_errors + np.random.normal(size=300)
        mdf1 = MixedLM(endog, exog, groups).fit()
        mdf2 = MixedLM(endog, exog, groups, np.ones(300)).fit()
        assert_almost_equal(mdf1.params, mdf2.params, decimal=8)

    def test_history(self):

        np.random.seed(3235)
        exog = np.random.normal(size=(300,4))
        groups = np.kron(np.arange(100), [1,1,1])
        g_errors = np.kron(np.random.normal(size=100), [1,1,1])
        endog = exog.sum(1) + g_errors + np.random.normal(size=300)
        mod = MixedLM(endog, exog, groups)
        rslt = mod.fit(full_output=True)
        assert_equal(hasattr(rslt, "hist"), True)

    def test_EM(self):

        np.random.seed(3298)
        exog = np.random.normal(size=(300,4))
        groups = np.kron(np.arange(100), [1,1,1])
        g_errors = np.kron(np.random.normal(size=100), [1,1,1])
        endog = exog.sum(1) + g_errors + np.random.normal(size=300)
        mdf1 = MixedLM(endog, exog, groups).fit(niter_em=10)

    def test_profile(self):
        # Smoke test
        np.random.seed(9814)
        k_fe = 4
        gsize = 3
        n_grp = 100
        exog = np.random.normal(size=(n_grp * gsize, k_fe))
        groups = np.kron(np.arange(n_grp), np.ones(gsize))
        g_errors = np.kron(np.random.normal(size=100), np.ones(gsize))
        endog = exog.sum(1) + g_errors + np.random.normal(size=n_grp * gsize)
        rslt = MixedLM(endog, exog, groups).fit(niter_em=10)
        prof = rslt.profile_re(0, dist_low=0.1, num_low=1, dist_high=0.1,
                               num_high=1)

    def test_formulas(self):
        np.random.seed(2410)
        exog = np.random.normal(size=(300,4))
        exog_re = np.random.normal(size=300)
        groups = np.kron(np.arange(100), [1,1,1])
        g_errors = exog_re * np.kron(np.random.normal(size=100),
                                     [1,1,1])
        endog = exog.sum(1) + g_errors + np.random.normal(size=300)

        mod1 = MixedLM(endog, exog, groups, exog_re)
        # test the names
        assert_(mod1.data.xnames == ["x1", "x2", "x3", "x4"])
        assert_(mod1.data.exog_re_names == ["Z1"])
        assert_(mod1.data.exog_re_names_full == ["Z1 RE"])

        rslt1 = mod1.fit()

        # Fit with a formula, passing groups as the actual values.
        df = pd.DataFrame({"endog": endog})
        for k in range(exog.shape[1]):
            df["exog%d" % k] = exog[:,k]
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
        # creation.
        exog_re = np.ones(len(endog), dtype=np.float64)
        mod4 = MixedLM(endog, exog, groups, exog_re)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rslt4 = mod4.fit(start_params=rslt2.params)
        from statsmodels.formula.api import mixedlm
        mod5 = mixedlm(fml, df, groups="groups")
        assert_(mod5.data.exog_re_names == ["Intercept"])
        assert_(mod5.data.exog_re_names_full == ["Intercept RE"])
        rslt5 = mod5.fit(start_params=rslt2.params)
        assert_almost_equal(rslt4.params, rslt5.params)

    def test_regularized(self):

        np.random.seed(3453)
        exog = np.random.normal(size=(400,5))
        groups = np.kron(np.arange(100), np.ones(4))
        expected_endog = exog[:,0] - exog[:,2]
        endog = expected_endog +\
                np.kron(np.random.normal(size=100), np.ones(4)) +\
                np.random.normal(size=400)

        # L1 regularization
        md = MixedLM(endog, exog, groups)
        mdf1 = md.fit_regularized(alpha=1.)
        mdf1.summary()

        # L1 regularization
        md = MixedLM(endog, exog, groups)
        mdf2 = md.fit_regularized(alpha=10*np.ones(5))
        mdf2.summary()

        # L2 regularization
        pen = penalties.L2()
        mdf3 = md.fit_regularized(method=pen, alpha=0.)
        mdf3.summary()

        # L2 regularization
        pen = penalties.L2()
        mdf4 = md.fit_regularized(method=pen, alpha=100.)
        mdf4.summary()

        # Pseudo-Huber regularization
        pen = penalties.PseudoHuber(0.3)
        mdf4 = md.fit_regularized(method=pen, alpha=1.)
        mdf4.summary()


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
        if not irf: # Free random effects covariance
            mdf = md.fit(gtol=1e-7, reml=reml)
        else: # Independent random effects
            k_fe = rslt.exog_fe.shape[1]
            k_re = rslt.exog_re.shape[1]
            free = MixedLMParams(k_fe, k_re)
            free.set_fe_params(np.ones(k_fe))
            free.set_cov_re(np.eye(k_re))
            mdf = md.fit(reml=reml, gtol=1e-7, free=free)

        assert_almost_equal(mdf.fe_params, rslt.coef, decimal=4)
        assert_almost_equal(mdf.cov_re, rslt.cov_re_r, decimal=4)
        assert_almost_equal(mdf.scale, rslt.scale_r, decimal=4)

        pf = rslt.exog_fe.shape[1]
        assert_almost_equal(rslt.vcov_r, mdf.cov_params()[0:pf,0:pf],
                            decimal=3)

        assert_almost_equal(mdf.llf, rslt.loglike[0], decimal=2)

        # Not supported in R
        if not irf:
            assert_almost_equal(mdf.random_effects.ix[0], rslt.ranef_postmean,
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
            for reml in False,True:
                for irf in False,True:
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
    smoke = result.summary()

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




if  __name__=="__main__":

    import nose

    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],
                   exit=False)
